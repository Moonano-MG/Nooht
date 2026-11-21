"""Native 生成循环：Jamba 黑盒 + tokenizer 解码、KV cache 与外层机制。"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import Callable

import torch

from .backend import Backend, resolve_backend
from .config import NativeConfig
from .copy_mechanism import CopyDecoder, CopySpan, FINAL_MARKER
from .memory_compression import SelectiveMemoryCompressor, split_segments
from .protected_zone import PersonaAdapter, ProtectedZone

_COPY_RE = re.compile(r"<copy:(\d+)-(\d+)>")


@dataclass
class GenerationResult:
    text: str
    thinking_text: str = ""
    compressed: bool = False
    copied_spans: int = 0
    steps: int = 0
    kv_cache_used: bool = False


class NativeGenerator:
    """Native 生成器：自回归贪心解码，优先使用模型原生 KV cache。"""

    def __init__(
        self,
        model: Callable,
        config: NativeConfig | None = None,
        protected_zone: ProtectedZone | None = None,
        compressor: SelectiveMemoryCompressor | None = None,
        noild_layer=None,
        tokenizer=None,
        persona_adapter: PersonaAdapter | None = None,
    ):
        self.model = model
        self.persona_adapter = persona_adapter
        self._persona_hooks = []
        if persona_adapter is not None and hasattr(model, "named_modules"):
            persona_adapter.register_targets(model)
            for name, module in model.named_modules():
                if name not in persona_adapter._name_to_key:
                    continue

                def apply_persona(_module, inputs, output, module_name=name):
                    adapter = self.persona_adapter
                    if not adapter.active or not isinstance(output, torch.Tensor):
                        return output
                    delta = adapter.delta(module_name, inputs[0])
                    if delta.shape != output.shape:
                        raise RuntimeError(f"PersonaAdapter shape mismatch at {module_name}")
                    return output + delta

                self._persona_hooks.append(module.register_forward_hook(apply_persona))
        self.tokenizer = tokenizer
        self.config = config or NativeConfig()
        self.backend: Backend = resolve_backend(self.config.backend)
        self.protected_zone = protected_zone or ProtectedZone()
        self.protected_zone.instruction.reinject_every = self.config.instruction_reinject_every
        self.protected_zone.thinking_memory.capacity = self.config.thinking_memory_capacity
        self.protected_zone.thinking_memory.min_interval = self.config.thinking_memory_min_interval
        self.compressor = compressor or SelectiveMemoryCompressor(
            threshold_ratio=self.config.compression_threshold_ratio,
            max_context_len=self.config.max_context_len,
            enabled=self.config.memory_compression_enabled,
        )
        self.copy_decoder = CopyDecoder()
        if self.config.noild_enabled and noild_layer is None:
            raise ValueError("noild_enabled=True 时必须传入 noild_layer。")
        self.noild_layer = noild_layer if self.config.noild_enabled else None
        self._memory_clock = 0

    def _model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except (AttributeError, StopIteration):
            return torch.device("cpu")

    def _encode(self, text: str, device: torch.device) -> torch.Tensor:
        if self.tokenizer is None:
            raise RuntimeError("真实生成需要 tokenizer；请传入 load_jamba_native 返回的 tokenizer。")
        encoded = self.tokenizer(text, return_tensors="pt")
        ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        if ids.ndim != 2 or ids.shape[1] == 0:
            raise ValueError("tokenizer 必须返回非空的二维 input_ids。")
        return ids.to(device)

    def _decode(self, ids: torch.Tensor) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _token_count(self, text: str) -> int:
        if self.tokenizer is None:
            raise RuntimeError("上下文 token 计数需要 tokenizer。")
        encoded = self.tokenizer(text, return_tensors="pt")
        ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        return int(ids.shape[-1])

    def _forward_signature(self):
        target = self.model.forward if hasattr(self.model, "forward") else self.model
        try:
            return inspect.signature(target)
        except (TypeError, ValueError):
            return None

    def _cache_call_support(self) -> tuple[bool, str]:
        signature = self._forward_signature()
        if signature is None:
            return False, "past_key_values"
        parameters = signature.parameters
        var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
        use_cache = "use_cache" in parameters or var_kwargs
        if "past_key_values" in parameters:
            cache_arg = "past_key_values"
        elif "cache" in parameters:
            cache_arg = "cache"
        else:
            cache_arg = "past_key_values"  # for **kwargs-style HF wrappers
        return use_cache and (cache_arg in parameters or var_kwargs), cache_arg

    @staticmethod
    def _extract_logits(output) -> torch.Tensor:
        if hasattr(output, "logits"):
            logits = output.logits
        elif isinstance(output, dict) and "logits" in output:
            logits = output["logits"]
        elif isinstance(output, (tuple, list)) and output:
            logits = output[0]
        else:
            logits = output
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise TypeError("model must return logits shaped (batch, sequence, vocab)")
        return logits

    @staticmethod
    def _extract_cache(output):
        cache = getattr(output, "past_key_values", None)
        if cache is None:
            cache = getattr(output, "cache", None)
        if cache is None and isinstance(output, dict):
            cache = output.get("past_key_values", output.get("cache"))
        return cache

    def _forward_logits(self, input_ids: torch.Tensor, step: int, past_cache=None, cache_arg: str = "past_key_values", use_cache: bool = False):
        kwargs = {"input_ids": input_ids}
        supports_cache, detected_arg = self._cache_call_support()
        if use_cache and supports_cache:
            kwargs["use_cache"] = True
            cache_arg = detected_arg
            if past_cache is not None:
                kwargs[cache_arg] = past_cache
        if self.noild_layer is None:
            output = self.model(**kwargs)
            return self._extract_logits(output), self._extract_cache(output)

        kwargs["output_hidden_states"] = True
        output = self.model(**kwargs)
        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states is None and isinstance(output, dict):
            hidden_states = output.get("hidden_states")
        if hidden_states is None or len(hidden_states) == 0:
            raise TypeError("Noild 插件要求模型返回 output_hidden_states=True 的 hidden_states。")
        output_embeddings = (
            self.model.get_output_embeddings()
            if hasattr(self.model, "get_output_embeddings")
            else getattr(self.model, "lm_head", None)
        )
        if output_embeddings is None:
            raise TypeError("Noild 插件需要模型提供 get_output_embeddings() 或 lm_head。")
        transformed = self.noild_layer(
            hidden_states[-1], step=step, return_aux_logits=False
        )
        return output_embeddings(transformed.hidden), self._extract_cache(output)

    def _render_model_copies(self, generated_text: str, thinking_text: str) -> tuple[str, int]:
        """只执行模型生成文本中的复制指令；用户 prompt 中的指令不参与解析。"""
        output = []
        cursor = 0
        count = 0
        for match in _COPY_RE.finditer(generated_text):
            output.append(generated_text[cursor:match.start()])
            span = CopySpan(int(match.group(1)), int(match.group(2)))
            copied = self.copy_decoder.copy_span(thinking_text, span)
            before = thinking_text[:span.start]
            last_fence = max(before.rfind("```"), before.rfind("~~~"))
            if last_fence >= 0:
                fence = "~~~" if before.rfind("~~~") > before.rfind("```") else "```"
                is_code = before.count(fence) % 2 == 1
            else:
                is_code = "```" in copied or "~~~" in copied
            if is_code:
                if FINAL_MARKER not in thinking_text[span.end:]:
                    raise ValueError("代码复制区间未定稿：区间后必须有 <final> 标记。")
                span.finalized = True
                copied = self.copy_decoder.copy_span(thinking_text, span, is_code=True)
            output.append(copied)
            cursor = match.end()
            count += 1
        output.append(generated_text[cursor:])
        return "".join(output), count

    def _assemble_initial_prompt(self, prompt_text: str) -> str:
        return self.protected_zone.assemble(prompt_text, step=0)

    def generate(
        self,
        prompt_text: str,
        thinking_text: str = "",
        max_new_tokens: int = 64,
        thinking_budget: int | None = None,
    ) -> GenerationResult:
        """逐 token 贪心解码，优先复用模型返回的 past_key_values/cache。

        若模型不支持/未返回 cache，则回退到完整上下文前向。复制区间指令必须
        出现在模型生成的文本中；其索引引用 ``thinking_text``。不接受 prompt 指令。
        """
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens 不得为负。")
        budget = thinking_budget if thinking_budget is not None else self.config.thinking_budget
        limit = max_new_tokens if budget is None else min(max_new_tokens, max(0, budget))
        segments = split_segments(prompt_text)
        compressed = False
        if self.compressor.should_compress(self._token_count(prompt_text)):
            segments = self.compressor.compress(segments)
            prompt_text = self.compressor.assemble(segments)
            compressed = True

        memory = self.protected_zone.thinking_memory
        if thinking_text:
            memory.queue_text(thinking_text)
            memory.apply_pending(self._memory_clock)
            self._memory_clock += 1

        device = self._model_device()
        base_prompt = self._assemble_initial_prompt(prompt_text)
        base_ids = self._encode(base_prompt, device)
        generated = torch.empty((base_ids.shape[0], 0), dtype=base_ids.dtype, device=device)
        steps = 0
        cache_supported, cache_arg = self._cache_call_support()
        cache = None
        pending_ids = base_ids
        cache_used = False
        self.model.eval() if hasattr(self.model, "eval") else None
        if self.noild_layer is not None:
            self.noild_layer.eval()

        with torch.no_grad():
            for step in range(limit):
                if cache is not None:
                    input_ids = pending_ids
                    # 只把指令锚点增量写入 cache，不重新编码整个会话。
                    if self.protected_zone.instruction.should_reinject(step):
                        anchor = self.protected_zone.instruction.get_anchor_text()
                        if anchor:
                            input_ids = torch.cat((input_ids, self._encode("\n" + anchor, device)), dim=1)
                    logits, new_cache = self._forward_logits(
                        input_ids, step, past_cache=cache, cache_arg=cache_arg, use_cache=True
                    )
                    cache = new_cache
                    cache_used = True
                elif cache_supported and step == 0:
                    logits, cache = self._forward_logits(
                        pending_ids, step, cache_arg=cache_arg, use_cache=True
                    )
                    cache_used = cache is not None
                else:
                    if step == 0:
                        input_ids = base_ids
                    else:
                        generated_text_so_far = self._decode(generated[0])
                        full_context = self.protected_zone.assemble(
                            prompt_text + generated_text_so_far, step=step
                        )
                        if self.protected_zone.instruction.should_reinject(step):
                            full_context += "\n" + self.protected_zone.instruction.get_anchor_text()
                        input_ids = self._encode(full_context, device)
                    logits, _ = self._forward_logits(input_ids, step, use_cache=False)
                if logits.ndim != 3:
                    raise TypeError("model output projection must return (batch, sequence, vocab) logits")
                next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True).to(device)
                generated = torch.cat((generated, next_id), dim=1)
                steps += 1
                pending_ids = next_id
                eos = getattr(self.tokenizer, "eos_token_id", None)
                if eos is not None and bool((next_id == eos).all()):
                    break

        raw_text = self._decode(generated[0]) if generated.shape[1] else ""
        text, copied_count = self._render_model_copies(raw_text, thinking_text) if self.config.copy_enabled else (raw_text, 0)
        return GenerationResult(
            text=text,
            thinking_text=thinking_text,
            compressed=compressed,
            copied_spans=copied_count,
            steps=steps,
            kv_cache_used=cache_used,
        )


def load_jamba_native(checkpoint: str, config: NativeConfig | None = None):
    """加载用户指定 checkpoint 与其 tokenizer，不自动下载或构建模型。

    本仓库不捆绑官方 Jamba 权重或 tokenizer。checkpoint 应指向用户可访问的
    本地目录或 Transformers 支持的模型标识；下载行为由调用方环境决定。
    """
    if not checkpoint:
        raise ValueError("必须显式提供本地 checkpoint 路径或模型标识；仓库不含权重/tokenizer。")
    cfg = config or NativeConfig(base_checkpoint=checkpoint)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError("加载 Jamba 需要安装 transformers。") from e
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    return model, tokenizer, cfg


def persona_consistency_check(generator: NativeGenerator, prompt: str) -> dict:
    """比较已启用适配器时有/无人设锚点的输出；未训练 adapter 不代表一致性通过。"""
    zone = generator.protected_zone
    saved = zone.persona_text
    try:
        with_persona = generator.generate(prompt).text
        zone.persona_text = ""
        without_persona = generator.generate(prompt).text
    finally:
        zone.persona_text = saved
    return {
        "with_persona": with_persona,
        "without_persona": without_persona,
        "persona_forced": bool(saved) and with_persona != without_persona,
        "adapter_trained": bool(generator.persona_adapter is not None and generator.persona_adapter.active),
    }
