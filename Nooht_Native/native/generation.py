"""Native 生成循环：Jamba 黑盒 + 外层优化层。

此实现使用 tokenizer 和自回归 logits 逐 token 解码；不会向模型传字符串，
不会吞掉模型异常，也不伪造生成文本。底座保持黑盒：仅调用 model(input_ids=...)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .backend import Backend, resolve_backend
from .config import NativeConfig
from .copy_mechanism import CopyDecoder, CopySpan, parse_copy_instruction
from .memory_compression import SelectiveMemoryCompressor, split_segments
from .protected_zone import PersonaAdapter, ProtectedZone


@dataclass
class GenerationResult:
    text: str
    thinking_text: str = ""
    compressed: bool = False
    copied_spans: int = 0
    steps: int = 0


class NativeGenerator:
    """Native 生成器：tokenizer 驱动的贪心解码与外层机制编排。"""

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

    def _model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except (AttributeError, StopIteration):
            return torch.device("cpu")

    def _forward_logits(self, input_ids: torch.Tensor, step: int) -> torch.Tensor:
        if self.noild_layer is None:
            return self._extract_logits(self.model(input_ids=input_ids))
        output = self.model(input_ids=input_ids, output_hidden_states=True)
        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states is None and isinstance(output, dict):
            hidden_states = output.get("hidden_states")
        if hidden_states is None or len(hidden_states) == 0:
            raise TypeError("Noild 插件要求模型返回 output_hidden_states=True 的 hidden_states。")
        output_embeddings = self.model.get_output_embeddings() if hasattr(self.model, "get_output_embeddings") else getattr(self.model, "lm_head", None)
        if output_embeddings is None:
            raise TypeError("Noild 插件需要模型提供 get_output_embeddings() 或 lm_head。")
        transformed = self.noild_layer(hidden_states[-1], step=step, return_aux_logits=False)
        return output_embeddings(transformed.hidden)

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

    def _validated_copies(self, thinking_text: str) -> list[str]:
        """复制指令必须来自思考区，不能从用户 prompt 取得。

        若复制内容含代码围栏，来源中必须包含 FINAL_MARKER（<final>）；
        非代码文本仍执行范围和边界检查。
        """
        from .copy_mechanism import FINAL_MARKER

        copies = []
        for span in parse_copy_instruction(thinking_text):
            copied = self.copy_decoder.copy_span(thinking_text, span)
            prefix = thinking_text[: span.start]
            last_fence = max(prefix.rfind("```"), prefix.rfind("~~~"))
            if last_fence >= 0:
                marker = "~~~" if prefix.rfind("~~~") > prefix.rfind("```") else "```"
                is_code = prefix.count(marker) % 2 == 1
            else:
                is_code = "```" in copied or "~~~" in copied
            if is_code:
                code_end = thinking_text.find(FINAL_MARKER, span.end)
                if code_end < 0:
                    raise ValueError("代码复制区间未定稿：区间后必须有 <final> 标记。")
                span.finalized = True
                copied = self.copy_decoder.copy_span(thinking_text, span, is_code=True)
            copies.append(copied)
        return copies

    def generate(
        self,
        prompt_text: str,
        thinking_text: str = "",
        max_new_tokens: int = 64,
        thinking_budget: int | None = None,
    ) -> GenerationResult:
        """逐 token 自回归解码，并按需应用压缩、保护区与复制区间。

        复制指令只从 ``thinking_text`` 解析。模型缺失 tokenizer、输出异常或
        返回形状错误时显式报错，不返回占位字符串。
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

        # 思考记忆只记录明确传入的新思考内容，不制造“继续生成”等伪信息。
        memory = self.protected_zone.thinking_memory
        if thinking_text and memory.entries.get("conclusions", {}).get("latest") != thinking_text:
            memory.mark_new_info()
            if memory.can_update(0):
                memory.edit_entry("conclusions", "latest", thinking_text, 0)

        copies = self._validated_copies(thinking_text) if self.config.copy_enabled else []
        device = self._model_device()
        base_ids = self._encode(prompt_text, device)
        generated = torch.empty((base_ids.shape[0], 0), dtype=base_ids.dtype, device=device)
        steps = 0
        self.model.eval() if hasattr(self.model, "eval") else None
        if self.noild_layer is not None:
            self.noild_layer.eval()
        with torch.no_grad():
            for step in range(limit):
                current_ids = torch.cat((base_ids, generated), dim=1)
                generated_text = self._decode(generated[0]) if generated.shape[1] else ""
                context_text = prompt_text + generated_text
                # 每步从稳定的基础上下文重建受保护区，避免上一步组装结果重复追加。
                assembled = self.protected_zone.assemble(context_text, step=step)
                input_ids = self._encode(assembled, device)
                logits = self._forward_logits(input_ids, step)
                if logits.ndim != 3:
                    raise TypeError("model output projection must return (batch, sequence, vocab) logits")
                next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True).to(device)
                generated = torch.cat((generated, next_id), dim=1)
                steps += 1
                eos = getattr(self.tokenizer, "eos_token_id", None)
                if eos is not None and bool((next_id == eos).all()):
                    break

        generated_text = self._decode(generated[0]) if generated.shape[1] else ""
        text = generated_text
        if copies:
            text = "".join(copies) + generated_text
        return GenerationResult(
            text=text,
            thinking_text=thinking_text,
            compressed=compressed,
            copied_spans=len(copies),
            steps=steps,
        )


def load_jamba_native(checkpoint: str, config: NativeConfig | None = None):
    """加载 Jamba checkpoint 与 tokenizer；不自动下载或构建模型。"""
    cfg = config or NativeConfig(base_checkpoint=checkpoint)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError("加载 Jamba 需要安装 transformers。") from e
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    return model, tokenizer, cfg


def persona_consistency_check(generator: NativeGenerator, prompt: str) -> dict:
    """比较开启/关闭人设锚点时的实际生成，并如实报告是否受人设条件化。"""
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
    }
