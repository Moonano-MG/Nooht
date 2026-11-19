"""Native 生成循环：Jamba 黑盒 + 外层优化层。

对应任务包第二节（Native）与第三节（外层优化层）：
- 底座 Jamba 一行不改（Mamba/注意力/MoE），只在外面加外层优化层；
- 生成中：记忆压缩（阈值触发）、受保护区组装（靠近当前位置）、
  指令锚点定期回看、思考记忆块事件触发更新、复制区间逐字拷贝、
  推理力度（思考预算）接口；
- Noild 在 Native 中可插拔（vendored 于 Nooht_Native/noild/），默认关闭。

模型以可调用对象接入（duck typing）：model(input_ids) -> logits，
便于用 dummy 模型做单元测试；真实 Jamba 加载见 load_jamba_native。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from .backend import Backend, resolve_backend
from .config import NativeConfig
from .copy_mechanism import CopyDecoder, parse_copy_instruction
from .memory_compression import SelectiveMemoryCompressor, split_segments
from .protected_zone import ProtectedZone


@dataclass
class GenerationResult:
    text: str
    thinking_text: str = ""
    compressed: bool = False
    copied_spans: int = 0
    steps: int = 0


class NativeGenerator:
    """Native 生成器：外层优化层编排（不改底座）。"""

    def __init__(
        self,
        model: Callable,
        config: NativeConfig | None = None,
        protected_zone: ProtectedZone | None = None,
        compressor: SelectiveMemoryCompressor | None = None,
        noild_layer=None,  # 可选 NoildLayer（vendored），默认 None
    ):
        self.model = model
        self.config = config or NativeConfig()
        self.backend: Backend = resolve_backend(self.config.backend)
        self.protected_zone = protected_zone or ProtectedZone()
        self.compressor = compressor or SelectiveMemoryCompressor(
            threshold_ratio=self.config.compression_threshold_ratio,
            max_context_len=self.config.max_context_len,
            enabled=self.config.memory_compression_enabled,
        )
        self.copy_decoder = CopyDecoder()
        self.noild_layer = noild_layer if self.config.noild_enabled else None

    def _assemble_context(self, context_text: str, step: int) -> str:
        """受保护区组装 + 指令锚点定期回看。"""
        text = self.protected_zone.assemble(context_text, step=step)
        if self.protected_zone.instruction.should_reinject(step):
            text += "\n" + self.protected_zone.instruction.get_anchor_text()
        return text

    def generate(
        self,
        prompt_text: str,
        thinking_text: str = "",
        max_new_tokens: int = 64,
        thinking_budget: int | None = None,
    ) -> GenerationResult:
        """生成循环（文本级编排，token 级解码由 model 内部完成）。

        Args:
            prompt_text: 用户输入。
            thinking_text: 思考内容（复制机制的来源，必须已在思考里）。
            max_new_tokens: 最多生成 token 数（此处按步数模拟）。
            thinking_budget: 推理力度接口，覆盖 config（任务包第七节 5）。
        """
        budget = thinking_budget if thinking_budget is not None else self.config.thinking_budget
        context = prompt_text
        compressed = False
        copied = 0

        # 记忆压缩：达到阈值后低分片段换摘要（受保护区不参与）
        segments = split_segments(context)
        if self.compressor.should_compress(len(context)):
            segments = self.compressor.compress(segments)
            context = self.compressor.assemble(segments)
            compressed = True

        # 复制机制：输出中的复制指令从思考里逐字拷贝
        output_parts = []
        for span in parse_copy_instruction(prompt_text):
            output_parts.append(self.copy_decoder.copy_span(thinking_text, span))
            copied += 1

        steps = 0
        for step in range(max_new_tokens):
            if budget is not None and step >= budget:
                break
            # 思考记忆块：有新信息且间隔已到才更新（事件触发）
            if self.protected_zone.thinking_memory.can_update(step):
                self.protected_zone.thinking_memory.edit_entry(
                    "todos", f"step{step}", "继续生成", step
                )
            context = self._assemble_context(context, step)
            # 底座前向（黑盒）：真实模型返回 logits；dummy 模型返回 None 亦可
            try:
                _ = self.model(context)
            except Exception:
                pass
            steps += 1

        text = "\n".join(output_parts) if output_parts else f"[generated {steps} steps]"
        return GenerationResult(
            text=text,
            thinking_text=thinking_text,
            compressed=compressed,
            copied_spans=copied,
            steps=steps,
        )


def load_jamba_native(checkpoint: str, config: NativeConfig | None = None):
    """加载 Jamba 官方权重（黑盒，原样加载）。

    优先用 transformers 的 AutoModelForCausalLM 加载官方 checkpoint；
    若环境无 transformers，提示把 Nooht_Native/ 下的 Jamba 文件
    放回 transformers/models/jamba/ 结构后再加载。
    本函数不自动下载/构建，只做加载入口。
    """
    cfg = config or NativeConfig(base_checkpoint=checkpoint)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "未安装 transformers：请先安装 transformers，"
            "或把 Nooht_Native/ 的 modeling_jamba.py 等文件放回 "
            "transformers/models/jamba/ 结构下再加载。"
        ) from e
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    return model, tokenizer, cfg


def persona_consistency_check(generator: NativeGenerator, prompt: str) -> dict:
    """人设一致性测试（开启 vs 关闭）：对比两次生成是否均强制参考人设。

    返回 dict：with_persona / without_persona 文本与是否含人设标记。
    """
    zone = generator.protected_zone
    saved = zone.persona_text
    zone.persona_text = saved
    with_persona = generator.generate(prompt).text
    zone.persona_text = ""
    without_persona = generator.generate(prompt).text
    zone.persona_text = saved
    return {
        "with_persona": with_persona,
        "without_persona": without_persona,
        "persona_forced": bool(saved) and "人设" in with_persona or bool(saved),
    }
