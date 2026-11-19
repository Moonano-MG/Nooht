"""外层优化层 3：受保护区。

对应任务包第三节 3：把三类内容合并成同一个机制，只做一遍：
1. 人设锚点：推理时可插拔轻量模块（adapter/LoRA 式），同一套底座权重，
   加载即开启，卸载即关闭，每次生成强制参考。Lumenoo 必须完整支持。
   交付含方案 A（两套权重）与方案 B（可插拔）对比（docs/persona_anchor_ab.md）。
2. 指令锚点：系统提示和用户硬约束，生成中定期回看。
3. 思考记忆块：见任务包第七节（固定区域、结构化条目、增量编辑）。

共同机制：不参与记忆压缩；强制可见（CSA2 稀疏注意力中强制选中）；
固定位置（靠近当前位置，更新后重新排到上下文末尾附近）；
完整注意力层可直接读到（线性层状态有损）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


# ---------- 1. 人设锚点 ----------

class PersonaAdapter(nn.Module):
    """方案 B（可插拔）：LoRA 式低秩适配器，加载即开启，卸载即关闭。

    同一套底座权重，适配器只保存低秩增量。
    """

    def __init__(self, target_modules: list[str], rank: int = 16, alpha: int = 32):
        super().__init__()
        self.target_modules = target_modules
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        self.active = False

    def register_targets(self, model: nn.Module) -> None:
        """为命中的 Linear 层创建低秩参数（按模块名匹配）。"""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and any(
                t in name for t in self.target_modules
            ):
                if name not in self.lora_A:
                    self.lora_A[name] = nn.Parameter(
                        torch.empty(self.rank, module.in_features)
                    )
                    self.lora_B[name] = nn.Parameter(
                        torch.zeros(module.out_features, self.rank)
                    )
                    nn.init.kaiming_uniform_(self.lora_A[name], a=5**0.5)

    def load(self) -> None:
        self.active = True

    def unload(self) -> None:
        self.active = False

    def delta(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """计算某层低秩增量；未激活或无此层时返回 0。"""
        if not self.active or name not in self.lora_A:
            return torch.zeros_like(x)
        return self.scaling * (x @ self.lora_A[name].T @ self.lora_B[name].T)


class PersonaSchemeA:
    """方案 A（两套权重）：base 权重与人设合并权重各存一份。

    切换时整体替换 state_dict；优点是推理零额外开销，缺点是显存翻倍且不可插拔。
    """

    def __init__(self, base_state: dict, persona_state: dict):
        self.base_state = base_state
        self.persona_state = persona_state
        self.active = False

    def load(self) -> dict:
        self.active = True
        return self.persona_state

    def unload(self) -> dict:
        self.active = False
        return self.base_state


# ---------- 2. 指令锚点 ----------

@dataclass
class InstructionAnchor:
    """系统提示 + 用户硬约束，生成中定期回看，防止被冲淡。"""

    system_prompt: str = ""
    hard_constraints: list[str] = field(default_factory=list)
    reinject_every: int = 256

    def get_anchor_text(self) -> str:
        parts = [self.system_prompt] + [f"【硬约束】{c}" for c in self.hard_constraints]
        return "\n".join(p for p in parts if p)

    def should_reinject(self, step: int) -> bool:
        return step > 0 and step % self.reinject_every == 0


# ---------- 3. 思考记忆块 ----------

@dataclass
class ThinkingMemoryBlock:
    """思考记忆块：固定区域、限定容量、结构化条目。

    - 条目类别：变量、约束、待办、已确认结论；
    - 只做增量编辑（改哪条写哪条），不整块重写（无 rewrite_all 方法）；
    - 最小更新间隔十多个 token，有新信息才更新（事件触发）；
    - 与路由专家重判节奏对齐（共用 interval）；
    - 属于受保护区；更新质量靠训练，本阶段只交付格式与接口。
    """

    capacity: int = 32
    min_interval: int = 16
    entries: dict[str, dict[str, str]] = field(default_factory=dict)
    _last_update_step: int = -10**9
    _has_new_info: bool = False

    CATEGORIES = ("variables", "constraints", "todos", "conclusions")

    def can_update(self, step: int) -> bool:
        return self._has_new_info and (step - self._last_update_step) >= self.min_interval

    def mark_new_info(self) -> None:
        self._has_new_info = True

    def edit_entry(self, category: str, key: str, value: str, step: int) -> bool:
        """增量编辑单条。返回是否成功（容量/间隔/类别校验）。"""
        if category not in self.CATEGORIES:
            raise ValueError(f"未知条目类别: {category!r}，可选: {self.CATEGORIES}")
        if not self.can_update(step):
            return False
        bucket = self.entries.setdefault(category, {})
        if key not in bucket and len(bucket) >= self.capacity:
            return False  # 容量上限：拒绝新增（可改已有条目）
        bucket[key] = value
        self._last_update_step = step
        self._has_new_info = False
        return True

    def render(self) -> str:
        lines = ["【思考记忆块】"]
        for cat in self.CATEGORIES:
            for k, v in self.entries.get(cat, {}).items():
                lines.append(f"- {cat}/{k}: {v}")
        return "\n".join(lines)


# ---------- 受保护区统一管理 ----------

@dataclass
class ProtectedZone:
    """三类受保护内容合并成同一机制，只做一遍组装。"""

    persona_text: str = ""
    instruction: InstructionAnchor = field(default_factory=InstructionAnchor)
    thinking_memory: ThinkingMemoryBlock = field(default_factory=ThinkingMemoryBlock)
    force_visible: bool = True  # CSA2 稀疏注意力中强制选中，不允许被稀疏掉

    def assemble(self, context_text: str, step: int = 0) -> str:
        """固定位置：受保护内容排到上下文末尾附近（靠近当前位置）。

        更新后重新排到末尾附近，保证完整注意力层直接读到。
        """
        protected_parts = []
        if self.persona_text:
            protected_parts.append(f"【人设锚点】{self.persona_text}")
        anchor = self.instruction.get_anchor_text()
        if anchor:
            protected_parts.append(anchor)
        memory = self.thinking_memory.render()
        if memory.strip() != "【思考记忆块】":
            protected_parts.append(memory)
        if not protected_parts:
            return context_text
        return context_text + "\n" + "\n".join(protected_parts)

    def is_protected_text(self, text: str) -> bool:
        """供记忆压缩判断：受保护区内容不参与压缩。"""
        markers = ("【人设锚点】", "【硬约束】", "【思考记忆块】")
        return any(m in text for m in markers)
