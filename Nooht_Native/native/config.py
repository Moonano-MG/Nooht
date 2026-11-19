"""Native 线配置：Lumenoo 档位与外层优化层开关。

对应任务包第二节（Native）、第三节（外层优化层）、第五节（档位）。
版本：v0.1.0-dev（与主仓库一致，本阶段不升版）。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NativeConfig:
    """Native（Lumenoo）配置。

    Native 底座为 Jamba 黑盒：Mamba/注意力/MoE 层一行不改，
    只在外面加外层优化层（任务包第二节）。本配置只管外层，不改底座结构。
    """

    # 档位：Saga（Max）/ Tale（Plus）/ Poem（Flash）
    # Lumenoo 没有专家路径之分，三档按底座规格或 LoRA 规格大小划分（任务包第五节）
    tier: str = "saga"
    base_checkpoint: str = ""  # Jamba 小规格 checkpoint，规格开工时确认（任务包待确认 3）
    lora_rank: int = 64  # LoRA 规格大小区分档位
    lora_alpha: int = 128

    # 外层优化层 1：多硬件后端（优先级 CUDA > ROCm > XLA、oneAPI）
    backend: str = "auto"  # auto / cuda / rocm / xla / oneapi / cpu

    # 外层优化层 2：选择性记忆压缩
    memory_compression_enabled: bool = True
    compression_threshold_ratio: float = 0.5  # 上下文达到一半触发
    max_context_len: int = 8192

    # 外层优化层 3：受保护区
    persona_scheme: str = "B"  # A=两套权重 / B=可插拔（见 docs/persona_anchor_ab.md）
    instruction_reinject_every: int = 256  # 指令锚点回看间隔（token）
    thinking_memory_capacity: int = 32  # 思考记忆块容量上限（条目数）
    thinking_memory_min_interval: int = 16  # 最小更新间隔（十多个 token）

    # 外层优化层 4：复制机制（本阶段只交付格式与接口）
    copy_enabled: bool = True

    # 推理力度接口（任务包第七节 5）：思考预算（token 上限），None 表示不限
    thinking_budget: int | None = None

    # Noild 在 Native 中为可插拔（vendored 于 Nooht_Native/noild/）：
    # Native 主线默认不走 Noild 路径（任务包第二节：Native 没有 Noild），
    # 开启后供 Fusion/Evolution 复用与对照实验。
    noild_enabled: bool = False
    noild_path: tuple[str, ...] = ("router", "draft", "vertical", "merge")

    def __post_init__(self):
        if self.tier.lower() not in ("saga", "tale", "poem"):
            raise ValueError("Lumenoo 档位只能是 saga/tale/poem。")
        if self.persona_scheme not in ("A", "B"):
            raise ValueError("persona_scheme 只能是 A 或 B。")
        if not 0 < self.compression_threshold_ratio <= 1:
            raise ValueError("compression_threshold_ratio 需在 (0, 1] 内。")


# Lumenoo 三档默认（任务包第五节：按底座规格或 LoRA 规格大小划分）
LUMENOO_TIERS: dict[str, dict] = {
    "saga": {"lora_rank": 64, "thinking_budget": None, "note": "Max：全额 LoRA 规格"},
    "tale": {"lora_rank": 32, "thinking_budget": 4096, "note": "Plus：Max 的缩小版"},
    "poem": {"lora_rank": 16, "thinking_budget": 2048, "note": "Flash：精简蒸馏版"},
}


def lumenoo_config(tier: str, **overrides) -> NativeConfig:
    """按档位生成 NativeConfig，可用 overrides 覆盖。"""
    key = tier.lower()
    if key not in LUMENOO_TIERS:
        raise KeyError(f"未知 Lumenoo 档位: {tier}，可选: {sorted(LUMENOO_TIERS)}")
    defaults = {k: v for k, v in LUMENOO_TIERS[key].items() if k != "note"}
    defaults["tier"] = key
    defaults.update(overrides)
    return NativeConfig(**defaults)
