"""Noild 配置与级联路径表。

对应任务包第四节（Noild 层）与第五节（三个模型与档位）。
版本：v0.1.0-dev（与主仓库一致，本阶段不升版）。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NoildConfig:
    """Noild 层配置。所有默认值均为「基础版」参数，训练后可调。"""

    hidden_size: int = 4096
    intermediate_size: int = 14336
    vocab_size: int = 65536
    hidden_act: str = "silu"

    # 领域与专家数量
    num_domains: int = 8  # 路由可判出的领域数 = 垂直专家数
    num_general_experts: int = 2  # 通用专家数量少
    top_vertical: int = 2  # 路由允许同时选 2 个垂直专家（上限）

    # 预生成专家：小而快
    draft_hidden_size: int = 1024  # 小于主 hidden_size
    draft_intermediate_size: int = 4096
    draft_num_tokens: int = 4  # 每次猜若干 token
    draft_adapter_rank: int = 8  # 按领域轻量适配器（LoRA 式低秩）

    # 通用专家 MTP
    mtp_num_tokens: int = 2  # MTP 头预测的未来 token 数

    # 合并专家窗口：首窗口小（2~3），逐步放大到 10 以上
    merge_window_start: int = 2
    merge_window_max: int = 10
    merge_window_growth_steps: int = 100  # 每多少步放大一次（线性增长）

    # 路由重判节奏：生成中每 N 个 token 重判一次
    router_rejudge_interval: int = 16  # 基础 N
    router_rejudge_max_interval: int = 64  # 负载大时扩大 N 的上限
    router_uncertainty_threshold: float = 0.9  # 归一化熵超过此值提前重判（0~1）
    router_uncertainty_calibrated: bool = False  # 仅在验证集校准后启用熵触发
    router_topic_shift_threshold: float = 0.5  # 与上次路由 hidden 余弦漂移超过此值提前重判

    # 训练：负载约束，防止垂直专家被饿死
    load_balance_coef: float = 0.01

    # 级联路径（见 get_cascade_path）。None 表示用默认 Veronoo Max 路径。
    path: tuple[str, ...] | None = None

    # 明确不采用：提前退出、动态深度（任务包第四节）。
    # 此处不提供开关，NoildLayer 不实现这两项。
    use_early_exit: bool = False
    use_dynamic_depth: bool = False

    def __post_init__(self):
        if self.use_early_exit or self.use_dynamic_depth:
            raise ValueError(
                "Noild 不采用提前退出和动态深度（任务包第四节），请保持为 False。"
            )
        if not 1 <= self.top_vertical <= 2:
            raise ValueError("top_vertical 允许为 1 或 2（路由可同时选 2 个垂直专家）。")
        if self.merge_window_start > self.merge_window_max:
            raise ValueError("merge_window_start 不得大于 merge_window_max。")
        if not 0.0 <= self.router_uncertainty_threshold <= 1.0:
            raise ValueError("router_uncertainty_threshold 必须位于 [0, 1]。")
        if self.draft_hidden_size >= self.hidden_size:
            raise ValueError("预生成专家应小于主 hidden_size（小而快）。")

    def merge_window_at_step(self, step: int) -> int:
        """合并窗口随训练步数增长：start -> max 线性放大。"""
        if self.merge_window_growth_steps <= 0:
            return self.merge_window_max
        grown = self.merge_window_start + step // self.merge_window_growth_steps
        return min(grown, self.merge_window_max)


# 级联路径表（任务包第五节）。路径元素为阶段名，NoildLayer 按顺序串联执行。
CASCADE_PATHS: dict[tuple[str, str], tuple[str, ...]] = {
    # Veronoo（Fusion）
    ("veronoo", "tutti"): ("router", "draft", "vertical", "merge"),
    ("veronoo", "ensemble"): ("router", "draft", "vertical", "merge"),
    ("veronoo", "duet"): ("router", "draft", "general"),
    # Soleinoo（Evolution）：末尾通用专家润色
    ("soleinoo", "tutti"): ("router", "draft", "general", "vertical", "merge", "general"),
    ("soleinoo", "ensemble"): ("router", "draft", "general", "vertical", "merge", "general"),
    ("soleinoo", "duet"): ("router", "draft", "general", "vertical", "merge"),
    # Lumenoo（Native）没有专家路径之分（任务包第五节），返回 None
    ("lumenoo", "saga"): (),
    ("lumenoo", "tale"): (),
    ("lumenoo", "poem"): (),
}


def get_cascade_path(model: str, tier: str) -> tuple[str, ...]:
    """按模型与档位返回级联路径。

    Args:
        model: "veronoo" / "soleinoo" / "lumenoo"（大小写不敏感）。
        tier: "tutti"/"ensemble"/"duet" 或 "saga"/"tale"/"poem"。

    Returns:
        阶段名元组；Lumenoo 返回空元组（无专家路径）。
    """
    key = (model.lower(), tier.lower())
    if key not in CASCADE_PATHS:
        raise KeyError(f"未知模型/档位组合: {key}，可选: {sorted(CASCADE_PATHS)}")
    return CASCADE_PATHS[key]


def scale_note() -> str:
    """适用范围说明（任务包第四节）。"""
    return (
        "生产部署不建议用于约 30B 以下的模型（专家拆分后每个更小）；"
        "小模型仅用于研究性验证专家分工与实现流程，不代表可部署能力或质量验收。"
    )
