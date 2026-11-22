"""路由专家：只输出领域，不输出档位。

对应任务包第四节「路由专家」：
- 每次用户输入必须判定领域；
- 生成中每 N 个 token 重判一次，负载大时扩大 N；
- 输出不确定度突升或话题转变时提前重判；
- 只输出领域，不输出档位（本模块不提供档位字段）。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import NoildConfig


@dataclass
class RouterOutput:
    """路由输出：仅领域信息，无档位。"""

    domain_ids: torch.Tensor  # (..., top_vertical) 选中的领域 id
    domain_logits: torch.Tensor  # (..., num_domains)
    entropy: torch.Tensor  # (...,) 领域分布熵（不确定度）
    top_weights: torch.Tensor  # (..., top_vertical) 归一化后的领域权重


class DomainRouter(nn.Module):
    """路由专家：hidden -> 领域分布。

    输入为（池化后的）hidden states，输出领域 logits 与 top-k 领域。
    """

    def __init__(self, config: NoildConfig):
        super().__init__()
        self.config = config
        self.num_domains = config.num_domains
        self.top_vertical = config.top_vertical
        self.weight = nn.Linear(config.hidden_size, config.num_domains, bias=False)
        # Entropy thresholds are not meaningful until calibrated on a trained router.
        self.calibrated = bool(config.router_uncertainty_calibrated)

    def forward(self, hidden: torch.Tensor) -> RouterOutput:
        # hidden: (..., hidden_size)
        logits = self.weight(hidden)
        probs = F.softmax(logits.float(), dim=-1)
        entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1)
        top_weights, top_ids = torch.topk(probs, k=self.top_vertical, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return RouterOutput(
            domain_ids=top_ids,
            domain_logits=logits,
            entropy=entropy,
            top_weights=top_weights,
        )

    def judge_user_input(self, hidden: torch.Tensor) -> RouterOutput:
        """每次用户输入必须判定领域：与 forward 同义，语义上强制先判。"""
        return self.forward(hidden)


class RejudgeController:
    """重判节奏控制：每 N token 重判，负载大时扩大 N，异常时提前重判。

    与思考记忆块的更新节奏对齐（任务包第七节），避免两套节奏打架：
    两者共用同一 interval 配置（见 Native protected_zone）。
    """

    def __init__(self, config: NoildConfig):
        self.base_interval = config.router_rejudge_interval
        self.max_interval = config.router_rejudge_max_interval
        self.uncertainty_threshold = config.router_uncertainty_threshold
        self.topic_shift_threshold = config.router_topic_shift_threshold
        # Python math 保持该控制器可在 meta 初始化/参数量统计期间构造。
        self.max_entropy = max(math.log(config.num_domains), 1e-9)
        self.last_rejudge_step: int = -10**9
        self.last_uncertainty_rejudge_step: int = -10**9
        self.last_hidden: torch.Tensor | None = None
        self.current_interval: int = self.base_interval

    def update_load(self, load_factor: float) -> int:
        """负载大时扩大 N：interval = min(base * (1 + load), max)。"""
        if load_factor < 0:
            raise ValueError("load_factor 不得为负。")
        self.current_interval = int(
            min(self.base_interval * (1.0 + load_factor), self.max_interval)
        )
        return self.current_interval

    def should_rejudge(
        self,
        step: int,
        entropy: float | torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
    ) -> tuple[bool, str]:
        """判断当前步是否需要重判。返回 (是否重判, 原因)。

        原因: "interval"（常规）/ "uncertainty"（不确定度突升）/
        "topic_shift"（话题转变）/ "none"（不重判）。
        """
        # 不确定度突升：提前重判
        if entropy is not None:
            ent = float(entropy) if not isinstance(entropy, torch.Tensor) else float(entropy.mean())
            normalized_entropy = ent / self.max_entropy
            # High-entropy random/untrained routers can remain uncertain forever.
            # Throttle repeated early rejudges; normal interval/topic-shift still applies.
            uncertainty_ready = (
                step - self.last_uncertainty_rejudge_step >= self.current_interval
            )
            if normalized_entropy > self.uncertainty_threshold and uncertainty_ready:
                self.last_uncertainty_rejudge_step = step
                return True, "uncertainty"
        # 话题转变：hidden 与上次路由 hidden 余弦漂移过大
        if hidden is not None and self.last_hidden is not None:
            h = hidden.detach().float().mean(dim=tuple(range(hidden.dim() - 1)))
            prev = self.last_hidden.detach().float().mean(dim=tuple(range(self.last_hidden.dim() - 1)))
            cos = F.cosine_similarity(h, prev, dim=-1)
            drift = float(1.0 - cos.mean())
            if drift > self.topic_shift_threshold:
                return True, "topic_shift"
        # 常规间隔
        if step - self.last_rejudge_step >= self.current_interval:
            return True, "interval"
        return False, "none"

    def mark_rejudged(self, step: int, hidden: torch.Tensor | None = None) -> None:
        self.last_rejudge_step = step
        if hidden is not None:
            self.last_hidden = hidden.detach().clone()
