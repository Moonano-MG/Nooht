"""NoildLayer：专家分角色、串联接力（不是同质并行分支）。

对应任务包第四节与第五节级联路径：
- Veronoo Max/Plus: router > draft > vertical > merge > output
- Veronoo Flash:    router > draft > general > output
- Soleinoo Max/Plus: router > draft > general > vertical > merge > general > output
- Soleinoo Flash:   router > draft > general > vertical > merge > output

不采用提前退出和动态深度（任务包第四节）：本层不实现这两项。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import NoildConfig, get_cascade_path
from .experts import DraftExpert, GeneralExpert, MergeExpert, VerticalExpert
from .loading import ExpertHeatManager
from .router import DomainRouter, RejudgeController, RouterOutput


@dataclass
class NoildOutput:
    hidden: torch.Tensor  # 级联后的 hidden
    router: RouterOutput  # 路由结果（仅领域）
    path_taken: tuple[str, ...]  # 实际走的路径
    draft_logits: torch.Tensor | None = None  # 预生成专家草稿 logits
    mtp_logits: torch.Tensor | None = None  # 通用专家 MTP logits（最后一次）
    rejudge_reason: str = "none"  # 本步重判原因


class NoildLayer(nn.Module):
    """Noild 稀疏层：按 path 串联执行各角色专家。

    Args:
        config: NoildConfig。
        path: 级联路径；None 时默认 Veronoo Tutti 路径。
            可用 get_cascade_path(model, tier) 生成。
        use_loading: 是否启用热度加载管理（默认 True，基础版）。
    """

    def __init__(
        self,
        config: NoildConfig,
        path: tuple[str, ...] | None = None,
        use_loading: bool = True,
    ):
        super().__init__()
        self.config = config
        self.path: tuple[str, ...] = (
            path if path is not None else get_cascade_path("veronoo", "tutti")
        )
        self.router = DomainRouter(config)
        self.rejudge = RejudgeController(config)
        self.draft = DraftExpert(config)
        self.generals = nn.ModuleList(
            [GeneralExpert(config) for _ in range(config.num_general_experts)]
        )
        self.verticals = nn.ModuleList(
            [VerticalExpert(config, d) for d in range(config.num_domains)]
        )
        self.merge = MergeExpert(config)
        self.loading: ExpertHeatManager | None = (
            ExpertHeatManager(config.num_domains, config.num_general_experts)
            if use_loading
            else None
        )
        self._general_cursor = 0  # 通用专家轮换（数量少，简单轮换）

    def _next_general(self) -> GeneralExpert:
        expert = self.generals[self._general_cursor % len(self.generals)]
        self._general_cursor += 1
        return expert

    def _run_verticals(
        self, hidden: torch.Tensor, router_out: RouterOutput
    ) -> torch.Tensor:
        """运行路由选中的垂直专家（1 或 2 个），按路由权重加权合并。"""
        # router_out.domain_ids: (..., top_vertical)
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_ids = router_out.domain_ids.reshape(-1, router_out.domain_ids.shape[-1])
        flat_w = router_out.top_weights.reshape(-1, router_out.top_weights.shape[-1])
        out = torch.zeros_like(flat_hidden)
        for k in range(flat_ids.shape[-1]):
            ids_k = flat_ids[:, k]
            for d in range(self.config.num_domains):
                mask = ids_k == d
                if not mask.any():
                    continue
                expert_out = self.verticals[d](flat_hidden[mask])
                out[mask] += expert_out * flat_w[mask, k : k + 1]
        return out.view_as(hidden)

    def forward(
        self,
        hidden: torch.Tensor,
        step: int = 0,
        domain_hint: torch.Tensor | None = None,
    ) -> NoildOutput:
        """串联执行。

        Args:
            hidden: (..., hidden_size) 输入 hidden。
            step: 生成步数（用于重判节奏与合并窗口）。
            domain_hint: 可选外部领域提示；提供时跳过路由重判（仍记录）。
        """
        # 1) 路由：每次用户输入必判；生成中按节奏重判
        # 按序列池化（只对序列维求均值，保留 batch 维），每条序列判一个领域
        pooled = hidden.mean(dim=-2) if hidden.dim() > 2 else hidden
        if domain_hint is not None:
            router_out = self.router.judge_user_input(pooled)
            reason = "hint"
        else:
            should, reason = self.rejudge.should_rejudge(step, hidden=hidden)
            router_out = self.router.judge_user_input(pooled)
            if should:
                self.rejudge.mark_rejudged(step, hidden)
        # 路由按序列池化判定，垂直/适配器按 token 执行：把领域广播到 token 级
        if router_out.domain_ids.dim() < hidden.dim():
            ids = router_out.domain_ids.unsqueeze(1)
            w = router_out.top_weights.unsqueeze(1)
            target = hidden.shape[:-1] + (ids.shape[-1],)
            router_out.domain_ids = ids.expand(target)
            router_out.top_weights = w.expand(target)
        if self.loading is not None:
            # 路由判出领域后异步预加载垂直专家
            domains = router_out.domain_ids.reshape(-1)[: self.config.top_vertical]
            self.loading.register_route([int(d) for d in domains.tolist()], step)
            self.loading.drain_preload()

        current = hidden
        draft_logits = None
        mtp_logits = None
        primary_source = "draft"  # 合并专家的 primary 来源（draft 或 general）
        primary_hidden = None

        # 2) 按路径串联（跳过 router 阶段，它已在上面执行）
        for stage in self.path:
            if stage == "router":
                continue
            elif stage == "draft":
                domain0 = router_out.domain_ids[..., 0]
                primary_hidden, draft_logits = self.draft(current, domain0)
                current = primary_hidden
                primary_source = "draft"
            elif stage == "general":
                current, mtp_logits = self._next_general()(current)
                primary_hidden = current
                primary_source = "general"
            elif stage == "vertical":
                current = self._run_verticals(current, router_out)
            elif stage == "merge":
                if primary_hidden is None:
                    raise RuntimeError("merge 阶段需要先经过 draft 或 general 阶段。")
                current = self.merge(
                    primary_hidden, current, source=primary_source, step=step
                )
            else:
                raise ValueError(f"未知路径阶段: {stage!r}，可选: router/draft/general/vertical/merge")

        return NoildOutput(
            hidden=current,
            router=router_out,
            path_taken=self.path,
            draft_logits=draft_logits,
            mtp_logits=mtp_logits,
            rejudge_reason=reason,
        )
