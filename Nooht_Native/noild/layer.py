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
        # 路由缓存（池化级）：重判间隔内复用，不每步重判
        self._cached_router: RouterOutput | None = None
        self._loading_clock = 0  # 单调加载时钟，不随新请求的 generation step 归零

    def set_load_factor(self, load_factor: float) -> int:
        """负载大时扩大重判间隔 N（供调用方按负载调用）。"""
        return self.rejudge.update_load(load_factor)

    def _general_for_domain(self, domain_id: int, stage_index: int = 0) -> GeneralExpert:
        """按领域与通用阶段确定性选择专家。

        同一领域在同一阶段可复现；Soleinoo 的前置与末尾通用阶段轮换到
        不同专家（当通用专家数大于 1），实现两个接力角色而非重复调用同一份权重。
        """
        return self.generals[(int(domain_id) + stage_index) % len(self.generals)]

    def _run_generals(
        self, hidden: torch.Tensor, router_out: RouterOutput, return_logits: bool,
        stage_index: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """按每个 token 的领域确定性选择通用专家，避免批次只用首样本领域。"""
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_ids = router_out.domain_ids[..., 0].reshape(-1)
        out = torch.zeros_like(flat_hidden)
        mtp = (
            hidden.new_zeros((flat_hidden.shape[0], self.config.mtp_num_tokens, self.config.vocab_size))
            if return_logits else None
        )
        for domain in range(self.config.num_domains):
            mask = flat_ids == domain
            if not bool(mask.any()):
                continue
            expert_out, expert_mtp = self._general_for_domain(domain, stage_index)(
                flat_hidden[mask], return_logits=return_logits
            )
            out[mask] = expert_out
            if mtp is not None:
                mtp[mask] = expert_mtp
        if mtp is not None:
            mtp = mtp.view(*hidden.shape[:-1], self.config.mtp_num_tokens, self.config.vocab_size)
        return out.view_as(hidden), mtp

    @staticmethod
    def _detach_router(router_out: RouterOutput) -> RouterOutput:
        return RouterOutput(
            domain_ids=router_out.domain_ids.detach(),
            domain_logits=router_out.domain_logits.detach(),
            entropy=router_out.entropy.detach(),
            top_weights=router_out.top_weights.detach(),
        )

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
        return_aux_logits: bool | None = None,
    ) -> NoildOutput:
        """串联执行。

        Args:
            hidden: (..., hidden_size) 输入 hidden。
            step: 生成步数（用于重判节奏与合并窗口）。
            domain_hint: 可选外部领域提示；提供时强制领域选择并更新路由缓存。
            return_aux_logits: 默认训练时计算 draft/MTP logits，推理时跳过未用辅助大词表头；
                可显式开启以用于验证/消融。
        """
        # step=0 表示新请求的首个输入：丢弃上一请求的缓存，保证每次用户输入必判。
        if not self.training and step == 0 and self._cached_router is not None:
            self._cached_router = None
            self.rejudge.last_rejudge_step = -10**9
            self.rejudge.last_uncertainty_rejudge_step = -10**9
            self.rejudge.last_hidden = None
        # 生成中按节奏重判，间隔内复用缓存；按序列池化并保留 batch 维。
        pooled = hidden.mean(dim=-2) if hidden.dim() > 2 else hidden
        if domain_hint is not None:
            pooled_router = self.router.judge_user_input(pooled)
            hint = torch.as_tensor(domain_hint, device=pooled.device, dtype=torch.long)
            hint = hint.clamp(0, self.config.num_domains - 1)
            if hint.ndim == 0:
                hint = hint.expand(pooled.shape[:-1])
            hint = hint.expand(pooled.shape[:-1])
            ids = torch.cat(
                (hint.unsqueeze(-1), pooled_router.domain_ids[..., 1:]), dim=-1
            )
            weights = pooled_router.top_weights.clone()
            weights[..., 0] = 1.0
            if weights.shape[-1] > 1:
                weights[..., 1:] = 0.0
            pooled_router = RouterOutput(ids, pooled_router.domain_logits, pooled_router.entropy, weights)
            self._cached_router = self._detach_router(pooled_router)
            self.rejudge.mark_rejudged(step, hidden)
            reason = "hint"
        elif self.training:
            # 训练中每步重新算路由并保留梯度；生成推理才应用节奏缓存。
            pooled_router = self.router.judge_user_input(pooled)
            self._cached_router = self._detach_router(pooled_router)
            self.rejudge.mark_rejudged(step, hidden)
            reason = "train"
        elif self._cached_router is None:
            pooled_router = self.router.judge_user_input(pooled)
            self._cached_router = self._detach_router(pooled_router)
            self.rejudge.mark_rejudged(step, hidden)
            reason = "first"
        else:
            # 用缓存路由的熵做不确定度触发（不确定度高时提前重判）
            should, why = self.rejudge.should_rejudge(
                step, entropy=self._cached_router.entropy, hidden=hidden
            )
            if should:
                pooled_router = self.router.judge_user_input(pooled)
                self._cached_router = self._detach_router(pooled_router)
                self.rejudge.mark_rejudged(step, hidden)
                reason = why
            else:
                pooled_router = self._cached_router
                reason = "cached"
        # 路由按序列池化判定，垂直/适配器按 token 执行：把领域广播到 token 级
        # 注意：广播产生新对象，不修改缓存（缓存保持池化级形状）
        router_out = RouterOutput(
            domain_ids=pooled_router.domain_ids,
            domain_logits=pooled_router.domain_logits,
            entropy=pooled_router.entropy,
            top_weights=pooled_router.top_weights,
        )
        if router_out.domain_ids.dim() < hidden.dim():
            ids = router_out.domain_ids.unsqueeze(1)
            w = router_out.top_weights.unsqueeze(1)
            target = hidden.shape[:-1] + (ids.shape[-1],)
            router_out.domain_ids = ids.expand(target)
            router_out.top_weights = w.expand(target)
        if self.loading is not None:
            # 独立单调时钟：用户请求的 generation step 会归零，加载热度不能归零。
            load_step = self._loading_clock
            self._loading_clock += 1
            # 路由判出领域后异步预加载垂直专家（延迟步数后生效）
            domains = router_out.domain_ids.reshape(-1)[: self.config.top_vertical]
            self.loading.register_route([int(d) for d in domains.tolist()], load_step)
            # 通用专家也需记录热度，防止热专家被误标为冷门卸载。
            primary_domains = router_out.domain_ids[..., 0].reshape(-1).unique().tolist()
            for domain in primary_domains:
                self.loading.register_general_use(int(domain) % self.config.num_general_experts, load_step)
            self.loading.drain_preload(load_step)
            self.loading.step(load_step)

        compute_aux_logits = self.training if return_aux_logits is None else return_aux_logits
        current = hidden
        draft_logits = None
        mtp_logits = None
        primary_source = "draft"  # 合并专家的 primary 来源（draft 或 general）
        primary_hidden = None
        general_stage_index = 0

        # 2) 按路径串联（跳过 router 阶段，它已在上面执行）
        for stage in self.path:
            if stage == "router":
                continue
            elif stage == "draft":
                domain0 = router_out.domain_ids[..., 0]
                primary_hidden, draft_logits = self.draft(
                    current, domain0, return_logits=compute_aux_logits
                )
                current = primary_hidden
                primary_source = "draft"
            elif stage == "general":
                current, mtp_logits = self._run_generals(
                    current, router_out, return_logits=compute_aux_logits,
                    stage_index=general_stage_index,
                )
                general_stage_index += 1
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
