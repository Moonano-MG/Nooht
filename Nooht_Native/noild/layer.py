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
import threading
from typing import Callable

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
        preload_expert/unload_expert: 可选的真实物理搬运回调；由后台线程调用，
            参数为 ``vertical:{domain}`` / ``general:{index}`` 专家 ID。必须成对传入。
        offload_device: 可选的推理卸载设备（例如 ``cpu``）；启用时自动将冷专家移到该设备，
            并在下一次使用前异步移回当前 hidden 所在设备。仅支持 eval 推理。
    """

    def __init__(
        self,
        config: NoildConfig,
        path: tuple[str, ...] | None = None,
        use_loading: bool = True,
        preload_expert: Callable[[str], None] | None = None,
        unload_expert: Callable[[str], None] | None = None,
        offload_device: str | torch.device | None = None,
    ):
        super().__init__()
        if not use_loading and (
            preload_expert is not None or unload_expert is not None or offload_device is not None
        ):
            raise ValueError("传入专家搬运选项时必须启用 use_loading。")
        if offload_device is not None and (preload_expert is not None or unload_expert is not None):
            raise ValueError("offload_device 与自定义搬运回调不能同时使用。")
        self.config = config
        self._offload_device = torch.device(offload_device) if offload_device is not None else None
        self._active_device: torch.device | None = None
        self._module_io_lock = threading.RLock()
        self._router_state_lock = threading.RLock()
        self.path: tuple[str, ...] = (
            path if path is not None else get_cascade_path("veronoo", "tutti")
        )
        # Do not allocate parameters for roles this cascade never executes.
        # Veronoo Tutti has no general stage; Veronoo Duet has no vertical/merge
        # stage. Unused modules would be dead, permanently gradient-free weights.
        uses_draft = "draft" in self.path
        uses_general = "general" in self.path
        uses_vertical = "vertical" in self.path
        uses_merge = "merge" in self.path
        if uses_general and config.num_general_experts <= 0:
            raise ValueError("包含 general 阶段的路径至少需要一个通用专家。")
        self.router = DomainRouter(config)
        self.rejudge = RejudgeController(config)
        self.draft = DraftExpert(config) if uses_draft else None
        self.generals = nn.ModuleList(
            [GeneralExpert(config) for _ in range(config.num_general_experts)]
            if uses_general else []
        )
        self.verticals = nn.ModuleList(
            [VerticalExpert(config, d) for d in range(config.num_domains)]
            if uses_vertical else []
        )
        self.merge: MergeExpert | None = MergeExpert(config) if uses_merge else None
        if self._offload_device is not None:
            preload_expert = self._preload_to_active_device
            unload_expert = self._offload_to_configured_device
        self.loading: ExpertHeatManager | None = (
            ExpertHeatManager(
                len(self.verticals), len(self.generals),
                preload_fn=preload_expert, unload_fn=unload_expert,
            )
            if use_loading
            else None
        )
        # 路由缓存（池化级）：重判间隔内复用，不每步重判；缓存只允许同请求/同池化批形状使用。
        self._cached_router: RouterOutput | None = None
        self._cache_request_id: str | int | None = None
        self._last_inference_step: int | None = None
        self._loading_clock = 0  # 单调加载时钟，不随新请求的 generation step 归零

    def _module_for_expert(self, expert_id: str) -> nn.Module:
        role, raw_index = expert_id.split(":", 1)
        index = int(raw_index)
        if role == "vertical" and 0 <= index < len(self.verticals):
            return self.verticals[index]
        if role == "general" and 0 <= index < len(self.generals):
            return self.generals[index]
        raise KeyError(f"未知专家 ID: {expert_id!r}")

    def _preload_to_active_device(self, expert_id: str) -> None:
        if self._active_device is None:
            raise RuntimeError("首次 forward 前无法确定专家目标设备。")
        with self._module_io_lock:
            self._module_for_expert(expert_id).to(self._active_device)

    def _offload_to_configured_device(self, expert_id: str) -> None:
        if self._offload_device is None:
            raise RuntimeError("未配置 offload_device。")
        with self._module_io_lock:
            self._module_for_expert(expert_id).to(self._offload_device)

    def set_load_factor(self, load_factor: float) -> int:
        """负载大时扩大重判间隔 N（供调用方按负载调用）。"""
        return self.rejudge.update_load(load_factor)

    def mark_router_uncertainty_calibrated(self, calibrated: bool = True) -> None:
        """只在已训练路由器的验证集校准后启用熵阈值提前重判。"""
        self.config.router_uncertainty_calibrated = bool(calibrated)
        self.router.calibrated = bool(calibrated)

    def reset_generation_state(self, request_id: str | int | None = None) -> None:
        """清除跨请求路由缓存；也可指定接下来请求的 ID。"""
        with self._router_state_lock:
            self._cached_router = None
            self._cache_request_id = request_id
            self._last_inference_step = None
            self.rejudge.last_rejudge_step = -10**9
            self.rejudge.last_uncertainty_rejudge_step = -10**9
            self.rejudge.last_hidden = None

    def _router_cache_matches(self, pooled: torch.Tensor) -> bool:
        """检查缓存路由是否与当前池化张量的前导形状和设备兼容。"""
        cached = self._cached_router
        return bool(
            cached is not None
            and cached.domain_ids.shape[:-1] == pooled.shape[:-1]
            and cached.domain_logits.shape[:-1] == pooled.shape[:-1]
            and cached.entropy.shape == pooled.shape[:-1]
            and cached.domain_ids.device == pooled.device
        )

    def _resolve_router(
        self,
        hidden: torch.Tensor,
        pooled: torch.Tensor,
        step: int,
        domain_hint: torch.Tensor | None,
        request_id: str | int | None,
    ) -> tuple[RouterOutput, str]:
        """Resolve this call's route atomically so interleaved requests cannot share state."""
        with self._router_state_lock:
            if not self.training:
                if request_id is None:
                    raise ValueError(
                        "eval 推理必须提供 request_id；新请求不得隐式复用其他请求的路由缓存。"
                    )
                request_changed = request_id != self._cache_request_id
                step_discontinuous = (
                    self._last_inference_step is not None
                    and step != self._last_inference_step + 1
                )
                if request_changed or step == 0 or step_discontinuous:
                    self.reset_generation_state(request_id=request_id)
                self._last_inference_step = step
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
                pooled_router = RouterOutput(
                    ids, pooled_router.domain_logits, pooled_router.entropy, weights
                )
                self._cached_router = None if self.training else self._detach_router(pooled_router)
                self.rejudge.mark_rejudged(step, hidden)
                return pooled_router, "hint"
            if self.training:
                # Training routes each batch afresh and never persists inference cache state.
                pooled_router = self.router.judge_user_input(pooled)
                self._cached_router = None
                self.rejudge.mark_rejudged(step, hidden)
                return pooled_router, "train"
            if self._cached_router is not None and not self._router_cache_matches(pooled):
                # Batch size/leading shape or device changed; do not broadcast stale route IDs.
                pooled_router = self.router.judge_user_input(pooled)
                self._cached_router = self._detach_router(pooled_router)
                self.rejudge.mark_rejudged(step, hidden)
                return pooled_router, "cache_mismatch"
            if self._cached_router is None:
                pooled_router = self.router.judge_user_input(pooled)
                self._cached_router = self._detach_router(pooled_router)
                self.rejudge.mark_rejudged(step, hidden)
                return pooled_router, "first"

            should, why = self.rejudge.should_rejudge(
                step,
                entropy=self._cached_router.entropy if self.router.calibrated else None,
                hidden=hidden,
            )
            if should:
                pooled_router = self.router.judge_user_input(pooled)
                self._cached_router = self._detach_router(pooled_router)
                self.rejudge.mark_rejudged(step, hidden)
                return pooled_router, why
            return self._cached_router, "cached"

    def close(self, wait_for_expert_io: bool = True) -> None:
        """等待并释放专家搬运工作线程。"""
        if self.loading is not None:
            self.loading.close(wait=wait_for_expert_io)

    def _general_for_domain(self, domain_id: int, stage_index: int = 0) -> GeneralExpert:
        """按领域与通用阶段确定性选择专家。

        同一领域在同一阶段可复现；Soleinoo 的前置与末尾通用阶段轮换到
        不同专家（当通用专家数大于 1），实现两个接力角色而非重复调用同一份权重。
        """
        if not self.generals:
            raise RuntimeError("当前级联未配置通用专家。")
        return self.generals[(int(domain_id) + stage_index) % len(self.generals)]

    def _run_generals(
        self, hidden: torch.Tensor, router_out: RouterOutput, return_logits: bool,
        stage_index: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """按每个 token 的领域确定性选择通用专家，避免批次只用首样本领域。"""
        if not self.generals:
            raise RuntimeError("当前级联未配置通用专家，不能执行 general 阶段。")
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
            expert_index = (int(domain) + stage_index) % len(self.generals)
            if self.loading is not None:
                self.loading.ensure_loaded(self.loading.general_id(expert_index))
            with self._module_io_lock:
                expert_out, expert_mtp = self.generals[expert_index](
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
        if not self.verticals:
            raise RuntimeError("当前级联未配置垂直专家，不能执行 vertical 阶段。")
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
                if self.loading is not None:
                    self.loading.ensure_loaded(self.loading.vertical_id(d))
                with self._module_io_lock:
                    expert_out = self.verticals[d](flat_hidden[mask])
                out[mask] += expert_out * flat_w[mask, k : k + 1]
        return out.view_as(hidden)

    def forward(
        self,
        hidden: torch.Tensor,
        step: int = 0,
        domain_hint: torch.Tensor | None = None,
        return_aux_logits: bool | None = None,
        request_id: str | int | None = None,
    ) -> NoildOutput:
        """串联执行。

        Args:
            hidden: (..., hidden_size) 输入 hidden。
            step: 生成步数（用于重判节奏与合并窗口）。
            domain_hint: 可选外部领域提示；提供时强制领域选择并更新路由缓存。
            return_aux_logits: 默认训练时计算 draft/MTP logits，推理时跳过未用辅助大词表头；
                可显式开启以用于验证/消融。
            request_id: eval 推理必填；同一生成请求的每次 forward 传相同 ID，变化时清空缓存。
                step=0 或步数不连续也会清空缓存；reset_generation_state() 可显式清空状态。
        """
        if self._offload_device is not None:
            if self.training or torch.is_grad_enabled():
                raise RuntimeError("offload_device 仅支持 no_grad 的 eval 推理。")
            if self._active_device is not None and hidden.device != self._active_device:
                raise RuntimeError("offload_device 模式不支持同一层在多个目标设备间切换。")
            self._active_device = hidden.device
        # 路由决策在锁内完成：共享层实例上的交错请求不会相互读取或覆盖未完成的状态。
        pooled = hidden.mean(dim=-2) if hidden.dim() > 2 else hidden
        pooled_router, reason = self._resolve_router(
            hidden, pooled, step, domain_hint, request_id
        )
        # 路由按序列池化判定，垂直/适配器按 token 执行：把领域广播到 token 级
        # 注意：广播产生新对象，不修改缓存（缓存保持池化级形状）
        router_out = RouterOutput(
            domain_ids=pooled_router.domain_ids,
            domain_logits=pooled_router.domain_logits,
            entropy=pooled_router.entropy,
            top_weights=pooled_router.top_weights,
        )
        if router_out.domain_ids.dim() < hidden.dim():
            ids = router_out.domain_ids.unsqueeze(-2)
            w = router_out.top_weights.unsqueeze(-2)
            target = hidden.shape[:-1] + (ids.shape[-1],)
            router_out.domain_ids = ids.expand(target)
            router_out.top_weights = w.expand(target)
        if self.loading is not None:
            # 独立单调时钟：用户请求的 generation step 会归零，加载热度不能归零。
            load_step = self._loading_clock
            self._loading_clock += 1
            # 路由后调度预加载；真实搬运由回调/内置 offload 完成，前向前会等待。
            if self.verticals:
                domains = router_out.domain_ids.reshape(-1).unique()
                self.loading.register_route([int(d) for d in domains.tolist()], load_step)
            # 通用专家也需记录热度，防止热专家被误标为冷门卸载。
            primary_domains = router_out.domain_ids[..., 0].reshape(-1).unique().tolist()
            for domain in primary_domains:
                if self.generals:
                    self.loading.register_general_use(int(domain) % len(self.generals), load_step)
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
                if self.draft is None:
                    raise RuntimeError("当前级联未配置预生成专家，不能执行 draft 阶段。")
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
                if self.merge is None:
                    raise RuntimeError("当前级联未配置合并专家，不能执行 merge 阶段。")
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
