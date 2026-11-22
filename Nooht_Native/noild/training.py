"""训练方式：共享底座分叉、分阶段联训、负载约束、对齐微调。

对应任务包第四节「训练方式」：
- 各专家共享同一底座（词嵌入加前几层）再分叉训练，最后做对齐微调，
  防止合并专家听不懂；
- 路由、预生成、合并分阶段联训，加负载约束防止垂直专家被饿死。

本模块提供可执行的辅助函数与阶段表；具体训练循环由调用方实现。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .config import NoildConfig
from .experts import ExpertMLP
from .layer import NoildLayer


def load_balance_loss(
    router_logits: torch.Tensor, num_domains: int, coef: float = 0.01
) -> torch.Tensor:
    """负载约束：鼓励领域使用均匀，防止垂直专家被饿死。

    router_logits: (N, num_domains)。返回标量 loss（已乘 coef）。
    """
    probs = F.softmax(router_logits.float(), dim=-1)
    mean_prob = probs.mean(dim=0)  # (num_domains,)
    # 均匀分布的熵最大；用 KL(mean_prob || uniform) 作为不均衡惩罚
    uniform = torch.full_like(mean_prob, 1.0 / num_domains)
    kl = (mean_prob * (mean_prob.clamp_min(1e-9).log() - uniform.log())).sum()
    return coef * kl


# 分阶段联训表：每阶段可训练的模块名（前缀匹配）
STAGED_TRAINING_PLAN: list[dict] = [
    {
        "stage": 1,
        "name": "router+vertical",
        "trainable": ["router", "verticals"],
        "note": "先学领域划分与垂直专家单领域能力",
    },
    {
        "stage": 2,
        "name": "+draft",
        "trainable": ["router", "verticals", "draft"],
        "note": "预生成专家从垂直专家蒸馏/适配，保证猜测口味对齐",
    },
    {
        "stage": 3,
        "name": "+merge alignment",
        "trainable": ["router", "verticals", "draft", "generals", "merge"],
        "note": "合并专家对齐微调：训练覆盖 draft+vertical 与 general+vertical 两种输入",
    },
]


def set_trainable(layer: NoildLayer, stage: int) -> list[str]:
    """按阶段表冻结/解冻模块。返回本阶段可训练模块名列表。"""
    plan = next((p for p in STAGED_TRAINING_PLAN if p["stage"] == stage), None)
    if plan is None:
        raise ValueError(f"未知 stage: {stage}，可选: {[p['stage'] for p in STAGED_TRAINING_PLAN]}")
    trainable_prefixes = tuple(plan["trainable"])
    for name, module in layer.named_children():
        train = name.startswith(trainable_prefixes)
        for p in module.parameters():
            p.requires_grad = train
    return list(plan["trainable"])


def upcycle_init(vertical_experts: nn.ModuleList, source_mlp: ExpertMLP) -> None:
    """Upcycling 初始化：把 MoE/共享 MLP 权重复制到各垂直专家。

    对应任务包第二节/第三节：Fusion/Evolution 权重来源可用 upcycling
    初始化 Noild 专家，能复用多少需实验，不保证。
    """
    for expert in vertical_experts:
        mlp: ExpertMLP = getattr(expert, "mlp", expert)
        mlp.gate_proj.weight.data.copy_(source_mlp.gate_proj.weight.data)
        mlp.up_proj.weight.data.copy_(source_mlp.up_proj.weight.data)
        mlp.down_proj.weight.data.copy_(source_mlp.down_proj.weight.data)


def future_token_targets(
    next_token_ids: torch.Tensor, num_tokens: int, ignore_index: int = -100
) -> torch.Tensor:
    """由按 hidden 位置对齐的 next-token labels 构造多步目标。

    ``next_token_ids[..., t]`` 是 hidden[..., t, :] 的一步目标。返回
    ``(..., sequence, num_tokens)``，后续预测越过样本末尾的位置置为 ignore_index。
    """
    if next_token_ids.ndim < 2 or num_tokens <= 0:
        raise ValueError("next_token_ids 至少需有 batch/sequence 两维，num_tokens 必须为正。")
    seq_len = next_token_ids.shape[-1]
    result = next_token_ids.new_full((*next_token_ids.shape, num_tokens), ignore_index)
    for horizon in range(num_tokens):
        valid = seq_len - horizon
        if valid > 0:
            result[..., :valid, horizon] = next_token_ids[..., horizon:]
    return result


def _horizon_cross_entropy(
    logits: torch.Tensor, target_ids: torch.Tensor, horizon_decay: float = 0.5,
    ignore_index: int = -100,
) -> torch.Tensor:
    if logits.ndim < 3 or target_ids.shape != logits.shape[:-1]:
        raise ValueError("预测 logits 与 target 形状必须分别为 (..., K, V) 与 (..., K)。")
    if not 0 < horizon_decay <= 1:
        raise ValueError("horizon_decay 必须位于 (0, 1]。")
    losses, weights = [], []
    horizons = target_ids.shape[-1]
    for horizon in range(horizons):
        labels = target_ids[..., horizon].reshape(-1)
        valid = labels != ignore_index
        if not bool(valid.any()):
            continue
        per_item = F.cross_entropy(
            logits[..., horizon, :].reshape(-1, logits.shape[-1]),
            labels,
            ignore_index=ignore_index,
            reduction="none",
        )
        losses.append(per_item[valid].mean())
        weights.append(horizon_decay ** horizon)
    if not losses:
        raise ValueError("所有预测目标均为 ignore_index，无法计算辅助 loss。")
    weight_tensor = logits.new_tensor(weights)
    return (torch.stack(losses) * weight_tensor).sum() / weight_tensor.sum()


def draft_loss(
    draft_logits: torch.Tensor, target_ids: torch.Tensor,
    horizon_decay: float = 0.5, ignore_index: int = -100,
) -> torch.Tensor:
    """草稿 token 交叉熵；较近预测获得较高权重，契合连续前缀接受目标。

    draft_logits: (..., draft_num_tokens, vocab_size)
    target_ids: (..., draft_num_tokens)，尾部无未来标签可用 ignore_index 屏蔽。
    """
    return _horizon_cross_entropy(draft_logits, target_ids, horizon_decay, ignore_index)


def mtp_loss(
    mtp_logits: torch.Tensor, target_ids: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    """MTP 头多步交叉熵，忽略样本边界后的无效预测位置。"""
    return _horizon_cross_entropy(mtp_logits, target_ids, horizon_decay=1.0, ignore_index=ignore_index)


def train_noild(
    layer: NoildLayer,
    lm_head: nn.Module,
    batches: Iterable[Mapping[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    draft_weight: float = 1.0,
    draft_horizon_decay: float = 0.5,
    mtp_weight: float = 0.1,
    max_grad_norm: float | None = 1.0,
) -> list[dict[str, float | str]]:
    """在已由底座产生的 hidden states 上训练 Noild 与输出头。

    ``target_ids`` 是按 hidden 序列位置对齐的一步 next-token labels；缺省时由它
    自动构造 draft/MTP 多步目标，尾部无效位置以 ignore_index 屏蔽。显式传入的辅助
    targets 可覆盖自动生成。路径启用了辅助头却未产出 logits 时立即报错；未启用 MTP
    的路径会记录 ``mtp_status=not_applicable``，绝不静默略过。返回逐步 loss 指标。
    """
    if draft_weight < 0 or mtp_weight < 0:
        raise ValueError("draft_weight 与 mtp_weight 不得为负。")
    if not 0 < draft_horizon_decay <= 1:
        raise ValueError("draft_horizon_decay 必须位于 (0, 1]。")
    uses_draft = "draft" in layer.path
    uses_mtp = "general" in layer.path
    layer.train()
    lm_head.train()
    history: list[dict[str, float | str]] = []
    for step, batch in enumerate(batches):
        if "hidden" not in batch or "target_ids" not in batch:
            raise KeyError("每个 batch 必须含 hidden 和 target_ids。")
        hidden = batch["hidden"]
        targets = batch["target_ids"]
        optimizer.zero_grad(set_to_none=True)
        output = layer(hidden, step=step)
        logits = lm_head(output.hidden)
        lm = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss = lm
        values: dict[str, float | str] = {"lm_loss": float(lm.detach())}
        if uses_draft:
            if output.draft_logits is None:
                raise RuntimeError("路径包含 draft，但 NoildLayer 未返回 draft_logits。")
            draft_targets = batch.get("draft_target_ids")
            if draft_targets is None:
                draft_targets = future_token_targets(targets, output.draft_logits.shape[-2])
            aux = draft_loss(output.draft_logits, draft_targets, horizon_decay=draft_horizon_decay)
            loss = loss + draft_weight * aux
            values["draft_loss"] = float(aux.detach())
            values["draft_status"] = "trained"
        else:
            if "draft_target_ids" in batch:
                raise ValueError("batch 提供了 draft_target_ids，但当前路径没有 draft 阶段。")
            values["draft_status"] = "not_applicable:no_draft_stage"
        if uses_mtp:
            if output.mtp_logits is None:
                raise RuntimeError("路径包含 general/MTP，但 NoildLayer 未返回 mtp_logits。")
            mtp_targets = batch.get("mtp_target_ids")
            if mtp_targets is None:
                mtp_targets = future_token_targets(targets, output.mtp_logits.shape[-2])
            aux = mtp_loss(output.mtp_logits, mtp_targets)
            loss = loss + mtp_weight * aux
            values["mtp_loss"] = float(aux.detach())
            values["mtp_status"] = "trained"
        else:
            if "mtp_target_ids" in batch:
                raise ValueError("batch 提供了 mtp_target_ids，但当前路径没有 general/MTP 阶段。")
            values["mtp_status"] = "not_applicable:no_general_stage"
        balance = load_balance_loss(
            output.router.domain_logits.reshape(-1, layer.config.num_domains),
            layer.config.num_domains,
            layer.config.load_balance_coef,
        )
        loss = loss + balance
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None],
                max_grad_norm,
            )
        optimizer.step()
        values["load_balance_loss"] = float(balance.detach())
        values["total_loss"] = float(loss.detach())
        history.append(values)
    return history


def alignment_check(merge_expert, hidden: torch.Tensor) -> dict:
    """对齐检查：合并专家对两种输入来源均可前向（训练覆盖两者的冒烟检查）。"""
    results = {}
    for source in ("draft", "general"):
        out = merge_expert(hidden, hidden, source=source, step=0)
        results[source] = {
            "shape_ok": tuple(out.shape) == tuple(hidden.shape),
            "finite": bool(torch.isfinite(out).all()),
        }
    return results
