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


def draft_loss(draft_logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """预生成专家训练 loss：草稿 token 的交叉熵。

    draft_logits: (..., draft_num_tokens, vocab_size)
    target_ids: (..., draft_num_tokens) 真实未来 token id。
    训练时必须把此 loss 加入总 loss，否则 draft_head 梯度恒为 0。
    """
    return F.cross_entropy(
        draft_logits.reshape(-1, draft_logits.shape[-1]),
        target_ids.reshape(-1),
    )


def mtp_loss(mtp_logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """MTP 头训练 loss：未来 k 个 token 的交叉熵（与 draft_loss 同形）。

    mtp_logits: (..., mtp_num_tokens, vocab_size)
    target_ids: (..., mtp_num_tokens)
    训练时必须加入总 loss，否则 mtp_head 梯度恒为 0。
    """
    return F.cross_entropy(
        mtp_logits.reshape(-1, mtp_logits.shape[-1]),
        target_ids.reshape(-1),
    )


def train_noild(
    layer: NoildLayer,
    lm_head: nn.Module,
    batches: Iterable[Mapping[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    draft_weight: float = 0.1,
    mtp_weight: float = 0.1,
    max_grad_norm: float | None = 1.0,
) -> list[dict[str, float]]:
    """在已由底座产生的 hidden states 上训练 Noild 与输出头。

    每个 batch 必须包含 ``hidden``、``target_ids``；若级联路径启用 draft 或
    general/MTP 阶段，还必须提供 ``draft_target_ids`` / ``mtp_target_ids``，
    避免预测头在没有 loss 的情况下静默训练。返回逐步 loss 指标。
    """
    layer.train()
    lm_head.train()
    history: list[dict[str, float]] = []
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
        values = {"lm_loss": float(lm.detach())}
        if output.draft_logits is not None:
            if "draft_target_ids" not in batch:
                raise KeyError("级联启用 draft 时需提供 draft_target_ids。")
            aux = draft_loss(output.draft_logits, batch["draft_target_ids"])
            loss = loss + draft_weight * aux
            values["draft_loss"] = float(aux.detach())
        if output.mtp_logits is not None:
            if "mtp_target_ids" not in batch:
                raise KeyError("级联启用 MTP 时需提供 mtp_target_ids。")
            aux = mtp_loss(output.mtp_logits, batch["mtp_target_ids"])
            loss = loss + mtp_weight * aux
            values["mtp_loss"] = float(aux.detach())
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
