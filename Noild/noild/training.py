"""训练方式：共享底座分叉、分阶段联训、负载约束、对齐微调。

对应任务包第四节「训练方式」：
- 各专家共享同一底座（词嵌入加前几层）再分叉训练，最后做对齐微调，
  防止合并专家听不懂；
- 路由、预生成、合并分阶段联训，加负载约束防止垂直专家被饿死。

本模块提供可执行的辅助函数与阶段表；具体训练循环由调用方实现。
"""

from __future__ import annotations

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
