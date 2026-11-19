"""Noild 四类专家：预生成 / 通用 / 垂直 / 合并。

对应任务包第四节。专家分角色、串联接力，不是同质并行分支：
- 预生成专家：小而快，猜若干 token 交后续校验；按领域加轻量适配器，
  或从垂直专家蒸馏，保证猜测口味对齐；
- 通用专家：速度与质量折中，数量少，加入 MTP；
- 垂直专家：单领域数据训练，慢但强；允许路由同时选 2 个；
- 合并专家：整合垂直专家输出，兼容「预生成+垂直」和「通用+垂直」两种输入，
  训练覆盖两者；首窗口合并 token 数小（2~3），逐步放大到 10 以上。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import NoildConfig
from .mtp import MTPHead, build_mtp_head


class ExpertMLP(nn.Module):
    """SwiGLU MLP（与 JambaMLP 同结构，便于 upcycling 初始化）。"""

    def __init__(self, hidden_size: int, intermediate_size: int, act: str = "silu"):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU() if act == "silu" else nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DomainAdapter(nn.Module):
    """按领域轻量适配器（LoRA 式低秩）：delta = B @ A @ x，A/B 按领域各一份。"""

    def __init__(self, hidden_size: int, num_domains: int, rank: int):
        super().__init__()
        self.num_domains = num_domains
        self.rank = rank
        self.A = nn.Parameter(torch.empty(num_domains, rank, hidden_size))
        self.B = nn.Parameter(torch.zeros(num_domains, hidden_size, rank))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)

    def forward(self, x: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        # x: (..., hidden_size); domain_id: (...) 或标量
        if not isinstance(domain_id, torch.Tensor):
            domain_id = torch.tensor(domain_id, device=x.device)
        A = self.A[domain_id]  # (..., rank, hidden)
        B = self.B[domain_id]  # (..., hidden, rank)
        # delta = B @ (A @ x)
        ax = torch.einsum("...rh,...h->...r", A, x)
        return torch.einsum("...hr,...r->...h", B, ax)


class DraftExpert(nn.Module):
    """预生成专家：小而快，输出草稿 hidden 与草稿 token logits。

    草稿 token 由后续校验（通用/垂直/合并路径）接受或拒绝，
    本模块只负责「猜」，不负责「验」。
    """

    def __init__(self, config: NoildConfig):
        super().__init__()
        self.config = config
        self.mlp = ExpertMLP(
            config.draft_hidden_size, config.draft_intermediate_size, config.hidden_act
        )
        self.in_proj = nn.Linear(config.hidden_size, config.draft_hidden_size, bias=False)
        self.out_proj = nn.Linear(config.draft_hidden_size, config.hidden_size, bias=False)
        self.adapter = DomainAdapter(
            config.draft_hidden_size, config.num_domains, config.draft_adapter_rank
        )
        self.draft_head = nn.Linear(
            config.draft_hidden_size, config.vocab_size * config.draft_num_tokens, bias=False
        )

    def forward(
        self, hidden: torch.Tensor, domain_id: torch.Tensor | int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (draft_hidden, draft_logits)。

        draft_hidden: (..., hidden_size) 供合并专家使用；
        draft_logits: (..., draft_num_tokens, vocab_size) 供校验使用。
        """
        h = self.in_proj(hidden)
        if domain_id is not None:
            h = h + self.adapter(h, domain_id)
        h = self.mlp(h)
        draft_logits = self.draft_head(h).view(
            *h.shape[:-1], self.config.draft_num_tokens, self.config.vocab_size
        )
        return self.out_proj(h), draft_logits


class GeneralExpert(nn.Module):
    """通用专家：速度与质量折中，数量少，带 MTP 头。"""

    def __init__(self, config: NoildConfig):
        super().__init__()
        self.mlp = ExpertMLP(config.hidden_size, config.intermediate_size, config.hidden_act)
        self.mtp_head: MTPHead = build_mtp_head(config)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.mlp(hidden)
        mtp_logits = self.mtp_head(out)
        return out, mtp_logits


class VerticalExpert(nn.Module):
    """垂直专家：单领域数据训练，慢但强。每个领域一个实例。"""

    def __init__(self, config: NoildConfig, domain_id: int):
        super().__init__()
        self.domain_id = domain_id
        self.mlp = ExpertMLP(config.hidden_size, config.intermediate_size, config.hidden_act)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden)


class MergeExpert(nn.Module):
    """合并专家：整合垂直专家输出。

    兼容两种输入（训练覆盖两者）：
    - source="draft":   预生成 + 垂直
    - source="general": 通用 + 垂直
    实现为门控融合：gate 决定 primary 与 vertical 的混合比例，
    避免合并专家「听不懂」任一来源（任务包第四节训练要求）。
    """

    VALID_SOURCES = ("draft", "general")

    def __init__(self, config: NoildConfig):
        super().__init__()
        self.config = config
        self.primary_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.vertical_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.gate = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.mlp = ExpertMLP(config.hidden_size, config.intermediate_size, config.hidden_act)

    def forward(
        self,
        primary_hidden: torch.Tensor,
        vertical_hidden: torch.Tensor,
        source: str = "draft",
        step: int = 0,
    ) -> torch.Tensor:
        """合并 primary 与 vertical。

        合并窗口：仅前 window 个 token（序列维）走合并 MLP，其余直通，
        窗口随 step 从 start（2~3）放大到 max（10+）。
        """
        if source not in self.VALID_SOURCES:
            raise ValueError(f"source 必须为 {self.VALID_SOURCES} 之一，得到 {source!r}")
        window = self.config.merge_window_at_step(step)
        cat = torch.cat([primary_hidden, vertical_hidden], dim=-1)
        gate = torch.sigmoid(self.gate(cat))
        fused = gate * self.primary_proj(primary_hidden) + (1 - gate) * self.vertical_proj(
            vertical_hidden
        )
        # 窗口内走 MLP，窗口外直通（保持形状不变）
        if fused.dim() >= 2 and fused.shape[-2] > window:
            head, tail = fused[..., :window, :], fused[..., window:, :]
            return torch.cat([self.mlp(head), tail], dim=-2)
        return self.mlp(fused)
