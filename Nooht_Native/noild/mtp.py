"""MTP 头：通用专家加入多 token 预测。

对应任务包第四节「通用专家：加入 MTP」与第七节「加速」：
预生成专家、DSpark、MTP 功能重叠（都是先猜后验），主线只保留一种，
先做消融（接受率与加速比）再定。本模块提供 MTP 接口，不预设它一定保留。
"""

from __future__ import annotations

import torch
from torch import nn

from .config import NoildConfig


class MTPHead(nn.Module):
    """多 token 预测头：一次预测未来 k 个 token 的 logits。

    Args:
        hidden_size: 输入 hidden 维度。
        vocab_size: 词表大小。
        num_tokens: 预测的未来 token 数（k）。
    """

    def __init__(self, hidden_size: int, vocab_size: int, num_tokens: int):
        super().__init__()
        self.num_tokens = num_tokens
        self.vocab_size = vocab_size
        self.proj = nn.Linear(hidden_size, vocab_size * num_tokens, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: (..., hidden_size) -> (..., num_tokens, vocab_size)
        logits = self.proj(hidden)
        shape = hidden.shape[:-1] + (self.num_tokens, self.vocab_size)
        return logits.view(shape)


def build_mtp_head(config: NoildConfig) -> MTPHead:
    return MTPHead(config.hidden_size, config.vocab_size, config.mtp_num_tokens)
