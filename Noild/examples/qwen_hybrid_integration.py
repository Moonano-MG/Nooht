"""Noild 接入 Qwen 混合底座示例（Evolution 线）。

任务包第三节：Gated DeltaNet + Transformer + Noild，
多数层为 Gated DeltaNet，少数层为完整注意力（约 3:1），MoE 换成 Noild。
基于 transformers 官方 Qwen3-Next / Qwen3.5 实现改造，
具体模块名开工时确认（任务包待确认 4）——本示例用通用适配器演示替换点，
不硬编码未确认的模块名。

用法（伪代码）：
    decoder_layer.mlp = NoildQwenFeedForward(noild_config, path)
"""

from __future__ import annotations

import torch
from torch import nn

from noild import NoildConfig, NoildLayer, get_cascade_path


class NoildQwenFeedForward(nn.Module):
    """把 NoildLayer 包装成 Qwen 混合层 MLP 接口（hidden -> hidden）。"""

    def __init__(
        self,
        config: NoildConfig,
        model: str = "soleinoo",
        tier: str = "tutti",
    ):
        super().__init__()
        self.noild = NoildLayer(config, path=get_cascade_path(model, tier))
        self._step = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out = self.noild(hidden_states, step=self._step)
        self._step += 1
        return out.hidden


if __name__ == "__main__":
    config = NoildConfig(
        hidden_size=256, intermediate_size=512, vocab_size=1024, num_domains=4,
        draft_hidden_size=64, draft_intermediate_size=128,
    )
    ff = NoildQwenFeedForward(config)
    x = torch.randn(2, 8, 256)
    y = ff(x)
    assert y.shape == x.shape
    print("QwenHybrid+Noild 前向 OK:", tuple(y.shape))
