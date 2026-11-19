"""Noild 接入 Jamba 底座示例（Fusion 线）。

任务包第二节：把 Jamba 的 MoE 层换成 Noild 层，注意力层换成 MLA/CSA2，
Mamba 层保留。本示例只演示「MoE FFN -> NoildLayer」的替换点，
MLA/CSA2 不在本示例范围（见任务包第六节，CSA2 待读技术报告）。

用法（伪代码，Jamba 文件在 Nooht_Native/ 中）：
    from Nooht_Native.modeling_jamba import JambaAttentionDecoderLayer
    layer.feed_forward = NoildJambaFeedForward(noild_config, path)

NoildJambaFeedForward 与 JambaSparseMoeBlock 签名一致（hidden -> hidden），
可直接替换 decoder layer 的 feed_forward，不改 Mamba/注意力层。
"""

from __future__ import annotations

import torch
from torch import nn

from noild import NoildConfig, NoildLayer, get_cascade_path


class NoildJambaFeedForward(nn.Module):
    """把 NoildLayer 包装成 Jamba feed_forward 接口。"""

    def __init__(
        self,
        config: NoildConfig,
        model: str = "veronoo",
        tier: str = "tutti",
        step: int = 0,
    ):
        super().__init__()
        self.step = step
        self.noild = NoildLayer(config, path=get_cascade_path(model, tier))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # JambaSparseMoeBlock 内部会 flatten，这里保持 (B, S, H) 接口不变
        out = self.noild(hidden_states, step=self.step)
        self.step += 1
        return out.hidden


def build_jamba_noild_stack(hidden_size: int = 256, intermediate_size: int = 512):
    """小规模验证用配置（任务包第四节：小规模只验证分工是否有效）。"""
    config = NoildConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        vocab_size=1024,
        num_domains=4,
        num_general_experts=2,
        draft_hidden_size=64,
        draft_intermediate_size=128,
    )
    return NoildJambaFeedForward(config)


if __name__ == "__main__":
    ff = build_jamba_noild_stack()
    x = torch.randn(2, 8, 256)
    y = ff(x)
    assert y.shape == x.shape, (y.shape, x.shape)
    print("Jamba+Noild 前向 OK:", tuple(y.shape))
