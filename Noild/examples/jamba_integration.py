"""Noild 接入 Jamba 底座示例（Fusion 线）。

任务包第二节：把 Jamba 的 MoE 层换成 Noild 层，注意力层换成 MLA/CSA2，
Mamba 层保留。本示例只演示「MoE FFN -> NoildLayer」的替换点，
MLA/CSA2 不在本示例范围（见任务包第六节，CSA2 待读技术报告）。

本模块提供 `install_noild_on_jamba_moe_layers`，会在传入的 Jamba causal LM 中
仅替换被识别为 `JambaSparseMoeBlock` 的 decoder FFN，保留密集 FFN 与 Mamba/注意力层。
替换范围仅限 Noild FFN，不包含任务包所述 MLA/CSA2，也不自动加载或验证 checkpoint；
每个新生成请求前必须调用 `reset_noild_generation(model)` 清除路由缓存。
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python Noild/examples/jamba_integration.py` from the repository root
# without requiring callers to set PYTHONPATH manually.
REPO_ROOT = Path(__file__).resolve().parents[2]
NOILD_ROOT = REPO_ROOT / "Noild"
if str(NOILD_ROOT) not in sys.path:
    sys.path.insert(0, str(NOILD_ROOT))

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

    def reset_generation(self) -> None:
        """在每个新请求前调用，重置层内路由缓存与步数。"""
        self.step = 0
        self.noild.reset_generation_state()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # JambaSparseMoeBlock 内部会 flatten，这里保持 (B, S, H) 接口不变
        out = self.noild(hidden_states, step=self.step)
        self.step += 1
        return out.hidden


def reset_noild_generation(model: nn.Module) -> None:
    """新请求开始前重置所有已安装 Noild Jamba FFN 的路由状态。"""
    found = False
    for module in model.modules():
        if isinstance(module, NoildJambaFeedForward):
            module.reset_generation()
            found = True
    if not found:
        raise ValueError("模型中没有已安装的 NoildJambaFeedForward。")


def install_noild_on_jamba_moe_layers(
    model: nn.Module,
    config: NoildConfig,
    *,
    model_name: str = "veronoo",
    tier: str = "tutti",
    layer_indices: set[int] | None = None,
) -> list[int]:
    """Replace actual Jamba decoder MoE FFNs with Noild, leaving dense FFNs alone.

    Works against the checked-in Jamba layout (decoder ``layers`` containing
    ``feed_forward`` modules named ``JambaSparseMoeBlock``). This swaps only
    the FFN: it does not implement MLA/CSA2, load weights, or verify an external
    Transformers checkpoint. Returns the replaced decoder layer indices.
    """
    base_config = getattr(model, "config", None)
    base_hidden_size = getattr(base_config, "hidden_size", None)
    if base_hidden_size is not None and int(base_hidden_size) != config.hidden_size:
        raise ValueError(
            f"Noild hidden_size={config.hidden_size} 与 Jamba hidden_size={base_hidden_size} 不匹配。"
        )
    candidates = None
    for path in ("model.layers", "layers"):
        try:
            candidates = model.get_submodule(path)
            break
        except AttributeError:
            continue
        except (KeyError, IndexError):
            continue
    if candidates is None or not isinstance(candidates, (nn.ModuleList, list, tuple)):
        raise ValueError("找不到 Jamba decoder layers；请传入包含 model.layers 的 causal LM。")

    replaced = []
    for index, decoder_layer in enumerate(candidates):
        if layer_indices is not None and index not in layer_indices:
            continue
        feed_forward = getattr(decoder_layer, "feed_forward", None)
        if feed_forward is None:
            continue
        is_sparse_moe = (
            feed_forward.__class__.__name__ == "JambaSparseMoeBlock"
            or (hasattr(feed_forward, "router") and hasattr(feed_forward, "experts"))
        )
        if not is_sparse_moe:
            continue
        decoder_layer.feed_forward = NoildJambaFeedForward(
            config, model=model_name, tier=tier
        )
        replaced.append(index)
    if not replaced:
        raise ValueError("未找到匹配的 Jamba MoE FFN；检查模型版本和 layer_indices。")
    return replaced


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
