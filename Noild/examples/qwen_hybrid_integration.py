"""Noild 接入 Qwen 混合底座示例（Evolution 线）。

任务包第三节：Gated DeltaNet + Transformer + Noild，
多数层为 Gated DeltaNet，少数层为完整注意力（约 3:1），MoE 换成 Noild。
基于 transformers 官方 Qwen3-Next / Qwen3.5 实现改造，
具体模块名开工时确认（任务包待确认 4）——本示例用通用适配器演示替换点，
不硬编码未确认的模块名。

模块提供 `install_noild_on_qwen_mlp_modules`，要求调用方按实际 Transformers
版本先核实 MLP 完全限定路径，再显式替换；此接口不推断层类型或处理 cache，
且每个新请求前需调用 `reset_noild_generation(model)`。
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow direct execution from the repository root without PYTHONPATH setup.
REPO_ROOT = Path(__file__).resolve().parents[2]
NOILD_ROOT = REPO_ROOT / "Noild"
if str(NOILD_ROOT) not in sys.path:
    sys.path.insert(0, str(NOILD_ROOT))

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

    def reset_generation(self) -> None:
        """在每个新请求前调用，清除路由缓存与步数。"""
        self._step = 0
        self.noild.reset_generation_state()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out = self.noild(hidden_states, step=self._step)
        self._step += 1
        return out.hidden


def reset_noild_generation(model: nn.Module) -> None:
    """新请求开始前重置所有已安装 Noild Qwen MLP 的路由状态。"""
    found = False
    for module in model.modules():
        if isinstance(module, NoildQwenFeedForward):
            module.reset_generation()
            found = True
    if not found:
        raise ValueError("模型中没有已安装的 NoildQwenFeedForward。")


def install_noild_on_qwen_mlp_modules(
    model: nn.Module,
    config: NoildConfig,
    module_paths: list[str],
    *,
    model_name: str = "soleinoo",
    tier: str = "tutti",
) -> list[str]:
    """Replace caller-verified Qwen MLP module paths with the Noild adapter.

    Qwen hybrid module names vary across Transformers/checkpoint versions, so
    this function deliberately requires explicit paths and does not guess which
    layers are Gated DeltaNet, attention, dense FFN, or MoE. It changes only the
    named MLP modules; architecture/cache integration still needs model-specific
    validation by the caller.
    """
    if not module_paths:
        raise ValueError("请传入已核实的 Qwen MLP module_paths。")
    base_hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
    if base_hidden_size is not None and int(base_hidden_size) != config.hidden_size:
        raise ValueError(
            f"Noild hidden_size={config.hidden_size} 与 Qwen hidden_size={base_hidden_size} 不匹配。"
        )
    resolved = []
    seen = set()
    for path in module_paths:
        if not path or path.startswith(".") or path.endswith(".") or path in seen:
            raise ValueError(f"无效或重复模块路径: {path!r}")
        seen.add(path)
        parent_path, _, leaf = path.rpartition(".")
        try:
            parent = model.get_submodule(parent_path) if parent_path else model
            if isinstance(parent, (nn.ModuleList, nn.Sequential)) and leaf.isdigit():
                target = parent[int(leaf)]
            elif isinstance(parent, nn.ModuleDict):
                target = parent[leaf]
            else:
                target = getattr(parent, leaf, None)
        except (AttributeError, IndexError, KeyError) as exc:
            raise ValueError(f"找不到模块路径 {path!r}") from exc
        if not isinstance(target, nn.Module):
            raise ValueError(f"{path!r} 不是已存在的 nn.Module")
        resolved.append((path, parent, leaf))

    # Resolve every target before mutation so a typo cannot leave a partial patch.
    for _, parent, leaf in resolved:
        replacement = NoildQwenFeedForward(config, model=model_name, tier=tier)
        if isinstance(parent, (nn.ModuleList, nn.Sequential)) and leaf.isdigit():
            parent[int(leaf)] = replacement
        elif isinstance(parent, nn.ModuleDict):
            parent[leaf] = replacement
        else:
            setattr(parent, leaf, replacement)
    return [path for path, _, _ in resolved]


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
