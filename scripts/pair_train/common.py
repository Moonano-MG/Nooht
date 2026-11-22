"""Model construction shared by Native and Noild pair training.

Native: plain Jamba (transformers implementation), 4 layers, hidden 128, 4 experts.
Noild:  the same Jamba, with the 2 MoE feed_forward blocks replaced by NoildLayer.
Parameter-matched: Native 1,369,776; Noild 1,368,752 (gap 0.075%) with intermediate=72,
draft hidden=96, draft intermediate=208. Noild's general experts are only built on paths that use them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import JambaConfig, JambaForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "Noild"))

from noild import NoildConfig, NoildLayer, get_cascade_path  # noqa: E402

H, L, V = 128, 4, 256  # hidden size, layers, byte-level vocab
NOILD_INTER = 72
NOILD_DRAFT_HIDDEN = 96
NOILD_DRAFT_INTER = 208
EXPECTED_NATIVE_PARAMS = 1369776
MAX_PARAM_GAP = 0.001  # Noild must be within 0.1% of Native


def jamba_cfg(inter: int = 256) -> JambaConfig:
    return JambaConfig(
        vocab_size=V, hidden_size=H, intermediate_size=inter, num_hidden_layers=L,
        num_attention_heads=4, num_key_value_heads=2, mamba_d_state=8, mamba_d_conv=4, mamba_expand=2,
        num_experts=4, num_experts_per_tok=2, attn_layer_offset=1, attn_layer_period=2,
        expert_layer_offset=1, expert_layer_period=2, max_position_embeddings=512,
    )


class NoildFF(torch.nn.Module):
    """Adapter: Jamba feed_forward interface (hidden -> hidden) backed by NoildLayer."""

    def __init__(self, cfg: NoildConfig):
        super().__init__()
        self.noild = NoildLayer(cfg, path=get_cascade_path("veronoo", "tutti"))
        self.last = None
        self.step = 0

    def forward(self, hidden_states):
        out = self.noild(hidden_states, step=self.step)
        self.step += 1
        self.last = out  # keep auxiliary outputs (draft logits, router) for the training loss
        return out.hidden


def build(arch: str, seed: int):
    torch.manual_seed(seed)
    model = JambaForCausalLM(jamba_cfg(256))
    if arch == "native":
        return model, []
    if arch != "noild":
        raise ValueError(arch)
    ncfg = NoildConfig(
        hidden_size=H, intermediate_size=NOILD_INTER, vocab_size=V, num_domains=4,
        num_general_experts=2, draft_hidden_size=NOILD_DRAFT_HIDDEN, draft_intermediate_size=NOILD_DRAFT_INTER,
    )
    wraps = []
    for layer in model.model.layers:
        if type(getattr(layer, "feed_forward", None)).__name__ == "JambaSparseMoeBlock":
            wrap = NoildFF(ncfg)
            layer.feed_forward = wrap
            wraps.append(wrap)
    if len(wraps) != 2:
        raise RuntimeError(f"expected 2 MoE blocks to replace, got {len(wraps)}")
    return model, wraps
