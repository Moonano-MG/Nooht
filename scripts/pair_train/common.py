"""Model construction shared by Native and Noild pair training.

Native: plain Jamba (transformers implementation), 4 experts per MoE block.
Noild:  the same Jamba, with every MoE feed_forward block replaced by NoildLayer.

Two parameter-matched sizes:
  small  : 4 layers, hidden 128. Native 1,369,776 params; Noild 1,368,752 (gap 0.075%).
  medium : 8 layers, hidden 256, 8 experts. Native 16,787,840 params; Noild 16,783,744 (gap 0.024%).
Noild general experts are only built on paths that use them (Veronoo Tutti does not).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import JambaConfig, JambaForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "Noild"))

from noild import NoildConfig, NoildLayer, get_cascade_path  # noqa: E402

V = 256  # byte-level vocab
SIZES: dict[str, dict] = {
    "small": dict(
        hidden=128, layers=4, heads=4, kv_heads=2, native_inter=256, experts=4,
        noild_inter=72, draft_hidden=96, draft_inter=208, native_params=1369776,
    ),
    "medium": dict(
        hidden=256, layers=8, heads=8, kv_heads=4, native_inter=512, experts=8,
        noild_inter=512, draft_hidden=224, draft_inter=832, native_params=16787840,
    ),
}
MAX_PARAM_GAP = 0.001  # Noild must be within 0.1% of Native


def jamba_cfg(size: dict) -> JambaConfig:
    return JambaConfig(
        vocab_size=V, hidden_size=size["hidden"], intermediate_size=size["native_inter"],
        num_hidden_layers=size["layers"], num_attention_heads=size["heads"],
        num_key_value_heads=size["kv_heads"], mamba_d_state=8, mamba_d_conv=4, mamba_expand=2,
        num_experts=size["experts"], num_experts_per_tok=2, attn_layer_offset=1, attn_layer_period=2,
        expert_layer_offset=1, expert_layer_period=2, max_position_embeddings=512,
    )


class NoildFF(torch.nn.Module):
    """Adapter: Jamba feed_forward interface (hidden -> hidden) backed by NoildLayer."""

    def __init__(self, cfg: NoildConfig):
        super().__init__()
        self.noild = NoildLayer(cfg, path=get_cascade_path("veronoo", "tutti"))
        self.last = None
        self.step = 0
        self.request_id = 0

    def reset_generation(self):
        """Start a fresh eval request without retaining route state from prior batches."""
        self.step = 0
        self.request_id += 1
        self.noild.reset_generation_state(request_id=self.request_id)

    def forward(self, hidden_states):
        out = self.noild(hidden_states, step=self.step, request_id=self.request_id)
        self.step += 1
        self.last = out  # keep auxiliary outputs (draft logits, router) for the training loss
        return out.hidden


def build(arch: str, seed: int, size_name: str):
    if size_name not in SIZES:
        raise ValueError(f"unknown size {size_name}; choose from {sorted(SIZES)}")
    size = SIZES[size_name]
    torch.manual_seed(seed)
    model = JambaForCausalLM(jamba_cfg(size))
    if arch == "native":
        return model, []
    if arch != "noild":
        raise ValueError(arch)
    ncfg = NoildConfig(
        hidden_size=size["hidden"], intermediate_size=size["noild_inter"], vocab_size=V,
        num_domains=4, num_general_experts=2,
        draft_hidden_size=size["draft_hidden"], draft_intermediate_size=size["draft_inter"],
    )
    wraps = []
    for layer in model.model.layers:
        if type(getattr(layer, "feed_forward", None)).__name__ == "JambaSparseMoeBlock":
            wrap = NoildFF(ncfg)
            layer.feed_forward = wrap
            wraps.append(wrap)
    if not wraps:
        raise RuntimeError("no MoE blocks found to replace")
    return model, wraps
