"""Noild 最小训练冒烟测试（随机数据，只验证流程与梯度，不代表训练效果）。

运行：PYTHONPATH=Noild python Noild/examples/train_noild_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "Noild") not in sys.path:
    sys.path.insert(0, str(ROOT / "Noild"))

import torch
import torch.nn.functional as F
from torch import nn

from noild import (
    NoildConfig, NoildLayer, draft_loss, future_token_targets,
    load_balance_loss, mtp_loss,
)


class SmokeLM(nn.Module):
    def __init__(self, config: NoildConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.noild = NoildLayer(
            config, path=("router", "draft", "general", "vertical", "merge"), use_loading=False
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=1.0 / (config.hidden_size**0.5))

    def forward(self, input_ids: torch.Tensor, step: int):
        hidden = self.embed(input_ids)
        out = self.noild(hidden, step=step)
        return self.lm_head(out.hidden), out


def main(steps: int = 20) -> None:
    torch.manual_seed(0)
    config = NoildConfig(
        hidden_size=64, intermediate_size=128, vocab_size=256, num_domains=4,
        draft_hidden_size=32, draft_intermediate_size=64,
    )
    model = SmokeLM(config)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(steps):
        batch, seq = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (batch, seq))
        targets = torch.randint(0, config.vocab_size, (batch, seq))
        logits, out = model(input_ids, step)
        lm = F.cross_entropy(logits.reshape(-1, config.vocab_size), targets.reshape(-1))
        future_draft = future_token_targets(targets, config.draft_num_tokens)
        future_mtp = future_token_targets(targets, config.mtp_num_tokens)
        dl = draft_loss(out.draft_logits, future_draft)
        ml = mtp_loss(out.mtp_logits, future_mtp)
        lb = load_balance_loss(
            out.router.domain_logits.reshape(-1, config.num_domains),
            config.num_domains, config.load_balance_coef,
        )
        loss = lm + 0.1 * dl + 0.1 * ml + lb
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step in (0, steps - 1):
            grad = model.noild.draft.draft_head.weight.grad
            grad_mean = float(grad.abs().mean()) if grad is not None else 0.0
            print(f"step {step}: loss={float(loss.detach()):.4f} lm={float(lm.detach()):.4f} draft_head_grad={grad_mean:.2e}")
    assert grad is not None and bool(torch.isfinite(grad).all()) and float(grad.abs().sum()) > 0
    print("冒烟训练 OK：草稿头与 MTP 头由辅助 loss 获得非零梯度。")


if __name__ == "__main__":
    main()
