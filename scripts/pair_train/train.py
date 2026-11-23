"""Same-budget trainer for Native (plain Jamba) and Noild (Jamba with 2 MoE blocks -> NoildLayer).

Both arches share: corpus, validation split, seed-controlled sampling, optimizer, steps, batch,
sequence length and evaluation set. Results are written after every evaluation, so a run that hits
the wall-clock guard (--max_minutes) still leaves a valid JSON with the completed evaluations.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from common import EXPECTED_NATIVE_PARAMS, MAX_PARAM_GAP, NOILD_INTER, V, build
from noild.training import load_balance_loss

K = 4  # draft_num_tokens (Noild draft head predicts t+1..t+K)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["native", "noild"], required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--draft_w", type=float, default=0.1)
    ap.add_argument("--max_minutes", type=float, default=45.0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    raw = torch.tensor(list(Path(args.corpus).read_bytes()), dtype=torch.long)
    CH = 2048
    chunks = raw[: len(raw) // CH * CH].reshape(-1, CH)
    val_mask = torch.arange(chunks.shape[0]) % 20 == 10  # 5% of 2KB chunks, interleaved
    train_d = chunks[~val_mask].reshape(-1)
    val_d = chunks[val_mask].reshape(-1)
    print(f"train_bytes={len(train_d)} val_bytes={len(val_d)}", flush=True)

    def sample(src, gen, n):
        starts = torch.randint(0, len(src) - args.seq - K - 2, (n,), generator=gen)
        return torch.stack([src[s:s + args.seq + K + 1] for s in starts.tolist()])

    VAL = sample(val_d, torch.Generator().manual_seed(1234), 128)  # identical for both arches

    model, wraps = build(args.arch, args.seed)
    n_params = sum(p.numel() for p in model.parameters())
    if args.arch == "native" and n_params != EXPECTED_NATIVE_PARAMS:
        raise RuntimeError(f"native param mismatch: {n_params} != {EXPECTED_NATIVE_PARAMS}")
    if args.arch == "noild" and abs(n_params - EXPECTED_NATIVE_PARAMS) / EXPECTED_NATIVE_PARAMS > MAX_PARAM_GAP:
        raise RuntimeError(f"noild param gap too large: {n_params} vs {EXPECTED_NATIVE_PARAMS}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(args.seed)

    @torch.no_grad()
    def evaluate() -> float:
        model.eval()
        tot, n = 0.0, 0
        for i in range(0, VAL.shape[0], 16):
            x = VAL[i:i + 16]
            # Validation batches are independent requests; reset router state via the public adapter API.
            for w in wraps:
                w.reset_generation()
            logits = model(input_ids=x[:, :args.seq]).logits
            tgt = x[:, 1:args.seq + 1]
            tot += float(F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), reduction="sum"))
            n += tgt.numel()
        model.train()
        return tot / n

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log: list[dict] = []
    status = "completed"

    def write(final: bool) -> None:
        best = min(log, key=lambda r: r["val_loss"])
        res = {
            "arch": args.arch, "seed": args.seed, "params": n_params, "steps_requested": args.steps,
            "steps_done": log[-1]["step"], "finished": final and status == "completed", "status": status,
            "noild_intermediate": NOILD_INTER if wraps else None, "batch": args.batch, "seq": args.seq,
            "lr": args.lr, "wall_s": round(time.time() - t0, 1), "final_val": log[-1]["val_loss"],
            "best_val": best["val_loss"], "best_step": best["step"], "log": log,
            "torch": torch.__version__,
        }
        out_path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    log.append({"step": 0, "val_loss": evaluate()})
    print(json.dumps(log[-1]), flush=True)
    write(False)
    model.train()
    deadline = t0 + args.max_minutes * 60
    for step in range(1, args.steps + 1):
        if time.time() > deadline:
            status = "stopped_time_limit"
            break
        batch = sample(train_d, gen, args.batch)
        x = batch[:, :args.seq]
        logits = model(input_ids=x).logits
        lm = F.cross_entropy(logits.reshape(-1, V), batch[:, 1:args.seq + 1].reshape(-1))
        loss = lm
        draft_val = 0.0
        if wraps:
            # Draft at position t predicts tokens t+1..t+K -> targets (B, L, K)
            tg = torch.stack([batch[:, 1 + k: 1 + k + args.seq] for k in range(K)], dim=-1)
            dl = 0.0
            bal = 0.0
            for w in wraps:
                o = w.last
                if o.draft_logits is not None:
                    dl = dl + F.cross_entropy(o.draft_logits.reshape(-1, V), tg.reshape(-1))
                bal = bal + load_balance_loss(o.router.domain_logits.reshape(-1, 4), 4, 0.01)
            loss = loss + args.draft_w * dl + bal
            draft_val = float(dl)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            rec = {"step": step, "train_lm": round(float(lm), 4), "aux_draft": round(draft_val, 4),
                   "grad_norm": round(float(gn), 3), "val_loss": round(evaluate(), 4),
                   "elapsed_s": round(time.time() - t0, 1)}
            log.append(rec)
            print(json.dumps(rec), flush=True)
            write(False)

    write(True)
    res = json.loads(out_path.read_text(encoding="utf-8"))
    print("FINAL", json.dumps({k: res[k] for k in ["arch", "seed", "params", "steps_done", "status",
                                                   "wall_s", "final_val", "best_val", "best_step"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
