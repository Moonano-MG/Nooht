# Scripts

## Linux build — `build_linux.sh`

Runs on `ubuntu-latest` through `.github/workflows/build-linux.yml`, or locally:

    bash scripts/build_linux.sh

Steps: byte-compile all sources, check that `Nooht_Native/noild` matches `Noild/noild`, build the
Noild sdist and wheel, install the wheel into a clean venv, run the Noild and Native unit tests and
the Noild integration examples, and write `dist-linux/artifacts/` with `SHA256SUMS.txt`.

Noild is pure Python (no C/CUDA extensions), so "compile" means byte-compilation plus packaging.

## Same-budget pair training — `pair_train/`

Trains Native (plain Jamba) and Noild (Jamba with its 2 MoE blocks replaced by `NoildLayer`) with
identical data, validation set, seed, steps, batch, sequence length and optimizer.

- `build_corpus.py` builds the byte-level corpus at runtime from installed `transformers` / `torch`
  docstrings and this repository's Markdown. The corpus is not committed.
- `common.py` builds both models. Both have 1,369,776 parameters at the reference setting; Noild
  uses intermediate 72, draft hidden 96, draft intermediate 208 (1,368,752 parameters, gap 0.075%).
- `train.py` writes its JSON after every evaluation, so a run stopped by `--max_minutes` still
  leaves completed results. The `status` field records `completed` or `stopped_time_limit`.

Run locally (CPU):

    python scripts/pair_train/build_corpus.py --out train-out/corpus.bin
    python scripts/pair_train/train.py --arch native --corpus train-out/corpus.bin --steps 2000 --out train-out/res_native.json
    python scripts/pair_train/train.py --arch noild  --corpus train-out/corpus.bin --steps 2000 --out train-out/res_noild.json

Dependencies: `torch` (CPU wheel is enough) and `transformers==5.24.0`.

Workflows: `.github/workflows/train-native.yml` and `train-noild.yml` (manual dispatch). Each run
cancels an older in-progress run of the same workflow (`concurrency.cancel-in-progress`).

Limitations: the data is docstrings and project Markdown, not the Nooht task data. The validation
split is held out from the same corpus. The models are tiny (4 layers, hidden 128) and byte-level.
