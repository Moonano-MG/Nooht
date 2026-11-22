#!/usr/bin/env bash
# Linux build for Nooht v0.1.0-dev.
#
# Steps:
#   1. compile check: byte-compile all Python sources
#   2. vendored-copy check: Nooht_Native/noild must equal Noild/noild (except README)
#   3. build the Noild package (sdist + wheel) with PEP 517 (setuptools)
#   4. install the wheel into a fresh venv and import it from outside the repo
#   5. unit tests (Noild, Native) and Noild integration examples
#   6. write artifacts + SHA256 manifest into $OUT_DIR (default: dist-linux/)
#
# Usage: scripts/build_linux.sh
#   PYTHON=python3.12 OUT_DIR=dist-linux scripts/build_linux.sh
#   SKIP_VENV=1 PYTHON=/path/to/venv/bin/python scripts/build_linux.sh   # local smoke only: reuse one env
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-python3}"
OUT_DIR="${OUT_DIR:-dist-linux}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
SKIP_VENV="${SKIP_VENV:-0}"
WORK="$REPO_ROOT/$OUT_DIR/work"
ART="$REPO_ROOT/$OUT_DIR/artifacts"
rm -rf "$REPO_ROOT/$OUT_DIR"
mkdir -p "$WORK" "$ART"

echo "== environment"
"$PY" --version
uname -sm

echo "== 1. compile check"
"$PY" -m compileall -q Noild/noild Nooht_Native scripts
echo "compileall OK"

echo "== 2. vendored copy check"
if ! diff -r -q Noild/noild Nooht_Native/noild -x __pycache__ -x README.md; then
  echo "ERROR: Nooht_Native/noild differs from Noild/noild" >&2
  exit 1
fi
echo "vendored copy OK"

if [ "$SKIP_VENV" = "1" ]; then
  BUILD_PY="$PY"
  CLEAN_PY="$PY"
  echo "SKIP_VENV=1: reusing $PY (smoke mode, step 4 is not isolated)"
else
  "$PY" -m venv "$WORK/venv_build"
  BUILD_PY="$WORK/venv_build/bin/python"
  "$BUILD_PY" -m pip install -q --upgrade pip build
  "$BUILD_PY" -m pip install -q --index-url "$TORCH_INDEX_URL" torch
  "$PY" -m venv "$WORK/venv_clean"
  CLEAN_PY="$WORK/venv_clean/bin/python"
  "$CLEAN_PY" -m pip install -q --upgrade pip
  "$CLEAN_PY" -m pip install -q --index-url "$TORCH_INDEX_URL" torch
  "$CLEAN_PY" -m pip install -q "transformers==5.24.0"
fi

echo "== 3. build Noild (sdist + wheel)"
"$BUILD_PY" -m build --outdir "$ART" Noild
ls -l "$ART"

echo "== 4. install wheel into a clean env and import outside the repo"
WHEEL="$(ls "$ART"/noild-*.whl)"
if [ "$SKIP_VENV" = "1" ]; then
  "$CLEAN_PY" -m pip install -q --no-deps --force-reinstall "$WHEEL"
else
  "$CLEAN_PY" -m pip install -q "$WHEEL"
fi
(cd "$WORK" && "$CLEAN_PY" -c "
import noild, torch
from noild import NoildConfig, NoildLayer, get_cascade_path
layer = NoildLayer(NoildConfig(hidden_size=64, intermediate_size=32, vocab_size=128, num_domains=4,
                               draft_hidden_size=32, draft_intermediate_size=64),
                   path=get_cascade_path('veronoo', 'tutti'))
out = layer(torch.randn(2, 4, 64), step=0)
assert out.hidden.shape == (2, 4, 64)
print('imported noild from', noild.__file__, '-> OK')
")

echo "== 5a. Noild unit tests"
"$CLEAN_PY" -W ignore -m unittest discover -s Noild/tests 2>&1 | tail -n 3

echo "== 5b. Native unit tests"
PYTHONPATH="$REPO_ROOT" "$CLEAN_PY" -W ignore -m unittest discover -s Nooht_Native/tests 2>&1 | tail -n 3

echo "== 5c. Noild integration examples"
for ex in Noild/examples/*.py; do
  echo "-- $ex"
  PYTHONPATH="$REPO_ROOT/Noild" "$CLEAN_PY" -W ignore "$ex" 2>&1 | grep -v "transformers\]" | tail -n 2
done

echo "== 6. manifest"
(cd "$ART" && sha256sum * > SHA256SUMS.txt)
cat "$ART/SHA256SUMS.txt"
echo "BUILD OK: $OUT_DIR/artifacts"
