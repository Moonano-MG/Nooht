"""Build the byte-level training corpus at runtime (the corpus itself is never committed).

Sources:
  - docstrings extracted with ast from installed transformers / torch.nn / torch.optim / torch.distributions
  - Markdown documents from this repository (Chinese project docs)
Usage: python build_corpus.py --out corpus.bin
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _pkg_dirs(name: str) -> list[Path]:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError(f"package {name} is not installed")
    return [Path(p) for p in spec.submodule_search_locations]


def english_docstrings() -> str:
    roots = _pkg_dirs("transformers") + _pkg_dirs("torch")
    sub = []
    for root in roots:
        if root.name == "torch":
            sub += [root / "nn", root / "optim", root / "distributions"]
        else:
            sub.append(root)
    docs, seen = [], set()
    for base in sub:
        for f in sorted(base.rglob("*.py")):
            try:
                tree = ast.parse(f.read_text(encoding="utf-8", errors="ignore"))
            except (SyntaxError, ValueError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef, ast.Module)):
                    d = ast.get_docstring(node)
                    if d and len(d) > 80:
                        d = d.strip()
                        if d not in seen:
                            seen.add(d)
                            docs.append(d)
    return "\n\n".join(docs)


def repo_markdown() -> str:
    parts = []
    for f in sorted(REPO_ROOT.rglob("*.md")):
        if ".git" in f.parts or "node_modules" in f.parts:
            continue
        parts.append(f.read_text(encoding="utf-8"))
    return "\n\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    en = english_docstrings()
    zh = repo_markdown()
    data = (en + "\n\n" + zh).encode("utf-8")
    if len(data) < 1_000_000:
        raise RuntimeError(f"corpus too small: {len(data)} bytes")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    print(f"corpus bytes={len(data)} english={len(en.encode())} repo_markdown={len(zh.encode())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
