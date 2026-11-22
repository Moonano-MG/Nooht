"""Native LoRA fine-tuning entry point.

Dry-run only parses the accelerator YAML. Training requires an explicit local or
caller-accessible checkpoint, JSONL text dataset, and verified Jamba target module
names. This script never builds a base model or downloads a checkpoint implicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_NATIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_NATIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_NATIVE_ROOT))


def load_config(path: str) -> dict:
    """Read the small scalar/nested YAML subset used by this repository."""
    config: dict = {}
    stack = [(config, -1)]
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.strip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip())
            key, _, value = line.strip().partition(":")
            value = value.split(" #", 1)[0].strip()
            while stack and indent <= stack[-1][1]:
                stack.pop()
            parent = stack[-1][0]
            if value == "":
                child: dict = {}
                parent[key] = child
                stack.append((child, indent))
            else:
                parent[key] = _parse_scalar(value)
    return config


def _parse_scalar(value: str):
    if value in ("true", "True"):
        return True
    if value in ("false", "False"):
        return False
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _load_jsonl_texts(path: str) -> list[str]:
    texts = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("text") if isinstance(record, dict) else None
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}:{line_no}: 每行 JSON 必须含非空 text 字段。")
            texts.append(text)
    if not texts:
        raise ValueError("训练数据没有可用的 text 记录。")
    return texts


def main() -> int:
    parser = argparse.ArgumentParser(description="Native Jamba LoRA fine-tuning")
    parser.add_argument("--config", required=True, help="train_small.yaml 或 train_large.yaml")
    parser.add_argument("--dry-run", action="store_true", help="只校验配置，不加载模型或训练")
    parser.add_argument("--checkpoint", help="显式提供 Jamba checkpoint 路径/模型标识")
    parser.add_argument("--data", help="JSONL 文件，每行包含 text 字段")
    parser.add_argument("--target-module", action="append", dest="target_modules",
                        help="经核实的 Jamba LoRA 模块名；可重复传入")
    parser.add_argument("--output-dir", help="PEFT adapter 输出目录")
    parser.add_argument("--tier", choices=("saga", "tale", "poem"), default="saga")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    print(f"scale={config.get('scale')} attention={config.get('attention')} "
          f"packing={config.get('sequence_packing')} parallelism={config.get('parallelism')}")
    if args.dry_run:
        print("dry-run：配置校验通过，未加载模型、权重或数据，也未启动训练。")
        return 0
    if not all((args.checkpoint, args.data, args.target_modules, args.output_dir)):
        parser.error("训练需显式提供 --checkpoint、--data、--target-module 和 --output-dir。")
    if args.max_steps <= 0:
        parser.error("--max-steps 必须为正整数。")

    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit("训练需安装 torch、transformers，且 LoRA 需安装 peft。") from exc

    # Delayed imports keep dry-run usable without ML dependencies.
    from native.config import lumenoo_config
    from native.generation import load_jamba_native
    from native.lora import train_causal_lm_steps

    texts = _load_jsonl_texts(args.data)
    native_config = lumenoo_config(args.tier, base_checkpoint=args.checkpoint)
    requested_attention = str(config.get("attention", "eager")).lower()
    attn_implementation = (
        "flash_attention_2"
        if requested_attention in ("flash_attention", "flash_attention_2") else None
    )
    model, tokenizer, _ = load_jamba_native(
        args.checkpoint, native_config,
        lora_target_modules=args.target_modules,
        attn_implementation=attn_implementation,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("tokenizer 需提供 pad_token 或 eos_token 以构造批次。")
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model, "config", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    training_cfg = config.get("training", {})
    autocast_dtype = None
    if training_cfg.get("bf16", False):
        if device.type == "cpu" or torch.cuda.is_bf16_supported():
            autocast_dtype = torch.bfloat16
        else:
            print("WARNING: 当前 CUDA 设备不支持 BF16，关闭混合精度。")
    strategy = config.get("parallelism", {}).get("strategy")
    if strategy not in (None, "", "none"):
        print(f"WARNING: parallelism={strategy!r} 尚未由本训练入口接入，按单进程运行。")
    if config.get("sequence_packing", False):
        print("WARNING: sequence_packing 当前未实现；样本独立 padding，不做跨样本 packing。")
    if config.get("checkpoint", {}).get("async", False):
        print("WARNING: async checkpoint 当前未实现；训练结束时同步保存 adapter。")
    if config.get("storage"):
        print("WARNING: configured distributed storage backend is not used by this single-process entry point.")
    max_length = int(config.get("data", {}).get("max_seq_len", 8192))
    batch_size = args.batch_size or int(config.get("training", {}).get("per_device_batch", 1))

    class TextDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(texts)
        def __getitem__(self, index):
            encoded = tokenizer(texts[index], truncation=True, max_length=max_length)
            encoded.pop("token_type_ids", None)
            return encoded

    def collate(features):
        return tokenizer.pad(features, padding=True, return_tensors="pt")

    loader = DataLoader(TextDataset(), batch_size=batch_size, shuffle=True, collate_fn=collate)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("PEFT 没有创建可训练参数；检查 target module 名称。")
    lr = float(config.get("optimizer", {}).get("lr", 2e-5))
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    history = train_causal_lm_steps(
        model, loader, optimizer, max_steps=args.max_steps,
        max_grad_norm=1.0,
        gradient_accumulation_steps=int(training_cfg.get("grad_accum", 1)),
        autocast_dtype=autocast_dtype,
    )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(
        f"micro_batches={len(history)} optimizer_steps={int(history[-1]['optimizer_steps'])} "
        f"final_loss={history[-1]['loss']:.6f}"
    )
    print(f"LoRA adapter saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
