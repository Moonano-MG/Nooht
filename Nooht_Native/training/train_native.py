"""Native 训练入口（配置切换版）：小规模验证，大规模只做接口验证。

对应任务包第十节：两套配置通过 config 切换，代码结构一致。
本脚本只做配置解析与流程骨架，不自动启动大规模训练，不自动构建。
用法：python training/train_native.py --config training/train_small.yaml
"""

from __future__ import annotations

import argparse
import sys


def load_config(path: str) -> dict:
    """极简 YAML 读取（仅本项目用到的标量/嵌套结构，避免引入 pyyaml 依赖）。"""
    config: dict = {}
    stack = [(config, -1)]
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.strip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip())
            key, _, value = line.strip().partition(":")
            value = value.split(" #", 1)[0].strip()  # 去掉行内注释
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Native 训练入口（配置切换）")
    parser.add_argument("--config", required=True, help="train_small.yaml 或 train_large.yaml")
    parser.add_argument("--dry-run", action="store_true", help="只校验配置，不启动训练")
    args = parser.parse_args()
    config = load_config(args.config)
    print(f"scale={config.get('scale')} attention={config.get('attention')} "
          f"packing={config.get('sequence_packing')} parallelism={config.get('parallelism')}")
    if args.dry_run:
        print("dry-run：配置校验通过，未启动训练。")
        return 0
    print("训练循环由调用方接入（需 torch/transformers 环境）；本阶段不自动启动。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
