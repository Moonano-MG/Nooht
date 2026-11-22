# Nooht_Native — Native 线（Lumenoo）

> 底座：Jamba 黑盒（Attention + Mamba + MoE），官方权重原样加载，
> Mamba/注意力/MoE 层一行不改，只在外面加外层优化层，再微调。
> 不需要预训练，不需要改模型结构。版本：`v0.1.0-dev`（本阶段不升版）。

## 目录

```
Nooht_Native/
├── modeling_jamba.py / modular_jamba.py / configuration_jamba.py  # Jamba 底座（一行不改）
├── noild/                  # Noild vendored 副本（同源于 ../Noild/noild，见 docs/noild_in_native.md）
├── native/                 # 外层优化层（任务包第三节）
│   ├── backend.py          # 后端检测/参考算子/注册接口；默认无优化 kernel
│   ├── memory_compression.py # 选择性记忆压缩
│   ├── protected_zone.py   # 受保护区（人设/指令/思考记忆块）
│   ├── copy_mechanism.py   # 复制机制（格式与接口）
│   ├── generation.py       # 生成循环编排
│   └── config.py           # Lumenoo 档位 Saga/Tale/Poem
├── configs/                # Lumenoo 三档配置
├── training/               # 训练加速栈两套配置（小规模验证）
├── docs/                   # 设计文档
└── tests/                  # 单元测试（unittest）
```

## 档位（任务包第五节）

Lumenoo 没有专家路径之分，三档按底座/LoRA 规格划分：
Saga（Max）、Tale（Plus）、Poem（Flash）。命名：`Lumenoo-Saga-xxB` 形式。

## LoRA 微调

`native.lora.attach_lora` 可在显式指定并核实的 Jamba 模块名上创建 PEFT LoRA；
`native.lora.train_causal_lm_steps` 提供调用方数据上的基本 causal-LM 训练循环。
`training/train_native.py` 的非 dry-run 模式要求显式传入 checkpoint、JSONL 数据
（每行含 `text`）、LoRA target module 和输出目录。请安装 `peft`，并自行核实
checkpoint 与 target module；仓库不含真实 Jamba 权重，也不自动下载或训练。

该实现补充了训练入口，不代表真实 checkpoint 上的训练/质量验收已完成。

## 测试

```
python -m unittest discover -s Nooht_Native/tests -v
```
