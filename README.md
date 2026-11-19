# Nooht（v0.1.0-dev）

> 以架构为核心的高性价比模型框架：同等能力下更低成本、更广硬件支持、更稳的角色扮演。
> 完整目标与边界见开发任务包（`Nooht_开发任务包.md`，v0.6）。
> 本阶段不升版、不改目标：三条产品线（Native/Fusion/Evolution）分工不变。

## 仓库结构

```
Nooht/
├── LICENSE                  # Apache 2.0
├── Noild/                   # 独立 Noild 层（Apache 2.0，不依赖 Nooht）
│   └── noild/               # 路由/预生成/通用/垂直/合并专家 + 级联层
├── Nooht_Native/            # Native 线（Lumenoo）：Jamba 黑盒 + 外层优化层
│   ├── modeling_jamba.py    # Jamba 底座（一行不改）
│   ├── noild/               # Noild vendored 副本（同源于 Noild/noild）
│   ├── native/              # 外层优化层：后端/记忆压缩/受保护区/复制/生成
│   ├── configs/             # Lumenoo Saga/Tale/Poem
│   ├── training/            # 训练加速栈两套配置
│   └── docs/                # 设计文档
├── Nooht_Evolution/         # Evolution 线占位（本阶段未开发）
└── Jamba-11172026/          # Jamba 官方参考副本
```

## 本阶段交付（先 Noild + 完整 Native）

1. **Noild 层**（`Noild/`）：角色分工、串联接力、路由重判节奏、
   热度加载、MTP、分阶段联训；含 Jamba/Qwen 集成示例与单元测试。
2. **完整 Native**（`Nooht_Native/`）：Jamba 黑盒加载入口、
   外层优化层四项（多硬件后端、选择性记忆压缩、受保护区、复制机制）、
   Lumenoo 三档配置、训练加速栈两套配置、设计文档与单元测试。

Fusion（MLA/CSA2）与 Evolution（Qwen 混合底座）按任务包顺序在后续阶段开发，
本阶段不展开。

## 约定

- 不自动构建：本仓库不自动触发构建/打包流程，构建由维护者手动执行。
- 不随意改版：版本号保持 `v0.1.0-dev`，升版需任务包更新。
- 不随意改目标：对外口径与产品线分工以任务包第十四节为准。
