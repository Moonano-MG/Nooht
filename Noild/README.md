# Noild — Role-Specialized Cascaded Experts

> 独立稀疏层，Apache 2.0 开源，不依赖 Nooht。
> 其他混合架构（Jamba、Qwen 混合底座等）均可接入。
> 版本与 Nooht 主仓库一致：`v0.1.0-dev`（本阶段不升版）。

Noild 全称 **Role-Specialized Cascaded Experts**：专家分角色、串联接力，
不是 MoE 那种同质并行分支。完整定义见 Nooht 开发任务包第四节。

## 角色分工

| 角色 | 职责 | 特点 |
|---|---|---|
| 路由专家 Router | 判定领域（只输出领域，不输出档位） | 每次用户输入必判；生成中每 N 个 token 重判，负载大时扩大 N；不确定度突升或话题转变时提前重判 |
| 预生成专家 Draft | 小而快，猜若干 token 交后续校验 | 按领域加轻量适配器，或从垂直专家蒸馏，保证猜测口味对齐 |
| 通用专家 General | 速度与质量折中 | 数量少，加入 MTP |
| 垂直专家 Vertical | 单领域数据训练 | 慢但强；路由可同时选 2 个 |
| 合并专家 Merge | 整合垂直专家输出 | 兼容「预生成+垂直」与「通用+垂直」两种输入；首窗口合并 token 数小（2~3），逐步放大到 10 以上 |

## 级联路径（任务包第五节）

- Veronoo Max/Plus：路由 > 预生成 > 垂直 > 合并 > 输出
- Veronoo Flash：路由 > 预生成 > 通用 > 输出
- Soleinoo Max/Plus：路由 > 预生成 > 通用 > 垂直 > 合并 > 通用 > 输出（末尾通用专家润色）
- Soleinoo Flash：路由 > 预生成 > 通用 > 垂直 > 合并 > 输出

路径由 `noild.config.get_cascade_path(model, tier)` 给出，`NoildLayer` 按路径串联执行。

## 加载策略（基础版）

默认冷启动全量加载，再按热度卸载冷门专家；路由判出领域后异步预加载垂直专家。
见 `noild.loading.ExpertHeatManager`。深度优化不在本阶段范围。

## 训练方式

各专家共享同一底座（词嵌入加前几层）再分叉训练，最后做对齐微调；
路由、预生成、合并分阶段联训，加负载约束防止垂直专家被饿死。
见 `noild.training`。

## 不采用

提前退出、动态深度——明确不做。

## 适用范围

专家拆分后每个更小，Noild 不建议用于约 30B 以下的模型；
小规模只用于验证分工是否有效。

## 目录

```
Noild/
├── LICENSE                  # Apache 2.0
├── pyproject.toml           # 仅依赖 torch，不依赖 Nooht
├── noild/                   # 层实现
│   ├── config.py            # NoildConfig + 级联路径表
│   ├── router.py            # 路由专家 + 重判节奏控制
│   ├── experts.py           # 预生成/通用/垂直/合并专家
│   ├── mtp.py               # 通用专家 MTP 头
│   ├── layer.py             # NoildLayer（串联接力）
│   ├── loading.py           # 热度加载/卸载/异步预加载
│   └── training.py          # 分阶段联训、负载约束、upcycling 初始化
├── examples/
│   ├── jamba_integration.py # 接入 Jamba 解码层（替换 MoE FFN）
│   └── qwen_hybrid_integration.py # 接入 Qwen 混合底座（示例）
├── docs/
│   └── DESIGN.md            # 路由设计、草稿消融方案
└── tests/                   # 单元测试（unittest，无需 pytest）
```

## 与 Native 的关系

`Nooht_Native/noild/` 是本包的 vendored 副本（同源同步），
Native 主线默认不走 Noild 路径（任务包第二节：Native 没有 Noild），
Noild 在 Native 中为可插拔，供 Fusion/Evolution 复用与对照实验。
