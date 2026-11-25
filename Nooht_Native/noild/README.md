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
eval 推理必须在每次 forward 提供稳定的 `request_id`；缺失时会立即报错，避免隐式复用其他请求的路由缓存。step=0 或步数不连续也会清空缓存；缓存只保留一个活跃请求，交错请求会重新路由而不会串用。

## 加载策略（基础版）

默认冷启动全量加载，再按热度管理冷门专家。`ExpertHeatManager` 默认只做状态/热度调度；
实际搬运可由 `NoildLayer(..., offload_device="cpu")` 在 `eval()` + `no_grad()` 推理时执行，或由调用方同时
传入 `preload_expert`/`unload_expert` 回调。搬运任务在线程池串行执行，专家前向前等待回调完成；
实际物理搬运由调用方提供的回调实现。
默认未配置搬运时只更新状态，不发生物理搬运，也不节省显存。

## 训练方式

各专家共享同一底座（词嵌入加前几层）再分叉训练，最后做对齐微调；
路由、预生成、合并分阶段联训，加负载约束防止垂直专家被饿死。Draft 输出头采用方差保持初始化，训练前应检查 CE 是否接近均匀基线 `ln(vocab_size)`；多层 Noild 的层级辅助 loss 应按层取均值，避免层数改变有效权重。
见 `noild.training`。

## 不采用

提前退出、动态深度——明确不做。

## 适用范围

当前仓库的验证以小规模研究性实验为主，尚无真实大规模 checkpoint 或部署基准；
现阶段不据参数量给出生产适用下限。特定规模是否适用，需要通过真实模型上的质量、性能与资源基准验证。

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
