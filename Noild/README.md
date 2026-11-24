# Noild — Role-Specialized Cascaded Experts

> 独立稀疏层，Apache 2.0 开源，不依赖 Nooht。
> 提供 Jamba MoE FFN 替换器与 Qwen 显式模块路径替换器；具体 Transformers checkpoint、模块名、隐藏态形状和缓存行为须由集成方按实际版本验证，不代表开箱即用的完整接入。每个新请求前必须调用相应示例的 `reset_noild_generation(model)` 清理路由缓存。
> 版本与 Nooht 主仓库一致：`v0.1.0-dev`（本阶段不升版）。

Noild 全称 **Role-Specialized Cascaded Experts**：专家分角色、串联接力，
不是 MoE 那种同质并行分支。完整定义见 Nooht 开发任务包第四节。

## 角色分工

| 角色 | 职责 | 特点 |
|---|---|---|
| 路由专家 Router | 判定领域（只输出领域，不输出档位） | 每次用户输入必判；生成中每 N 个 token 重判，负载大时扩大 N；话题变化时提前重判；熵阈值仅在验证集校准后显式启用 |
| 预生成专家 Draft | 小而快，猜若干 token 交后续校验 | 按领域加轻量适配器，或从垂直专家蒸馏，保证猜测口味对齐 |
| 通用专家 General | 速度与质量折中 | 数量少，加入 MTP |
| 垂直专家 Vertical | 单领域数据训练 | 慢但强；路由可同时选 2 个 |
| 合并专家 Merge | 整合垂直专家输出 | 兼容「预生成+垂直」与「通用+垂直」两种输入；首窗口合并 token 数小（2~3），逐步放大到 10 以上 |

## 级联路径（任务包第五节）

- Veronoo Max/Plus：路由 > 预生成 > 垂直 > 合并 > 输出
- Veronoo Flash：路由 > 预生成 > 通用 > 输出
- Soleinoo Max/Plus：路由 > 预生成 > 通用 > 垂直 > 合并 > 通用 > 输出（末尾通用专家润色）
- Soleinoo Flash：路由 > 预生成 > 通用 > 垂直 > 合并 > 输出

路径由 `noild.config.get_cascade_path(model, tier)` 给出，`NoildLayer` 按路径串联执行。推理路由缓存会检查池化 batch 形状与设备，并在路由状态锁内处理请求切换；可传入稳定的 `request_id` 在请求变化时自动失效缓存。缓存只保留一个活跃请求，交错使用多个 ID 时会重新路由但不会串用。未传 ID 时，`step=0` 或相对上一步不连续会使缓存失效；如果新请求恰好以连续的非零 step 开始，调用方仍须传 ID 或调用 `reset_generation_state()`，否则无法从张量本身推断请求边界。

## 加载策略（基础版）

默认冷启动全量加载，再按热度管理冷门专家。`ExpertHeatManager` 默认只做状态/热度调度；
实际搬运可由 `NoildLayer(..., offload_device="cpu")` 在 `eval()` + `no_grad()` 推理时执行，或由调用方同时
传入 `preload_expert`/`unload_expert` 回调。搬运任务在线程池串行执行，专家前向前等待加载完成。
默认未配置搬运时只更新状态，不发生物理搬运，也不节省显存。异步搬运并不自动预加载当前下一专家，
调用方需要显式调用预加载接口；吞吐收益、并发安全与峰值显存仍需按目标设备实测。

## 训练方式

各专家共享同一底座（词嵌入加前几层）再分叉训练，最后做对齐微调；
路由、预生成、合并分阶段联训，加负载约束防止垂直专家被饿死。Draft 输出头零初始化，使训练初期 logits 为零、loss 从 ln(vocab_size) 起步，随后由辅助损失学习非均匀预测。
`train_noild` 从 next-token `target_ids` 自动构造 draft/MTP 未来目标并屏蔽序列尾部无效标签；
辅助阶段被路径启用但未返回 logits 会报错，未启用阶段会显式记为 not applicable。
更高预测步采用递减权重（`train_noild` 参数 `draft_horizon_decay`，默认 0.5），默认近端步权重大；这只是训练目标改进，
是否提高验证质量或推测解码接受率仍需真实数据与目标模型评估。见 `noild.training`。

## 不采用

提前退出、动态深度——明确不做。

## 适用范围

生产部署时，专家拆分后容量更小，Noild 不建议用于约 30B 以下的模型。
小模型仅用于研究性验证分工与实现流程，不代表可部署能力、质量验收或低参数档支持。

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
│   ├── jamba_integration.py # 替换真实 Jamba MoE FFN；不含 MLA/CSA2
│   └── qwen_hybrid_integration.py # 替换调用方核实并显式指定的 Qwen MLP 路径
├── docs/
│   └── DESIGN.md            # 路由设计、草稿消融方案
└── tests/                   # 单元测试（unittest，无需 pytest）
```

## 与 Native 的关系

`Nooht_Native/noild/` 是本包的 vendored 副本（同源同步），
Native 主线默认不走 Noild 路径（任务包第二节：Native 没有 Noild），
Noild 在 Native 中为可插拔，供 Fusion/Evolution 复用与对照实验。
