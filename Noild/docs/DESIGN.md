# Noild 设计文档（v0.1.0-dev）

## 1. 路由设计

- 路由专家只输出领域，不输出档位（`RouterOutput` 无档位字段）。
- 每次用户输入必判（`judge_user_input`）；推理时生成中每 N token 重判，
  `NoildLayer` 在间隔内复用缓存，负载大时 `set_load_factor` 扩大 N（上限 64）。
- 提前重判两种触发：归一化领域熵超过阈值（由冷却期限制最多每 N 步触发一次）、
  hidden 与上次路由 hidden 余弦漂移超过阈值（话题转变）。未训练路由需校准阈值；
  训练模式每次重算路由并保留梯度。
- 重判节奏与思考记忆块更新节奏共用同一 interval 配置，
  避免两套节奏打架（任务包第七节）。

## 2. 专家分工与级联

串联接力而非并行分支：`NoildLayer.forward` 按 `get_cascade_path` 顺序执行。
预生成专家先猜（draft hidden + draft logits），垂直专家按路由领域（1~2 个）
加权输出，合并专家门控融合 primary（draft 或 general）与 vertical。
合并窗口从 2 起步随步数放大到 10+；窗口外直接透传 primary，
不经融合门控与合并 MLP。通用专家按“领域 + 通用阶段序号”确定性映射；
Soleinoo 的前置通用与末尾润色阶段会轮到不同通用专家（专家数大于 1 时），
同一领域同一阶段仍保持确定性。

## 3. 加载策略（基础版）

`ExpertHeatManager`：冷启动全量加载；`register_route` 提升热度并排入预加载；
`drain_preload(step)` 只在搬运完成后标记加载。推理时可将 `NoildLayer` 配置为
`offload_device="cpu"`，执行内置模块迁移；也可提供成对的自定义搬运回调。搬运任务
在线程池串行执行，专家前向前由 `ensure_loaded` 等待。默认不搬运，不节省显存。

## 4. 训练

分阶段联训（`STAGED_TRAINING_PLAN`）：router+vertical → +draft → +merge 对齐。
负载约束 `load_balance_loss` 用 KL(mean_prob || uniform) 惩罚领域不均衡，
防止垂直专家被饿死。`draft_loss`/`mtp_loss` 必须由训练循环显式加入，
否则对应预测头没有梯度。`upcycle_init` 把共享 MLP 权重复制到垂直专家，
供 Fusion/Evolution 权重复用实验（复用效果需实验，不保证）。
`train_noild` 提供基于底座 hidden states 的 batch 训练循环；调用方负责准备真实数据与预测目标。
`examples/train_noild_smoke.py` 是随机数据的梯度冒烟示例，不代表训练质量。

## 5. 草稿机制消融方案（任务包第七节）

预生成专家、DSpark、MTP 功能重叠（先猜后验），主线只保留一种。
消融指标：草稿接受率（draft logits 与目标输出的连续前缀一致比例）、
加速比（同等输出 token 数下的延迟比）。`speculative_greedy_decode` 提供
`speculative_greedy_decode` 支持普通 causal logits（逐 token 草稿）及未来多 token logits，
包括 DraftExpert 的四维输出；具备贪心验证/bonus token 逻辑，但本函数本身不使用 KV cache，
也未接入 Native 生成路径。概率式采样与基准加速评估仍未实现，不宣称降延迟或降成本。

## 6. 不做

提前退出、动态深度（明确不采用）。当前验证以小规模研究性实验为主，尚无真实大规模 checkpoint 或部署基准；
现阶段不据参数量给出生产适用下限，特定规模是否适用需通过真实模型上的质量、性能与资源基准验证。
