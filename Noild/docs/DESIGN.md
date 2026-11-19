# Noild 设计文档（v0.1.0-dev）

## 1. 路由设计

- 路由专家只输出领域，不输出档位（`RouterOutput` 无档位字段）。
- 每次用户输入必判（`judge_user_input`）；生成中每 N token 重判，
  N 由 `RejudgeController` 控制：负载大时 `update_load` 扩大 N（上限 64）。
- 提前重判两种触发：领域分布熵超过阈值（不确定度突升）、
  hidden 与上次路由 hidden 余弦漂移超过阈值（话题转变）。
- 重判节奏与思考记忆块更新节奏共用同一 interval 配置，
  避免两套节奏打架（任务包第七节）。

## 2. 专家分工与级联

串联接力而非并行分支：`NoildLayer.forward` 按 `get_cascade_path` 顺序执行。
预生成专家先猜（draft hidden + draft logits），垂直专家按路由领域（1~2 个）
加权输出，合并专家门控融合 primary（draft 或 general）与 vertical。
合并窗口从 2 起步随步数放大到 10+，窗口外 token 直通。

## 3. 加载策略（基础版）

`ExpertHeatManager`：冷启动全量加载；`register_route` 提升热度并把
垂直专家加入异步预加载队列，`drain_preload` 标记加载完成；
`step` 衰减热度并卸载冷门专家（冷却期与最小保留数保护）。
只管理热度与集合，不做显存搬运；深度优化不在本阶段。

## 4. 训练

分阶段联训（`STAGED_TRAINING_PLAN`）：router+vertical → +draft → +merge 对齐。
负载约束 `load_balance_loss` 用 KL(mean_prob || uniform) 惩罚领域不均衡，
防止垂直专家被饿死。`upcycle_init` 把共享 MLP 权重复制到垂直专家，
供 Fusion/Evolution 权重复用实验（复用效果需实验，不保证）。

## 5. 草稿机制消融方案（任务包第七节）

预生成专家、DSpark、MTP 功能重叠（先猜后验），主线只保留一种。
消融指标：草稿接受率（draft logits 与校验结果一致比例）、
加速比（同等输出 token 数下的延迟比）。
只在小批量低延迟场景宣称降延迟，不宣称降成本。
本阶段交付接口（`DraftExpert`、`MTPHead`），不承诺训练效果。

## 6. 不做

提前退出、动态深度（明确不采用）；30B 以下模型不建议使用，
小规模只验证分工是否有效。
