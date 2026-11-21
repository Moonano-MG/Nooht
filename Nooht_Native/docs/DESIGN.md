# Native（Lumenoo）设计文档（v0.1.0-dev）

底座：Jamba 黑盒（Attention + Mamba + MoE），官方权重原样加载，
Mamba/注意力/MoE 层一行不改，只在外面加外层优化层，再微调。
不需要预训练，不需要改模型结构。Native 主线默认关闭 Noild；显式配置并传入
`noild_layer` 后，生成循环在模型 hidden states 与输出投影之间调用 vendored Noild。

## 1. 多硬件后端（第三节 1）

`native/backend.py`：运行时按 CUDA > ROCm > XLA > oneAPI 检测，
缺省 CPU 回退。算子覆盖 Mamba selective scan 与 Gated DeltaNet；
后者依赖第三方线性注意力算子库，非 CUDA 后端可能只有慢速回退，需确认。
`check_correctness` 将注册实现与纯 PyTorch 参考实现比较前向/反向结果，
`benchmark` 记录当前设备基础性能；未注册专用 kernel 时使用的仍是参考公式，
不是上游 Mamba/Gated DeltaNet 算子的替代品。深度 kernel 优化只留接口。

## 2. 选择性记忆压缩（第三节 2）

`native/memory_compression.py`：只处理输入上下文，不碰模型内部状态。
上下文达到一半（`threshold_ratio=0.5`）后按语义片段打分，
低分片段换为确定性的抽取式摘要或降权。打分 = 辅助打分头（目前未训练）
+ 规则兜底（新近性/长度/与指令锚点关键词重合）；辅助头需真实训练和特征接入后
才可用于质量结论。受保护区不参与压缩。消融：`ablation()` 对比长度与片段数。

## 3. 受保护区（第三节 3）

`native/protected_zone.py`：人设锚点 + 指令锚点 + 思考记忆块合并成同一机制。
共同机制：不参与压缩、强制可见、固定位置（排到上下文末尾附近）、
完整注意力层可直接读到。人设方案 A/B 对比见 `persona_anchor_ab.md`。

## 4. 复制机制（第三节 4）

`native/copy_mechanism.py`：思考里写出内容后，输出阶段发复制区间指令
`<copy:start-end>`，解码器逐字拷贝。Native 生成循环只从思考区解析指令，
代码范围需 `<final>` 定稿标记。复制不纠错；KV 并行写入尚未实现。
训练数据自动构造：`find_longest_overlap` 找最长逐字重合区间。


## 5. 推理力度接口（第七节 5）

`NativeConfig.thinking_budget`：当前解码器将其作为生成步数上限截断；
真正分离的思考/回答预算控制与策略训练尚未接入。

## 6. 训练加速栈（第十节）

`training/train_small.yaml`（FlashAttention + Packing + ZeRO-2/FSDP + 异步 CKPT）
与 `train_large.yaml`（拓扑感知 + DualPipe/1F1B + 3FS + 异步 CKPT + Packing），
通过配置切换，代码结构一致。小规模验证，大规模只做接口与流程验证。

## 7. 测试（第十二节相关项）

- 各后端正确性与基准：`backend.check_correctness` / `benchmark`
- 记忆压缩开启/关闭消融：`compressor.ablation`
- 人设一致性（开启/关闭）：`generation.persona_consistency_check`
- 复制区间逐字匹配率：`CopyDecoder.copy_span` 逐字断言（见 tests）
- 思考记忆保真率：`ThinkingMemoryBlock.edit_entry` 增量编辑断言
