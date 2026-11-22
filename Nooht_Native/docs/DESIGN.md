# Native（Lumenoo）实现状态（v0.1.0-dev）

## 底座与训练

Native 通过 `load_jamba_native` 从调用方指定 checkpoint 加载 Transformers 模型与 tokenizer；
仓库不捆绑权重，也不自动下载。可显式传入已经核实的 `lora_target_modules` 创建 PEFT LoRA。
`native.lora.train_causal_lm_steps` 和 `training/train_native.py` 提供基本 causal-LM LoRA
训练路径，训练数据由调用方提供。真实 Jamba checkpoint、target module、训练结果和质量
尚未在此仓库验收；不把随机初始化模型或接口测试当作权重验收。

## 1. 多硬件后端（第三节 1）

`native/backend.py` 提供设备检测、kernel 注册表和纯 PyTorch 参考算子。默认没有注册
CUDA/ROCm/XLA/oneAPI 优化 kernel，`run_op` 会退回参考实现。`backend_status` 区分
“检测到设备”和“某算子确有注册实现”。这不是完整的多后端 Mamba/Gated DeltaNet 集成；
注册实现仍需按设备做前向、反向和基准测试。`NativeGenerator.backend` 目前也不接管模型内核。

## 2. 选择性记忆压缩（第三节 2）

`native/memory_compression.py` 支持规则评分、抽取式摘要和压缩长度统计。`ScoringHead`
可通过 `fit_scoring_head` 使用调用方提供的片段 pooled hidden 与重要性标签训练；推理 head
需要传入真实 `segment_hidden`，不会再把文本长度伪装成 hidden。普通 Native 生成路径目前
没有自动提取片段 hidden/打分标签，因此默认使用规则评分；任务质量消融尚未完成。

## 3. 受保护区（第三节 3）

`native/protected_zone.py` 维护人设、指令与结构化思考记忆；`protect_segments` 在压缩前
保护带标记的区段和思考记忆尾部。PersonaAdapter 是独立的 LoRA 式人设模块，hook 会按激活
设备/精度适配参数；本模块不等同于对 Jamba 底座的 PEFT LoRA 训练。现有一致性 helper
只比较两次生成文本是否不同，不是语义人设一致性验收；需用训练后的模型和评测集验证。

## 4. 复制机制（第三节 4）

`native/copy_mechanism.py` 支持 `<copy:start-end>` 格式、范围校验与代码 `<final>` 标记。
当前 KV cache 与复制区间并行融合未实现；复制质量及输出耗时未验收。

## 5. 推理预算（第七节 5）

`NativeConfig.thinking_budget` 当前作为生成 token 上限；真正分离的思考/回答预算策略与训练
尚未接入。

## 6. 训练加速栈（第十节）

`train_native.py` 可执行基本 LoRA causal-LM 训练，支持配置中的序列长度、batch size、
gradient accumulation、learning rate、BF16，以及请求 Transformers 使用 FlashAttention 2。
DeepSpeed/FSDP、跨样本 sequence packing、异步 checkpoint 与大规模并行尚未接入；入口对这些
配置显式告警，而不是报告为已启用。大规模 YAML 仍是目标配置/流程占位，未做实际训练。

## 7. Noild 可插拔路径

`NoildLayer` 按显式级联路径构造专家；没有 `general` 阶段的路径不再分配通用专家参数。
通用阶段的专家由对应路径调用。专家加载管理默认只管理状态；若要实际搬运，须显式提供成对的
`preload_expert`/`unload_expert` 回调，回调由串行后台线程执行，前向前通过 `ensure_loaded`
等待搬运完成；`eval()` + `no_grad()` 推理也可直接设置 `offload_device="cpu"` 使用内置模块迁移。
未配置搬运时不发生物理搬运。

## 8. 验收边界

单测、mock 与随机初始化小模型只验证局部代码路径。真实 Jamba 权重验证、后端性能、记忆任务
质量、人设一致性、真实 drafter 接受率/加速、复制 KV 并行耗时和大规模训练仍需各自独立测量。
