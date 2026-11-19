# Noild 在 Native 中的位置（v0.1.0-dev）

## 文件夹关系

- 独立版：仓库根目录 `Noild/`（Apache 2.0，不依赖 Nooht，其他架构可接入）。
- Native 内置版：`Nooht_Native/noild/`（vendored 副本，与独立版同源同步）。
- 两者文件夹分开，Native 里带一份 Noild（按用户要求）。

## Native 主线与 Noild 的关系（目标不变）

任务包第二节：Native 底座 Jamba 黑盒，不需要改模型结构，Native 没有 Noild。
因此 Native 主线（Lumenoo）默认 `noild_enabled=False`，不走 Noild 路径；
`Nooht_Native/noild/` 存在于 Native 中，供 Fusion/Evolution 复用与对照实验，
以及小规模验证 Noild 分工是否有效（任务包第四节适用范围）。

开启方式：`NativeConfig(noild_enabled=True)` 后，
`NativeGenerator` 接受 vendored `NoildLayer`（见 `native/generation.py`）。
