# DA V2 RGB-D 轻量 student A4 R0

日期：2026-08-05（训练和 P1 输出前冻结）

## 决策

A4 不再延续 A3 的 teacher-only 时序调参，而是固定同一个 `1.27M` 参数 MobileNetV3-Small
student，引入 ARKitScenes `confidence == 2` 的真实米制深度监督。DA V2 teacher 只以 `0.1`
权重约束中心化 log-depth 相对结构，不充当米制真值。

这是一个新信息源 arm，不是对已经看到的 A3 失败样本调阈值。A1、A2、A3 的 R0 终点保持
不变，也不在 R1 下重算。

## 冻结训练

- 2,400 train / 600 parent-disjoint validation RGB-D 帧；
- sensor truth：深度毫米转米、`confidence == 2`、范围 `[0.25, 6] m`；
- 固定 seed `20260805`、5 epochs、batch 8、AdamW `2e-4`；
- 主损失为 masked log-depth SmoothL1 与 masked log-gradient；
- teacher centered-log 结构正则权重 `0.1`；
- 只按固定 validation total 选择 checkpoint，平局取更早 epoch；
- 不做 augmentation、seed/head/loss 搜索。

## 判定

checkpoint 与 P1 深度缓存先锁哈希，再一次性运行前瞻 P1 R1。R1 同时拦截 false-clear 和
false-block/保守占用塌缩。只有全部质量门通过，才允许进入 Android 同 APK 端到端 profile；
否则直接停止该 arm。

ARKitScenes 与 TUM 都已属于 Development 消费数据。即使 A4 过门，也只能支持工程 Pareto
和设备 profile，不支持产品或安全结论。
