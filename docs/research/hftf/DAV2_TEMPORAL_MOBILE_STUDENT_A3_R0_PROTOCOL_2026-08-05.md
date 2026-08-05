# DA V2 Temporal Mobile Student A3 R0

日期：2026-08-05，在 A3 模型初始化、训练或 P1 cache 生成前冻结。

A2 已证明单帧 teacher distillation 能恢复 metric scale 与 false-clear，但 temporal delta 和
几何 transition 失败。A3 不重调 A2；它换成真正轻量的 MobileNetV3-Small encoder 和固定
三层 FPN depth head，并只在 training-role 连续 teacher frames 上增加 log-depth delta 监督。

固定模型输入为 `294x392`，使用 ImageNet V1 权重
`047DCFF4...994F4E1F`，encoder taps 为 `24/48/576` channels，decoder 固定为 96-channel
additive FPN。总参数必须不超过 1.6M，不搜索 width、tap、head 或输入尺寸。

训练使用 immutable 518 teacher cache。相邻同视频、同 role、`0 < dt <= 0.5 s` 形成
1,937 train pairs 与 417 validation pairs。固定五轮、seed `20260805`、AdamW `2e-4`，
per-frame depth/gradient/scale loss 外加权重 `0.75` 的 teacher log-depth-delta SmoothL1。
checkpoint 只按 teacher-only validation total loss选择。

训练和 P1 RGB cache 锁哈希前禁止读取 TUM 注册深度、clearance、false-clear 或状态结果。
A3 只运行一次 P1；全部工程门通过才可进行端侧 profile，否则保持失败终态。
