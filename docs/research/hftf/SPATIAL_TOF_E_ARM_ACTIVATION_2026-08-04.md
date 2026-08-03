# Spatial / multizone ToF E-arm activation

Spatial Calibration Head R1 触发了预冻结的 ToF 切换条件：`0/4` 折联合优于常数，
validation false-clear `19.45%`，ECE `0.601`。因此 E 臂已正式激活，但当前只到
硬件与采集协议准备阶段，不代表已采购或得到 ToF 性能证据。

比较必须保持同一最终摄像头、安装位姿、session 与评价门：

- D 臂：冻结的 DA V2 + Spatial Calibration Head R1，不在 E 臂评价 session 上重训；
- E 臂：同一 RGB 路径 + 已注册多区 ToF；
- 真值：独立源生米制真值，禁止用 Metric3D 或其他模型输出替代；
- 硬件：沿用现有 `VL53L8CX` 默认、`VL53L5CX` 供货回退的选择，不允许看结果后切换。

开始采集前仍必须冻结最终相机、同步与注册、ToF zone 质量拒绝、融合规则、parent
名单和 sealed 激活回执。机器可读激活记录见
[`SPATIAL_TOF_E_ARM_ACTIVATION_2026-08-04.json`](SPATIAL_TOF_E_ARM_ACTIVATION_2026-08-04.json)。
