# JRDB person 3D trajectory sensor support and bias canary R0 目标（2026-07-25）

状态：`FROZEN_BEFORE_EXECUTION`

最大权限：`SEEN_DEVELOPMENT_DIAGNOSTIC_ONLY`

## 唯一研究问题

在父 R1 已证明“可计算”的 `1,350` 个 source-annotation-derived 3D object-frame 与 `1,336` 个相邻 motion pair 中，有多少得到真实双 LiDAR PCD 的独立点云支持，annotation center 与点云质心偏差多大，偏差是否随 3D-only、2D-only、遮挡、距离和稀疏点云系统变化？

## 冻结输入与唯一主要变量

- 输入固定为 `meyer-green-2019-03-16_0 / 000000..000119` immutable observation packet、R1 eligibility ledger 与 packet 已绑定的 upper/lower PCD；
- 逐帧分别解码双 PCD，按官方静态变换进入 `logical_rgb360`，再在 annotation 3D oriented box 内查询点；
- 唯一主要变量是“annotation-derived object-frame/pair 是否得到独立 LiDAR 点支持及其残差”；不改 label、box、pose、轨迹或 R1 分母。

## 四类局部处置

- `sensor-supported`：双 PCD 均可审计，融合 box 内有限点 `>=3`；
- `annotation-only`：双 PCD 均可审计，但 box 内为 `0` 点；
- `abstained`：box 内仅 `1..2` 点、缺对应 PCD、缺 3D box/pose/time，或 pair 任一端不足；只弃权对应 object-frame/pair；
- `invalid`：结构损坏、hash 漂移、重复 ID 或非有限值；只有不可局部化传播才允许全局失败。

`2D-only` 进入 union object-frame ledger 并因缺 3D box 局部弃权；`3D-only` 仍可独立接受 LiDAR 支持审计。不得以交集缩分母。

## 指标

- object-frame / pair 四类计数与守恒；
- upper-only / lower-only / both / neither 支持模式；
- 点云质心减 annotation center 的 xyz、水平与 3D 残差分布；
- annotation 与 sensor centroid 的相邻位移、速度、加速度与差值；
- dynamic odom 与 left-pose-frozen 位移之差作为 pose 敏感性诊断；
- 按 cross-modal presence、2D occlusion、点数、距离分层的 coverage、残差和最差组。

跳变 `>0.5m/pair`、速度 `>4.5m/s`、加速度 `>12m/s²` 只作冻结诊断 flag，不是人体运动真值或 promotion gate。

## 合法终态

1. `INVALID_GLOBAL_INTEGRITY`；
2. `NOT_EVALUABLE_POINTCLOUD_SUPPORT`；
3. `SENSOR_SUPPORT_AND_BIAS_PROFILE_AVAILABLE_WITH_ABSTENTION`；
4. `SENSOR_SUPPORT_AND_BIAS_PROFILE_AVAILABLE_COMPLETE`。

## 明确不能声称

box 查询本身由 annotation 定义，因此 LiDAR 点是独立 sensor evidence，但支持判定不是 annotation-free detector。该结果不能证明 annotation ground truth 精度、人体真实速度、intended route、event lifecycle、USTRF 算法效果、提醒有效性、Android、人身安全、独立行走或生产权限。
