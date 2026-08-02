# HFTF Stage C D45 phone metric-depth source canary protocol

状态：

`STAGE_C_D45_PHONE_METRIC_DEPTH_SOURCE_CANARY_FROZEN`

## 单一问题

D44 已证明：同一目标连续 7 个 causal relative metric observations，足以在
JRDB Development 上把一秒后水平位置误差降低 `56.36%`。D45 不再改模型、阈值
或 alert kernel，只回答：

> 一台真实 Android 手机能否为 current person target 产生已注册、带质量和时间戳的
> metric-depth observations，并稳定组成同一目标的 7 点 causal history？

## 防火墙

- 只读 camera/depth source capability、current camera frame、current detector person
  box 和人工量测距离；
- 不读 event、alert、碰撞、用户反馈或主线 evaluation outcome；
- 不把 depth 写入 `Detection.distanceEvidence`，不调用 risk/feedback seam；
- 默认 App、主线和 D35 均不改变；
- ARCore 仅进入 isolated instrumentation/source canary，不能因本协议成为运行时
  必需依赖。

## 冻结 source 顺序

1. 先探测仓库已锁定的 `com.google.ar:core:1.33.0`：
   `AUTOMATIC`、`RAW_DEPTH_ONLY`、camera config 及 hardware depth usage；
2. 若 `AUTOMATIC` 可用，首选 registered automatic depth；存在 hardware-depth
   camera config 时只作为 source 属性记录，不另开 arm；
3. 若只有 raw depth，必须先完成 raw-to-camera registration receipt，之后才可进入
   measurement；
4. ARCore 不可用时报告 source-unavailable，不用未校准的逐帧 relative monocular
   depth补位。

Capability probe 不启动 CameraX、不请求安装 ARCore、不写 alert。

## 冻结 measurement contract

每个 accepted person observation 必须携带：

- source frame id、capture/produce timestamp 与 `ANDROID_ELAPSED_REALTIME` clock；
- source kind、registration transform id、depth raster size 和 camera intrinsics；
- person target key、registered current box；
- median optical-axis depth meters；
- valid sample count、box coverage、mean confidence、relative IQR、quality score；
- source-to-receipt latency。

固定 sampler：

- current detector label 必须为 `person`；
- 在 box 的中心 `60% × 60%` 区域采样；
- 有效范围 `[0.20 m, 20.0 m]`，sample confidence `>=0.50`；
- 至少 12 个有效 samples，coverage `>=0.25`；
- mean confidence `>=0.50`，relative IQR `<=0.50`；
- receipt age `<=150 ms`；
- depth 取有效样本 median，不在结果后改 percentile 或 crop。

固定 history solver：

- exact same target key、source 和 registration；
- 只取最近连续 7 个 accepted observations；
- history span `>=200 ms`，相邻 gap `<=200 ms`；
- 对 camera-relative `x/y/z` 分别做普通最小二乘，预测 `+1.0 s`；
- solver 输出仅写 shadow receipt，不进入 decision seam。

## 首个物理 canary

同一手机、同一后摄、同一构建，人工量测 person torso plane 到 camera optical
center 的 `1/2/3/5 m` 四个距离；每个距离至少 20 个 accepted observations。
固定报告：

- capability receipt；
- frame/person/accepted coverage；
- absolute error median/P90 与 relative error median；
- source-to-receipt latency P50/P95；
- 7-point history availability；
- 每个距离分层结果。

支持门：

- 四个距离均有 `>=20` accepted observations；
- overall accepted-person coverage `>=0.60`；
- median absolute error `<=0.50 m` 且 median relative error `<=0.20`；
- P90 absolute error `<=1.00 m`；
- latency P95 `<=150 ms`；
- eligible 7-point history availability `>=0.50`；
- 无任何 risk/feedback invocation，baseline artifact hash 不变。

## 终态

- 所有门通过：
  `D45_PHONE_METRIC_DEPTH_SOURCE_SUPPORTED_DEVELOPMENT_ONLY`
- capability 可用但任一 measurement gate 失败：
  `D45_PHONE_METRIC_DEPTH_SOURCE_NOT_SUPPORTED`
- device/source/registration 不可用，未读 measurement outcome：
  `D45_PHONE_METRIC_DEPTH_SOURCE_NOT_EVALUABLE`
- 当前无可执行设备：
  `D45_NOT_EVALUATED_NO_READY_DEVICE`

任何终态都不建立 event utility、独立泛化、产品或安全主张；只有支持终态才允许
另行冻结 source-to-D44 shadow forecast parity，仍不得接 alert。
