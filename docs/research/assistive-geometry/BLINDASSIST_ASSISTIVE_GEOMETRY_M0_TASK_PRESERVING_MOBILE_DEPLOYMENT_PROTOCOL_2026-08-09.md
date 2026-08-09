# Assistive Geometry M0 任务保持型移动部署协议

状态：`MECHANICS_FROZEN_SELECTED_MODEL_AND_MOBILE_COHORT_NOT_AVAILABLE`

本协议把“能导出/能在 HTP 跑”与“保持助盲通行任务”分开。未来选定的单帧 checkpoint 必须先导出
portrait `608x448` 与 landscape `448x608` 静态 ONNX，保留五个 SelectiveScan、四级外部 camera
prompt 和 dense-depth/ground/clearance/occupancy/confidence 五个 raw 输出；gravity、transform、UNKNOWN
与最终三态继续由 host 掌握。只允许一个 outcome 前冻结的 QAIRT 2.47 fixed-mixed recipe，目标为
SM8650 HTP v75 全图无 CPU fallback。

现有 DepthART D1 roster 明确排除了 Assistive Geometry B0/B1，不能复用。M0 要求新的 8-primary +
8-reserve `MOBILE_DEVELOPMENT` parent/session/capture-disjoint cohort；先过 raw parity 和完整任务质量门，
再测 `QNN P95 <=150 ms`、full GeometryState `P95 <=180 ms` 与 sustained `>=5 Hz`。质量失败时禁止用
延迟绕过，也不改写既有 strict G4-D negative。

当前选定模型、移动 cohort、QAIRT recipe 与设备执行均不存在；不授权转换、outcome、默认 App、
DA2 替换、产品或 safety。
