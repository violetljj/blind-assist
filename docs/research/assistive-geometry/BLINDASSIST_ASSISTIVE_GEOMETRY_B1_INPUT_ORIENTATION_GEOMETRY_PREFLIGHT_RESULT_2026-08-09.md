# BlindAssist Assistive Geometry B1 orientation geometry preflight

状态：`ATTEMPT_01_SUPERSEDED_PRE_OUTCOME / NO_IMPLEMENTATION_OR_TRAINING_AUTHORITY`

Attempt 1 的内部 schema 校验虽然通过，但 implementation 前的 pose-only audit 发现它把全部
ARKitScenes 帧静默假定为 portrait `608×448`。冻结 TRAIN 的真实逐帧 upright geometry 是：

| orientation family | frames | fraction |
|---|---:|---:|
| portrait `608×448` | 2,724 | 56.75% |
| landscape `448×608` | 2,076 | 43.25% |

继续单一 portrait shape 必须丢弃 43.25% TRAIN、裁掉 landscape 大量 full FOV、把重力方向
旋歪，或非等价拉伸。四种修复均被拒绝。

另一个更严重的问题是 Attempt 1 的 DEVELOPMENT_CALIBRATION 四个 parent 只有 30/1,200
portrait 帧，且没有 portrait-dominant parent，不能校准产品 portrait confidence threshold。
该问题在任何 RGB/depth/task outcome 或模型输出打开前发现，因此可以安全重冻数据角色。

完整机器结果见
[JSON](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_INPUT_ORIENTATION_GEOMETRY_PREFLIGHT_RESULT_2026-08-09.json)，
receipt 位于
`artifacts.local/evidence/hftf/assistive-geometry-b1-orientation-geometry-audit-20260809/receipt.json`
（SHA-256 `DC3F9A68...B0A8`）。

Attempt 1 保留为历史负结果，不得继续授权 implementation/training。当前路线转 Attempt 2：
dual-orientation full-FOV tensor、orientation-bucketed batch、逐 shape K update、orientation
strata reporting，并在 calibration/selection 各保留一个 portrait-dominant parent。
