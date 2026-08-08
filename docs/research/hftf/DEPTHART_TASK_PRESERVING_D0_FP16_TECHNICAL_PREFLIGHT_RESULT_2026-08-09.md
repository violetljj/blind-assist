# DepthART D0 FP16 technical preflight result

终态：`D0_ARM_TECHNICALLY_INELIGIBLE_NO_RECIPE_REPAIR`

机器结果：[`DEPTHART_TASK_PRESERVING_D0_FP16_TECHNICAL_PREFLIGHT_RESULT_2026-08-09.json`](DEPTHART_TASK_PRESERVING_D0_FP16_TECHNICAL_PREFLIGHT_RESULT_2026-08-09.json)

FP16 arm 完成了 converter、SM8650/v75 context 生成和 saved-context 单帧执行，输出也全部
finite；但它仍在预冻结技术前门失败。DLC inventory 中 5 个 SelectiveScan 和 23 个
DepthArtLayerNorm 的输入输出都从冻结 supplemental OpDef 与既有 parity evidence 所覆盖的
Float32 漂移成 Float16，整个 DLC 没有 Float32 tensor。因此“能 compose/execute”不能证明
这两个 custom families 的 Float16 语义兼容。

同一 synthetic canary 上，FP16 output 对 PyTorch 的 raw-depth MAE 为 `1.07783 m`、max-abs
为 `1.44105 m`。这只作为故障邻近诊断，不是任务等价 gate；FP16 已由更早的 dtype contract
失败淘汰，不进入 clearance/false-clear/temporal，也不测性能。

本结果不修改 strict G4-D 负终态，也不授权修改 custom FP16 kernel 来救 arm。D0 只继续
冻结 recipe 的 W8A16 与 INT8，两者必须共享同一个独立于 R2 cohort 的 Development
calibration roster。
