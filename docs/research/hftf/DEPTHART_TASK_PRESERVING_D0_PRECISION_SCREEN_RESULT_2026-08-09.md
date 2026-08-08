# DepthART task-preserving D0 precision screen result

终态：`D0_NO_TASK_PRESERVING_CANDIDATE_R2_NOT_ACTIVATED`

机器结果：[`DEPTHART_TASK_PRESERVING_D0_PRECISION_SCREEN_RESULT_2026-08-09.json`](DEPTHART_TASK_PRESERVING_D0_PRECISION_SCREEN_RESULT_2026-08-09.json)

三条冻结 recipe 均在任务 outcome 前的技术前门关闭，因此没有 arm 进入
clearance/false-clear/false-block/temporal，也没有测性能：

| arm | 技术结果 | 首个失败点 |
|---|---|---|
| FP16 | `INELIGIBLE` | 5×SelectiveScan、23×LayerNorm 从冻结 Float32 custom contract 漂移为 Float16 |
| W8A16 | `INELIGIBLE` | Windows host quantizer 缺少可注册 SelectiveScan 的 CPU op package，无法构图 |
| INT8 | `INELIGIBLE` | 与 W8A16 相同，在 bitwidth 生效前即失败 |

W8A16/INT8 确实共用了同一 16-frame TUM calibration roster；它排除了既有 consumed R0
120 rows，未读取 truth/model outcome，也未访问 R2 cohort。两个量化臂的失败不是 calibration
数据漂移，而是 QAIRT host calibration runtime 缺少对应 custom-op execution package。

本终态只关闭当前三条 recipe，不否定 task-preserving deployment。既有 G4-C fixed-mixed 图
仍有 full context/device evidence，但它不是本 D0 的事后救援 arm，也没有自动成为 R2 候选。
唯一干净 successor 是另立 fixed-mixed Development screen。既有 G4-C 是固定 `448×448`
数值 canary 图，不能直接承载矩形 CameraX/TUM 的 left/center/right FOV 主张；必须先冻结产品
纵横比输入尺寸、resize/crop、intrinsics 传播、truth 对齐及 depth→ground→clearance→risk
后处理，再建立新 Development roster。不能回写 D0 或改 custom kernel/host package 来救本轮。
