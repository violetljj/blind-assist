# BlindAssist Assistive Geometry

状态：`current / RESEARCH_MAINLINE / B0_DATA_CAPABILITY_AND_ROSTER_LOCK_PASS / B1_TRAINING_NOT_AUTHORIZED`

本路线把 DepthART-S 从研究终点降为可替换的轻量 encoder/initialization 候选，核心问题改为：

> 能否学习直接面向身体通行空间的 Ground、Clearance、Confidence、UNKNOWN 与
> Body-swept Occupancy，同时保持移动端可部署性？

当前真源：

- [路线决策](BLINDASSIST_ASSISTIVE_GEOMETRY_PROGRAM_ROUTE_DECISION_2026-08-09.md)
- [机器合同](BLINDASSIST_ASSISTIVE_GEOMETRY_PROGRAM_ROUTE_DECISION_2026-08-09.json)
- [B0 task contract](BLINDASSIST_ASSISTIVE_GEOMETRY_B0_TASK_CONTRACT_2026-08-09.md)
- [B0 machine contract](BLINDASSIST_ASSISTIVE_GEOMETRY_B0_TASK_CONTRACT_2026-08-09.json)
- [B0 input/data preflight result](BLINDASSIST_ASSISTIVE_GEOMETRY_B0_INPUT_DATA_PREFLIGHT_RESULT_2026-08-09.md)
- [B0 preflight machine result](BLINDASSIST_ASSISTIVE_GEOMETRY_B0_INPUT_DATA_PREFLIGHT_RESULT_2026-08-09.json)
- [B0 runtime geometry receipt](BLINDASSIST_ASSISTIVE_GEOMETRY_B0_RUNTIME_GEOMETRY_RECEIPT_2026-08-09.md)
- [B0 runtime geometry machine receipt](BLINDASSIST_ASSISTIVE_GEOMETRY_B0_RUNTIME_GEOMETRY_RECEIPT_2026-08-09.json)
- [B0 data capability and roster lock result](BLINDASSIST_ASSISTIVE_GEOMETRY_B0_DATA_CAPABILITY_AND_ROSTER_LOCK_RESULT_2026-08-09.md)
- [B0 data capability and roster machine result](BLINDASSIST_ASSISTIVE_GEOMETRY_B0_DATA_CAPABILITY_AND_ROSTER_LOCK_RESULT_2026-08-09.json)
- [B0 data-use authorization receipt](BLINDASSIST_ASSISTIVE_GEOMETRY_B0_DATA_USE_AUTHORIZATION_RECEIPT_2026-08-09.md)
- [算法研究入口](../ALGORITHM_RESEARCH_CURRENT.md)
- [DepthART 部署支线](../hftf/README.md)

## 唯一 successor

`BLINDASSIST_ASSISTIVE_GEOMETRY_B0_TRUTH_READER_AND_REGISTRATION_LOCK`

`1×3×608×448` shape/export 与 SM-S9280 benchmark geometry receipt 已通过。新的 ARKitScenes
`16/8/8` visit/video-disjoint roster 已冻结；Attempt 4 已物化 `9,600` 个 pose-covered 帧，
并通过 `28,800` 张图像实际解码、`9,600` 个内参映射及 trajectory 包络审计。下一步只冻结
depth unit、RGB-depth/K registration、pose interpolation、ground 与 body-swept clearance truth
reader。reader 关闭前不训练 student、不读取 DEVELOPMENT/CONFIRMATION outcome、不启动
DA3 + Metric3D 双教师全套，也不改默认 App。
