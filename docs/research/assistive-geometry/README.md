# BlindAssist Assistive Geometry

状态：`current / RESEARCH_MAINLINE / B0_GEOMETRY_RECEIPT_PASS / DATA_ROSTERS_UNRESOLVED / EXECUTION_NOT_AUTHORIZED`

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
- [算法研究入口](../ALGORITHM_RESEARCH_CURRENT.md)
- [DepthART 部署支线](../hftf/README.md)

## 唯一 successor

`BLINDASSIST_ASSISTIVE_GEOMETRY_B0_DATA_CAPABILITY_AND_ROSTER_LOCK`

`1×3×608×448` 已完成 PyTorch shape PASS 与 ONNX graph/checker PASS；ONNX output metadata
仍因 SelectiveScan shape inference 缺失而 symbolic。SM-S9280 隔离 device benchmark 已观测实际
CameraX geometry 并派生 `448×608` tensor K，但 authority 仅 benchmark-only。下一步只做新数据
label-capability/identity/license/roster lock。blocker 关闭前不训练 student、不读取
独立 outcome、不启动 DA3 + Metric3D 双教师全套，也不改默认 App。
