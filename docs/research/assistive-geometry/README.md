# BlindAssist Assistive Geometry

状态：`current / RESEARCH_MAINLINE / B0_CONTRACT_NOT_FROZEN / NO_TRAINING_AUTHORITY`

本路线把 DepthART-S 从研究终点降为可替换的轻量 encoder/initialization 候选，核心问题改为：

> 能否学习直接面向身体通行空间的 Ground、Clearance、Confidence、UNKNOWN 与
> Body-swept Occupancy，同时保持移动端可部署性？

当前真源：

- [路线决策](BLINDASSIST_ASSISTIVE_GEOMETRY_PROGRAM_ROUTE_DECISION_2026-08-09.md)
- [机器合同](BLINDASSIST_ASSISTIVE_GEOMETRY_PROGRAM_ROUTE_DECISION_2026-08-09.json)
- [算法研究入口](../ALGORITHM_RESEARCH_CURRENT.md)
- [DepthART 部署支线](../hftf/README.md)

## 唯一 successor

`BLINDASSIST_ASSISTIVE_GEOMETRY_B0_TASK_CONTRACT`

先冻结产品成像几何、身体扫掠包络、`GeometryState` 输出语义、truth/pseudo-label 角色、
UNKNOWN 规则、基线与消融、指标和停止门。B0 完成前不训练 student、不读取独立 outcome、
不启动 DA3 + Metric3D 双教师全套，也不改默认 App。
