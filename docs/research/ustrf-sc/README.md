# USTRF-SC 研究文档索引

状态：research index
当前生产授权：无；正式 App 状态仍以 `docs/SANPO_CURRENT_STATUS.md`、当前协议和可复现门禁报告为准。

## 首要入口

- [项目工作记录与恢复入口（2026-07-20）](USTRF_SC_PROJECT_RECORD_AND_RESUME_2026-07-20.md)：本轮总体思路、实验结果、GPU 中断边界和下一步。
- [新窗口交接（2026-07-20）](USTRF_SC_WINDOW_HANDOFF_2026-07-20.md)：当前暂停点、证据入口、GPU 边界、参考文档清单和可复制续接提示词。
- [双环实施状态与证据边界](USTRF_SC_IMPLEMENTATION_STATUS.md)：逐模块实现、证据、缺口和授权。
- [研究型离线量化基线（2026-07-20）](USTRF_SC_RESEARCH_METRICS_2026-07-20.md)：SANPO 表征、公开 source-native 几何/轨迹、REveL detector、range/radial-motion 分层与 V13 门禁。
- [安全内核实验方案（2026-07-20）](USTRF_SC_SAFETY_KERNEL_EXPERIMENT_2026-07-20.md)：本轮详细实现与实验演化。

## 协议与专项记录

- [路线条件化无类别风险场主线](../../ROUTE_CONDITIONED_OBJECT_AGNOSTIC_RISK_FIELD_PLAN_2026-07-20.md)：当前优先研究路线；typed route-risk seam 已建立，真实事件与设备米制几何硬门仍阻塞，未授权训练或接入 App。
- [REveL YOLO11n 8/32 帧 crop/tiling 配对实验](USTRF_SC_REVEL_CROP_TILING_PAIRED_2026-07-20.md)：8 帧 canary 恢复 4/8 small miss，但 FP 从 4 增至 14，按预注册停止并跳过 32 帧。
- [设备阶段策略](USTRF_SC_DEVICE_PHASE_POLICY.md)
- [设备几何校准与证据协议](USTRF_SC_CALIBRATION_PROTOCOL.md)
- [临时手持刚体标定执行单](USTRF_SC_PROVISIONAL_HANDHELD_CALIBRATION_RUNBOOK.md)
- [离线安全仿真与自动验证](USTRF_SC_OFFLINE_SAFETY_SIMULATION.md)
- [SANPO 数据回放接入](USTRF_SC_SANPO_REPLAY_INTEGRATION.md)

## 阅读规则

- 想知道“现在做到哪里、下一步从哪里接”：先读项目工作记录，再读实施状态。
- 想复核具体数字：读取对应 `artifacts.local/evidence/` JSON/HTML receipt，不从概述反推。
- 想改变设备、训练或生产权限：以仓库 current 协议和 promotion gate 为准，日期化研究文档不能单独授权。
