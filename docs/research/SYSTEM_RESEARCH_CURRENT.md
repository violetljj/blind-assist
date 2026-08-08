# 系统与平台研究入口

状态：`current / DEVELOPMENT_PLATFORM_WORKSTREAM / PARALLEL_OR_COUPLED`

## 当前总表

| 研究面 | 当前问题 | 状态 | 唯一真源 | 唯一 successor |
|---|---|---|---|---|
| 通信链路与端到端延迟 | 外设、网络、拷贝、排队、推理、反馈、TTL、丢帧和 latest-only 时序 | `DEVELOPMENT_PLATFORM_WORKSTREAM` | 本页；测量证据见 [AtomS3R/ToF4M E2E result](hftf/ATOMS3R_ANDROID_E2E_TIMING_R0_RESULT_2026-08-06.md) | `MEASURED_BOTTLENECK_SUCCESSOR`：bounded smoke 后只优化已测得的主瓶颈 |
| 性能优化 | CPU/GPU/HTP、内存、热、稳定性和 runtime attribution | `DEVELOPMENT_PLATFORM_WORKSTREAM` | 本页及对应日期化 benchmark | `SINGLE_VARIABLE_ATTRIBUTION_SUCCESSOR`：冻结输入和测量口径后做单变量比较 |
| 部署可行性 | ONNX/QNN graph、operator/lowering、backend 和设备约束 | `STRICT_G4D_NEGATIVE_TERMINAL / TASK_PRESERVING_R2_PROTOCOL_FROZEN_NOT_ACTIVATED`（DepthART） | [DepthART A3 result](hftf/DEPTHART_ADMISSION_R1_A3_RESULT_2026-08-07.md)；[task-preserving R2 protocol](hftf/DEPTHART_TASK_PRESERVING_DEPLOYMENT_R2_PROTOCOL_2026-08-09.md) | `DEPTHART_TASK_PRESERVING_DEPLOYMENT_R2`：不重开或改写 strict G4-D；先冻结新独立 cohort、单一 HTP-friendly candidate 与任务非劣/绝对门，显式激活后才访问 outcome。只有任务质量 PASS，才进入该候选自己的 partition/performance gate |

这些研究可以独立推进，也可以在接口和证据边界冻结后与算法路线耦合。性能、RTT、
accelerator occupancy 或导出成功都不能单独证明算法准确率、产品安全，也不改变默认 App。
