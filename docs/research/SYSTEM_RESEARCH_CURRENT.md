# 系统与平台研究入口

状态：`current / DEVELOPMENT_PLATFORM_WORKSTREAM / PARALLEL_OR_COUPLED`

## 当前总表

| 研究面 | 当前问题 | 状态 | 唯一真源 | 唯一 successor |
|---|---|---|---|---|
| 通信链路与端到端延迟 | 外设、网络、拷贝、排队、推理、反馈、TTL、丢帧和 latest-only 时序 | `DEVELOPMENT_PLATFORM_WORKSTREAM` | 本页；测量证据见 [AtomS3R/ToF4M E2E result](hftf/ATOMS3R_ANDROID_E2E_TIMING_R0_RESULT_2026-08-06.md) | `MEASURED_BOTTLENECK_SUCCESSOR`：bounded smoke 后只优化已测得的主瓶颈 |
| 性能优化 | CPU/GPU/HTP、内存、热、稳定性和 runtime attribution | `DEVELOPMENT_PLATFORM_WORKSTREAM` | 本页及对应日期化 benchmark | `SINGLE_VARIABLE_ATTRIBUTION_SUCCESSOR`：冻结输入和测量口径后做单变量比较 |
| 部署可行性 | ONNX/QNN graph、operator/lowering、backend 和设备约束 | `G4-A_PACKAGE_REGISTRATION_PASS / G4-B_OPERATOR_PARITY_PASS_SM8650_V75 / G4-C_FULL_CONTEXT_PASS_SM8650_V75 / G4-D_FULL_GRAPH_NUMERICAL_PARITY_FAIL_SM8650_V75`（DepthART A3） | [DepthART A3 result](hftf/DEPTHART_ADMISSION_R1_A3_RESULT_2026-08-07.md) | `TWO_STAGE_NUMERICAL_REPAIR_SUCCESSOR`：先关闭 PyTorch→canonical ONNX 漂移；HTP 分支只在首个 patch-embed Conv 的 layout/precision lowering 边界做单节点族修复，再用同一冻结 canary 重跑 G4-D；G4-E/F 继续未评价 |

这些研究可以独立推进，也可以在接口和证据边界冻结后与算法路线耦合。性能、RTT、
accelerator occupancy 或导出成功都不能单独证明算法准确率、产品安全，也不改变默认 App。
