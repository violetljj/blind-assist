# 系统与平台研究入口

状态：`current / DEVELOPMENT_PLATFORM_WORKSTREAM / PARALLEL_OR_COUPLED`

## 当前总表

| 研究面 | 当前问题 | 状态 | 唯一真源 | 唯一 successor |
|---|---|---|---|---|
| 通信链路与端到端延迟 | 外设、网络、拷贝、排队、推理、反馈、TTL、丢帧和 latest-only 时序 | `DEVELOPMENT_PLATFORM_WORKSTREAM` | 本页；测量证据见 [AtomS3R/ToF4M E2E result](hftf/ATOMS3R_ANDROID_E2E_TIMING_R0_RESULT_2026-08-06.md) | `MEASURED_BOTTLENECK_SUCCESSOR`：bounded smoke 后只优化已测得的主瓶颈 |
| 性能优化 | CPU/GPU/HTP、内存、热、稳定性和 runtime attribution | `DEVELOPMENT_PLATFORM_WORKSTREAM` | 本页及对应日期化 benchmark | `SINGLE_VARIABLE_ATTRIBUTION_SUCCESSOR`：冻结输入和测量口径后做单变量比较 |
| 部署可行性 | ONNX/QNN graph、operator/lowering、backend 和设备约束 | `DEPTHART_STATE_DELEGATED_TO_ROUTE_CURRENT` | [DepthART current](hftf/README.md) | `DEPTHART_SUCCESSOR_DELEGATED_TO_ROUTE_CURRENT`：只执行唯一真源声明的 successor；本页不复制阶段、门禁、断点或 outcome 状态 |

这些研究可以独立推进，也可以在接口和证据边界冻结后与算法路线耦合。性能、RTT、
accelerator occupancy 或导出成功都不能单独证明算法准确率、产品安全，也不改变默认 App。
DepthART 同时跨算法与部署 workstream，其动态状态和唯一 successor 只在路线 current 维护；
本页只负责系统分类，避免形成第二份可漂移的执行状态。
