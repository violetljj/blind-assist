# 系统与平台研究入口

状态：`current / DEVELOPMENT_PLATFORM_WORKSTREAM / PARALLEL_OR_COUPLED`

## 当前总表

| 研究面 | 当前问题 | 状态 | 唯一真源 | 唯一 successor |
|---|---|---|---|---|
| 通信链路与端到端延迟 | 外设、网络、拷贝、排队、推理、反馈、TTL、丢帧和 latest-only 时序 | `DEVELOPMENT_PLATFORM_WORKSTREAM` | 本页；测量证据见 [AtomS3R/ToF4M E2E result](hftf/ATOMS3R_ANDROID_E2E_TIMING_R0_RESULT_2026-08-06.md) | `MEASURED_BOTTLENECK_SUCCESSOR`：bounded smoke 后只优化已测得的主瓶颈 |
| 性能优化 | CPU/GPU/HTP、内存、热、稳定性和 runtime attribution | `DEVELOPMENT_PLATFORM_WORKSTREAM` | 本页及对应日期化 benchmark | `SINGLE_VARIABLE_ATTRIBUTION_SUCCESSOR`：冻结输入和测量口径后做单变量比较 |
| 部署可行性 | ONNX/QNN graph、operator/lowering、backend 和设备约束 | `STRICT_G4D_NEGATIVE_TERMINAL / D0_NO_ELIGIBLE_PRECISION_ARM / D1_MEDIA_PREFLIGHT_PASS / D1_608X448_SINGLE_CANDIDATE_HOST_CONVERSION_PASS / SM8650_CONTEXT_NOT_BUILT_DEVICE_ABSENT / R2_CANDIDATE_NOT_SELECTED`（DepthART） | [DepthART current](hftf/README.md)；[D1 technical preflight](hftf/DEPTHART_TASK_PRESERVING_D1_PRODUCT_ASPECT_TECHNICAL_PREFLIGHT_RESULT_2026-08-10.md)；[D0 terminal](hftf/DEPTHART_TASK_PRESERVING_D0_PRECISION_SCREEN_RESULT_2026-08-09.md) | 连接冻结 SM8650/v75 设备，对 exact `608×448` DLC 建 context 并关闭 synthetic full-graph parity，再做无 outcome activation validation；通过前不得计算 task outcome、测性能、修改 candidate 或访问 R2 cohort |

这些研究可以独立推进，也可以在接口和证据边界冻结后与算法路线耦合。性能、RTT、
accelerator occupancy 或导出成功都不能单独证明算法准确率、产品安全，也不改变默认 App。
