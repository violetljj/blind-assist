# 算法研究入口

状态：`current / ALGORITHM_FOCUS=DEPTHART-S`

## 路线总表

“下一动作”即该路线唯一 successor；写“无”表示路线只能保持 `closed`、`paused` 或
`diagnostic`，不能从旧文档推导隐含下一步。

| 路线 | 主张 | 当前状态 | 唯一真源 | 下一动作（唯一 successor） | 禁止动作 | 影响默认 App |
|---|---|---|---|---|---|---|
| DepthART-S | 当前主要算法候选 | `R0_QUALITY_NOT_ADMITTED / R1_RESEARCH_MAINLINE / G4-C_FULL_CONTEXT_PASS_SM8650_V75 / G4-D_FULL_GRAPH_NUMERICAL_PARITY_FAIL_SM8650_V75` | [DepthART current](hftf/README.md) | `TWO_STAGE_NUMERICAL_REPAIR_SUCCESSOR`：先关闭 PyTorch→canonical ONNX 漂移；再只围绕首个 patch-embed Conv 的 HTP layout/precision lowering 边界做单节点族修复并重跑 G4-D；通过后才评价 G4-E/F，并另行激活 `DEPTHART_PARENT_DISJOINT_ADMISSION_SUCCESSOR` | consumed 数据回救；用部署结果替代算法 admission；在 G4-D 通过前进入 partition/performance gate | 否 |
| DA2 | teacher、baseline、reference、fallback | `CLOSED / FROZEN_DEVELOPMENT_REFERENCE` | [DA2 closure](hftf/DAV2_P1_P2_EXECUTION_CLOSURE_2026-08-05.md) | 无；新问题必须建立新版本和独立数据路线 | 把 reference 写成 active candidate 或沿用旧“下一步” | 否 |
| YOLO + 语义分割双环 | 论文次线 | `THESIS_DEVELOPMENT_SECONDARY / RISKSEG_R0_NEGATIVE_NOT_PROMOTABLE` | [dual-loop current](dual-loop/README.md) | `RISKSEG_EVENT_EVAL_DATA_REPAIR_SUCCESSOR` | truth 未冻结时训练、选模或晋级 | 否 |
| RCLE-RF | 历史风险场研究 | `PAUSED / NO_ACTIVE_EXECUTION` | [RCLE current](rcle/README.md) | 无；只有用户明确重开后才能建立新 scoped successor | 消费暂停前授权或把旧 README 的“下一步”视为当前权限 | 否 |
| USTRF-SC route-conditioned | 历史路线代理 | `CLOSED / HISTORICAL_DIAGNOSTIC` | [USTRF closure](ustrf-sc/USTRF_ROUTE_CONDITIONED_PROGRAM_CLOSURE_R1_2026-07-25.md) | 无；实质不同的假设必须登记为新路线 | 把实验 App、旧协议或诊断结果写成当前主线 | 否 |

## 算法边界

数据、链路、性能和部署结果可以为算法提供输入，但不能直接替代算法 admission。算法研究结果也不能反向证明产品安全或默认 App 可用。
