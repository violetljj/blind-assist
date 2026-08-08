# 算法研究入口

状态：`current / ALGORITHM_FOCUS=DEPTHART-S`

## 路线总表

“下一动作”即该路线唯一 successor；写“无”表示路线只能保持 `closed`、`paused` 或
`diagnostic`，不能从旧文档推导隐含下一步。

| 路线 | 主张 | 当前状态 | 唯一真源 | 下一动作（唯一 successor） | 禁止动作 | 影响默认 App |
|---|---|---|---|---|---|---|
| DepthART-S | 当前主要算法候选 | `R0_QUALITY_NOT_ADMITTED / R1_RESEARCH_MAINLINE / STRICT_G4D_NEGATIVE_TERMINAL / TASK_PRESERVING_R2_PROTOCOL_FROZEN_NOT_ACTIVATED` | [DepthART current](hftf/README.md) | `DEPTHART_TASK_PRESERVING_DEPLOYMENT_R2`：保持 strict G4-D 终态不变；先建立新独立 parent/session-disjoint cohort 与冻结候选，再按任务质量合同比较 canonical FP32 reference 和 HTP-friendly candidate。质量通过后才可评价该候选自己的 partition/performance | 继续 custom 化标准 Conv/Norm/activation 来回救 strict G4-D；复用 consumed R0 做授权；用部署结果替代算法 admission；未过任务质量即测性能 | 否 |
| DA2 | teacher、baseline、reference、fallback | `CLOSED / FROZEN_DEVELOPMENT_REFERENCE` | [DA2 closure](hftf/DAV2_P1_P2_EXECUTION_CLOSURE_2026-08-05.md) | 无；新问题必须建立新版本和独立数据路线 | 把 reference 写成 active candidate 或沿用旧“下一步” | 否 |
| YOLO + 语义分割双环 | 论文次线 | `THESIS_DEVELOPMENT_SECONDARY / RISKSEG_R0_NEGATIVE_NOT_PROMOTABLE` | [dual-loop current](dual-loop/README.md) | `RISKSEG_EVENT_EVAL_DATA_REPAIR_SUCCESSOR` | truth 未冻结时训练、选模或晋级 | 否 |
| RCLE-RF | 历史风险场研究 | `PAUSED / NO_ACTIVE_EXECUTION` | [RCLE current](rcle/README.md) | 无；只有用户明确重开后才能建立新 scoped successor | 消费暂停前授权或把旧 README 的“下一步”视为当前权限 | 否 |
| USTRF-SC route-conditioned | 历史路线代理 | `CLOSED / HISTORICAL_DIAGNOSTIC` | [USTRF closure](ustrf-sc/USTRF_ROUTE_CONDITIONED_PROGRAM_CLOSURE_R1_2026-07-25.md) | 无；实质不同的假设必须登记为新路线 | 把实验 App、旧协议或诊断结果写成当前主线 | 否 |

## 算法边界

数据、链路、性能和部署结果可以为算法提供输入，但不能直接替代算法 admission。算法研究结果也不能反向证明产品安全或默认 App 可用。
