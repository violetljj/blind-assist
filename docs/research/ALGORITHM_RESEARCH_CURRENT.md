# 算法研究入口

状态：`current / PRODUCT_MAINLINE=GOAL_DRIVEN_VISUAL_COPILOT / ALGORITHM_MAINLINE=GRAIL_R / R1C_L_TASK_TRAINED_PAIRWISE_OWNER_COORDINATE_AUTHORIZED / FINAL_TEST_UNOPENED / DEFAULT_APP_UNCHANGED`

本页只保留可操作的当前路线表。详细指标、轮次和终态由“唯一真源”承担；历史路线不因仍有代码或文档而恢复执行权限。

| 路线 | 主张 | 当前状态 | 唯一真源 | 下一动作（唯一 successor） | 禁止动作 | 影响默认 App |
|---|---|---|---|---|---|---|
| GRAIL-R | task-trained pairwise RGB/mask coordinate 能否恢复 privileged owner-local slot mechanism | `ACTIVE_R1C_L / VALIDATION_GATED / FINAL_UNOPENED` | [Goal Copilot current](goal-copilot/README.md) | 执行 frozen R1C-L train/validation；过 `+8` 门才打开一次 final | 调 consumed R1C-P；architecture sweep；depth、M2、App | 否 |
| Goal Copilot 历史证据 | public-real、identity、semantic anchor 与 last-mile geometry 的结论是否仍可追溯 | `ARCHIVED_OR_CLOSED / NO_EXECUTION` | [Goal Copilot current](goal-copilot/README.md) | 无 | 从旧 successor 恢复 P1、SAGE、SAGE-LM 或 L10M | 否 |
| D-ORACLE-1 | oracle intervention 定位 representation 与 policy stack 损失 | `PAUSED / NO_EXECUTION` | [D-ORACLE current](failure-synthesis/README.md) | 无 | 读取 action truth、增加新臂或调 policy | 否 |
| Assistive Geometry | task geometry representation 是否有独立新信息 | `CLOSED / NO_ACTIVE_SUCCESSOR` | [Assistive Geometry current](assistive-geometry/README.md) | 无 | 重跑或调已消费 Q-Plane、boundary、correction 路线 | 否 |
| TARO | 主动观测是否提高风险可辨识性 | `CLOSED / NO_RESCUE` | [TARO current](taro/README.md) | 无 | 重开 R31–R38 或 outcome-guided successor | 否 |
| DepthART / HFTF | bounded deferral 与部署前置是否提供可晋级增量 | `PAUSED / NO_ACTIVE_EXECUTION` | [DepthART current](hftf/README.md) | 无 | 重调 opened parent、读取 sealed R2 或晋级 App | 否 |
| Dual-loop | segmentation/risk-event 次线是否能形成独立增量 | `THESIS_DEVELOPMENT_SECONDARY` | [Dual-loop current](dual-loop/README.md) | `RISKSEG_EVENT_EVAL_DATA_REPAIR_SUCCESSOR` | truth 未修复时训练、选模或晋级 | 否 |
| RCLE-RF | 历史风险场研究 | `PAUSED / NO_ACTIVE_EXECUTION` | [RCLE current](rcle/README.md) | 无 | 消费旧授权或恢复历史 successor | 否 |
| USTRF-SC | route-conditioned 历史诊断 | `CLOSED / HISTORICAL_DIAGNOSTIC` | [USTRF closure](ustrf-sc/USTRF_ROUTE_CONDITIONED_PROGRAM_CLOSURE_R1_2026-07-25.md) | 无 | 把旧实验 App、协议或代理结果写成主线 | 否 |

## 算法边界

数据、链路、性能和部署证据可以支撑算法，但不能替代算法 admission；算法结果也不能反向证明
产品安全、默认 App 可用性或真实用户效果。`UNKNOWN` 与 `NOT_EVALUABLE` 不得写成 negative。
