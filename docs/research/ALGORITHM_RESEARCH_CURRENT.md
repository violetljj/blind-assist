# 算法研究入口

状态：`current / ALGORITHM_FOCUS=BLINDASSIST_ASSISTIVE_GEOMETRY`

## 路线总表

“下一动作”即该路线唯一 successor；写“无”表示路线只能保持 `closed`、`paused` 或
`diagnostic`，不能从旧文档推导隐含下一步。

| 路线 | 主张 | 当前状态 | 唯一真源 | 下一动作（唯一 successor） | 禁止动作 | 影响默认 App |
|---|---|---|---|---|---|---|
| BlindAssist Assistive Geometry | 当前算法主线：学习 Ground / Clearance / Confidence / UNKNOWN / Body-swept Occupancy | `A0_FORMAL_RUNNER_AND_HOST_PERFORMANCE_PILOT_PASS / WORKERS_1_SELECTED / SEED_17_GUARDED_EXECUTION_NOT_STARTED / DEVELOPMENT_AND_CONFIRMATION_SEALED` | [Assistive Geometry current](assistive-geometry/README.md) | `BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_SEED_17_GUARDED_FORMAL_TRAIN_EXECUTION`：按冻结 seed 17、20 epoch、6,000 steps 执行 TRAIN-only A0；完成后依次执行 29/43 | 训练中读取 DEVELOPMENT/CONFIRMATION outcome；best-seed 早停/选择；运行 A1–A4、双教师、HTP 或默认 App | 否 |
| DepthART-S | Assistive Geometry 的优先 encoder/initialization、depth baseline 与部署使能线，不是算法终点 | `R1_RESEARCH_MAINLINE / STRICT_G4D_NEGATIVE_TERMINAL / D0_NO_ELIGIBLE_PRECISION_ARM / R2_CANDIDATE_NOT_SELECTED / ARKIT_ROSTER_8_LOCKED_UNOPENED` | [DepthART current](hftf/README.md) | `DEPTHART_TASK_PRESERVING_D1_FIXED_MIXED_DEVELOPMENT_SCREEN`：消费 B0 冻结的产品纵横比、FOV/resize、intrinsics/truth 对齐与 task postprocess；再重建 fixed-mixed 图并在新 Development roster 评价 clearance/risk | 用 `448×448` canary 替代产品视场；用独立 R2 cohort 选模；事后把 G4-C 塞回 D0；为量化补写 host/custom execution engine | 否 |
| DA2 | teacher、baseline、reference、fallback | `CLOSED / FROZEN_DEVELOPMENT_REFERENCE` | [DA2 closure](hftf/DAV2_P1_P2_EXECUTION_CLOSURE_2026-08-05.md) | 无；新问题必须建立新版本和独立数据路线 | 把 reference 写成 active candidate 或沿用旧“下一步” | 否 |
| YOLO + 语义分割双环 | 论文次线 | `THESIS_DEVELOPMENT_SECONDARY / RISKSEG_R0_NEGATIVE_NOT_PROMOTABLE` | [dual-loop current](dual-loop/README.md) | `RISKSEG_EVENT_EVAL_DATA_REPAIR_SUCCESSOR` | truth 未冻结时训练、选模或晋级 | 否 |
| RCLE-RF | 历史风险场研究 | `PAUSED / NO_ACTIVE_EXECUTION` | [RCLE current](rcle/README.md) | 无；只有用户明确重开后才能建立新 scoped successor | 消费暂停前授权或把旧 README 的“下一步”视为当前权限 | 否 |
| USTRF-SC route-conditioned | 历史路线代理 | `CLOSED / HISTORICAL_DIAGNOSTIC` | [USTRF closure](ustrf-sc/USTRF_ROUTE_CONDITIONED_PROGRAM_CLOSURE_R1_2026-07-25.md) | 无；实质不同的假设必须登记为新路线 | 把实验 App、旧协议或诊断结果写成当前主线 | 否 |

## 算法边界

数据、链路、性能和部署结果可以为算法提供输入，但不能直接替代算法 admission。算法研究结果也不能反向证明产品安全或默认 App 可用。
