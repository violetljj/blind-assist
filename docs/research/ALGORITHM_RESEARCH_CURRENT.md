# 算法研究入口

状态：`current / ALGORITHM_FOCUS=BLINDASSIST_ASSISTIVE_GEOMETRY`

## 路线总表

“下一动作”即该路线唯一 successor；写“无”表示路线只能保持 `closed`、`paused` 或
`diagnostic`，不能从旧文档推导隐含下一步。

| 路线 | 主张 | 当前状态 | 唯一真源 | 下一动作（唯一 successor） | 禁止动作 | 影响默认 App |
|---|---|---|---|---|---|---|
| BlindAssist Assistive Geometry | 当前算法主线：学习 Ground / Clearance / Confidence / UNKNOWN / Body-swept Occupancy | `A0_THREE_SEED_COMPLETE / DEVELOPMENT_SELECTION_CONSUMED / A0_FAIL_TASK_GATES / A1_A4_NOT_AUTHORIZED / CALIBRATION_AND_CONFIRMATION_SEALED` | [Assistive Geometry current](assistive-geometry/README.md) | 无；A0 已到冻结负终态。重开必须先建立实质不同的 pre-outcome 假设与独立选择证据 | 复用已消费 Selection 调参/选模；事后放宽 A0 门；激活旧 A1、双教师、HTP、时序或默认 App | 否 |
| AG-QSF / Queryable Survival Geometry | 并行 WILD_LAB：以 profile-queryable body-swept robust q-contact 生存分布统一 clearance 与 horizon-consistent occupancy | `H1_IMPLEMENTED / TRAIN_CANARY_LOCKED_NOT_RUN / RUNTIME_DEFERRED_WHILE_FOREIGN_FORMAL_TRAIN_ACTIVE / LEARNABILITY_NOT_ESTABLISHED` | [AG-QSF current](assistive-geometry-qsf/README.md) | `BLINDASSIST_ASSISTIVE_GEOMETRY_QSF_H1_TRAIN_CANARY_PERFORMANCE_PILOT_THEN_RUN_WHEN_FOREIGN_GPU_IDLE`：foreign formal GPU 空闲后先跑 16-frame performance pilot，合格才运行 12/4 parent-disjoint TRAIN-only canary | 读 B1 Development/Confirmation 或 active checkpoint/progress；改 B1 successor/seed/gate；共享可变输出；与正式 seed 竞争 GPU/重 I/O；H1 有效终态前实现 H2；H1/H2 各自通过前训练组合版 | 否 |
| DepthART-S | Assistive Geometry 的优先 encoder/initialization、depth baseline 与部署使能线，不是算法终点 | `R1_RESEARCH_MAINLINE / STRICT_G4D_NEGATIVE_TERMINAL / D0_NO_ELIGIBLE_PRECISION_ARM / D1_CONTRACT_AND_METADATA_ROSTER_LOCKED_OUTCOME_NONE / PRODUCT_ASPECT_GRAPH_NOT_BUILT / R2_CANDIDATE_NOT_SELECTED` | [DepthART current](hftf/README.md) | `DEPTHART_TASK_PRESERVING_D1_LICENSE_SCOPE_AND_LABEL_BLIND_MEDIA_PREFLIGHT`：先扩展 reviewed ARKitScenes use scope，再按冻结主备顺序检查 portrait/pose/RGB-D 连续性；通过后才重建 `608×448` fixed-mixed 单候选 | 下载或读取未授权 D1 media/outcome；用 `448×448` canary 替代产品视场；用独立 R2 cohort 选模；事后把 G4-C 塞回 D0 | 否 |
| DA2 | teacher、baseline、reference、fallback | `CLOSED / FROZEN_DEVELOPMENT_REFERENCE` | [DA2 closure](hftf/DAV2_P1_P2_EXECUTION_CLOSURE_2026-08-05.md) | 无；新问题必须建立新版本和独立数据路线 | 把 reference 写成 active candidate 或沿用旧“下一步” | 否 |
| YOLO + 语义分割双环 | 论文次线 | `THESIS_DEVELOPMENT_SECONDARY / RISKSEG_R0_NEGATIVE_NOT_PROMOTABLE` | [dual-loop current](dual-loop/README.md) | `RISKSEG_EVENT_EVAL_DATA_REPAIR_SUCCESSOR` | truth 未冻结时训练、选模或晋级 | 否 |
| RCLE-RF | 历史风险场研究 | `PAUSED / NO_ACTIVE_EXECUTION` | [RCLE current](rcle/README.md) | 无；只有用户明确重开后才能建立新 scoped successor | 消费暂停前授权或把旧 README 的“下一步”视为当前权限 | 否 |
| USTRF-SC route-conditioned | 历史路线代理 | `CLOSED / HISTORICAL_DIAGNOSTIC` | [USTRF closure](ustrf-sc/USTRF_ROUTE_CONDITIONED_PROGRAM_CLOSURE_R1_2026-07-25.md) | 无；实质不同的假设必须登记为新路线 | 把实验 App、旧协议或诊断结果写成当前主线 | 否 |

## 算法边界

数据、链路、性能和部署结果可以为算法提供输入，但不能直接替代算法 admission。算法研究结果也不能反向证明产品安全或默认 App 可用。
