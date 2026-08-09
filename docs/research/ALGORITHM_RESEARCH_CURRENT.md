# 算法研究入口

状态：`current / ALGORITHM_FOCUS=BLINDASSIST_ASSISTIVE_GEOMETRY`

## 路线总表

“下一动作”即该路线唯一 successor；写“无”表示路线只能保持 `closed`、`paused` 或
`diagnostic`，不能从旧文档推导隐含下一步。

| 路线 | 主张 | 当前状态 | 唯一真源 | 下一动作（唯一 successor） | 禁止动作 | 影响默认 App |
|---|---|---|---|---|---|---|
| BlindAssist Assistive Geometry | 当前算法主线：连续视觉几何因子 → deterministic body-swept task geometry | `R2_F0_SYNTHETIC_REDUCER_PASS / F1_P_PROTOCOL_FROZEN / SUPERVISION_FRONTDOOR_UNSATISFIED / F1_EXECUTION_NOT_AUTHORIZED` | [Assistive Geometry current](assistive-geometry/README.md) | `BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_SUPERVISION_SOURCE_AND_LABEL_CONTRACT_LOCK`：只冻结 parent-complete TRAIN 监督源、continuous boundary、residual uncertainty、provenance 与角色 roster，不授权物化或训练 | 复用 B1 Selection/threshold；让 learned graph 输出 final task state；把 F0 mechanics 当 factor learnability；用 aggregate/reducer task metric 拯救 factor failure；提前物化、训练、F2、teacher、HTP、时序或默认 App | 否 |
| AG-QSF / Queryable Survival Geometry | 并行 WILD_LAB：以 profile-queryable body-swept robust q-contact 生存分布统一 clearance 与 horizon-consistent occupancy | `CLOSED_DATA_SUPPORT_INSUFFICIENT / H1_NOT_EVALUABLE / H2_AND_COMBINATION_NOT_AUTHORIZED` | [AG-QSF current](assistive-geometry-qsf/README.md) | 无；重开必须先有 pre-outcome 新 target/data contract，使至少两个 parent identity 具有 right-censor support，再建立独立新路线版本 | 从单一 censor parent 构造伪 parent-disjoint 评价；把 event-only 当 censored evaluation；实现 H2/组合；复用旧执行权限；声称数学假设被反证 | 否 |
| AG-CBF / Corridor Bottleneck Field | 并行 WILD_LAB：检验 ground-aligned、body-profile inflated、拓扑连通的 corridor bottleneck 表示是否保留三带摘要丢失的信息 | `R0_CLOSED_DATA_SUPPORT_NOT_EVALUABLE / ORACLE_NEVER_AUTHORIZED / MODEL_AND_TRAINING_NEVER_AUTHORIZED` | [AG-CBF current](assistive-geometry-cbf/README.md) | 无；重开必须另立 pre-outcome source-geometry/target contract 与路线版本，从 DATA SUPPORT 重新开始 | 事后降低 gate/缩短网格；把 UNKNOWN 填成 free；实现当前 R0 oracle/模型；读取 A0 consumed Development/Confirmation；把并行路线升为主线 successor | 否 |
| DepthART-S | Assistive Geometry 的优先 encoder/initialization、depth baseline 与部署使能线，不是算法终点 | `R1_RESEARCH_MAINLINE / STRICT_G4D_NEGATIVE_TERMINAL / D0_NO_ELIGIBLE_PRECISION_ARM / D1_CONTRACT_AND_METADATA_ROSTER_LOCKED_OUTCOME_NONE / PRODUCT_ASPECT_GRAPH_NOT_BUILT / R2_CANDIDATE_NOT_SELECTED` | [DepthART current](hftf/README.md) | `DEPTHART_TASK_PRESERVING_D1_LICENSE_SCOPE_AND_LABEL_BLIND_MEDIA_PREFLIGHT`：先扩展 reviewed ARKitScenes use scope，再按冻结主备顺序检查 portrait/pose/RGB-D 连续性；通过后才重建 `608×448` fixed-mixed 单候选 | 下载或读取未授权 D1 media/outcome；用 `448×448` canary 替代产品视场；用独立 R2 cohort 选模；事后把 G4-C 塞回 D0 | 否 |
| DA2 | teacher、baseline、reference、fallback | `CLOSED / FROZEN_DEVELOPMENT_REFERENCE` | [DA2 closure](hftf/DAV2_P1_P2_EXECUTION_CLOSURE_2026-08-05.md) | 无；新问题必须建立新版本和独立数据路线 | 把 reference 写成 active candidate 或沿用旧“下一步” | 否 |
| YOLO + 语义分割双环 | 论文次线 | `THESIS_DEVELOPMENT_SECONDARY / RISKSEG_R0_NEGATIVE_NOT_PROMOTABLE` | [dual-loop current](dual-loop/README.md) | `RISKSEG_EVENT_EVAL_DATA_REPAIR_SUCCESSOR` | truth 未冻结时训练、选模或晋级 | 否 |
| RCLE-RF | 历史风险场研究 | `PAUSED / NO_ACTIVE_EXECUTION` | [RCLE current](rcle/README.md) | 无；只有用户明确重开后才能建立新 scoped successor | 消费暂停前授权或把旧 README 的“下一步”视为当前权限 | 否 |
| USTRF-SC route-conditioned | 历史路线代理 | `CLOSED / HISTORICAL_DIAGNOSTIC` | [USTRF closure](ustrf-sc/USTRF_ROUTE_CONDITIONED_PROGRAM_CLOSURE_R1_2026-07-25.md) | 无；实质不同的假设必须登记为新路线 | 把实验 App、旧协议或诊断结果写成当前主线 | 否 |

## 算法边界

数据、链路、性能和部署结果可以为算法提供输入，但不能直接替代算法 admission。算法研究结果也不能反向证明产品安全或默认 App 可用。
