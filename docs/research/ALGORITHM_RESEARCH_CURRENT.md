# 算法研究入口

状态：`current / ALGORITHM_FOCUS=DUAL_LANE_GRADUATION_PORTFOLIO / ACTIVE_REVERSIBLE_LANES=2 / SCHEDULE=EVIDENCE_GATED_NO_DAY_QUOTA`

新增算法预算只投向两条可由 Codex 并行的可逆支线，不设固定天数、周数或日期晋级：Assistive Geometry
负责较高成功率的 factor-wise no-regret composition，TARO 负责较高创新上限的 task-directed
observability。每线同一时刻只有一个最小判别实验；晋级靠证据，失败按停止条件收缩，不用文档数量代替进展。

表中“下一动作”就是唯一 successor；完整历史、数值和所有禁止项留在路线真源。`无` 表示
路线保持 closed、paused 或 diagnostic，不能从旧文档推导隐含下一步。

| 路线 | 主张 | 当前状态 | 唯一真源 | 下一动作（唯一 successor） | 禁止动作 | 影响默认 App |
|---|---|---|---|---|---|---|
| BlindAssist Assistive Geometry | 连续因子 → selective no-regret → body-swept geometry | `R2_F0_SYNTHETIC_REDUCER_PASS / F1_SUPERVISION_FRONTDOOR_SATISFIED / AG_R2_SUPERTEACHER_TO_AG_FINAL_V2_SEAM_PASS / AG_R2_CROSS_SENSOR_CALIBRATION_CONTROL_R0_AND_R1_FAIL_CLOSED_CONSUMED / FACTORWISE_NO_REGRET_R0_ACTIVE / SCIENTIFIC_NOT_RUN` | [Assistive Geometry current](assistive-geometry/README.md) | `AG_FACTORWISE_NO_REGRET_ORACLE_AND_PARENT_GATE_CANARY_R0`：比较 frozen prior、expert、signed-advantage oracle 与 selector；oracle 有安全 coverage 才训练下一 router | 重跑 consumed R0/R1；读 ETH3D outcome；在已看 EVAL 调门；用 macro gain 或 fallback 冒充成功 | 否 |
| TARO / Task-directed Active Risk Observability | 同预算观测使 body/path query 更早可识别 | `PARALLEL_WILD_LAB / R11_NOT_EVALUABLE_DUAL_CLASS_COVERAGE / TASK_OBSERVABILITY_BONN_POSE_PAIR_CAPABILITY_PASS / TASK_OBSERVABILITY_POSITIVE_ORACLE_R1_NOT_EVALUABLE_DENOMINATOR / LEARNED_SCORER_NOT_JUSTIFIED` | [TARO current](taro/README.md) | `TARO_TASK_OBSERVABILITY_BALANCED_POSE_SOURCE_FRONTDOOR_R0`：冻结 pose/depth/intrinsics/label contract，先满足 48 references、4 recovery parents、4 CLEAR parents，再允许另立五臂 R2 | 在 Bonn 调门或解释表面 2/2 recovery；缺分母时重跑/训练 scorer；输出 CLEAR、把 UNKNOWN 当 negative 或产品化 | 否 |
| AG-QSF / Queryable Survival Geometry | profile-queryable q-contact 生存分布统一 clearance 与 horizon occupancy | `CLOSED_DATA_SUPPORT_INSUFFICIENT / H1_NOT_EVALUABLE` | [AG-QSF current](assistive-geometry-qsf/README.md) | 无；只有新的 pre-outcome target/data contract 可重开 | 伪造 parent-disjoint censor support、实现未授权 H2 或声称数学假设被反证 | 否 |
| AG-CBF / Corridor Bottleneck Field | 检验 ground-aligned corridor bottleneck 是否保留三带摘要丢失的信息 | `R0_CLOSED_DATA_SUPPORT_NOT_EVALUABLE / ORACLE_NEVER_AUTHORIZED` | [AG-CBF current](assistive-geometry-cbf/README.md) | 无；新版本必须从 source-geometry/data support 重开 | 填补 UNKNOWN、事后改 gate、实现 oracle/model 或复用旧权限 | 否 |
| DepthART-S | parent-relative risk + fixed-budget UNKNOWN deferral，降低危险 false-clear 且封顶 coverage/false-block 代价 | `R1_RESEARCH_MAINLINE / D3R3_EXACT64_CENSUS_PASS_9597_OF_9600_PAIRED / D3R4_D3R5_DIRECT_VETO_NEGATIVES_PRESERVED / D3R6_BUDGETED_UNKNOWN_DEFERRAL_FRESH_CONFIRMATION_PASS / DEVELOPMENT_CANDIDATE_ONLY / USER_PAUSED / R2_CANDIDATE_NOT_AUTHORIZED` | [DepthART current](hftf/README.md) | 无；按用户要求在 D3R6 fresh confirmation 后暂停 | 在已打开 parent 上重调 2% budget/checkpoint；把 UNKNOWN 当 negative；读 sealed R2 或写成默认 App/安全结论 | 否 |
| DA2 | teacher、baseline、reference、fallback | `CLOSED / FROZEN_DEVELOPMENT_REFERENCE` | [DA2 closure](hftf/DAV2_P1_P2_EXECUTION_CLOSURE_2026-08-05.md) | 无；新问题必须建立新版本和独立数据路线 | 把 reference 写成 active candidate 或沿用旧下一步 | 否 |
| YOLO + 语义分割双环 | 论文次线：事件级风险/可通行性增量 | `THESIS_DEVELOPMENT_SECONDARY / RISKSEG_R0_NEGATIVE_NOT_PROMOTABLE` | [dual-loop current](dual-loop/README.md) | `RISKSEG_EVENT_EVAL_DATA_REPAIR_SUCCESSOR` | truth 未冻结时训练、选模或晋级 | 否 |
| RCLE-RF | 暂停的历史风险场研究 | `PAUSED / NO_ACTIVE_EXECUTION` | [RCLE current](rcle/README.md) | 无；只有用户明确重开后才能建立新 scoped successor | 消费暂停前授权或把旧 snapshot 的下一步当当前权限 | 否 |
| USTRF-SC route-conditioned | 已收口的历史路线代理 | `CLOSED / HISTORICAL_DIAGNOSTIC` | [USTRF closure](ustrf-sc/USTRF_ROUTE_CONDITIONED_PROGRAM_CLOSURE_R1_2026-07-25.md) | 无；实质不同的假设必须登记为新路线 | 把实验 App、旧协议或诊断结果写成当前主线 | 否 |

## 算法边界

数据、链路、性能和部署证据可以支撑算法，但不能替代 admission；算法结果也不能反向证明
产品安全、默认 App 可用性或真实用户效果。
