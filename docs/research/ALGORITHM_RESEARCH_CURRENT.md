# 算法研究入口

状态：`current / ALGORITHM_FOCUS=BLINDASSIST_ASSISTIVE_GEOMETRY`

## 路线总表

“下一动作”即该路线唯一 successor；写“无”表示路线只能保持 `closed`、`paused` 或
`diagnostic`，不能从旧文档推导隐含下一步。

| 路线 | 主张 | 当前状态 | 唯一真源 | 下一动作（唯一 successor） | 禁止动作 | 影响默认 App |
|---|---|---|---|---|---|---|
| BlindAssist Assistive Geometry | 当前算法主线：连续视觉几何因子 → deterministic body-swept task geometry | `R2_F0_SYNTHETIC_REDUCER_PASS / F1_P_PROTOCOL_FROZEN / FACTORTENSOR_ADAPTER_SYNTHETIC_CANARY_PASS / SUPERVISION_FRONTDOOR_UNSATISFIED / F1_EXECUTION_NOT_AUTHORIZED` | [Assistive Geometry current](assistive-geometry/README.md) | `BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_SUPERVISION_SOURCE_AND_LABEL_CONTRACT_LOCK`（execution=false）：只允许另锁 pre-outcome source/continuous-label/provenance/parent-role 合同，不物化或训练 | 改写 F0/F1-P/adapter 冻结字节；覆盖或重跑 adapter evidence；让 learned graph 输出 final task state；提前物化、训练、F2、teacher、HTP、时序或默认 App；把 synthetic seam mechanics 写成 real factor learnability/headroom | 否 |
| TARO / Task-directed Active Risk Observability | 独立并行 WILD_LAB：在声明的米制锚与冻结 factor/reducer 下，以低维 residual gauge posterior、可观测子空间和受限相机微基线，使 body/path-specific clearance query 先于完整场景达到局部可识别 | `PARALLEL_WILD_LAB / R6_UNTOUCHED_CONFIRMATION_PASS / R6_FACTOR_POLICY_PROMOTION_ALLOWED / R6_EVIDENCE_VERIFIED / R6_PROSPECTIVE_RUNTIME_PROTOCOL_FROZEN / R6_PROSPECTIVE_RUNTIME_IMPLEMENTATION_FROZEN / R6_FORMATION_REPLAY_COMPLETE / NO_ACTIVE_EXECUTION` | [TARO current](taro/README.md) | `TARO_O1R_R6_PROSPECTIVE_FACTOR_REDUCER_INTEGRATION_PROTOCOL_LOCK`：只允许冻结 factor-to-uncertainty/reducer integration；不得用 formation outcome 调参或执行新的真实数据评估 | 用 prior/LM/regularizer 伪造可观测；把 UNKNOWN 当 negative；覆盖或重跑已消费 root；复用 formation 或 confirmation parents 冒充新证据；让 truth/knownness 参与 factor ownership；改变 R6 exact-copy compositor、factor depth lineage 或 gates；把 formation replay 写成 confirmation；训练、主动提示、Android/HTP 或默认 App；未冻结新协议即继续执行 | 否 |
| AG-QSF / Queryable Survival Geometry | 并行 WILD_LAB：以 profile-queryable body-swept robust q-contact 生存分布统一 clearance 与 horizon-consistent occupancy | `CLOSED_DATA_SUPPORT_INSUFFICIENT / H1_NOT_EVALUABLE / H2_AND_COMBINATION_NOT_AUTHORIZED` | [AG-QSF current](assistive-geometry-qsf/README.md) | 无；重开必须先有 pre-outcome 新 target/data contract，使至少两个 parent identity 具有 right-censor support，再建立独立新路线版本 | 从单一 censor parent 构造伪 parent-disjoint 评价；把 event-only 当 censored evaluation；实现 H2/组合；复用旧执行权限；声称数学假设被反证 | 否 |
| AG-CBF / Corridor Bottleneck Field | 并行 WILD_LAB：检验 ground-aligned、body-profile inflated、拓扑连通的 corridor bottleneck 表示是否保留三带摘要丢失的信息 | `R0_CLOSED_DATA_SUPPORT_NOT_EVALUABLE / ORACLE_NEVER_AUTHORIZED / MODEL_AND_TRAINING_NEVER_AUTHORIZED` | [AG-CBF current](assistive-geometry-cbf/README.md) | 无；重开必须另立 pre-outcome source-geometry/target contract 与路线版本，从 DATA SUPPORT 重新开始 | 事后降低 gate/缩短网格；把 UNKNOWN 填成 free；实现当前 R0 oracle/模型；读取 A0 consumed Development/Confirmation；把并行路线升为主线 successor | 否 |
| DepthART-S | Assistive Geometry 的优先 encoder/initialization、depth baseline 与部署使能线，不是算法终点 | `R1_RESEARCH_MAINLINE / STRICT_G4D_NEGATIVE_TERMINAL / D1_TASK_QUALITY_FAIL_TERMINAL / D2_MECHANICS_CANARY_PASS / D2R1_SOURCE_SUPPORT_PASS_16_OF_16 / D2_4_TRAIN_4_DEVELOPMENT_SEALED / D2_PHASE_C_SOURCE_MATERIALIZATION_PASS / D2_TRAIN_ONLY_24_OF_24_COMPLETE / D2_STEP500_HEAD_LOCKED / D2_DEVELOPMENT_QUALITY_AWAITING_EXPLICIT_SCOPE / R2_CANDIDATE_NOT_AUTHORIZED` | [DepthART current](hftf/README.md) | `EXPLICIT_D2_DEVELOPMENT_BASELINE_AND_FROZEN_HEAD_QUALITY_ACTIVATION`（execution=false）：只允许冻结 4 DEVELOPMENT / 1,200 帧上的 same-base no-head baseline 与 SHA `7D889744...B017C8` frozen-head 一次性质量 screen；不得再训练、调参或选 checkpoint | 在已消费 D1 outcome 上回救；修改 D2R1 数据/门；改变已锁 head/recipe/features/checkpoint；用 Development 训练或校准；重开 strict G4-D；访问 R2、测性能或接入默认 App | 否 |
| DA2 | teacher、baseline、reference、fallback | `CLOSED / FROZEN_DEVELOPMENT_REFERENCE` | [DA2 closure](hftf/DAV2_P1_P2_EXECUTION_CLOSURE_2026-08-05.md) | 无；新问题必须建立新版本和独立数据路线 | 把 reference 写成 active candidate 或沿用旧“下一步” | 否 |
| YOLO + 语义分割双环 | 论文次线 | `THESIS_DEVELOPMENT_SECONDARY / RISKSEG_R0_NEGATIVE_NOT_PROMOTABLE` | [dual-loop current](dual-loop/README.md) | `RISKSEG_EVENT_EVAL_DATA_REPAIR_SUCCESSOR` | truth 未冻结时训练、选模或晋级 | 否 |
| RCLE-RF | 历史风险场研究 | `PAUSED / NO_ACTIVE_EXECUTION` | [RCLE current](rcle/README.md) | 无；只有用户明确重开后才能建立新 scoped successor | 消费暂停前授权或把旧 README 的“下一步”视为当前权限 | 否 |
| USTRF-SC route-conditioned | 历史路线代理 | `CLOSED / HISTORICAL_DIAGNOSTIC` | [USTRF closure](ustrf-sc/USTRF_ROUTE_CONDITIONED_PROGRAM_CLOSURE_R1_2026-07-25.md) | 无；实质不同的假设必须登记为新路线 | 把实验 App、旧协议或诊断结果写成当前主线 | 否 |

## 算法边界

数据、链路、性能和部署结果可以为算法提供输入，但不能直接替代算法 admission。算法研究结果也不能反向证明产品安全或默认 App 可用。
