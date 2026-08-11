# 算法研究入口

状态：`current / ALGORITHM_FOCUS=BLINDASSIST_ASSISTIVE_GEOMETRY`

## 路线总表

“下一动作”即该路线唯一 successor；写“无”表示路线只能保持 `closed`、`paused` 或
`diagnostic`，不能从旧文档推导隐含下一步。

| 路线 | 主张 | 当前状态 | 唯一真源 | 下一动作（唯一 successor） | 禁止动作 | 影响默认 App |
|---|---|---|---|---|---|---|
| BlindAssist Assistive Geometry | 当前算法主线：连续视觉几何因子 → deterministic body-swept task geometry | `R2_F0_SYNTHETIC_REDUCER_PASS / F1_P_PROTOCOL_FROZEN / FACTORTENSOR_ADAPTER_SYNTHETIC_CANARY_PASS / SUPERVISION_FRONTDOOR_UNSATISFIED / F1_EXECUTION_NOT_AUTHORIZED` | [Assistive Geometry current](assistive-geometry/README.md) | `BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_SUPERVISION_SOURCE_AND_LABEL_CONTRACT_LOCK`（execution=false）：只允许另锁 pre-outcome source/continuous-label/provenance/parent-role 合同，不物化或训练 | 改写 F0/F1-P/adapter 冻结字节；覆盖或重跑 adapter evidence；让 learned graph 输出 final task state；提前物化、训练、F2、teacher、HTP、时序或默认 App；把 synthetic seam mechanics 写成 real factor learnability/headroom | 否 |
| TARO / Task-directed Active Risk Observability | 独立并行 WILD_LAB：在声明的米制锚与冻结 factor/reducer 下，以低维 residual gauge posterior、可观测子空间和受限相机微基线，使 body/path-specific clearance query 先于完整场景达到局部可识别 | `PARALLEL_WILD_LAB / P0_PASS / O0M_SYNTHETIC_ANALYTIC_MECHANICS_PASS / O0R_SOURCE_TRUTH_MATERIALIZED / O0R_NOT_EVALUABLE_SOURCE_TRUTH_OR_INTERFACE / DEPTHART_CANDIDATES_239_SEALED / PARTIAL_FACTOR_CANARY_COMPLETE / APPLE_SCALE_SOURCE_CANARY_COMPLETE / SOURCE_ANCHORED_FACTOR_CANARY_R1_COMPLETE / CANDIDATE_REFIT_REJECTED / DIRECT_APPLE_SUPPORT_R3_COMPLETE / DIRECT_APPLE_SUPPORT_PARTIAL_HEADROOM / NO_ACTIVE_EXECUTION` | [TARO current](taro/README.md) | `DIRECT_APPLE_SUPPORT_FULL_COHORT_R4`：保持 R3 的 Apple-only SUPPORT 与 source-scaled dense boundary ownership 不变，无阈值扩到全部 171 frames / 1,539 queries，量化 coverage 和 height-normal tradeoff；无效 plane 仍为 UNKNOWN | 用 prior/LM/regularizer 伪造可观测；把 K 混入 S/P/B factorial；继承 B1 outcome/threshold；把 UNKNOWN 当 negative；覆盖或重跑已消费 root；让 candidate refit/veto Apple SUPPORT；从同一 16-parent outcome 事后选 selector/threshold；把 AppleDepth 写成 RGB-only 能力；训练、主动提示、TwinScene/AC4D、Android/HTP 或默认 App；把 descriptive canary 写成 formal O0R PASS；跳过完整 query truth admission进入 G0/G1/A0/A1/J0 | 否 |
| AG-QSF / Queryable Survival Geometry | 并行 WILD_LAB：以 profile-queryable body-swept robust q-contact 生存分布统一 clearance 与 horizon-consistent occupancy | `CLOSED_DATA_SUPPORT_INSUFFICIENT / H1_NOT_EVALUABLE / H2_AND_COMBINATION_NOT_AUTHORIZED` | [AG-QSF current](assistive-geometry-qsf/README.md) | 无；重开必须先有 pre-outcome 新 target/data contract，使至少两个 parent identity 具有 right-censor support，再建立独立新路线版本 | 从单一 censor parent 构造伪 parent-disjoint 评价；把 event-only 当 censored evaluation；实现 H2/组合；复用旧执行权限；声称数学假设被反证 | 否 |
| AG-CBF / Corridor Bottleneck Field | 并行 WILD_LAB：检验 ground-aligned、body-profile inflated、拓扑连通的 corridor bottleneck 表示是否保留三带摘要丢失的信息 | `R0_CLOSED_DATA_SUPPORT_NOT_EVALUABLE / ORACLE_NEVER_AUTHORIZED / MODEL_AND_TRAINING_NEVER_AUTHORIZED` | [AG-CBF current](assistive-geometry-cbf/README.md) | 无；重开必须另立 pre-outcome source-geometry/target contract 与路线版本，从 DATA SUPPORT 重新开始 | 事后降低 gate/缩短网格；把 UNKNOWN 填成 free；实现当前 R0 oracle/模型；读取 A0 consumed Development/Confirmation；把并行路线升为主线 successor | 否 |
| DepthART-S | Assistive Geometry 的优先 encoder/initialization、depth baseline 与部署使能线，不是算法终点 | `R1_RESEARCH_MAINLINE / STRICT_G4D_NEGATIVE_TERMINAL / D0_NO_ELIGIBLE_PRECISION_ARM / D1_QUALITY_SCREEN_48_OF_48_COMPLETE / D1_TASK_QUALITY_FAIL_TERMINAL / R2_CANDIDATE_NOT_AUTHORIZED` | [DepthART current](hftf/README.md) | 无 active successor；新版本必须先显式冻结新的 pre-outcome protocol、候选与数据角色 | 在已消费 D1 Development outcome 上修改 candidate/data/postprocess/known-coverage denominator/gates；把相对改善写成 gate PASS；重开 strict G4-D；访问独立 R2 cohort；测性能或接入默认 App | 否 |
| DA2 | teacher、baseline、reference、fallback | `CLOSED / FROZEN_DEVELOPMENT_REFERENCE` | [DA2 closure](hftf/DAV2_P1_P2_EXECUTION_CLOSURE_2026-08-05.md) | 无；新问题必须建立新版本和独立数据路线 | 把 reference 写成 active candidate 或沿用旧“下一步” | 否 |
| YOLO + 语义分割双环 | 论文次线 | `THESIS_DEVELOPMENT_SECONDARY / RISKSEG_R0_NEGATIVE_NOT_PROMOTABLE` | [dual-loop current](dual-loop/README.md) | `RISKSEG_EVENT_EVAL_DATA_REPAIR_SUCCESSOR` | truth 未冻结时训练、选模或晋级 | 否 |
| RCLE-RF | 历史风险场研究 | `PAUSED / NO_ACTIVE_EXECUTION` | [RCLE current](rcle/README.md) | 无；只有用户明确重开后才能建立新 scoped successor | 消费暂停前授权或把旧 README 的“下一步”视为当前权限 | 否 |
| USTRF-SC route-conditioned | 历史路线代理 | `CLOSED / HISTORICAL_DIAGNOSTIC` | [USTRF closure](ustrf-sc/USTRF_ROUTE_CONDITIONED_PROGRAM_CLOSURE_R1_2026-07-25.md) | 无；实质不同的假设必须登记为新路线 | 把实验 App、旧协议或诊断结果写成当前主线 | 否 |

## 算法边界

数据、链路、性能和部署结果可以为算法提供输入，但不能直接替代算法 admission。算法研究结果也不能反向证明产品安全或默认 App 可用。
