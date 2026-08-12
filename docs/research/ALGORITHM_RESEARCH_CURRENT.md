# 算法研究入口

状态：`current / ALGORITHM_FOCUS=BLINDASSIST_ASSISTIVE_GEOMETRY`

## 路线总表

“下一动作”即该路线唯一 successor；写“无”表示路线只能保持 `closed`、`paused` 或
`diagnostic`，不能从旧文档推导隐含下一步。

| 路线 | 主张 | 当前状态 | 唯一真源 | 下一动作（唯一 successor） | 禁止动作 | 影响默认 App |
|---|---|---|---|---|---|---|
| BlindAssist Assistive Geometry | 当前算法主线：连续视觉几何因子 → deterministic body-swept task geometry | `R2_F0_SYNTHETIC_REDUCER_PASS / F1_SUPERVISION_FRONTDOOR_SATISFIED / AG_ST_DIRECT_TEACHER_TO_AG_REAL_SEAM_PASS / F1_STUDENT_ATTEMPT17_FAIL_NO_PROMOTION / AG_R2_SUPERTEACHER_TO_AG_FINAL_V2_SEAM_PASS / AG_R2_CROSS_SENSOR_FACTOR_ACCURACY_CONFIRMATION_F2_FROZEN / CONFIRMATION_OUTCOMES_UNOPENED` | [Assistive Geometry current](assistive-geometry/README.md) | `BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_CONFIRMATION_EXECUTOR_IMPLEMENTATION_LOCK`：只实现并以 synthetic/metadata fixture 验证已冻结 ETH3D 三 session source adapter、roster、prediction-before-truth firewall、factor-only scorer 与独立 validator；不打开 payload outcome 或运行 Confirmation | 枚举/解压已封存 archive 或运行模型/Confirmation；改 ETH3D parent/roster/gate；用 reducer/CLEAR/OCCUPIED 反传、校准或选模型；重选/重跑 Attempt17；在已消费 walking_xyz/sitting_rpy 上调参；把 Tier B/C 称 truth；把 protocol/data lock 写成跨传感器 PASS、HTP、默认 App、产品或安全证明 | 否 |
| TARO / Task-directed Active Risk Observability | 独立并行 WILD_LAB：在声明的米制锚与冻结 factor/reducer 下，以低维 residual gauge posterior、可观测子空间和受限相机微基线，使 body/path-specific clearance query 先于完整场景达到局部可识别 | `PARALLEL_WILD_LAB / R10_NOT_EVALUABLE_DUAL_CLASS_COVERAGE / R10_NO_PROMOTION / R11_WEAK_DISTAL_ABSTENTION_DEVELOPMENT_ONLY / R11_HEAD_144_OF_144_PASS_ONE_SHOT_CONSUMED / R11_DOWNLOAD_LOCK_AUTHORIZED_UNCONSUMED / R11_SOURCE_UNOPENED / R11_SCIENTIFIC_NOT_RUN / DEFAULT_APP_UNCHANGED` | [TARO current](taro/README.md) | `TARO_O1R_R11_FRESH_POOL_SOURCE_DOWNLOAD_INTEGRITY_PASS`：仅消费已提交的 exact download lock 一次，封存 144 个 HEAD-bound source blob 与逐文件 hash/CRC receipt；不得解压或读取 FARO | 用 prior/LM/regularizer 伪造可观测；把 UNKNOWN 当 negative；覆盖、重跑、改门或救活已消费 R10；复用 R10 全 32-source-pool parents 冒充 fresh evidence；让 truth/knownness 进入 source phase；越过 R11 阶段锁读 source/FARO、运行模型、训练、主动提示、Android/HTP 或默认 App；把 WILD_LAB 结果写成部署、产品或安全证明 | 否 |
| AG-QSF / Queryable Survival Geometry | 并行 WILD_LAB：以 profile-queryable body-swept robust q-contact 生存分布统一 clearance 与 horizon-consistent occupancy | `CLOSED_DATA_SUPPORT_INSUFFICIENT / H1_NOT_EVALUABLE / H2_AND_COMBINATION_NOT_AUTHORIZED` | [AG-QSF current](assistive-geometry-qsf/README.md) | 无；重开必须先有 pre-outcome 新 target/data contract，使至少两个 parent identity 具有 right-censor support，再建立独立新路线版本 | 从单一 censor parent 构造伪 parent-disjoint 评价；把 event-only 当 censored evaluation；实现 H2/组合；复用旧执行权限；声称数学假设被反证 | 否 |
| AG-CBF / Corridor Bottleneck Field | 并行 WILD_LAB：检验 ground-aligned、body-profile inflated、拓扑连通的 corridor bottleneck 表示是否保留三带摘要丢失的信息 | `R0_CLOSED_DATA_SUPPORT_NOT_EVALUABLE / ORACLE_NEVER_AUTHORIZED / MODEL_AND_TRAINING_NEVER_AUTHORIZED` | [AG-CBF current](assistive-geometry-cbf/README.md) | 无；重开必须另立 pre-outcome source-geometry/target contract 与路线版本，从 DATA SUPPORT 重新开始 | 事后降低 gate/缩短网格；把 UNKNOWN 填成 free；实现当前 R0 oracle/模型；读取 A0 consumed Development/Confirmation；把并行路线升为主线 successor | 否 |
| DepthART-S | Assistive Geometry 的优先 encoder/initialization、depth baseline 与部署使能线，不是算法终点 | `R1_RESEARCH_MAINLINE / STRICT_G4D_NEGATIVE_TERMINAL / D1_TASK_QUALITY_FAIL_TERMINAL / D2_DEVELOPMENT_FROZEN_HEAD_QUALITY_FAIL_TERMINAL / D3_PHASE_A_FAIL_21_OF_32_NO_SELECTION / D3R1_PHASE_A_BODY_127_PROCESSED_53_ELIGIBLE_32_LOCKED / D3R1_PHASE_B_EXECUTION_INVALID_INCOMPLETE_NO_SCIENTIFIC_TERMINAL / R2_CANDIDATE_NOT_AUTHORIZED` | [DepthART current](hftf/README.md) | `NONE`：D3R1 Phase-B r0 在 exact-frame source coverage hard gate 停止，scientific terminal 与 successor 均为 null；未来恢复必须另行授权并另立版本 | resume/repair/覆盖或同版本重跑 r0；替换邻帧、身份或扩大 pool；把 integrity stop 写成科学 FAIL；训练或 Development；重开 strict G4-D；访问 R2、测性能或接入默认 App | 否 |
| DA2 | teacher、baseline、reference、fallback | `CLOSED / FROZEN_DEVELOPMENT_REFERENCE` | [DA2 closure](hftf/DAV2_P1_P2_EXECUTION_CLOSURE_2026-08-05.md) | 无；新问题必须建立新版本和独立数据路线 | 把 reference 写成 active candidate 或沿用旧“下一步” | 否 |
| YOLO + 语义分割双环 | 论文次线 | `THESIS_DEVELOPMENT_SECONDARY / RISKSEG_R0_NEGATIVE_NOT_PROMOTABLE` | [dual-loop current](dual-loop/README.md) | `RISKSEG_EVENT_EVAL_DATA_REPAIR_SUCCESSOR` | truth 未冻结时训练、选模或晋级 | 否 |
| RCLE-RF | 历史风险场研究 | `PAUSED / NO_ACTIVE_EXECUTION` | [RCLE current](rcle/README.md) | 无；只有用户明确重开后才能建立新 scoped successor | 消费暂停前授权或把旧 README 的“下一步”视为当前权限 | 否 |
| USTRF-SC route-conditioned | 历史路线代理 | `CLOSED / HISTORICAL_DIAGNOSTIC` | [USTRF closure](ustrf-sc/USTRF_ROUTE_CONDITIONED_PROGRAM_CLOSURE_R1_2026-07-25.md) | 无；实质不同的假设必须登记为新路线 | 把实验 App、旧协议或诊断结果写成当前主线 | 否 |

## 算法边界

数据、链路、性能和部署结果可以为算法提供输入，但不能直接替代算法 admission。算法研究结果也不能反向证明产品安全或默认 App 可用。
