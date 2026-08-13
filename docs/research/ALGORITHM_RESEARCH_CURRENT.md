# 算法研究入口

状态：`current / ALGORITHM_FOCUS=BLINDASSIST_ASSISTIVE_GEOMETRY`

表中“下一动作”就是唯一 successor；完整历史、数值和所有禁止项留在路线真源。`无` 表示
路线保持 closed、paused 或 diagnostic，不能从旧文档推导隐含下一步。

| 路线 | 主张 | 当前状态 | 唯一真源 | 下一动作（唯一 successor） | 禁止动作 | 影响默认 App |
|---|---|---|---|---|---|---|
| BlindAssist Assistive Geometry | 连续视觉几何因子 → deterministic body-swept task geometry | `R2_F0_SYNTHETIC_REDUCER_PASS / F1_SUPERVISION_FRONTDOOR_SATISFIED / AG_R2_SUPERTEACHER_TO_AG_FINAL_V2_SEAM_PASS / AG_R2_CROSS_SENSOR_CALIBRATION_CONTROL_R0_AND_R1_FAIL_CLOSED_CONSUMED / R1_INDEPENDENT_REPLAY_CONFIRMED_PRODUCER_FAILURE / SCIENTIFIC_NOT_RUN` | [Assistive Geometry current](assistive-geometry/README.md) | 无；未来恢复须另立基于官方 archive-format evidence 的新版本和 fresh root，并单独授权 | 重跑/重释 consumed R0/R1；把 null 当 0/negative；访问 session/model/Confirmation；把 control failure 当 factor 科学结论 | 否 |
| TARO / Task-directed Active Risk Observability | 让 body/path-specific clearance query 先于完整场景达到局部可识别 | `PARALLEL_WILD_LAB / R10_NOT_EVALUABLE_DUAL_CLASS_COVERAGE / R11_SOURCE_ONLY_TOP24_ONE_SHOT_CONSUMED_PASS / R11_TOP24_INDEPENDENT_VALIDATION_PASS / R11_SELECTED_TOP24_FARO_PHASE_B_IMPLEMENTATION_LOCK_PASS_NON_EXECUTING / R11_SCIENTIFIC_NOT_RUN` | [TARO current](taro/README.md) | `TARO_O1R_R11_SELECTED_TOP24_FARO_PHASE_B_ONE_SHOT_EXECUTION_LOCK`：绑定 sealed selected 24 与冻结 implementation | 修改 selected 24；读取 unselected FARO/highres；execution lock 前读 FARO；训练、产品或安全晋级 | 否 |
| AG-QSF / Queryable Survival Geometry | profile-queryable q-contact 生存分布统一 clearance 与 horizon occupancy | `CLOSED_DATA_SUPPORT_INSUFFICIENT / H1_NOT_EVALUABLE` | [AG-QSF current](assistive-geometry-qsf/README.md) | 无；只有新的 pre-outcome target/data contract 可重开 | 伪造 parent-disjoint censor support、实现未授权 H2 或声称数学假设被反证 | 否 |
| AG-CBF / Corridor Bottleneck Field | 检验 ground-aligned corridor bottleneck 是否保留三带摘要丢失的信息 | `R0_CLOSED_DATA_SUPPORT_NOT_EVALUABLE / ORACLE_NEVER_AUTHORIZED` | [AG-CBF current](assistive-geometry-cbf/README.md) | 无；新版本必须从 source-geometry/data support 重开 | 填补 UNKNOWN、事后改 gate、实现 oracle/model 或复用旧权限 | 否 |
| DepthART-S | Assistive Geometry 的 encoder/depth baseline 与部署使能线 | `R1_RESEARCH_MAINLINE / D3R2_PHASE_B_COVERAGE_CENSUS_EXECUTION_INVALID_INCOMPLETE / D3R3_FRESH_HEAD_64_OF_64_PASS_ZERO_HEADER_DRIFT / D3R3_EXACT64_CENSUS_ACTIVATED / R2_CANDIDATE_NOT_AUTHORIZED` | [DepthART current](hftf/README.md) | `RUN_D3R3_EXACT64_SOURCE_MEMBER_COVERAGE_CENSUS`：fresh root 全 64 GET，只允许 matching-header premature EOF 从 byte 0 重试 | 复用 D3R2 bodies；Range/redirect/member payload/pixel/truth/selection；训练、R2 或默认 App | 否 |
| DA2 | teacher、baseline、reference、fallback | `CLOSED / FROZEN_DEVELOPMENT_REFERENCE` | [DA2 closure](hftf/DAV2_P1_P2_EXECUTION_CLOSURE_2026-08-05.md) | 无；新问题必须建立新版本和独立数据路线 | 把 reference 写成 active candidate 或沿用旧下一步 | 否 |
| YOLO + 语义分割双环 | 论文次线：事件级风险/可通行性增量 | `THESIS_DEVELOPMENT_SECONDARY / RISKSEG_R0_NEGATIVE_NOT_PROMOTABLE` | [dual-loop current](dual-loop/README.md) | `RISKSEG_EVENT_EVAL_DATA_REPAIR_SUCCESSOR` | truth 未冻结时训练、选模或晋级 | 否 |
| RCLE-RF | 暂停的历史风险场研究 | `PAUSED / NO_ACTIVE_EXECUTION` | [RCLE current](rcle/README.md) | 无；只有用户明确重开后才能建立新 scoped successor | 消费暂停前授权或把旧 snapshot 的下一步当当前权限 | 否 |
| USTRF-SC route-conditioned | 已收口的历史路线代理 | `CLOSED / HISTORICAL_DIAGNOSTIC` | [USTRF closure](ustrf-sc/USTRF_ROUTE_CONDITIONED_PROGRAM_CLOSURE_R1_2026-07-25.md) | 无；实质不同的假设必须登记为新路线 | 把实验 App、旧协议或诊断结果写成当前主线 | 否 |

## 算法边界

数据、链路、性能和部署证据可以支撑算法，但不能替代 admission；算法结果也不能反向证明
产品安全、默认 App 可用性或真实用户效果。
