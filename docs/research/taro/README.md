# BlindAssist TARO

状态：`current / PARALLEL_WILD_LAB / P0_PASS / O0M_SYNTHETIC_ANALYTIC_MECHANICS_PASS / O0R_SOURCE_TRUTH_MATERIALIZED / DEPTHART_CANDIDATES_239_SEALED / SOURCE_SCALE_239_OF_239 / DIRECT_APPLE_HYBRID_R4A_COMPLETE / R5_TASK_METRIC_CONFIRMATION_FAIL / R6_UNTOUCHED_CONFIRMATION_PASS / R6_FACTOR_POLICY_PROMOTION_ALLOWED / R6_EVIDENCE_VERIFIED / R6_PROSPECTIVE_RUNTIME_PROTOCOL_FROZEN / R6_PROSPECTIVE_RUNTIME_IMPLEMENTATION_FROZEN / R6_FORMATION_REPLAY_COMPLETE / R6_REDUCER_INTEGRATION_NOT_EVALUABLE_ALL_UNKNOWN / R6_REDUCER_EVIDENCE_VERIFIED / R7_FIT_LOPO_CANARY_PASS / R7_POSITIVE_OCCUPANCY_ADVANCES / R7_CLEAR_NOT_EVALUABLE / R7_EVIDENCE_VERIFIED / R7_FRESH_CONFIRMATION_NOT_EVALUABLE_DUAL_CLASS_COVERAGE / R8_SELECTED_PHASE_B_COMPLETE / R8_SPARSE_RAY_INTERFACE_FAIL / R8_DENSE_TRUTH_OWNED_FALLBACK_COMPLETE / R9_SOURCE_ONLY_SELECTOR_FROZEN / R10_FRESH_32_PARENT_PIPELINE_COMPLETE / R10_POSITIVE_OCCUPANCY_GATES_PASS / R10_NOT_EVALUABLE_DUAL_CLASS_COVERAGE / R10_NO_PROMOTION / R11_WEAK_DISTAL_ABSTENTION_DEVELOPMENT_ONLY / R11_FRESH_48_PARENT_PROTOCOL_LOCKED / R11_EXACT_DATA_USE_AUTHORIZED / R11_HEAD_144_OF_144_PASS_ONE_SHOT_CONSUMED / R11_DOWNLOAD_IMPLEMENTATION_READY / R11_DOWNLOAD_LOCK_AUTHORIZED_UNCONSUMED / R11_SOURCE_UNOPENED / R11_SCIENTIFIC_NOT_RUN / DEFAULT_APP_UNCHANGED`

## 需求、使用场景与效果合同

TARO 不试图定义全部视障出行需求。它只选择一个与当前算法对象直接对应的窄场景：用户已经通过
白杖、定向行走技能或外部导航掌握宏观方向，需要判断前方几步内，自己的身体沿声明路径是否具有
足够净空。它不承担全局导航、道路穿越、自动接管行走或“安全路线”认证。

| 需求 | 系统应形成的效果 | 研究主指标 | 对算法的直接约束 |
|---|---|---|---|
| 不把走不通说成能走 | 路径/身体特定的完整净空区间 | `false-clear`、query error | 不能只用均值越阈值 |
| 不把能走的地方普遍阻断 | 在不增加错误放行时减少保守阻断 | `false-block`、known coverage | 不能靠 all-`UNKNOWN` 通过 |
| 证据不足时不装作知道 | 校准的 `UNKNOWN` 与 reason code | interval calibration、错误高置信率 | 只更新可观测状态方向 |
| 不要求用户冒险迈步取证 | 先复用被动历史，必要时才考虑站定相机微基线 | query-risk reduction、prompt/time cost | 禁止 body motion 取证 |

这张表是需求到研究效果的追踪合同，不是已经验证的用户效果。当前 O0M 只证明冻结 synthetic
analytic family 上的 mechanics；真实 query Pareto、交互收益和用户结果仍未建立。

## 当前主张

TARO（Task-directed Active Risk Observability，任务定向主动风险可观测性）研究：

> 在有效相机/裁剪/旋转 receipt 与至少一个独立米制锚存在时，即使完整场景与完整
> gauge 不能被唯一恢复，body-swept clearance 查询能否先达到可识别、可校准状态；
> 当查询仍不可识别时，受限的被动帧复用或站定相机微基线，能否比通用熵、最大视差
> 或普通 next-best-view 更有效地降低任务决策风险，并在证据不足时保持 `UNKNOWN`？

本版本以 `task-query identifiability` 为主科学命题，并用两个受共同协议约束的组件检验它：

- `GaugeFix`：metadata-first 的低维残余 gauge posterior、协方差与可观测子空间更新；
- `PARA`：只有被动 query 仍不可识别且 action oracle 先通过时，才研究以
  body/path-specific clearance query 为目标的受限观察与证据选择。

TARO 明确接受**组合式创新**：factor encoder、残余 gauge posterior、可观测子空间求解、
选择性校准和 action scorer 都可以继承或替换优秀前作。原创性不要求每个零件从零发明，而要求
它们服从同一条可证伪合同：`clearance functional → identifiability → calibrated uncertainty →
evidence value → deterministic reducer`。组合是否成立，必须用同预算强基线、可插拔替换、单组件/
联合 factorial ablation 和联合增益归因来证明，不能把各模块单独有效直接写成系统有效。

两个组件不得在 outcome 后任意拆分或互相背书；active branch 若失败，任何 passive-only 延续都必须
另立版本。这样可保持贡献主次清楚，同时不把主动提示预设为 TARO 必须成立的用户行为。

TARO 是与 [Assistive Geometry](../assistive-geometry/README.md) 并列的独立
`WILD_LAB` 算法路线，不是其 F1/F2 successor，也不修改其 frozen factor schema、
`FactorTensorAdapter` blocker、deterministic reducer、数据角色、终态或唯一 successor。

## 唯一真源与稳定入口

- 本页：路线状态、权限、唯一 successor、禁止动作与 claim ceiling；
- [TARO R0 详细路线指南](TARO_R0_RESEARCH_ROUTE_GUIDE_2026-08-10.md)：研究命题、
  数学定义、接口、阶段、数据、基线、指标、拟议 kill gates 与停止条件；
- [TARO P0 协议锁](TARO_P0_TASK_QUERY_IDENTIFIABILITY_AND_FACTOR_ORACLE_CANARY_PROTOCOL_LOCK_2026-08-10.md)：
  四个 schema、measurement-only identifiability、有限 task ambiguity、八臂 factorial、负控、
  数据角色、gate 与权限；
- [TARO P0 lock result](TARO_P0_PROTOCOL_LOCK_RESULT_2026-08-10.md)：33 个静态/mutation tests
  通过；科学状态仍为 `NOT_RUN`；
- [TARO O0M protocol lock](TARO_O0M_SYNTHETIC_IDENTIFIABILITY_AND_FACTORIAL_MECHANICS_PROTOCOL_LOCK_2026-08-10.md)：
  冻结 10 个 identifiability cases、5 scenes × 8 arms × 2 modes、十门与 one-shot 预算；
- [TARO O0M protocol result](TARO_O0M_PROTOCOL_LOCK_RESULT_2026-08-10.md)：33/33 mutation tests；
  protocol lock 时 implementation、runner 与 scientific artifact 尚不存在；
- [TARO O0M implementation lock](TARO_O0M_IMPLEMENTATION_LOCK_2026-08-10.md)：独立 NumPy runtime
  与 13/13 disjoint unit tests 已 hash-bound；
- [TARO O0M one-shot execution lock](TARO_O0M_ONE_SHOT_EXECUTION_LOCK_2026-08-10.md)：exact
  code/fixture/argv/environment/resource/root 已绑定；该 one-shot 现已消费；
- [TARO O0M signed result](TARO_O0M_SYNTHETIC_ANALYTIC_MECHANICS_RESULT_2026-08-10.md)：
  `92/92` records、`10/10` gates 与两次 byte-identical replay PASS；真实 O0R 仍不可评估；
- [TARO O0R ARKitScenes source-and-adapter contract](TARO_O0R_ARKITSCENES_SOURCE_AND_ADAPTER_CONTRACT_LOCK_2026-08-10.md)：
  新的 `8 ADAPTER_FIT + 16 O0R_EVAL_CANDIDATE` fresh TRAIN parent、FARO truth、uncertainty、
  query、factor injection、统计、预算与 failure scope；
- [TARO O0R source-and-adapter lock result](TARO_O0R_SOURCE_AND_ADAPTER_CONTRACT_LOCK_RESULT_2026-08-10.md)：
  pinned exclusion/roster 与关键 implementation seam 可重算，21/21 mutation tests；scientific status 仍为 `NOT_RUN`；
- [TARO O0R source-adapter implementation lock](TARO_O0R_ARKITSCENES_SOURCE_ADAPTER_IMPLEMENTATION_LOCK_2026-08-10.md)：
  纯内存 runtime 与 44/44 synthetic focused tests 已 hash-bound；source I/O 与 scientific execution 仍为 false；
- [TARO O0R truth-only preflight lock](TARO_O0R_ARKITSCENES_TRUTH_ONLY_ONE_SHOT_PREFLIGHT_LOCK_2026-08-10.md)：
  精确冻结 24-parent × 3-asset 的 72 个 HEAD target、静态 argv、环境、预算、授权缺口和四个 absent roots；
  `HEAD_NOT_RUN / ONE_SHOT_UNCONSUMED / EXECUTION_NOT_AUTHORIZED`；
- [TARO O0R data-use authorization](TARO_O0R_ARKITSCENES_DATA_USE_AUTHORIZATION_RECEIPT_2026-08-10.json)：
  用户原文授权锁定 24 个 Training video 的 HEAD、bounded source/truth-only WILD_LAB 使用；授权本身不激活 runner；
- [TARO O0R materializer amendment](TARO_O0R_ARKITSCENES_MATERIALIZER_INPUT_AND_PERSISTENCE_AMENDMENT_LOCK_2026-08-10.md)：
  冻结 all-exact frame roster、fit-before-eval、per-query lookup、original-member envelope 与 ndarray reload gate；
- [TARO O0R truth materializer implementation lock](TARO_O0R_ARKITSCENES_TRUTH_MATERIALIZER_IMPLEMENTATION_LOCK_2026-08-10.md)：
  HEAD/source/truth runner、atomic writer 与 25/25 synthetic focused tests 已 hash-bound；HEAD/source/truth 仍未运行；
- [TARO O0R Content-Length HEAD execution lock](TARO_O0R_ARKITSCENES_CONTENT_LENGTH_HEAD_EXECUTION_LOCK_2026-08-10.md)：
  Attempt 01 的精确 72 URL、zero-body、argv/environment/budget 与 exclusive root；现已 pre-start superseded；
- [TARO O0R HEAD Attempt 01 pre-start incident](TARO_O0R_ARKITSCENES_CONTENT_LENGTH_HEAD_ATTEMPT_01_PRESTART_INCIDENT_2026-08-10.md)：
  `artifacts.local` junction 被旧 path guard 误判；发生在 root/HEAD 前，request=0、one-shot 未消费；
- [TARO O0R Content-Length HEAD execution lock Attempt 02](TARO_O0R_ARKITSCENES_CONTENT_LENGTH_HEAD_EXECUTION_LOCK_ATTEMPT_02_2026-08-10.md)：
  绑定 junction-aware implementation commit `2c0fdef8` 与同一 72-URL plan；当前 `AUTHORIZED_UNCONSUMED`；
- [TARO O0R Content-Length HEAD result Attempt 02](TARO_O0R_ARKITSCENES_CONTENT_LENGTH_HEAD_RESULT_ATTEMPT_02_2026-08-10.md)：
  71/72 available；`47333152/lowres_wide.traj` 3/3 HTTP 403；zero body、HEAD one-shot consumed、no replacement；
- [TARO O0R Apple scale source canary R0 result](TARO_O0R_ARKITSCENES_APPLE_SCALE_SOURCE_CANARY_R0_RESULT_2026-08-11.md)：
  239/239 source-only estimates、166 paired frames、16/16 parents improved；parent-macro absolute log-scale error
  从 `0.30498` 降到 `0.01561`，但仍仅为 retrospective WILD_LAB diagnostic；
- [TARO O0R source-anchored factor canary R1 result](TARO_O0R_ARKITSCENES_SOURCE_ANCHORED_FACTOR_CANARY_R1_RESULT_2026-08-11.md)：
  171 frames / 1,539 queries；尺度后 SUPPORT height 与 BOUNDARY error 显著改善，但新增 112 个
  extractor loss、恢复 0，故不采用 unconditional pre-scale；R1A 仅修正 canonical summary round-trip；
- [TARO O0R Apple-seeded support recovery R2 result](TARO_O0R_ARKITSCENES_APPLE_SEEDED_SUPPORT_RECOVERY_R2_RESULT_2026-08-11.md)：
  在 R1 的 14 lost frames / 112 queries 上仅恢复 1 frame / 2 queries，height+normal no-regret 为 0；
  拒绝让 monocular candidate refit 或 veto Apple metric support；
- [TARO O0R direct Apple SUPPORT R3 result](TARO_O0R_ARKITSCENES_DIRECT_APPLE_SUPPORT_R3_RESULT_2026-08-11.md)：
  strict raw-source Phase A 下 8/14 frames 获得物理可信 Apple SUPPORT，58/112 queries 可评估、
  20 queries 同时 height+normal no-regret；normal parent-macro 仍变差，故只建立 partial headroom；
- [TARO O0R direct Apple SUPPORT R4 full-cohort result](TARO_O0R_ARKITSCENES_DIRECT_APPLE_SUPPORT_R4_RESULT_2026-08-11.md)：
  固定 R3 方法扩到 171 frames / 1,539 queries；高度和法向 parent-macro 均改善，但 direct-only
  相对 baseline 恢复 36、丢失 108 个 extraction-evaluable queries，拒绝无条件替换；
- [TARO O0R direct Apple hybrid R4A result](TARO_O0R_ARKITSCENES_DIRECT_APPLE_HYBRID_R4A_RESULT_2026-08-11.md)：
  source-only plane 有效则 direct、否则回退 R1 baseline；零参数、零阈值、零训练，恢复 36、丢失 0，
  高度与法向均为 16/16 parents 改善，但仍是同 cohort retrospective headroom；
- [TARO O0R direct Apple hybrid R5 amendment](TARO_O0R_ARKITSCENES_DIRECT_APPLE_HYBRID_ADAPTER_FIT_CONFIRMATION_R5_AMENDMENT_2026-08-11.md)：
  outcome 前冻结 8 个 parent-disjoint former ADAPTER_FIT identities / 211 frames / 1,899 slots、
  Phase-A/Phase-B firewall 与确认门；execution=false，唯一后继为 hash-bound implementation lock；
- [R5 pre-implementation transform-ID repair](TARO_O0R_ARKITSCENES_DIRECT_APPLE_HYBRID_ADAPTER_FIT_CONFIRMATION_R5_PRE_IMPLEMENTATION_TRANSFORM_ID_REPAIR_2026-08-11.json)：
  保留原 amendment 字节不变，只把错误序列化的 candidate transform 标签校正为已封存 runner 的
  `RGB_CUBIC_IMAGENET_V1` 与 bilinear `align_corners=true`；修复时 R5 root 不存在、inference/metric 均为 0；
- [R5 implementation lock](TARO_O0R_ARKITSCENES_DIRECT_APPLE_HYBRID_ADAPTER_FIT_CONFIRMATION_R5_IMPLEMENTATION_LOCK_2026-08-11.json)：
  已绑定独立 R5 role API、phase-scoped reader、one-shot runner、future execution-lock validator 与 22 个
  implementation-surface focused tests；加上 implementation-lock mutation tests 合计 26/26 PASS，仍为 execution=false；
- [R5 formal result and R6 factor-split canary](TARO_O0R_ARKITSCENES_DIRECT_APPLE_HYBRID_R5_RESULT_AND_R6_FACTOR_SPLIT_CANARY_2026-08-11.md)：
  R5 在 8 parents / 211 frames / 1,899 slots 上有效执行并以 query-knownness `7 → 5` 触发正式 FAIL；
  SUPPORT/BOUNDARY 的 8/8 parent 正向 headroom 保留。固定 `QUERY_CLEARANCE=R1 baseline` 的 R6
  factor-split post-hoc landscape 消除该 regret，但 promotion=false，必须在 untouched parents 独立确认；
- [R6 factor-split protocol lock](TARO_O0R_R6_FACTOR_SPLIT_UNTOUCHED_PARENT_CONFIRMATION_PROTOCOL_LOCK_2026-08-11.md)：
  outcome 后形成的候选已在下一版本中冻结；SUPPORT/BOUNDARY 与 QUERY_CLEARANCE 分别绑定 selected component
  和 R1 baseline，至少 8 个 untouched parents，24 个 formation parents 全部禁止复用；
- [R6 factor-split implementation lock](TARO_O0R_R6_FACTOR_SPLIT_IMPLEMENTATION_LOCK_2026-08-11.md)：
  roster-independent factor schema、depth lineage、exact-copy compositor、role firewall 和 parent-macro reducer
  已实现；15 个 focused tests 与 4 个 lock mutation tests 通过，1,899-query formation replay PASS，但
  `promotion_allowed=false`；
- [R6 untouched cohort/data-use lock](TARO_O0R_R6_UNTOUCHED_COHORT_AND_DATA_USE_LOCK_2026-08-11.md)：
  用户授权已约束到 exact 8 个新 Training parents 与 24 个 source assets；授权不自动激活执行；
- [R6 untouched source preflight result](TARO_O0R_R6_UNTOUCHED_SOURCE_PREFLIGHT_RESULT_2026-08-11.md)：
  HEAD 24/24、下载完整性 24/24、container CRC 与 exact pose plan 均 PASS；冻结 120 frames，模型与 truth 未运行；
- [R6 untouched confirmation executor implementation lock](TARO_O0R_R6_UNTOUCHED_CONFIRMATION_EXECUTOR_IMPLEMENTATION_LOCK_2026-08-11.json)：
  独立 R6 source receipt、DepthART candidate、source-only decision、FARO truth binding、9-query factor compositor、
  `UNKNOWN` retention、真实 ZIP I/O 与 one-shot runner 已 hash-bound；8 个 focused tests PASS，execution=false；
- [R6 untouched confirmation one-shot execution lock](TARO_O0R_R6_UNTOUCHED_CONFIRMATION_ONE_SHOT_EXECUTION_LOCK_2026-08-11.json)：
  精确绑定 8 parents / 120 frames / 1,080 slots、DepthART commit/checkpoint/transform、Phase-A/Phase-B firewall、
  唯一 argv、资源上限与 absent evidence root；该 one-shot 已消费且不得重跑；
- [R6 untouched confirmation result](TARO_O0R_R6_UNTOUCHED_CONFIRMATION_RESULT_2026-08-11.md)：
  8 parents / 120 frames / 1,080 slots 的九门全部 PASS；extraction evaluability `725 → 765`、boundary
  evaluability `45 → 50`、query-known `6 → 6`，height/normal 为 8/8 parents 联合正向；725 个 evidence
  文件已独立验签，R6 factor policy 获得 research-route promotion；
- [R6 factor-policy adoption and prospective-runtime protocol](TARO_O0R_R6_FACTOR_POLICY_ADOPTION_AND_PROSPECTIVE_RUNTIME_PROTOCOL_LOCK_2026-08-11.md)：
  明确 R6 的 FARO common-support 仅是归因 surface，不是 runtime input；下一实现必须从 factor owner depth、
  K、重力与 query 重算 source-defined surface，public API 禁止 FARO/truth/outcome，且禁用 8 个 R6 untouched
  parents 做 implementation/formation/调参；
- [R6 prospective-runtime query-frame repair](TARO_O0R_R6_PROSPECTIVE_RUNTIME_QUERY_FRAME_PRE_IMPLEMENTATION_REPAIR_2026-08-11.json)：
  原协议字节不变；在 implementation lock 与真实执行前冻结 query frame 为 direct support → baseline support →
  unavailable 九个 UNKNOWN，避免实现自行选择；
- [R6 prospective factor-runtime implementation lock](TARO_O0R_R6_PROSPECTIVE_FACTOR_RUNTIME_IMPLEMENTATION_LOCK_2026-08-11.md)：
  source-defined surface、factor-depth/pixel-ID lineage、九 query retention 与 no-result-side public API 已实现；
  8/8 synthetic mechanics/mutation tests PASS，真实 frame/model/truth/reducer execution 均为 0；
- [R6 prospective runtime formation replay result](TARO_O0R_R6_PROSPECTIVE_RUNTIME_FORMATION_REPLAY_RESULT_2026-08-11.json)：
  24 parents / 450 frames / 4,050 slots 的 source-first 两阶段 replay 已有效完成；16 个 eval parents 上
  support height 为 16/16 正向、normal 为 15 正/1 平，boundary XYZ 为 12/12 正向；clearance 严格沿用
  R1，effect 为 0/UNKNOWN。该结果支持 research factor extractor adoption，但不产生 final state 或产品权限；
- [R6 prospective factor-reducer integration protocol](TARO_O1R_R6_PROSPECTIVE_FACTOR_REDUCER_INTEGRATION_PROTOCOL_LOCK_2026-08-11.md)：
  冻结 source-only uncertainty lookup、区间合成、九 query retention、独立 seal 与 UNKNOWN/abort 边界；
- [R6 reducer integration implementation lock](TARO_O1R_R6_PROSPECTIVE_FACTOR_REDUCER_INTEGRATION_IMPLEMENTATION_LOCK_2026-08-12.md)：
  绑定真实 8-parent/211-frame uncertainty artifact、source-only runtime 与 13 个 focused tests；
- [R6 reducer integration one-shot execution lock](TARO_O1R_R6_PROSPECTIVE_FACTOR_REDUCER_INTEGRATION_ONE_SHOT_EXECUTION_LOCK_2026-08-12.md)：
  精确绑定 16 eval parents / 239 frames / 2,151 queries、唯一 argv 与 absent evidence root；该 one-shot 已消费；
- [R6 reducer integration result](TARO_O1R_R6_PROSPECTIVE_FACTOR_REDUCER_INTEGRATION_RESULT_2026-08-12.json)：
  执行与 243 个 evidence 文件验签有效，但 2,151/2,151 final states 全为 UNKNOWN；29 个数值区间也因
  clearance 下限 -0.30 m 与最小 uncertainty 0.4595 m 无法形成 occupied，正式终态为 NOT_EVALUABLE；
- [R7 positive occupancy and clear coverage task lock](TARO_O1R_R7_POSITIVE_OCCUPANCY_AND_CLEAR_COVERAGE_TASK_LOCK_2026-08-12.md)：
  冻结独立正占用证据与 far-censored clear coverage 两个可证伪假设，只允许在 8 个 ADAPTER_FIT parents
  上做留一父级 CPU canary；已观察的 16 eval parents 不得用于 R7 promotion；
- [R7 canary implementation lock](TARO_O1R_R7_POSITIVE_OCCUPANCY_AND_CLEAR_COVERAGE_CANARY_IMPLEMENTATION_LOCK_2026-08-12.md)：
  冻结 972 个候选组合、正占用优先与 clear veto、source-first 两阶段防泄漏和 10 个 focused tests；
- [R7 canary one-shot execution lock](TARO_O1R_R7_POSITIVE_OCCUPANCY_AND_CLEAR_COVERAGE_CANARY_ONE_SHOT_EXECUTION_LOCK_2026-08-12.md)：
  绑定 8 ADAPTER_FIT parents / 211 frames / 1,899 queries、唯一 argv 与 absent root；one-shot 已消费；
- [R7 fit-only LOPO canary result](TARO_O1R_R7_POSITIVE_OCCUPANCY_AND_CLEAR_COVERAGE_FIT_LOPO_CANARY_RESULT_2026-08-12.md)：
  426 个 evidence 文件全量验签；8 折均选中 2-pixel / 0.08 m / 2.0 m 正占用规则，恢复 1,619 个
  OCCUPIED 状态并保持 280 UNKNOWN。fit gates PASS，但 FARO clear label=0、clear output=0，故只推进正占用分支；
- [R7 fresh dual-class confirmation protocol](TARO_O1R_R7_FRESH_PARENT_DISJOINT_DUAL_CLASS_CONFIRMATION_PROTOCOL_LOCK_2026-08-12.md)：
  冻结全新 parent/visit-disjoint 确认与 definite-clear 负控门；clear 分支禁用，数据不足为 NOT_EVALUABLE；
- [R7 fresh cohort and data-use lock](TARO_O1R_R7_FRESH_CONFIRMATION_COHORT_AND_DATA_USE_LOCK_2026-08-12.md)：
  以冻结 exclusion snapshot 与 SHA 排序选出 8 个新 Training parents / 24 assets；用户授权已约束到该
  exact roster，当前 HEAD/body/model/truth 均未执行且 outcome 后不得替换；
- [R7 fresh HEAD one-shot lock](TARO_O1R_R7_FRESH_CONFIRMATION_CONTENT_LENGTH_HEAD_ONE_SHOT_EXECUTION_LOCK_2026-08-12.md)：
  zero-body HEAD 24/24 可达，总 Content-Length 494,703,329 bytes；one-shot 已消费且无 replacement；
- [R7 fresh source download one-shot lock](TARO_O1R_R7_FRESH_CONFIRMATION_SOURCE_DOWNLOAD_ONE_SHOT_EXECUTION_LOCK_2026-08-12.md)：
  24/24 GET 与逐文件 size/SHA-256 重验通过；未解压、未解码、未运行模型或 FARO；
- [R7 fresh HEAD/download result](TARO_O1R_R7_FRESH_HEAD_AND_SOURCE_DOWNLOAD_RESULT_2026-08-12.md)：
  封存 494,703,329 bytes 的 availability 与 integrity；唯一后继为本地 inventory/CRC/frame-plan；
- [R7 fresh confirmation result](TARO_O1R_R7_FRESH_CONFIRMATION_RESULT_2026-08-12.md)：
  24 parents / 402 frames 的 source-first confirmation 已消费；正占用实现保留，但 dual-class coverage
  不足，正式终态为 `TARO_O1R_R7_FRESH_CONFIRMATION_NOT_EVALUABLE_DUAL_CLASS_COVERAGE`；
- [R8 clear-negative-control cohort lock](TARO_O1R_R8_SOURCE_ONLY_CLEAR_NEGATIVE_CONTROL_COHORT_ENRICHMENT_PROTOCOL_LOCK_2026-08-12.md)：
  outcome-blind 冻结 clear-enriched source pool；后续 source-only 筛选完成，并仅对已选择的 8 parents /
  133 frames 打开 FARO Phase B；
- [R8 sparse ray-space interface lock](TARO_O1R_R8_FARO_RAY_SPACE_TRUTH_INTERFACE_CANARY_ONE_SHOT_EXECUTION_LOCK_2026-08-12.md)：
  已消费的 V1 得到 54 个 clear labels，但破坏 frozen occupied compatibility，故该接口保持 FAIL；
- [R8 dense truth-owned fallback lock](TARO_O1R_R8_DENSE_FARO_TRUTH_OWNED_FALLBACK_CANARY_ONE_SHOT_EXECUTION_LOCK_2026-08-12.md)：
  已消费；同一 8 parents / 133 frames 的 dense fallback 只保留为 R8 证据，不得覆盖或重跑；
- [R9 clear-enrichment development result](TARO_O1R_R9_CLEAR_ENRICHMENT_DEVELOPMENT_RESULT_2026-08-12.md)：
  冻结一个仅使用 pre-opened source features 的 parent selector；旧 50-clear 目标不可达，selector 只获准在
  全新 parent pool 做 outcome-blind 排名；
- [R10 fresh clear-enriched confirmation protocol](TARO_O1R_R10_FRESH_PARENT_SOURCE_ONLY_CLEAR_ENRICHED_CONFIRMATION_PROTOCOL_LOCK_2026-08-12.md)：
  冻结 32-parent / 96-asset source-first 流程、top-eight 防泄漏选择、selected-only FARO 与 dual-class 门；
- [R10 fresh clear-enriched confirmation result](TARO_O1R_R10_FRESH_CLEAR_ENRICHED_CONFIRMATION_RESULT_2026-08-12.md)：
  完整执行 710 source frames，并只对 sealed top eight 的 260 frames 读取 FARO；正占用侧冻结门全部通过，
  但 definite `CLEAR` 仅覆盖 3 个 parents 且 Wilson clear-specificity 下界失败，正式终态为 `NOT_EVALUABLE`，
  不产生路线、部署、产品或安全晋级；
- [R11 weak-distal abstention development result](TARO_O1R_R11_WEAK_DISTAL_ABSTENTION_DEVELOPMENT_RESULT_2026-08-12.md)：
  在 consumed R10 上只读形成 source-only 候选；抑制唯一 clear false positive，同时损失 1 个 occupied
  true positive。该结果严格为 development-only，不改写 R10 终态；
- [R11 positive-occupancy abstention and fresh dual-class protocol](TARO_O1R_R11_POSITIVE_OCCUPANCY_ABSTENTION_AND_FRESH_DUAL_CLASS_CONFIRMATION_PROTOCOL_LOCK_2026-08-12.md)：
  冻结 48-parent metadata-only fresh pool、top-24 source-only selection、selected-only FARO、frame/parent-aware
  dual-class gates 与 execution=false 权限；
- [R11 exact 48-parent data-use authorization](TARO_O1R_R11_FRESH_48_PARENT_DATA_USE_AUTHORIZATION_RECEIPT_2026-08-12.json)：
  用户已明确授权 exact 48 parents × 3 assets 的 frozen source-first 序列；receipt 不自行激活任何 runner，
  且不授予训练、设备、部署、产品、安全或再分发权限；
- [R11 zero-body HEAD one-shot execution lock](TARO_O1R_R11_FRESH_48_PARENT_ZERO_BODY_HEAD_ONE_SHOT_EXECUTION_LOCK_2026-08-12.json)：
  精确绑定 implementation commit、7 份文件、144-request plan、12 GiB ceiling、最多 288 attempts 与 exclusive
  root；现已消费且不得重跑；
- [R11 zero-body HEAD result](TARO_O1R_R11_FRESH_48_PARENT_ZERO_BODY_HEAD_RESULT_2026-08-12.md)：
  144/144 assets 首试可达、zero body，总 Content-Length `2,960,390,828 bytes`；source 仍未打开；
- [R11 bounded source download one-shot execution lock](TARO_O1R_R11_FRESH_48_PARENT_BOUNDED_SOURCE_DOWNLOAD_ONE_SHOT_EXECUTION_LOCK_2026-08-12.json)：
  精确绑定 implementation commit `399b53ec`、11 份代码/HEAD evidence、144-row plan、总字节、共享 deadline、
  transient-only retries 与双 root failure terminal；当前 `AUTHORIZED_UNCONSUMED`；
- [算法研究入口](../ALGORITHM_RESEARCH_CURRENT.md)：项目级算法路线登记；
- [R2 factorized geometry protocol](../assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_FACTORIZED_GEOMETRY_HYPOTHESIS_PROTOCOL_2026-08-09.md)：
  可只读复用的 factor/reducer/UNKNOWN 上游合同；
- [A0 failure anatomy](../assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_FAILURE_ANATOMY_RESULT_2026-08-09.md)：
  只作为系统性 conservative-bias 诊断，不作为 TARO 选模或阈值来源；
- [AG-DCA result](../assistive-geometry-data-capability/BLINDASSIST_ASSISTIVE_GEOMETRY_DCA_R0_RESULT_2026-08-10.md)：
  当前 target-atlas timestamp/pose/factor-support 缺口的能力真源。

## 唯一 successor

R10 已消费且正式 `NOT_EVALUABLE`。它保留了强正占用信号，但未完成 dual-class confirmation：
definite `CLEAR` 为 13 queries / 3 parents，且其中 1 个 false occupied 使单侧 95% Wilson 下界为
`0.717742`。不得通过修改 R10 selector、threshold、denominator 或 gate 救活该结果。

R11 已完成该 non-execution successor：弱、低、远端 R7 positive 只有在 16-pixel、0.15 m height 或
1.5 m forward 三个既有相邻强度 cell 至少一个成立时才保留 `OCCUPIED`，否则变为 `UNKNOWN`；它不输出
`CLEAR`。consumed R10 replay 只用于 development formation，不能成为 confirmation 或改写 R10。

exact 48 个 Training parents × 3 assets 的独立数据使用授权现已记录并绑定 pool/request SHA；zero-body
HEAD 已以 144/144 首试成功、zero body 正式 PASS 并消费。唯一 successor 是
独立 `TARO_O1R_R11_FRESH_48_PARENT_BOUNDED_SOURCE_DOWNLOAD_ONE_SHOT_EXECUTION_LOCK`：download implementation
已实现 144-row HEAD-bound GET、逐文件 SHA-256/CRC32、仅 transient transport retry 和 reservation 后 sealed
failure terminal，并以真实总量 `2,960,390,828 bytes` 为硬上限；独立 execution lock 已提交且当前
`AUTHORIZED_UNCONSUMED`，唯一 successor 是消费锁内固定 argv 一次并形成
`TARO_O1R_R11_FRESH_POOL_SOURCE_DOWNLOAD_INTEGRITY_PASS` 或 sealed invalid terminal。source body、DepthART 和 FARO 仍为 0，也没有 TARO
route/deployment/device/product/safety promotion。

O0M implementation、one-shot lock 与唯一正式执行均已完成。正式 `10+80+2` synthetic canary
终态为 `TARO_O0M_SYNTHETIC_ANALYTIC_MECHANICS_PASS`；exclusive root 已创建并消费，结果不得
覆盖、删除或重跑。

前一版本的真实 O0R 硬终态 `TARO_O0R_NOT_EVALUABLE_DATA_AND_INTERFACE` 已由新的、outcome 前
冻结的 ARKitScenes source-and-adapter contract 在**协议层**关闭：24 个全新且 visit-disjoint 的
TRAIN parent 已按 pinned exclusion + SHA 排序分配为 `8 ADAPTER_FIT + 16 O0R_EVAL_CANDIDATE`；
FARO-only factor/query truth、real-residual uncertainty、exact timestamp/K/pose receipt、八臂
deterministic injection、parent-level statistics、预算和 failure scope 均已冻结并通过静态验证。

纯内存 adapter 已以 44/44 synthetic focused tests 关闭其 implementation seam：覆盖 exact
timestamp/pose watermark、冻结 role/roster/asset/decoded-content binding、由 FARO support geometry 生成的
source+9-query receipt、由绑定 FARO/AppleDepth/confidence 内推的 fit-only uncertainty、FARO 与
candidate-depth 双 extractor、deep-read-only whole geometry/immutable base/common support、sparse
BOUNDARY、固定 model/checkpoint/metric-zero candidate output receipt、独立 TARO query reducer、genuine
truth-only 9/9 bundle 与带逐组件 lineage/parent context 的 8×2 injection；公开 API 不提供 caller residual、
factor reseal 或 self-signed truth-only scale correction。该 Module 本身不包含 downloader、archive reader、
materializer、DepthART runner、scientific evaluator 或 artifact writer；source I/O 由独立 materializer Module 承担。

truth-only preflight lock 现已静态通过：它从合同重算 24 个 parent，并以 3 个 URL template 展开
72 个唯一 Training HEAD target；request-plan SHA、离线复验 argv、Python/NumPy/SciPy/Pillow 环境、
20 GiB/50 GiB/12 h/12 GiB/2 GiB 上限和四个 absent roots 均已冻结。该锁没有发送 HEAD，未读取
Content-Length，也未创建或消费 one-shot root；绑定的旧 Assistive Geometry receipt 只覆盖 6 个旧视频且
缺 trajectory，不能授权 TARO body access。此外 `47333152` 被官方 downloader 列为缺少 3DOD assets，
未来 trajectory HEAD 若非 `200 + Content-Length`，R0 必须 `NOT_EVALUABLE`，不得换 parent。

用户随后以原文授权该锁定 roster 的 HEAD 与 bounded source/truth-only WILD_LAB 使用；outcome-blind
amendment 进一步冻结 all-exact denominator、8-parent fit-before-eval、每 query 独立 FARO/confidence lookup、
official member provenance 与 content-addressed ndarray reload。独立 materializer Module 已实现 HEAD/source/truth
runner、atomic one-shot writer，并通过 25/25 focused tests（含消费后故障与 junction containment）与 6/6 lock-validator mutation tests；未发送 HEAD/GET、
未打开 selected source、未创建 root、未拟合真实 uncertainty 或物化 truth。

HEAD Attempt 01 已提交，但在任何 HEAD 前由旧 `safe_join` 将仓库授权的 F-backed `artifacts.local`
junction 误判为 `PATH_ESCAPE`。失败发生在 output root 解析阶段，因此 `HEAD/GET/body/source/truth=0`、
五个正式 root 仍不存在、one-shot 未消费；Attempt 01 原锁不得原地重跑。实现现已仅对受信任
`artifacts.local` namespace 做 junction-aware containment，并保留 lexical repository receipt path，25/25 tests PASS。

Attempt 02 已消费：72 个冻结 HEAD target 中 71 个返回正 Content-Length；ADAPTER_FIT video `47333152`
的 `lowres_wide.traj` 在 3/3 attempts 均为 HTTP 403，无 Content-Length。响应体读取为 0，可用资产总长度
`1,105,086,109 bytes`。HEAD evidence manifest 已复核；source/work/truth/factor 四个 root 仍不存在，
truth one-shot 未消费。

后续用户授权与 recovery execution 已使锁定 source/truth 路径实际完成；当前 model-free R3 终态为
`TARO_O0R_NOT_EVALUABLE_SOURCE_TRUTH_OR_INTERFACE`，原因是 complete query truth admission 未满足，
不是 source body 缺失。239 个 DepthART eval candidate 已在 truth join 前封存；描述性 partial-factor
canary 随后完成。新的 Apple scale R0 又在独立两阶段 replay 中先封存 239/239 source-only estimates，
再与 166 帧、1,494 query 的 FARO scale oracle 比较：16/16 parents、163/166 frames 改善，parent-macro
absolute log error 从 `0.3049765` 降至 `0.0156090`。该结果只授权下一步 source-anchored factor
injection canary，不改变正式 O0R NOT_EVALUABLE 终态。

source-anchored R1 随后在全部 171 个 compact-truth frames / 1,539 queries 上完成：成功分支的
SUPPORT height parent-macro error 从 `0.34618 m` 降至 `0.10818 m`，BOUNDARY XYZ error 从
`0.42575 m` 降至 `0.10959 m`；但 hard support extractor 新失去 112 queries / 14 frames，恢复
0 个 baseline failures，且 paired support normal 多数变差。因此 Apple scale estimator 保留，
unconditional pre-extraction injection 不采用。唯一 successor 改为先在这 14 个 lost frames 上做
source-only Apple support seed recovery；无法恢复或一致性不足必须继续 UNKNOWN，不得从同一 eval
cohort 事后选部署阈值。原 R1 summary 的 12-decimal round-trip 末位漂移已由不重算几何的 R1A
reconciliation 封闭；逐帧/query 证据未变。

Apple-seeded candidate refit R2 随后只在这 14 个 lost frames 上完成：13/14 source frames 仍失败，
仅 2/112 queries 可作 post-hoc 比较，且 height+normal no-regret 为 0。该方法已拒绝；R2 虽未 hydrate
FARO/query arrays，但曾为 source metadata 解析 compact-truth package，因此不声称 strict byte-level
source-only。R3 已用 raw AppleDepth/confidence、exact `.pincam` 与 trajectory 重建窄 source receipt，
在 source completion seal 前不读取 R1 query records、compact truth 或 FARO。

direct Apple SUPPORT R3 在同一 lost cohort 上保留 8/14 物理可信 source planes，使 58/112 queries
恢复 SUPPORT evaluability，20 queries / 3 frames 同时不劣于 R1 baseline 的 height 与 normal；但 normal
parent-macro reduction 为 `-0.03906 rad`，6 frames 继续 UNKNOWN，且该 cohort 的 truth query knownness
本身均不满足，所以不能形成 final clearance claim。

R4 已把相同、冻结且无阈值的 factor ownership 扩到全部 171 frames / 1,539 queries。direct-only
分支相对 baseline 恢复 36 个 extraction-evaluable queries，却丢失 108 个；因此不作无条件替换。
R4A 随后冻结 `DIRECT_WHEN_SOURCE_SUPPORT_AVAILABLE_ELSE_R1_BASELINE_V1`：selection 只读取 Phase-A
的 source plane availability，绝不读取 truth-derived evaluability/error/knownness。该零参数回退使
extraction-evaluable 达到 1,530/1,539，相对 baseline 恢复 36、丢失 0；height 和 normal
parent-macro 均为 16/16 parents 改善。仍有 2 个 baseline-known query 的 knownness 损失，且全部结果
来自已观察的同一 16-parent cohort，所以不能写成 fresh confirmation 或 final clearance 增益。

R5 已在 8 个 visit-disjoint former `ADAPTER_FIT` parents / 211 source frames / 1,899 query slots 上完成
唯一正式执行。height/normal 为 8/8 parents 正向，extraction 恢复 44、丢失 0，但 query knownness
恢复 5、丢失 7，冻结门 `QUERY_KNOWN_COVERAGE_NO_REGRET` 失败；正式终态因此是
`TARO_O0R_DIRECT_APPLE_HYBRID_R5_TASK_METRIC_CONFIRMATION_FAIL`。R4A frame-level policy 不得晋级。

R6 把 ownership 拆到 factor level：SUPPORT/BOUNDARY 沿用 R5 source-only branch，QUERY_CLEARANCE 固定沿用
R1 baseline。protocol 与 factor compositor 先以 1,899-query formation replay 冻结，随后在 outcome-blind
选出的 8 个 untouched parents / 120 exact frames / 1,080 slots 上完成唯一正式执行。Phase A 只读取
RGB/AppleDepth/confidence/K/trajectory，模型只接收 RGB/K；全部 candidate 与 source decision 封存并重载
completion 后，Phase B 才首次读取 FARO。

正式终态为 `TARO_O0R_R6_FACTOR_SPLIT_UNTOUCHED_CONFIRMATION_PASS`：九门全部通过，extraction evaluability
`725 → 765`、boundary evaluability `45 → 50`、query-known `6 → 6`，height 与 normal 均为 8/8 parents
联合正向。34 个 FARO support-unobservable frames 保留为 `UNKNOWN`。725 个 evidence 文件与 1,080 条
query lineage 已独立复核；one-shot root 已消费，不得覆盖或重跑。唯一后继是冻结 factor policy 如何进入
prospective TARO research runtime；不得从本 cohort 事后拟合 selector/阈值，也不得外推为部署、产品或安全证明。

该 prospective protocol 现已冻结。它明确关闭 R5/R6 evaluator 的最后一条 runtime interface seam：旧
common-support pixels 与 local-valid fraction 由 FARO 定义，只能用于 task-metric 归因；public runtime
必须从 factor owner 自己的 sealed depth、K、重力和 query receipt 重建 source-defined local surface。

source-defined runtime implementation 现已 hash-bound。pre-implementation repair 先关闭 query-frame owner 歧义，
随后实现以 8/8 synthetic tests 证明 direct/baseline/unavailable 三分支、factor-depth lineage、source pixel-ID
binding、九 slot retention、candidate mutation rejection 与 deterministic roundtrip。24-parent / 450-frame
formation replay 随后先封存 4,050 个 source-only slots，再打开 FARO 评分；最终 R6 evidence 全量完成且执行有效。
16 个 eval parents 上 support height 为 16/16 正向、normal 为 15 正/1 平，boundary XYZ 为 12/12 正向；
boundary Jaccard 为 7 正/5 负/1 平，故不声明 no-regret。clearance 始终沿用 R1，最终 effect 只能为 0/UNKNOWN。
prospective factor 到 uncertainty/deterministic reducer 的 integration protocol、实现与 one-shot replay 均已完成。
执行有效且 evidence 全量验签，但 2,151/2,151 final states 全为 UNKNOWN；这不是效果 PASS。旧 clearance
区间的数值下限与真实 uncertainty 使 occupied 结构性不可达，因此不得重跑同一 reducer 或事后缩小 uncertainty。
R7 fit-only one-shot 也已完成并通过全部冻结门。8 个留一父级折叠均选择同一最弱正占用规则，R7 从
1,899 个 baseline UNKNOWN 中形成 1,619 个 OCCUPIED、保留 280 个 UNKNOWN；1,438/1,450 definite
occupied labels 被恢复，source 阶段在 FARO=0 reads 时已封存重载。但 fit FARO 标签中 CLEAR=0，且 R7
clear output=0，所以 far-censored clear 假设正式为 NOT_EVALUABLE；occupied precision 也缺少 definite-clear
负类，不能解释为部署精度。唯一后继是 outcome-blind 的全新 parent/visit-disjoint cohort/data-use lock；
fresh protocol 已冻结 dual-class evaluability guard，且明确禁用 clear 分支。

fresh cohort 也已用 commit `59023049` 的官方身份 exclusion snapshot 在任何新 media/model/truth 读取前冻结：
8 个 Training visit、24 个 asset URL 与用户数据使用授权均已绑定。数据锁本身没有发送 HEAD、下载 body、
运行 DepthART 或读取 FARO。其后 HEAD 24/24 与 494,703,329 source bytes 下载/哈希重验均已 PASS；
当前仍未解压或解码。唯一后继是 hash-bound 的本地 inventory、ZIP CRC 与 exact pose-frame plan one-shot lock。

原 amendment 的两条 candidate transform 描述与已封存 DepthART runner 不一致；该歧义已由
pre-implementation transform-ID repair 在任何 R5 inference/metric/root 产生前关闭。实现锁必须同时绑定
原 amendment、repair 与实际 runner，不能自行选择另一套 resize 语义。

P0 的 analytic fixture 不是标签清单：validator 会从 measurement-only Jacobian 重算强/弱子空间、
finite task ambiguity 与非光滑分支，并重算 `8 arms × 2 modes × 6 cases = 96` 份
payload/output/common-support hash。通用治理验证的两条 sealed-future-partition warning 已在 result
披露，不构成 O0M/O0R 执行权限。

## 路线隔离与共享边界

- TARO 只能只读复用冻结的相机几何 receipt、R2 factor 字段、deterministic reducer、
  truth-reader 约定、analytic fixture、公开文献和明确标注的数据能力结论。
- 不得写入 Assistive Geometry、AG-QSF、AG-CBF、DepthART/HFTF、RCLE 或 USTRF 的 active
  artifact、checkpoint、progress、optimizer、scheduler、target cache 或 outcome 目录。
- P0 静态 Module 位于 `scripts/research/taro/`；O0M 的独立解析 runtime 位于
  `scripts/research/taro_o0m_runtime/`，不含模型、materializer 或 trainer。唯一 scientific evidence
  固定在 `artifacts.local/evidence/taro/o0m-analytic-mechanics-r0/`，禁止覆盖、删除或重跑。
- O0R 契约 validator 位于 `scripts/research/taro_o0r_source_adapter/`；hash-bound 纯内存实现位于
  `scripts/research/taro_o0r_source_adapter_runtime/`，无 source I/O、model runner 或 artifact writer。
- O0R truth-only preflight 静态 validator 位于 `scripts/research/taro_o0r_truth_preflight/`；它只重算
  exact roster/URL/request digest/binding/budget/authority/root absence，无网络代码或 artifact writer。
- O0R HEAD/source/truth implementation 位于 `scripts/research/taro_o0r_truth_materializer_runtime/`；两个
  production runner 都必须先验证各自未来的 hash-bound execution lock，当前调用只能 fail closed。
- B1 Selection 已消费，Calibration/Confirmation 继续 sealed；A0 anatomy 只能
  `DIAGNOSTIC_ONLY`。TARO 不继承 B1 threshold、seed outcome、best checkpoint 或晋级权限。
- 当前 ARKitScenes TRAIN 原始 K/pose/depth 能力不等于 TARO target 已物化，也不自动获得
  task-query、主动观测或新 Development 权限；任何内容读取必须由后续 source-specific
  protocol 单独授权。

## 与其他新想法的关系

- `TwinScene` 是未来可独立立项的真实锚定 3D 反事实数据/训练引擎；它可以为 TARO 提供
  paired view 与 factor treatment-effect 数据，但不属于 TARO P0，也不能用 synthetic pair
  自动建立真实因果或 Confirmation claim。
- `AC4D` 是未来可独立立项的动态 first-contact 世界信念；它只有在 D44、Kalman/IMM 等
  metric-track oracle 之外证明困难分层增益后，才可能消费 TARO 的 metric posterior。
- TARO 首轮不预测 future pixels、不建 4D occupancy world model、不学习社会反应，也不把
  TwinScene/AC4D 作为组合通过条件。

## 当前允许

- 维护本路线 current 与详细路线指南；
- outcome-blind 的文献去重、接口设计、数据字段映射和解析公式检查；
- 重放 hash-bound O0R adapter 的 44 项、materializer 的 25 项 synthetic tests、6 项 implementation-lock
  mutation tests，以及 truth-only preflight lock 的 8 项静态/mutation tests；
- 只读重放 HEAD receipt/manifest hash、72-row identity、attempt budget 与 zero-body validator；不得发送新请求；
- 只读审计候选公开数据源的文档、许可、字段与能力，并记录 `CANDIDATE_METADATA_MAPPED` /
  `GAP_OPEN` / `NOT_ADMITTED`；该映射不等于下载、source admission 或 O0R successor；
- 重放 P0/O0M 静态 validator、mutation tests 与 O0M runtime unit tests；
- 对已签署 O0M evidence 做只读 hash、manifest、record 与 replay 审计；
- 只读审计已消费 R4/R4A evidence 的 manifest、逐 query 外部绑定与 canonical summary replay；
- R10 正式 `NOT_EVALUABLE` 保持不可改写；R11 development-only abstention、48-parent fresh protocol 与
  exact 数据授权已冻结。zero-body HEAD 已消费，download implementation 与 one-shot lock 已就绪；当前只可
  消费锁内固定 argv 一次；模型与 FARO 仍须依阶段
  顺序另立 one-shot lock，且 Phase A FARO 与 unselected FARO 必须为 0；
- 只读引用历史负结果、数据能力、现有 reducer 和运行时 receipt 的已签署结论。

## 当前禁止

- 从 RGB-only、纯旋转或无独立米制锚的输入输出有限高置信 metric scale；
- 从零学习或覆盖有效的 Camera2/ARCore K、crop、rotation、resize receipt；
- 把不可观测方向、缺字段、track 丢失、动态污染或 unsupported factor 当作 clear/negative；
- 让 learned graph、gauge solver 或 action scorer直接输出最终
  `CLEAR/OCCUPIED/UNKNOWN`，或绕过 deterministic body-swept reducer；
- 把 conformal/CRC 的分布内统计保证写成真实助行安全保证，或用它修复无效 receipt、缺失 metric
  anchor、数据漂移和不满足 exchangeability 的输入；
- 把 VGGT、MapAnything、MASt3R-SLAM 等几何基础模型当作独立米制锚、O0R truth、因果裁判或最终
  三态输出；它们最多是显式 provenance 的 teacher/proposal/initializer/upper bound；
- 用 task metric 选择/拯救 R2 factor backbone checkpoint；
- 要求用户向前或侧向迈步来获取证据，或把计划动作当作已执行基线；
- 读取受保护 outcome、重标 consumed cohort、启动训练/Teacher/TwinScene/AC4D、接
  Android/QNN/HTP、修改默认 App 或宣称助盲安全、独立行走、产品有效性。
- 从现有 16 个 eval parents 事后拟合 selector/threshold，或把 R4A retrospective replay 写成 fresh
  confirmation；
- 覆盖、删除或重跑已消费的 O0M one-shot，或把历史 execution lock 解释为剩余权限；
- 把 synthetic mechanics 写成真实 factor causal headroom，或跳过 real O0R 数据/adapter 前门进入
  G0/G1、A0/A1、J0。
- 借旧 `ADAPTER_FIT` role 对 211 fit frames 运行 DepthART、生成 task metric 或读取其 FARO outcome；
  这些动作只能由新的 R5 role amendment、hash-bound implementation lock 与 one-shot execution lock 授权。

## Claim ceiling

当前证明两个受限算法结果。第一，source-visible AppleDepth 零参数尺度锚点能把 parent-macro
absolute log-scale error 从 `0.30498` 降到 `0.01561`，239/239 source frames 可估计。第二，固定
R4A direct-when-valid/baseline-fallback policy 在同一已观察的 16-parent cohort 上使
extraction-evaluable 从 baseline 的 1,494 提高到 1,530，且 height/normal parent-macro 均为
16/16 parents 改善。独立 R5 又确认 SUPPORT/BOUNDARY 在 8/8 parents 正向，但 frame-level ownership
因 query knownness `7 → 5` 正式 FAIL。后续 R6 factor split 与 source-defined formation replay 在 16 个 eval
parents 上得到 support height 16/16 正向、normal 15 正/1 平、boundary XYZ 12/12 正向，但 boundary Jaccard
仍有 5 个负向 parent，clearance 只证明冻结 R1 owner 的零差异。它仍不证明 fresh confirmation、正式 O0R
PASS、RGB-only operation、真实 evidence dedup/whitening、完整 query truth
admission、选择性风险校准、observation-withholding/sensing regret、真实 geometry-anchored
counterfactual pair、被动/主动视角收益、跨设备泛化、移动端可行、产品有效性或真实用户安全。

默认 App、正式 YOLO、Assistive Geometry 主线、DepthART 路线以及所有产品/安全权限均不变。
