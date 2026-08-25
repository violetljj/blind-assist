# Goal-Driven Visual Copilot

状态：`current / PRODUCT_MAINLINE=GOAL_DRIVEN_VISUAL_COPILOT / ALGORITHM_MAINLINE=GRAIL / LAST_METER_ALGORITHM_MAINLINE_REOPENED / PROCTHOR_NATIVE_M0_V2_ALL_GATES_PASS / M1_V1_TARGET_CENTERING_LEAK_CLOSED_BEFORE_TEST / M1_V2_FROZEN_TEST_UNOPENED / PASSIVE_EXACT_INSTANCE_CLOSED / FOUR_BOUNDARY_MAINLINE_CLOSED / MARKER_POSE_CANARY_ONLY / DYNAMIC_RISK_AUXILIARY / DEFAULT_APP_UNCHANGED`

完整系统蓝图见 [`V2 路线图`](BLINDASSIST_GOAL_DRIVEN_VISUAL_COPILOT_V2_ROADMAP_2026-08-21.md)。本页是
Goal Copilot 动态执行状态唯一真源；日期化 protocol/result、archive、旧 handoff 与历史 successor 不产生当前权限。

## 当前结论

[`GRAIL M0`](GRAIL_M0_ORACLE_INTERACTION_POSE_RESULT_2026-08-25.md) 已用新的任务定义重开最后十米算法主线：

```text
goal-conditioned + set-valued interaction-pose prediction
referent != affordance != reachability != visibility != arrival
```

fresh 程序化 metric 2.5D M0 的 36 个 held-out 建筑/实例中，24 个 positive 的 oracle set-field 位姿与简单闭环均
`24/24`，12 个当前无合法位姿的场景 false commit=`0/12`，几何微扰稳定 `24/24`，四类结构化反事实各拒绝
`36/36`。B1 最近自由点仅 `15/24` 且 `6/12` 无解场景强行提交，证明“可走”不足以定义“可交互”。

后续 [`natural-3D M0`](GRAIL_M0_NATURAL_3D_DERIVED_TEACHER_RESULT_2026-08-25.md) 在 8 个 fresh ARKitScenes scene、79 个实例上仅生成 `20/79` 非空 set，未过 50% coverage 门；该 derived source 保持关闭。改变信息源后的 [`ProcTHOR native M0 V2`](GRAIL_M0_PROCTHOR_NATIVE_INTERACTION_V2_RESULT_2026-08-25.md) 在冻结的 12 个 held-out house、205 个 target 上得到 pose coverage=`199/205`、oracle pose/path=`199/199`、local stability=`191/199`、action canary=`12/12`、NONE false commit=`0/18` 与 counterfactual=`572/572`，全部预注册门通过。因此 M1 frozen-encoder `B0/B1/B2/GRAIL` 比较已授权；这仍不构成 RGB、自然场景、Android、产品或安全证据。旧路线、hidden canary、动态风险和默认 App 角色不变。

## 当前终态

[`PUBLIC_REAL_EPISODE_MINING_V0`](BLINDASSIST_PUBLIC_REAL_EPISODE_MINING_V0_RESULT_2026-08-23.md) 已执行并封存，属于 `REVERSIBLE_EXPLORATION`：

1. 实现 current-frame-only selective guidance、争议/弃权、handoff 与用户确认责任合同；
2. 在 pixel/truth 前冻结 OSM/Overture public goal 与 entrance candidate set，再由 Mapillary sequence 的 GPS、heading、
   capture time 自动构造真实 approach episodes；
3. truth 按 `NATIVE_GT / MAP_TRAJECTORY_DERIVED / TEACHER_SUPPORTED / TEACHER_ONLY_WEAK / UNKNOWN`、人工最后手段排序；
4. ADT 支持 calibration/mechanism，Ego4D 支持 domain realism，Habitat 支持 explicit-goal mechanics；它们不互相冒充；
5. 物理采集不是当前 blocker；只有公开数据无法回答一个单独声明的高价值问题时才作为最后手段。

Pilot 按 speech/action decision 时刻和 episode 评估 visibility、proposal Recall@K、referent selection、confident wrong
guidance、abstention/contested/lost/stale、range、handoff、用户确认/否认、完成时间、指令与纠正。所有条件指标必须保留
conditioned denominator；provider/evaluator 分离、goal-before-truth、provenance、fail-closed 与 claim ceiling 保持有效。

Claim ceiling：`EXPLORATORY_MECHANICS_AND_FAILURE_ATTRIBUTION_ONLY`。

已消费 Development smoke run 自动适配 `6 episodes / 29 observations`，但 exact region visibility truth 全部不可用；
因此 27 次 confident guidance 为 `UNKNOWN` 而非错误负例，6 个 episode 均归因
`TRUTH_OR_CONTRACT_INSUFFICIENT`。该 run 只证明工具链可运行，不证明性能或 freshness。

prospective V1 的 `8 episodes / 89 observations` 已完整执行并封存。Truth coverage 为
`native/map-only strong=0 / teacher-supported weak usable=4 / teacher-only weak=19 / UNKNOWN=66`；三 teacher
agreement 为 `5 / 18 / 66`，原始输出和 disagreement 均保留。Truth 冻结后原 V0 baseline 完成
`89/89` provider call，`0 in_doubt`。

仅 4 个弱可用 observation 上 proposal Recall@10=`4/4`、selection=`1/4`、referent-selection failure=`3/4`；
但 8/8 episode 仍由 `TRUTH_OR_CONTRACT_INSUFFICIENT` 主导。因此当前不授权任何算法 successor；若继续，只能另行
授权 substrate/truth-source 工作，不得补抽、换 teacher/provider/prompt/threshold 或把局部 3/4 信号写成 H1 已成立。

该 substrate 工作已由 [`public functional-truth substrate audit`](BLINDASSIST_PUBLIC_FUNCTIONAL_TRUTH_SUBSTRATE_AUDIT_2026-08-23.md)
执行。ABotN-POIBench 的 11 scenes / 163 tasks 全部具有 named goal、metric endpoint 与 trajectory，但显式 entrance
frame/pixel region 为 `0/163`；它只能承担 arrival-truth canary。官方 evaluator 默认向 agent 暴露
`target_position`/`distance_to_goal`，未来 provider firewall 必须移除。官方 3DGS runtime 要求 Linux + CUDA compiler +
≥24 GB VRAM，本机 8151 MiB 因而保持 `NOT_EVALUABLE`。独立的 unofficial WebGL mechanics canary 已在最小 scene
固定首帧完成 `1,865,491/1,865,491` retained-splat submission 和非退化 1280×720 RGB；renderer envelope 未包含 goal、
endpoint、distance、teacher 或 private truth，provider/baseline call 均为 0。该 PASS 不建立官方 renderer equivalence 或
functional pixel truth。Short-Horizon OVON 缺独立
HM3D/OVON 资产，DoorFront 数据读取受 token 保护且无公开 export；不得绕过访问控制或把 shop sign 当 functional entrance。
固定官方 release tree 的补充 inventory 进一步确认 182 张 PNG 全是 163 张成功轨迹图、8 张失败轨迹图和 11 张
occupancy map，预渲染 camera observation RGB 为 0；封存“大众浴池”同名 PNG 是俯视轨迹图，不能替代官方 renderer。
当前已配置 AutoDL worker 的只读 preflight 仍不可达，故去混杂的下一执行依赖仍是可用的官方 24 GB renderer host。

正式 run 前的 validity hardening 已实现：Annotation V1 保存五级 truth authority、三个 teacher 的原始独立输出、
agreement/disagreement、functional authority 及其 native/map sources；teacher-only consensus 不能建立 functional
truth。Evaluator 拒绝未冻结 truth，并按 tier 分层报告 denominator、UNKNOWN coverage 与 failure attribution；没有
新增 performance gate。

[`SUN3D native-door approach V0`](BLINDASSIST_SUN3D_NATIVE_DOOR_APPROACH_V0_RESULT_2026-08-24.md) 的 object-45
polygon/range truth 覆盖 `15/15`，原始 private-object 计数为 `VISIBLE=4 / NOT_VISIBLE=11`、visible proposal
availability=`3/4`、selection=`0/3`。但后续
[`Referent Identifiability Audit`](BLINDASSIST_SUN3D_REFERENT_IDENTIFIABILITY_AUDIT_V0_RESULT_2026-08-24.md)
证明公开 goal `the door` 没有唯一绑定 private object 45：官方 sequence 另有 `object 57 = door: bathroom`，三个
usable-proposal 帧都至少出现两扇合理门，三个 object-45-absent confident commits 也都有另一扇门可见。故
`11/15` 只保留为 private-object visibility descriptor，`0/3 selection` 与 `4/15 wrong` 均为 public-goal
`NOT_EVALUABLE`。当前不授权 independent cohort、Active Referent Search、FSM、同 episode 调参或 P1；下一前门是
在 pixels/provider output 前冻结 independently public-identifiable 的 `UNIQUE / SET_VALUED / AMBIGUOUS` contract。

该前门的 C0/C1 mechanics 已实现为
[`PUBLIC_IDENTIFIABLE_REFERENT_CONTRACT_V1`](BLINDASSIST_PUBLIC_IDENTIFIABLE_REFERENT_CONTRACT_V1_2026-08-24.md)：
reference-image instance goal 被强制为 `UNIQUE`，参考图必须是全帧单实例或公开 target region；pre-observation
evaluator-private lock 固定 physical instance、world anchor、binding authority 与 source hash，provider-public receipt
只暴露 reference evidence 和 opaque anchor。Later per-frame visibility/regions 必须绑定 public/private hashes；
`SET_VALUED` 保留完整 legal set，`AMBIGUOUS` 不携带 target，teacher/model 不能建立 identity authority。

随后按单独冻结的
[`C2 small-roster protocol V1`](BLINDASSIST_PUBLIC_IDENTIFIABLE_REFERENT_C2_SMALL_ROSTER_PROTOCOL_V1_2026-08-24.md)
完成唯一一次 materialization，并封存
[`C2 result`](BLINDASSIST_PUBLIC_IDENTIFIABLE_REFERENT_C2_SMALL_ROSTER_RESULT_2026-08-24.md)：排除已消费 hotel source 后，
固定 7 个 SUN3D pose-corrected sources 全部形成 source-disjoint `REFERENCE_IMAGE_INSTANCE + UNIQUE` episode；
7/7 identity locks 在任何 later image GET 前完成，21/21 later observations 通过 frozen viewpoint 与 truth-binding gate，
28/28 image hashes 唯一，6/7 episodes 含 same-class distractor。Provider/teacher/detector/matcher/baseline calls 全为 0。

C2 只建立 `SMALL_ROSTER_MATERIALIZABLE`，不建立同实例重找能力。C2 到此关闭，不继续扩建 contract，也不自动运行
C3 或 Active Referent Search。若另行授权，唯一下一边界是一个独立版本的极简单 passive baseline，用 failure anatomy
区分 acquisition、proposal、identity、false commit 与 honest abstention；当前 `passive_baseline / algorithm` authority
仍为 false。

用户随后依次授权三把 consumed、visible-only Discovery 探针，均不修改或重跑 C2。固定
`GPT-5.6-Sol/high` 的 localization probe 在 21 个 observation 上得到 `FOUND=20 / ABSTAIN=1`；20 次 commit 为
`SAME_INSTANCE=16 / SAME_CLASS_DISTRACTOR=4 / UNRELATED=0 / BACKGROUND=0`，仅 `3/7` episodes 三视角全对。
四个错误均以 `IoU=0.58--0.96` 命中另一 native same-class instance，直接暴露 identity competition 缺口。

同一 VLM 的 17-pair oracle competition 在四个历史错例上原排列选对 `2/4`，13 个原正确 controls 为
`12 TARGET / 1 CONTESTED / 0 DISTRACTOR`。只交换 A/B candidate images 后，physical-instance 配对为
`ROBUST_TARGET=1 / STABLE_DISTRACTOR=2 / ORDER_SENSITIVE=1`；因此 competition 有信号，但可靠 verifier 未建立。

固定 `facebook/dinov2-small@ed25f3a` 的 order-free local probe 对每个 candidate 独立计算 bbox 内 patch 的双向
mean-nearest cosine，无 threshold/training/fusion。结果为 `TARGET_OUTRANKS=13/17`；四个历史错例 `3/4`，其中
`ROBUST_TARGET=1/1 / ORDER_SENSITIVE=1/1 / STABLE_DISTRACTOR=1/2`，原正确 controls 仅 `10/13`。两个 stable
distractor 被拆开：`c2-ref-006-later-03=+0.01698`，`c2-ref-001-later-01=-0.04007`；三个 collateral 为
`-0.00107 / -0.00115 / -0.06206`。Local evidence 具有互补信号但不能独立承担 verifier；不续做 threshold、
fusion、belief、tracker 或 Active Search。详细口径见脚本包 README；reports：

随后授权的 matched two-reference Development probe 排除旧 C2 图像、target 与 distractor IDs，从 5 条复用 SUN3D
source 冻结 5 个新 target、10 张 reference 与 14 个 competition frame。固定 `max(R1,R2)` 从 single=`14/14`
降至 `11/14`，paired transition=`0 rescue / 3 collateral`、median margin delta=`-0.03808`；naive max 被拒绝。
之后的 source-disjoint T-LESS/BOP19 hard-error probe 在 RGB 前冻结 roster；原 DINOv2-S baseline 为 `27/30`，并冻结
6 hard pairs（3 wrong + 3 lowest-margin correct）、6 matched controls 与 8 target-absent audits。唯一 PDM PerMIR unary
arm 得到 `1 rescue / 4 collateral / control retention=4/6`，未过 `rescue > collateral` 且 retention `>=80%` 的门；
终态为 `PDM_UNARY_MIXED_RESCUE_WITH_COLLATERAL_DEVELOPMENT`。absence 因无冻结 NONE threshold 全部 `NOT_EVALUABLE`。
这只覆盖 T-LESS textureless near-instance oracle-candidate ranking，不产生 native same-class、proposal 或产品声明。
只读 [`identity failure-layer audit`](PUBLIC_IDENTIFIABLE_REFERENT_IDENTITY_FAILURE_LAYER_AUDIT_V0_RESULT_2026-08-24.md) 将 12 对分为 `REPRESENTATION_COLLAPSE=4 / LOCAL_LAYOUT_LOST=3 / UNKNOWN=5`，由此授权的 [`NearID-style unary V0`](NEAR_IDENTITY_HARD_NEGATIVE_UNARY_V0_RESULT_2026-08-24.md) 在全新 CORe50 三重 source-disjoint split 上得到 `4 rescue / 17 collateral / control retention 1/18 / coverage 5/135`，终态 `MIXED_WITH_COLLATERAL`；这只拒绝冻结的小型 DINO projection arm，不等同于运行官方 NearID checkpoint，layout 仍不预先 fusion。

随后独立执行的 [`Spatial-Layout Identity Verification V0`](SPATIAL_LAYOUT_IDENTITY_VERIFICATION_V0_RESULT_2026-08-24.md)
在全新 Washington RGB-D Object Dataset 的 300 个 physical instances / 900 个 paired decisions 上，把固定 DINO
mean-nearest baseline 的 `702/900` 降到 `558/900`，paired transition=`74 rescue / 218 collateral`、control retention
`484/702=68.9%`；23 个 stable-distractor instances 上仅 `29/69=42.0%`。direction 与 candidate permutation
invariance 均 100%，所以 analytic layout arm 有效失败。按预声明 stop rule，passive single-reference RGB exact-instance
mainline 关闭；不再换 backbone/head/layout，且 identity signal 未通过前不启动 open-set calibration。新研究必须改变输入合同，
转向主动 distinctive evidence 或独立身份来源；P1 与 App 不启动。

[`Active Distinctive Evidence Acquisition V0`](ACTIVE_DISTINCTIVE_EVIDENCE_ACQUISITION_V0_RESULT_2026-08-24.md) 的两个
storefront、商品、个人物品三帧 sweep 最终与 passive 完全相同：top-1=`11/16`、wrong lock=`9/20`、reacquisition=`3/4`，
故 `APPEARANCE_DERIVED_DISTINCTIVE_ANCHOR_NO_UPLIFT`。它关闭 patch/threshold successor，不关闭项目 Demo；
`NO_P1 / DEFAULT_APP_UNCHANGED` 只禁本 lane 晋升。当时唯一下一动作是独立 OCR/logo/marker semantic anchor；V0 执行时
OCR runtime 不可执行，故保持 `NOT_EVALUABLE`，也没有用旧部分 canary 输出补分母。
随后完成的 [`Semantic Distinctive Anchor V1`](SEMANTIC_DISTINCTIVE_ANCHOR_V1_RESULT_2026-08-24.md) 在相同四目标、16 个 present decision、candidate role 与四次 lost 节奏上加入自然 OCR、公开-reference distinctive sign、包装码和 ArUco，零 appearance fallback 得到 top-1=`16/16`、wrong lock=`0/20`、reacquisition=`4/4`、lost `ABSTAIN=4/4`。这是 controlled derived mechanism demo，不是 same-pixel matcher 或 general OCR/logo confirmation。后续 [`V2-MARKER-POSE Android seam`](SAGE_LM_V2_MARKER_POSE_LIVE_SEAM_IMPLEMENTATION_2026-08-25.md) 已在独立 demo app 接通 live QR exact ID、Camera2 actual-focal intrinsics、四角 planar pose、target-front waypoint、center baseline、PnP controller 与 LOST/fresh reacquire；JVM mechanics `8/8`、APK build 通过，但无 ready device，真实相机与 18-run 指标仍未运行。

[`Semantic Anchor Graph + Referent Belief V2`](SEMANTIC_ANCHOR_GRAPH_AND_BELIEF_V2_RESULT_2026-08-24.md) 随后在
23 帧 synthetic OCR-stage hard cohort 上，把同输入 substring + two-frame FSM 的 correct terminal `3 -> 12`、wrong lock
`6 -> 0`、correct `NONE 0 -> 3`、low-quality `UNKNOWN 0/3 -> 3/3`。V2 已实现 relational lexical/layout/candidate
association、scene distinctiveness、candidate+NONE belief 和 correlated-burst suppression；结果只建立 synthetic mechanism。
[`V2.1 RapidOCR transfer`](SEMANTIC_ANCHOR_GRAPH_AND_BELIEF_V2_1_REAL_OCR_TRANSFER_RESULT_2026-08-24.md) 随后保持 V2
scorer、belief 与阈值不变，在 16 张自动生成门牌式像素图上实际运行 RapidOCR polygon/confidence，把 correct terminal
`1 -> 9`、wrong lock `4 -> 0`、correct `NONE 0 -> 3`、low-quality `UNKNOWN 0/2 -> 2/2`。它建立 generated-pixel
RapidOCR transfer，不建立 natural-photo/open-set calibration authority。当前唯一 successor 是 V3 小型 learned relational
scorer：synthetic graph training，未参与训练的自然照片门牌/directory Development test；V4 active information gain、
Android/P1 与默认 App 均不启动。

[`Typed Semantic-Referent Graph V3-A`](TYPED_SEMANTIC_REFERENT_GRAPH_V3_A_RESULT_2026-08-25.md) 已把手工 candidate scorer
替换为三类节点、14 类有向 edge、3 层 relational attention/message passing。1,600 synthetic graph train、320 validation 后，
在同一 V2.1 RapidOCR Development rows 上，V2 heuristic=`9/16 correct, target 6/7, NONE 3/7, wrong 0`；no-relation=`7/16,
target 0/7, NONE 7/7`；full=`14/16, target 7/7, NONE 7/7, wrong 0, UNKNOWN 2/2`，candidate permutation 16/16。
V2 observability/novelty/belief/NONE/threshold 全冻结，learned reliability/NONE head 仅诊断。由于 directory failure inspection
影响了 generator coverage，结果是 `TUNED_ON_DEVELOPMENT`。该结果当时的 successor 是冻结 full scorer 后运行未参与开发的
自然照片 source-disjoint cohort；该 successor 已由下述 natural-photo Development 执行。

[`V3-A natural-photo source-disjoint Development`](SAGE_R_V3_A_NATURAL_PHOTO_SOURCE_DISJOINT_RESULT_2026-08-24.md)
随后在 OCR/模型打开前封存 9 张独立 Wikimedia Commons 自然照片、12 个 target query 与 36 个派生 observation。目标类型为
`PLATFORM / EXIT / EMERGENCY EXIT / LAB / CLASSROOM`，与 synthetic generator 的 target type 隔离。V2 / no-relation /
full correct terminal=`4/36 / 0/36 / 3/36`，evaluable target=`2/27 / 0/27 / 3/27`；full wrong lock=`9`、NONE=`0/6`、
directory false binding=`2`，仅 candidate permutation `36/36` 保持。relation path 仍提供 `+3` target observation，但没有建立
可迁移的 authority assignment，状态为 `DO_NOT_ENTER_V3_B`。下一问题收窄到 natural OCR grouping、sign-to-destination
association、directional geometry normalization 与 missing decisive token 的 open-set behavior；不得在本 sealed cohort 上换图、
改 candidate/truth、调阈值或重跑，V3-B/V4 继续不启动。

[`SAGE-R V3-C Authority-Typed Sign-Destination Graph`](SAGE_R_V3_C_AUTHORITY_TYPED_NATURAL_RESULT_2026-08-24.md)
随后完成唯一 representation pivot：五类节点、显式 carrier/cue、`LABELS / POINTS / LISTS / NEAR / UNRELATED` 与 decisive-token
completeness。旧 cohort diagnostic 为 `11/36 correct, wrong=2`，但全新预冻结 6-source / 11-query / 33-observation cohort 上，
V2=`11 correct, wrong=1, directional=2/3, directory false=1`，V3-C full=`13 correct, wrong=5, directional=0/3,
directory false=5`。candidate permutation `33/33`，所以新增 recall 仍以错误 authority 为代价。状态
`CLOSE_NATURAL_SAGE_R`：不调本 cohort、不进入 V3-B/V4、不再另立 natural SAGE-R 修补；仅保留 controlled QR/OCR exact-anchor
demo 与失败 harness。

[`SAGE-LM V0 controlled geometry`](SAGE_LM_V0_CONTROLLED_GEOMETRY_RESULT_2026-08-24.md) 固定 exact semantic identity，
在 36 个 procedural episode 上把 target-front arrival 从 `7` 提至 `33`、median lateral error 从 `0.592` 降至 `0.094 m`；
这是 synthetic mechanism evidence，不是 real-RGB、导航、安全或产品证据。随后
[`SAGE-LM V1 controlled real-RGB observation`](SAGE_LM_V1_CONTROLLED_REAL_RGB_OBSERVATION_RESULT_2026-08-24.md) 在 24 个
curated ARKitScenes episode 上得到 arrival `7/24 -> 2/24`、median lateral error `0.219 -> 0.261 m`。source-pose audit
发现原 materializer 把 rotation-vector 列误作 camera positions，故原 active baseline 无效，结果只说明该 adapter 未保留 uplift。
随后同 cohort 的
[`V1-A all-oracle ceiling`](SAGE_LM_V1_A_ALL_ORACLE_OBSERVATION_CEILING_RESULT_2026-08-24.md) 得到 `24/24` arrival、
`0.000 m` median error、completion `24/24`、controls `6/6`，原八条标准全过。因此 downstream policy ceiling 已建立，
[`V1-B source-pose two-view`](SAGE_LM_V1_B_SOURCE_POSE_TWO_VIEW_BOUNDARY_GEOMETRY_RESULT_2026-08-24.md) 实现 B0/B1/B2
且移除 LK/metric depth，但冻结 pair 仅 `2/24` 通过正确 source-pose motion gate，同 window 只有 `13/24` 存在可替代 pair，
故 `NOT_EVALUABLE`；B1/B2 raw outcome 不是 boundary negative。新 cohort 须另行显式授权，不接 Android/P1/default App。

[`V1-B-R2`](SAGE_LM_V1_B_R2_CORRECT_POSE_BOUNDARY_RESULT_2026-08-24.md) 冻结正确-pose 24 episodes；B0=`24/24`，B1 geometry=`2/24`、missing=`21/24`。[`R3/R4`](SAGE_LM_V1_B_R3_R4_DENSE_BOUNDARY_RESULT_2026-08-25.md) 中 R3 提至 true pair=`15/24`、geometry=`13/24`，R4 退化为 `9/24`、`8/24`；R3 保持当前冠军，R4 objective rejected，B2 不运行。
[`R5/R5S`](SAGE_LM_V1_B_R5_ANCHOR_PAIR_COVERAGE_RESULT_2026-08-25.md) 分别为 true pair/geometry=`12/24`、`11/24`，均低于 R3。[`V1-C`](SAGE_LM_V1_C_TASK_SPECIFIC_APERTURE_BOUNDARY_FIELD_RESULT_2026-08-25.md) 的同域 C0/C1 四边界 Recall@8=`1/24`、`4/24`，true pair/geometry=`1/24`、`3/24`；当前 CNN + automatic opening-proxy supervision 正式关闭。

[`V1-D active parallax boundary field`](SAGE_LM_V1_D_ACTIVE_PARALLAX_BOUNDARY_FIELD_RESULT_2026-08-25.md)
固定同一 24 episodes，以 RAFT-Small residual-parallax discontinuity 产生 LEFT/RIGHT top-8；四边界 Recall@8=`4/24`、
true pair/geometry=`4/24`、R3 missing rescue=`0/9`。不做融合，当前 parallax 实现关闭；当时唯一 successor 为 V1-E
mesh/Faro-depth privileged boundary teacher，现已由下述 E0 执行。随后
[`V1-E0 privileged geometry ceiling`](SAGE_LM_V1_E0_PRIVILEGED_GEOMETRY_TEACHER_CEILING_RESULT_2026-08-25.md) 使用全 24 条可对齐的官方 ARKit 3DOD mesh，得到四边界 Recall@8=`10/24`、true pair/geometry=`9/24`、R3 missing rescue=`3/9`、retention=`6/15`；三个继续门全部失败，停止于 E0，不训练 student。
随后 [`V1-F portal interior field`](SAGE_LM_V1_F_ANCHOR_CONDITIONED_PORTAL_INTERIOR_FIELD_RESULT_2026-08-25.md) 在与旧 cohort raw-frame identity 零重叠的 fresh 24-case Development 上，把 primary teacher target 改为 anchor 支撑平面后的跨视图 connected free-space。冻结 R3 同批 true pair/geometry=`18/24, 7/24`，V1-F D2=`0/24, 10/24`，retention=`0/18`、missing rescue=`0/6`；仅 geometry uplift 过门，且 D2 经 D1 outcome 后调整，明确为 `TUNED_ON_DEVELOPMENT`。
因此停止于 teacher ceiling，不训练 student、不补 Faro、不恢复 fusion/R6/B2；四离散边界恢复不再作为主线表示继续，R3 仅保留为冻结参考。该结果拒绝当前 D2 teacher，不反证所有 portal-interior 表示。

## V0 与 P1 边界

`FOUND / CONTESTED / NOT_VISIBLE / ABSTAIN / STALE / HANDOFF_READY / COMPLETED_BY_USER` 是 current-frame 状态；`LOST`
只由 `VISIBLE -> NOT_VISIBLE_AFTER_VISIBLE` 派生，不构成 persistence。当前不得让 tracker、re-ID、gallery、world anchor、
VIO/SLAM 或 scene graph 获得 identity authority，也不得静默修改默认 App。只有真实 pilot 同时证明连续可见 selection
可靠、episode failure 仍显著、主导失败确为出画/遮挡后的错锁或无法恢复，且 pointing/proposal/range/interaction 均非
主导层，才可另行提出 P1 successor。

## 关键历史证据入口

- [`Goal-semantic proposal + RGB-D servo`](BLINDASSIST_GOAL_RGBD_SERVO_RESULT_2026-08-23.md)：proposal availability
  established；fresh action availability not established。
- [`TartanGround route servo`](BLINDASSIST_TARTANGROUND_ROUTE_SERVO_RESULT_2026-08-23.md)：Top-10 target 多数存在，
  但 route/functional selection 与 STOP 未建立，waypoint 不是 destination truth。
- [`S0v11 visual servo`](BLINDASSIST_LAST_10M_VISUAL_SERVO_S0V11_RESULT_2026-08-22.md)：`9/13` false completion，
  bbox extent responsibility rejected。
- [`S2–S5 current-frame completion`](BLINDASSIST_LAST_10M_CURRENT_FRAME_COMPLETION_S2_S5_RESULT_2026-08-22.md) 与
  [`D1C`](BLINDASSIST_LAST_10M_FUNCTIONAL_REGION_D1C_RESULT_2026-08-22.md) / [`D3`](BLINDASSIST_LAST_10M_FUNCTIONAL_REGION_D3_RESULT_2026-08-22.md)：
  synthetic exact-door/ground-connected proxy 的边界。
- [`P0 grounding contract`](P0_GROUNDING_PROTOCOL_V1.md)：`UNIQUE / SET_VALUED / AMBIGUOUS`、provider、selection 与
  evaluator 分离；既有 provider/threshold 不因本 successor 改变。
- [`Prospective recorder result`](P1_PROSPECTIVE_DEVICE_RECORDER_IMPLEMENTATION_RESULT_2026-08-22.md)：独立 CameraX
  recorder 已实现且不进入默认 App。
- P1 A1–A4、W1/W2、AMRM0 历史终态保持原样；旧文件只用于追溯，不恢复执行权限。

## 默认 App 与声明边界

默认 App 保持当前 YOLO/risk 正式路径不变。本 successor 是 research/experimental integration；任何正结果都不自动授权
Android/default-App 接线、模型晋级、导航/安全、用户有效性或产品成功声明。
