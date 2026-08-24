# 算法研究入口

状态：`current / PRODUCT_RESEARCH_MAINLINE=GOAL_DRIVEN_VISUAL_COPILOT / PASSIVE_SINGLE_REFERENCE_RGB_EXACT_INSTANCE_MAINLINE_STOP / SEMANTIC_ANCHOR_V1_CONTROLLED_DEMO / CLOSE_NATURAL_SAGE_R / CONTROLLED_EXACT_ANCHOR_RETAINED / SAGE_LM_V0_CONTROLLED_36_POSITIVE / SAGE_LM_V1_REAL_RGB_24_FAIL_2_VS_7 / OBSERVATION_DIAGNOSTIC_ONLY / LIVE_ANDROID_NOT_RUN / NO_P1 / DEFAULT_APP_UNCHANGED`

Goal-Driven Visual Copilot 现为 BlindAssist 的上位产品/研究主线。P0 commitment-policy discovery 已以
`COMPLEXITY_ONLY_BUYS_ABSTENTION` 收口；现有 P0 grounding/provider 与 evaluator 保持冻结。用户此前显式授权
`P1-AMRM0 Adaptive Multi-view Referent Memory`；它把 P1-VF0 作为 verifier foundation，并新增严格
tentative/verified 分离、verified-only retrieval、按 distance × viewpoint × scale/context 新颖度增长的 bounded memory。
31 项 contract tests 通过；matched canary 已以 `P1_AMRM0_MEMORY_POISONING_FAIL` 终止。旧 A1-A4、W1/W2 终态继续有效；AMRM0 不是 P1-W3，也不
覆盖或续跑 consumed evidence。随后显式授权的 P1-PA0/PA1 独立上移到 proposal availability，不修改 AMRM0。

当前算法 evidence boundary 已增加一组用户授权的 referent identity 探针。C2 visible-only passive identity probe 中，
固定强单模型在 21 个 visible observations 上 `FOUND=20 / SAME_INSTANCE=16 / SAME_CLASS_DISTRACTOR=4 / ABSTAIN=1`，
且四个错指全部落到 native same-class instance。随后固定同一模型的 oracle competing-identity diagnostic 在四个历史
错例上选对 `2/4`，并在 13 个原正确、有同类竞争者的 control 上保留 `12/13`、另 `1/13 CONTESTED`；但 A/B swap
反事实只留下 `1/4` robust target、`2/4` stable distractor、`1/4` order-sensitive。它把下一问题从泛化 persistence
进一步收窄到 instance-discriminative appearance/local evidence 与 order-invariant comparison。固定 DINOv2-S 的
order-free local probe 随后在同 17 对上得到 `13/17` target outrank；四个历史错例为 `3/4`，其中两个 stable
distractor 被拆成 `1/2` recovered、`1/2` still wrong；但原正确 controls 只有 `10/13`。因此 local evidence 有互补
信号却不足以单独形成 verifier；可靠 verifier 尚未建立，也不自动授权 threshold/fusion、belief、Active Search 或 P1。
随后单独授权的 two-reference matched Development probe 在新 RGB GET 前排除旧 C2 图像、target 与 same-class distractor
IDs，从 5 条复用 SUN3D capture source 中冻结 5 个新 target 与 14 个新 competition frame。Single arm 为 `14/14`
target outrank，固定 `max(R1,R2)` arm 反而为 `11/14`，paired transition=`0 rescue / 3 collateral`、median target-margin
delta=`-0.03808`；R2 为 distractor 提供最大分数 `9/14`，为 target 仅 `2/14`。因此新增 exemplar 并不自动增加
instance authority，naive multi-reference max 被拒绝。该 cohort 没有 single-arm hard-error 分母，不能证明单 reference
信息完备，也不授权在打开 outcome 后换 aggregation 追结果。以上探针都不回答 `NOT_VISIBLE` 或 calibration；上游
public-real miner 与 C2 终态保持不变。
新授权的 source-disjoint T-LESS/BOP19 Development probe 随后先冻结 roster，再由同一 DINOv2-S baseline 建立
`27/30` 与 3 个真实 hard errors。固定 PDM PerMIR unary arm 在 6 hard/6 control 上仅 `1 rescue / 4 collateral`，
control retention=`4/6`，未过预注册门；8 个 absence 仍因无 NONE threshold 为 `NOT_EVALUABLE`。PDM unary 路线关闭，
且 T-LESS 的 near-instance industrial objects 不支持 native same-class 外推。
只读 residual identity failure-layer audit 随后复核冻结原图、bbox、instance truth 与 raw scores，没有调用模型或修改结果。
12 对中 `representation collapse=4 / local layout lost=3 / UNKNOWN=5`；现有证据没有建立 pixel information insufficient、
background shortcut 或 quality failure。后续 NearID-style small head 在全新 CORe50 上为 `4 rescue / 17 collateral`，
再后的 analytic spatial-layout arm 在全新 Washington 300-instance roster 上为 `74 rescue / 218 collateral`。因此
passive single-reference RGB exact-instance mainline 已按 stop rule 关闭；不再以新 backbone/head/layout 续跑，identity
signal 未建立前也不启动 open-set calibration。下一研究必须改变输入合同，转向主动 distinctive evidence 或独立身份来源。
PA2 后的 `Proposal–Identity Responsibility Mismatch` 只是待验证解释；当前 7-case 没有 provider-public goal semantics，
private category 不能回填成 text prompt。既有 P1-D0/PA0、Silver-B 与 Last-10m 均无可证明的
user/product goal-before-truth provenance，历史 eligible episode 为 `0`。本轮已新建 pre-truth product goal 与
entrance-anchor development cohort；PA3 在 2 个 visible case 上 IoU >= 0.30 的 Recall@10 为 `0/2`，FRG1 frozen
functional prompt 为 `1/2`。结果只属于 consumed development mechanics；fresh hierarchical confirmation 尚未建立。
AMRM、verifier、reacquisition、VLM 与 App 均未授权。随后 S0v3 在 metadata/pixel/truth/provider 前冻结 12 个
fresh product goals 与 public spatial candidate set，物化 8 个 episode、22 帧；private truth 只有 `6` 个 visible
episodes、`7` 个 visible frames，未同时达到预注册 `5/8` 门，因此以零模型调用终止。该结果只诊断 observation
denominator，不是 proposal provider 的负结果。
本 successor 在 pixel/truth 前冻结 public Goal Contract 与 OSM/Overture entrance candidate set，再自动挖掘 Mapillary
continuous sequence 的真实 approach episode；ADT/Ego4D/Habitat 分别只承担 calibration、domain realism、explicit-goal
mechanics。物理采集不是当前 blocker。Synthetic arrival authority、
P1 persistence/reacquisition、模型搜索和默认 App promotion 均关闭。

Assistive Geometry 的 factor-wise、obstacle 与 Q-Plane 表示支线均已按停止条件关闭；TARO R38 也已在
parent-disjoint ARKitScenes Validation confirmation 上有效失败并关闭。用户已明确新开 SATOM-A，
它通过主动稀疏 metric range 与 causal task-space memory 改变输入信息、时间结构、表示和动作，
不是旧 selector 的 rescue continuation。SATOM-R0 的 Bonn Real E0 在任何 arm metric 产生前，
因 frozen DepthART prior 无法稳定提供 ground-height observability 而 `NOT_EVALUABLE` 并关闭。
GA-SATOM 的 physical-ToF G0 协议保留但按用户的无 ToF 选择暂停；VI-Task Geometry 的 RGB+IMU
G0 协议也按进一步的纯 RGB 选择暂停。SVRF-O0 的协议、机制与 source/index 权限仅作历史保留，不执行
bus canary、member index、payload materialization、truth writer 或 O0。Failure Synthesis 冻结的
D-ORACLE-1 同样暂停，不执行 source/action truth/policy lock 或 outcome access。
当前唯一 active algorithm lane 是 SAGE-LM V1 的 observation-channel diagnostic；24 个 controlled real-RGB episode 未保留
V0 synthetic uplift，只允许依次定位 boundary association、reciprocal flow survival 与 metric-depth range。任何
policy/threshold/baseline/cohort 修改、prospective goal intake、fresh confirmation、正式模型选择、新 proposal arm 或 App
集成都需要用户另行授权，不能由旧文档中的 successor 或历史优先级自行恢复。
DepthART D3R6 仍保持暂停；其 bounded deferral contract 与 fresh gate 保留，但 post-hoc
same-domain random audit 不支持 risk ranking 的增量收益，不恢复 D3R6 执行权限。

表中“下一动作”就是唯一 successor；完整历史、数值和所有禁止项留在路线真源。`无` 表示
路线保持 closed、paused 或 diagnostic，不能从旧文档推导隐含下一步。

| 路线 | 主张 | 当前状态 | 唯一真源 | 下一动作（唯一 successor） | 禁止动作 | 影响默认 App |
|---|---|---|---|---|---|---|
| Goal Copilot / passive exact-instance identity closure | generic appearance、diffusion、multi-reference、learned head、layout 是否建立可迁移的单参考 RGB identity rule | `NEARID_SMALL_HEAD: RESCUE=4 COLLATERAL=17 / LAYOUT: BASELINE=702/900 CHALLENGER=558/900 RESCUE=74 COLLATERAL=218 CONTROL_RETENTION=68.9% STABLE=42.0% / PASSIVE_SINGLE_REFERENCE_RGB_EXACT_INSTANCE_MAINLINE_STOP` | [layout result](goal-copilot/SPATIAL_LAYOUT_IDENTITY_VERIFICATION_V0_RESULT_2026-08-24.md) | 已由 active distinctive V0 改变输入合同；passive 路线保持 closed | 新 passive backbone/head/layout、threshold/fusion/Deep Sets；先跑 open-set calibration；从本终态晋升 P1/App | 否 |
| Goal Copilot / active distinctive evidence V0 | 三帧 reference sweep 与 candidate-unique local anchors 是否获得 passive 单图没有的新信息 | `4 TARGETS / 16 PRESENT DECISIONS / ACTIVE=PASSIVE: TOP1 11/16, WRONG_LOCK 9/20, REACQUISITION 3/4 / APPEARANCE_DERIVED_DISTINCTIVE_ANCHOR_NO_UPLIFT / OCR_NOT_EVALUABLE` | [active result](goal-copilot/ACTIVE_DISTINCTIVE_EVIDENCE_ACQUISITION_V0_RESULT_2026-08-24.md) | 仅在可执行独立 OCR/logo/marker evidence runtime 建立后另立 semantic-anchor V1 | 调 patch/backbone/aggregation/cosine/margin/lock threshold；用纯弃权当 uplift；旧部分 OCR 输出补分母；identity/P1/default-App 晋升 | 否 |
| Goal Copilot / semantic distinctive anchor V1 | 独立 OCR/sign/marker 信息是否能在相同 sequence 与 candidate roles 上产生 appearance-only 没有的可展示增益 | `CONTROLLED_DERIVED_DEMO / PASSIVE 11/16 -> SEMANTIC 16/16 / WRONG_LOCK 9 -> 0 / REACQUISITION 3/4 -> 4/4 / LOST_ABSTAIN 4/4 / LIVE_ANDROID_NOT_RUN` | [semantic result](goal-copilot/SEMANTIC_DISTINCTIVE_ANCHOR_V1_RESULT_2026-08-24.md) | research-only live `SEARCH -> SEMANTIC LOCK -> LOST -> FRESH REACQUIRE` seam；marker device canary 先行 | 把 derived canary 当自然分布/产品效果；重复 logo 冒充 physical identity；appearance fallback/tracker 自建 identity；默认 App/P1/导航/安全晋升 | 否 |
| Goal Copilot / SAGE-R V3-C authority graph | sign role + decisive token 能否增加自然 referent correct 且不增加 wrong lock | `FRESH 6 SOURCES/11 QUERIES/33 OBS / V2 11 CORRECT,1 WRONG / V3-C 13 CORRECT,5 WRONG,DIRECTION 0/3,DIRECTORY FALSE 5 / CLOSE_NATURAL_SAGE_R` | [V3-C result](goal-copilot/SAGE_R_V3_C_AUTHORITY_TYPED_NATURAL_RESULT_2026-08-24.md) | 无；保留 controlled exact-anchor，另选有正信号的 Goal Copilot 能力 | 改 fresh cohort 或重跑；V3-B/V4；natural graph 修补；Android/P1/App 晋升 | 否 |
| Goal Copilot / SAGE-LM last mile | exact identity 后，真实 RGB observation 能否保留 synthetic aperture uplift | `V0 7->33 POSITIVE / V1 RGB 24: BASELINE=7,SAGE=2,FLOW_PASS=0 / UPLIFT_NOT_PRESERVED` | [V1 result](goal-copilot/SAGE_LM_V1_CONTROLLED_REAL_RGB_OBSERVATION_RESULT_2026-08-24.md) | 只分解 flow -> boundary -> range | 改 policy/门/baseline/cohort/anchor；Android/P1/App | 否 |
| Goal Copilot / DINOv2-S matched two-reference probe | 固定 scorer、candidate 与 evaluator，只增加 R2 和 exemplar-set max，是否增加 identity rank | `SOURCE_REUSED_DEVELOPMENT / SINGLE=14/14 / TWO=11/14 / RESCUE=0 / COLLATERAL=3 / NET=-3 / MEDIAN_DELTA=-0.03808 / NAIVE_MAX_REJECTED` | [Goal Copilot current](goal-copilot/README.md) | 无自动 successor；新授权须改为 distinctive-anchor/correspondence representation，或建立新的 hard-error denominator | 同 cohort 改 aggregation/threshold/fusion 或换 crop/layer/model；升格 Confirmation；belief/tracker/Active Search/P1/App | 否 |
| Goal Copilot / source-disjoint PDM hard-error unary probe | 新 T-LESS/BOP19 roster 上，固定 PDM PerMIR unary score 能否救 DINO hard errors 且不击穿 controls | `DINO=27/30 / HARD=6 / CONTROL=6 / RESCUE=1 / COLLATERAL=4 / CONTROL_RETENTION=4/6 / GATE_FAIL / ABSENCE_NOT_EVALUABLE / PDM_UNARY_REJECTED` | [Goal Copilot current](goal-copilot/README.md) | residual audit 已完成；PDM 无 successor，NearID 仍须另行授权 | outcome 后改 PDM layer/timestep/prompt/top-k/crop/threshold/fusion；把 T-LESS 当 native same-class 或 Confirmation；P1/App | 否 |
| Goal Copilot / DINOv2-S order-free local appearance probe | 独立 local patch evidence 能否在无 A/B positional bias 下区分 frozen target 与 same-class distractor | `TARGET_OUTRANKS=13/17 / HISTORICAL_WRONG=3/4 / ROBUST_TARGET=1/1 / ORDER_SENSITIVE=1/1 / STABLE_DISTRACTOR=1/2 / CONTROLS=10/13 / COMPLEMENTARY_NOT_SUFFICIENT / CONSUMED_ORACLE_CANDIDATE_DISCOVERY` | [Goal Copilot current](goal-copilot/README.md) | 已由一次新 target/frame、source-reused two-reference matched Development probe 检验；naive max 失败，不自动续跑 | outcome 后扫 threshold/crop/layer/model/fusion；把 raw rank 当 verifier；belief/tracker/Active Search/P1/App | 否 |
| Goal Copilot / oracle competing-identity diagnostic | 显式加入 target-vs-same-class 竞争假设后，能否救回四个 wrong-instance commit 且保留原正确 case | `ORIGINAL_ORDER: WRONG_CASE_TARGET=2/4; CONTROL_TARGET=12/13; CONTROL_CONTESTED=1/13 / A-B_SWAP: ROBUST_TARGET=1/4; STABLE_DISTRACTOR=2/4; ORDER_SENSITIVE=1/4 / RELIABLE_VERIFIER_NOT_ESTABLISHED / CONSUMED_ORACLE_CANDIDATE_DISCOVERY` | [Goal Copilot current](goal-copilot/README.md) | 已由一次固定 DINOv2-S order-free local probe 检验；不自动续跑 | 把 oracle candidates 当产品 proposal；prompt/model/threshold sweep；直接上 belief/tracker/Active Search/P1/App | 否 |
| Goal Copilot / public referent visible-only passive identity probe | reference + public target region 在真实视角变化和同类干扰下能否指出同一 physical instance | `20/21 FOUND / 16/20 SAME_INSTANCE / 4/20 SAME_CLASS_DISTRACTOR / 1 ABSTAIN / 3/7 THREE_VIEW_STABLE / CONSUMED_DISCOVERY` | [Goal Copilot current](goal-copilot/README.md) | 已由一次 oracle competing-identity diagnostic 攻击四个错误；不自动续跑 | 扩模型/阈值/arm sweep；用 visible-only 结果声称 NOT_VISIBLE/calibration；Active Search/P1/App | 否 |
| Goal Copilot / public-real episode mining + selective guidance V0 | 自动挖掘公开真实 approach episode，并定位 current-frame guidance failure layer | `8X89_SEALED / ABOTN_WEBGL_TRANSPORT_PASS / ARRIVAL_ONLY / NO_FUNCTIONAL_REGION` | [public-real result](goal-copilot/BLINDASSIST_PUBLIC_REAL_EPISODE_MINING_V0_RESULT_2026-08-23.md) / [substrate audit](goal-copilot/BLINDASSIST_PUBLIC_FUNCTIONAL_TRUTH_SUBSTRATE_AUDIT_2026-08-23.md) | 无算法 successor；仅同 task arrival-only provider-firewall canary，或 source-native entrance-region export | 换 goal/补抽/挑 teacher；WebGL 冒充官方 renderer；endpoint/sign 冒充入口；绕过访问控制；P1；模型/阈值/provider sweep | 否 |
| Goal Copilot / P1-PA3-S0v3 observation contract | 合法 public goal + spatial candidate set 能否产生足够的 entrance-visible PA3 denominator | `6_VISIBLE_EPISODES / 7_VISIBLE_FRAMES / NOT_EVALUABLE / PROVIDER_CALLS=0` | [P1-PA3-S0v3 result](goal-copilot/P1_PA3_S0V3_PUBLIC_SPATIAL_CANDIDATE_SET_RESULT_2026-08-22.md) | 转入 public-real miner：先冻结 Goal Contract，再自动挖 Mapillary approach sequence，最后私有建 truth | 补抽/替换 sealed frames；强造 UNIQUE/visibility；运行 PA3/verifier；把 UNKNOWN 写成 negative | 否 |
| Goal Copilot / P1-PA3 + FRG1 proposal | 合法 goal semantics / frozen functional prompt 能否建立 bounded target availability | `PA3=0/2@10 / FRG1=1/2@10 / CONSUMED_DEVELOPMENT_ONLY` | [P1-PA3 + FRG1 result](goal-copilot/P1_PA3_GOAL_SEMANTIC_AND_FUNCTIONAL_REGION_RESULT_2026-08-22.md) | 由新的合格 prospective observation denominator 重新授权后才可另行冻结 proposal run | 同 cohort 调 prompt/threshold/model/拼框；identity verifier、AMRM、App；fresh confirmation claim | 否 |
| Goal Copilot / P1-PA3-C0 public Goal Contract | 在 capture/truth 前形成 provider-public goal 与全局 canonical prompt | `S0V3_FRESH_GOALS=12 / MATERIALIZATION_VALID / PA3_INFERENCE_NOT_AUTHORIZED` | [P1-PA3-C0 result](goal-copilot/P1_PA3_C0_PUBLIC_GOAL_CONTRACT_COHORT_MATERIALIZATION_RESULT_2026-08-22.md) | 保持同一 C0 contract，转入 prospective first-person observation cohort | 回填旧类别/target name；伪造 user intent；在 observation gate 前跑 PA3、AMRM/verifier/App | 否 |
| Goal Copilot / P1-PA2 representation audit | oracle ROI 下 target/context prompt 是否仍有 proposal signal | `WEAK_CONTEXT_SIGNAL_1_OF_7 / REPRESENTATION_MISMATCH_PRIMARY / APP_UNCHANGED` | [P1-PA2 result](goal-copilot/P1_PA2_TARGET_REPRESENTATION_OBSERVABILITY_AUDIT_RESULT_2026-08-22.md) | 无自动 successor；若另行启动，审计或改变 representation/prompt interface | outcome 后调 crop/prompt/provider；直接 parent-first、adaptive search、model zoo、AMRM/App 或扩大 claim | 否 |
| D-ORACLE-1 causal ladder | 三臂 matched oracle intervention 定位 downstream target-policy stack 与 estimated representation 的损失 | `PAUSED_BY_BA_ADT_PRODUCT_RESEARCH_MAINLINE / PROTOCOL_FROZEN / NO_EXECUTION / NO_SEARCH` | [Failure diagnosis current](failure-synthesis/README.md) | 无；只有用户显式改变主线后才可重开 | 执行 source/action truth/policy lock；增加第四竞争臂；提前拆H3/H4；训练/调policy/threshold；读取outcome后换parent/gate | 否 |
| SVRF / Scale-free Visual Risk Field | 纯 RGB 派生的相对深度动态、局部扩张和视觉通道侵入能否形成稳定相对风险排序 | `PAUSED_BY_BA_ADT_PRODUCT_RESEARCH_MAINLINE / RGB_ONLY / A2D2_SPRING_SOURCE_LOCK_VALID / STREAM_INDEX_NOT_ACTIVE / REAL_O0_NOT_RUN / NO_TRAINING` | [SVRF current](svrf/README.md) | 无；只有用户显式改变主线并有新的 representation-headroom 前置证据后，才可另行恢复 | bus canary、member index、payload/truth materialization、O0、训练或接 Android | 否 |
| VI-Task Geometry | 同刚体 RGB+IMU 自校准 metric pose/ground，再为未来 task geometry 分配 computation/parallax budget | `PAUSED_BY_PURE_RGB_SELECTION / G0_PROTOCOL_RETAINED / REAL_G0_NOT_RUN / NO_TOF / NO_TRAINING` | [VI-Task Geometry current](vi-task-geometry/README.md) | 无；只有用户明确恢复 RGB+IMU metric-frame 路线才可重开 preflight | 实现/采集/运行 G0；用手机 IMU配眼镜视频；重跑 ARCore D45；提前训练、主动分配或接 Android | 否 |
| GA-SATOM | 固定总稀疏测距信息预算内，先建立 metric ground frame，再保留剩余预算给未来 task sensing | `PAUSED_BY_NO_EXTERNAL_TOF_SELECTION / G0_PROTOCOL_RETAINED / REAL_G0_NOT_RUN / NO_PROCUREMENT` | [GA-SATOM current](ga-satom/README.md) | 无；只有用户明确恢复 external-ToF 路线才可重开 preflight | 采购/采集/运行 physical G0；用单区/模拟/旧输出替代；提前运行 G1/arm/训练 | 否 |
| SATOM-A | metric pose + active sparse ToF + frozen dense prior → causal task-space occupancy memory | `WILD_LAB / SATOM_R0_REAL_E0_NOT_EVALUABLE / DEPTHART_GROUND_HEIGHT_OBSERVABILITY_FAIL / NO_ARM_METRIC / CLOSED_NO_TUNING` | [SATOM-A current](satom/README.md) | 无；新的 pre-outcome 协议须先提供独立 ground-height observability 或 materially different representation | 用 truth height/scale 修补 candidate；在已打开 Bonn/DepthART 输出上放宽高度门、改 winner rule、训练或接 Android | 否 |
| BlindAssist Assistive Geometry | 连续因子 → selective no-regret → body-swept geometry | `CORRECTION_GAIN_LOPO_FAIL_STOP / ANGULAR_BOUNDARY_FAIL_CLOSED_SAFE_BUT_TASK_INERT / SUPPORT_VALIDITY_FAIL_OPEN / OBSTACLE_RGB_INTERACTION_FAIL_STOP / POSE_ANALYTIC_FAIL_STOP / QPLANE_O0A_REPRESENTATION_HEADROOM_FAIL_CLOSE_NO_TRAINING / CURRENT_OBSTACLE_TASK_ROUTE_CLOSED / SCIENTIFIC_NOT_RUN` | [Assistive Geometry current](assistive-geometry/README.md) | 无 active successor；需新增 source-native obstacle supervision，或 Q-Plane 家族之外的 materially different representation 后另立协议 | 调参/重跑 Q-Plane 或进入 O0-B/训练；继续调旧 obstacle 候选；让 support 创建 validity；重开 correction/boundary/fresh3；把 consumed negative 写成 task success | 否 |
| TARO / Task-directed Active Risk Observability | 同预算观测使 body/path query 更早可识别 | `R31_V6_LEAK_INVALIDATED / R36_FRESH_PARENT_FAIL / R37_CLOSED / R38_REFERENCE_PREFLIGHT_PASS / R38_R32_FRESH_PARENT_CONFIRMATION_FAIL / TARO_CLOSED_NO_RESCUE` | [TARO current](taro/README.md) | 无；R38 已有效消费，论文主线盘点前不建立 R39/R40/R41 | 覆盖/重跑 R38；回调 R31/R32/R35/R36/R37；以 R38 outcome rescue tuning；接默认 App/risk/guidance；夸大为 fresh-source、broad breakthrough、安全或产品成功 | 否 |
| AG-QSF / Queryable Survival Geometry | profile-queryable q-contact 生存分布统一 clearance 与 horizon occupancy | `CLOSED_DATA_SUPPORT_INSUFFICIENT / H1_NOT_EVALUABLE` | [AG-QSF current](assistive-geometry-qsf/README.md) | 无；只有新的 pre-outcome target/data contract 可重开 | 伪造 parent-disjoint censor support、实现未授权 H2 或声称数学假设被反证 | 否 |
| AG-CBF / Corridor Bottleneck Field | 检验 ground-aligned corridor bottleneck 是否保留三带摘要丢失的信息 | `R0_CLOSED_DATA_SUPPORT_NOT_EVALUABLE / ORACLE_NEVER_AUTHORIZED` | [AG-CBF current](assistive-geometry-cbf/README.md) | 无；新版本必须从 source-geometry/data support 重开 | 填补 UNKNOWN、事后改 gate、实现 oracle/model 或复用旧权限 | 否 |
| DepthART-S | parent-relative risk + fixed-budget UNKNOWN deferral，降低危险 false-clear 且封顶 coverage/false-block 代价 | `PAUSED_BY_USER_AND_BA_ADT_MAINLINE / D3R6_FROZEN_GATE_PASS / BOUNDED_DEFERRAL_CONTRACT_PRESERVED / POSTHOC_SAME_DOMAIN_RANDOM_AUDIT_NO_INCREMENTAL_RANKING_SUPPORT / R2_CANDIDATE_NOT_AUTHORIZED` | [DepthART current](hftf/README.md) | 无；按用户要求暂停 | 在已打开 parent 上重调 2% budget/checkpoint；把 fresh gate 写成 ranking 增量证据；把 UNKNOWN 当 negative；读 sealed R2 或写成默认 App/安全结论 | 否 |
| DA2 | teacher、baseline、reference、fallback | `CLOSED / FROZEN_DEVELOPMENT_REFERENCE` | [DA2 closure](hftf/DAV2_P1_P2_EXECUTION_CLOSURE_2026-08-05.md) | 无；新问题必须建立新版本和独立数据路线 | 把 reference 写成 active candidate 或沿用旧下一步 | 否 |
| YOLO + 语义分割双环 | 论文次线：事件级风险/可通行性增量 | `THESIS_DEVELOPMENT_SECONDARY / RISKSEG_R0_NEGATIVE_NOT_PROMOTABLE` | [dual-loop current](dual-loop/README.md) | `RISKSEG_EVENT_EVAL_DATA_REPAIR_SUCCESSOR` | truth 未冻结时训练、选模或晋级 | 否 |
| RCLE-RF | 暂停的历史风险场研究 | `PAUSED / NO_ACTIVE_EXECUTION` | [RCLE current](rcle/README.md) | 无；只有用户明确重开后才能建立新 scoped successor | 消费暂停前授权或把旧 snapshot 的下一步当当前权限 | 否 |
| USTRF-SC route-conditioned | 已收口的历史路线代理 | `CLOSED / HISTORICAL_DIAGNOSTIC` | [USTRF closure](ustrf-sc/USTRF_ROUTE_CONDITIONED_PROGRAM_CLOSURE_R1_2026-07-25.md) | 无；实质不同的假设必须登记为新路线 | 把实验 App、旧协议或诊断结果写成当前主线 | 否 |

## 算法边界

数据、链路、性能和部署证据可以支撑算法，但不能替代 admission；算法结果也不能反向证明
产品安全、默认 App 可用性或真实用户效果。
