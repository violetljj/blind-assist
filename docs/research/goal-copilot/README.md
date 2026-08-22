# Goal-Driven Visual Copilot

状态：`current / PRODUCT_AND_RESEARCH_MAINLINE / PROSPECTIVE_FIRST_PERSON_DEVICE_RECORDER_READY / PA3_DENOMINATOR_GATE_ENFORCED / REAL_DEVICE_COHORT_NOT_CAPTURED / PA3_INFERENCE_NOT_AUTHORIZED / IDENTITY_NOT_AUTHORIZED / DEFAULT_APP_UNCHANGED`

完整系统蓝图见 [`V2 路线图`](BLINDASSIST_GOAL_DRIVEN_VISUAL_COPILOT_V2_ROADMAP_2026-08-21.md)。本页是
Goal Copilot 动态执行状态真源；历史协议与数字只通过链接保留，不再授予执行权限。

## 当前研究实现

当前算法基础设施结果真源是
[`P1 PA3 execution-gate hardening result`](P1_PA3_EXECUTION_GATE_HARDENING_RESULT_2026-08-22.md)。physical capture manifest
现在必须验证自身 body hash、C0/设备 receipt/plan 绑定、device timeline 和 outcome-blind fixed-offset 语义；随后只有私有
denominator 同时达到 `>=5 visible episodes / >=8 visible frames` 才能生成绑定唯一 prediction/journal 的授权 receipt。
YOLOE runner 在授权前不能导入 provider，失败或中断后禁止 replay；evaluator 也只接收完成 journal 绑定的 prediction。

设备 producer 仍由 [`P1 prospective device recorder result`](P1_PROSPECTIVE_DEVICE_RECORDER_IMPLEMENTATION_RESULT_2026-08-22.md)
定义。本机检查到 Android SDK/ADB 正常，但 `0` 个 ready device、`0` 个 AVD，因此真实 cohort 尚未采集；provider calls
仍为 `0`，PA3 不授权，默认 App 未改动。

当前 observation evidence 真源仍是
[`P1-PA3-S0v3 public spatial candidate-set result`](P1_PA3_S0V3_PUBLIC_SPATIAL_CANDIDATE_SET_RESULT_2026-08-22.md)。
12 个 fresh product goals 与 public spatial candidate set 均在 metadata/pixel/truth/provider 前冻结；统一采集物化
8 个 episode、22 帧，private pre-provider truth 得到 `6` 个 visible episodes、`7` 个 visible frames。由于预注册
authorization 同时要求 `5/8`，frame denominator 差 1，PA3 以零模型调用判为 `NOT_EVALUABLE`。不得把它写成
YOLOE 负结果，也不得继续 retrospective Mapillary resampling；唯一 successor 是 prospective、明确面向目标入口的
第一视角 Goal Contract cohort。

其前一个算法结果是
[`P1-HRG2 fresh public-anchor global-local reranking result`](P1_HRG2_FRESH_PUBLIC_ANCHOR_GLOBAL_LOCAL_RERANKING_RESULT_2026-08-22.md)。
7 个 museum Goal Contract 在 pixel/truth 前冻结；只用公开命名场所 anchor 的 multi-view acquisition 得到 16 帧，但 pre-provider
truth 只有 `2 VISIBLE / 14 NOT_VISIBLE`，覆盖 2/7 goal episodes。IoU `0.30` 下，HRG0 与 HRG2 都在 Top-3 达到
`2/2`；HRG2 没有提高任何预注册 Recall@K，并把 Recall@1 从 `1/2` 降到 `0/2`。这证明两帧上的 bounded proposal
mechanics，但 denominator 太小，不能建立跨场所 coverage 或授权正式 identity verifier。`SET_VALUED` 入口任务与 `UNIQUE`
实例 identity 也必须分开；AMRM 与 App 仍不授权。

后继 observation 扩展必须使用显式的 `P1-PA3-S0` public spatial Goal Contract：导航侧公开的 OSM place、parent building 与
route-endpoint candidate 在 Mapillary metadata、project pixel 和 truth 前冻结并绑定 C0，再作为 provider-public context 进入
materialized input。它不是 evaluator truth，也不携带 visibility/bbox/mask；任何 hash、source-role、precedence 或 endpoint 漂移
都 fail closed。不得再把内部 entrance anchor 隐式当作合法输入。

首个 S0 v2 fresh cohort 在 14 个目标中只有 5 个 parent-bound entrance tag、1 个可 materialize episode、3 帧且
`0 VISIBLE`，因此以零模型调用终止。后继 v3 不修改该 sealed cohort，而把公开 spatial contract 改成 bounded entrance
candidate set；无 entrance tag 时使用最多 4 个显式标注为非 truth 的 building-frontage midpoint fallback，并在所有 candidate
上做统一 geometry-only frame ranking 与 image-id 去重。该 v3 已由上述 S0v3 result 消费，不再补抽或重跑。

其前一个 fresh paired 结果是
[`P1-HRG1 fresh parent-bound local-refinement result`](P1_HRG1_FRESH_PARENT_BOUND_LOCAL_REFINEMENT_RESULT_2026-08-22.md)：
7 个 visible frame 上 HRG0 Recall@10 为 `2/7`，冻结的 HRG1 Top-5 coarse-to-local refinement 为 `0/7`。该结果解释了
HRG2 的全 Top-10 parent 与全局 local-score reranking 设计，但已观察 cohort 不作为 HRG2 confirmation denominator。

其前一个
[`P1-HRG0 fresh single-visible-case result`](P1_HRG0_FRESH_HIERARCHICAL_FUNCTIONAL_CONTEXT_RESULT_2026-08-22.md)
在唯一 visible Haarlem city-hall case 上 rank 1、IoU `0.8955`，只能作为 fresh 单例 observation，已被当前更大的 paired
cohort 取代为动态决策依据。

其 predecessor 是
[`P1-PA3 + FRG1 result`](P1_PA3_GOAL_SEMANTIC_AND_FUNCTIONAL_REGION_RESULT_2026-08-22.md)：2 个 consumed-development
visible case 上 YOLOE semantic-only PA3 为 `0/2`，FRG1 Recall@10 为 `1/2`。它们只用于说明 HRG0 设计来源，不能
与 fresh 单例拼接成确认分母。

C0 真源仍是
[`P1-PA3-C0 public Goal Contract cohort materialization`](P1_PA3_C0_PUBLIC_GOAL_CONTRACT_COHORT_MATERIALIZATION_RESULT_2026-08-22.md)。
它不运行模型，只要求 user/product task semantics 在 capture 与 target truth 前形成 immutable public receipt，并由全局
`goal_type -> canonical_prompt` exact mapping 派生 prompt。既有 P1-D0/PA0、Silver-B 与 Last-10m 均无法证明该
precedence，历史合格 episode 为 `0`，没有历史回填。prospective intake mechanics 已就绪；后续 fresh cohorts 已沿用
同一 contract 形成 immutable receipt，但任何 goal receipt 本身仍固定 `pa3_inference_authorized=false`。早期 development
inference 由另行冻结的 run manifest 授权；最新 S0v3 因 observation denominator 不足没有获得授权。两者都不反向改变
C0 receipt 的 authority。AMRM、verifier 与 App 均未授权。

PA2 后的 `Proposal–Identity Responsibility Mismatch` 只登记为有效待验证解释，不是 YOLOE instance-ReID 机制事实。
只有合法 C0 cohort 后，PA3 才能单独测试 goal-semantic bounded candidate availability。

上一执行面已以
[`P1-PA2 target representation observability audit`](P1_PA2_TARGET_REPRESENTATION_OBSERVABILITY_AUDIT_RESULT_2026-08-22.md)
终止，没有自动 successor。PA2 明确是 consumed-Development oracle autopsy：GT 只为 exact target crop、3x target-centred
ROI 与 evaluator 提供位置；provider checkpoint、visual prompt API、640 输入与 score floor 不变。AMRM、reacquisition、
identity selection、verifier、VLM、VIO/SLAM、geometry 与 App 全部拔除。

三臂在 IoU 0.30 完整 rank 的 recall 为 target crop target-only `0/7`、oracle ROI target-only `0/7`、同一 ROI
target+context `1/7`。唯一 context hit 是 wine rack，rank 4 首次达标、best IoU `0.3769`；其余 6/7 未恢复。
因此结果保留一个弱的 case-local context interaction，但主归因仍是 target representation / target-conditioned grounding
mismatch；不授权 parent-first、adaptive search 或 model-zoo sweep。后续若另行启动，应直接改变或审计 target
representation/prompt interface，并把该 1 positive + 6 negative 作为 consumed diagnostic counterexamples。

PA1 的历史结果是：与 sealed PA0 使用同一 YOLOE visual-prompt provider 和 7 帧 cohort，只把 full-frame 640 替换成固定
`2x2 / 20% overlap / tile-to-640` 搜索。结果见
[`P1-PA1 result`](P1_PA1_TARGET_PROPOSAL_RESCUE_RESULT_2026-08-22.md)：预注册的 IoU >= 0.30 Recall@10
在 bounded pool 和完整 postprocessed rank 中均为 `0/7`，PA0 的 5 个 IoU >= 0.10 absent cases 被救回 `0/5`。
固定 tiled zoom 增加了 4 倍输入图像和 proposal 竞争，但没有建立新的足够质量 target candidate。终态为
`P1_PA1_FIXED_TILED_SCALE_RESCUE_NOT_SUPPORTED_ON_FAILURE_COHORT`；不在 outcome 上继续搜索 tile、overlap、
resolution、threshold、NMS 或 K，也不自动进入 parent-first。

PA0 的历史结果是：
[`P1-PA0 result`](P1_PA0_TARGET_CANDIDATE_AVAILABILITY_RESULT_2026-08-22.md) 在 IoU >= 0.10 的
Recall@1/3/5/10 为 `0/7, 0/7, 0/7, 2/7`，两个弱 candidate 首次位于 rank 9/10；IoU >= 0.30 与 0.50
在所有 K 均为 `0/7`。这只是 failure-cohort 上的弱 `TOP1_COLLAPSE_SIGNAL`，不是 proposal availability pass、
模型选择或泛化证据。

## 已关闭的 AMRM0 实验

用户此前明确将 `P1-AMRM0 Adaptive Multi-view Referent Memory` 定为主实验路径。准确状态是“当时最值得优先
验证的新研究假设”，不是已经证明有效。核心假设为：在第一视角最后十米任务中，相比持续维护 2D correspondence，
积累经过身份验证的多距离、多视角 referent memory，是否能提高真实同实例重捕获，同时降低 wrong-instance
reacquisition。

其冻结实现是
[`P1-AMRM0`](../../../scripts/research/goal_copilot_bridge/p1_verifier_first/README.md)。P1-VF0 作为其 verifier foundation，实现
`GoalContract`、有限且不可变的 `ReferentLedger`、常驻 `H_other`、parent/child-slot relation、distractor registry、
observability-gated negative evidence、appearance cap，以及 `CONFIRMED_VISIBLE / VERIFYING / AMBIGUOUS / STALE /
REBOUND_TO_NEW_VALID_INSTANCE / DISPROVED` 裁决。

候选源只能 proposal，只有 verifier 可更新 active referent 或 identity gallery；确认要求独立 prediction/context
支持与 distractor exclusion，appearance 永远不能单独授权身份。`UNIQUE / SET_VALUED / AMBIGUOUS`、identity 与
current goal validity 分开保存。`memory.py` 进一步保存 target/context/full-frame immutable refs、orientation 与
distance × viewpoint × scale coverage；tentative 与 verified 严格分离，只有匹配 verifier confirmation receipt 的观察
才能晋升，retrieval 不暴露 tentative。重复观察丢弃，新的 coverage/context 才进入 bounded bank；出画、遮挡、
stale、disproved 或 referent rebound 时停止写入旧 bank。主动取证只允许保持静止、旋转扫描或纳入 parent context。
31 项 contract tests 已通过；这只证明 mechanics/invariants，不是 real-data utility 或 scientific result。

P1-AMRM0 与已消费的 W1-T0/W2 实质不同，不修改、续跑或覆盖旧 cohort，也不继承旧 execution authority。AMRM0
本身未引入新 RGB provider、数据 roster 或 performance experiment；VIO、SLAM、metric 3D、POMDP、主动平移、自动到达距离、
Android/default App 均禁止。

## 已关闭的当前帧工程里程碑

上一执行面 `BLINDASSIST_LAST_10M_REGROUNDING_V0` 已关闭：它复用且不修改现有 P0 named-building entrance
grounding/provider，完成“入口寻找—引导—重新观测—确认”的当前帧机械闭环。只支持清晰、相对唯一的建筑入口。

最小状态机：

```text
SCAN
-> CURRENT_CANDIDATE
-> ALIGN
-> ADVANCE_AND_REOBSERVE
-> ARRIVAL_CONFIRM
-> COMPLETE / RESCAN / ABSTAIN / EXHAUSTED
```

每次转向、前进或重扫后必须提交新的 frame 并重新调用 P0 grounding。控制 state 不保存 candidate id、bbox、
图像、特征、score、handoff 或 identity，也不比较相邻帧；P0 persistence handoff 只校验当前帧绑定后丢弃。
无唯一可靠 candidate 时固定输出“没有可靠找到入口，请停下并缓慢重新扫描。”；连续三次无法确认后进入
`ABSTAIN`；即使持续有候选，12 条指令仍未完成也必须停止。两种停止都提供现场工作人员或可信任真人协助出口。
任何指令都不得输出“前方安全”。

到达不是历史 identity 延续：当前帧出现居中、近距机械 cue 后先停下，再用一个新的当前帧重新 grounding；
只有新的输出仍独立满足当前帧条件才能 `COMPLETE`。该 bbox cue 只是机械任务规则，不是距离或安全模型。

稳定实现与网络场景命令见 [`last_10m_regrounding_v0`](../../../scripts/research/goal_copilot_bridge/last_10m_regrounding_v0/README.md)。
单帧 adapter 直接复用已冻结 Grounding DINO 与 Terra Brain 函数/身份，hash/config 漂移即 fail closed；不形成新
model/checkpoint 选择。当前实现专项 tests 覆盖实际 P0 output shape、fresh-frame、跨帧/stale fail-close、无
candidate 连续 abstain、二次到达确认、错误确认优先归因和 3x5 汇总形状。Android/default App 不变。

## 网络场景执行结果与报告边界

按用户更新，本里程碑不再需要真实设备。已使用 3 个真实世界 Mapillary 地点的 9 张既有公开场景帧，各执行 5 个
固定序列机械 episodes，共 15 次。完整结果见
[`BLINDASSIST_LAST_10M_REGROUNDING_V0 result`](BLINDASSIST_LAST_10M_REGROUNDING_V0_RESULT_2026-08-22.md)：错误入口确认
`0`、完成 `0/15`、首次可靠发现 median `9,745 ms`、方向指令 `40`、重扫 `5`。15 次全部在固定视角耗尽后
fail closed；由于 playlist 不响应方向指令，该结果只属于网络场景机械回放，不能冒充真实用户控制闭环。

## Action-responsive sanity closeout

后续最小 sanity check 已以 1 个 Mapillary sequence、22 张真实 pose/heading frame、110 个预冻结 viewport states 和
6 个固定 starts 一次性执行。完整结果见
[`responsive sanity result`](BLINDASSIST_LAST_10M_RESPONSIVE_SANITY_RESULT_2026-08-22.md)：完成 `0/6`、false arrival
`0`、observations `29`、可靠 grounding `27`、方向指令 `27`、重扫 `2`、exhausted `6/6`，终态
`CONTROL_POLICY_BOTTLENECK`。一个 episode 从 `15.55 m` 推进到 `5.56 m`，但控制在 viewport 间持续左转/振荡，
未进入 `ARRIVAL_CONFIRM`。该 deterministic viewport replay 不是实际转头或真实用户 walk-through。

每次 observation、candidate、direction、rescan、abstention、completion 和 evaluator 错误确认进入 append-only JSONL/
episode summary。最终报告首先单列错误入口确认数，再报告任务完成率、完成时间、首次发现时间、指令数和重扫数。
每个已 adjudicate episode 只允许以下三个归因之一：

1. `CURRENT_FRAME_GROUNDING_BOTTLENECK`
2. `INTERACTION_OR_CONTROL_BOTTLENECK`
3. `REGROUNDING_LOOP_MECHANICALLY_USEFUL`

错误入口确认无条件计入第一类。只有恰好 3 locations x 5 adjudicated episodes 才能标记
`MECHANICAL_EXECUTION_COMPLETE`。本里程碑是机械工程结果，不是 scientific confirmation、用户安全、导航有效性或
默认 App 准入证据。

## 复用的 P0 权威

P0 冻结合同见 [`Protocol V1`](P0_GROUNDING_PROTOCOL_V1.md) / [`JSON`](p0_grounding_protocol_v1.json)。它分离
`UNIQUE / SET_VALUED / AMBIGUOUS`、Provider availability、Brain selection 和 end-to-end outcome。当前闭环只接受
其 source-frame-bound 输出，不改变 provider、model、checkpoint、threshold、cohort 或 evaluator。

P0 commitment-policy discovery 已以
[`P0-A2`](P0_A2_COMPACT_AMBIGUITY_POLICY_DISCOVERY_RESULT_2026-08-21.md) 的
`COMPLEXITY_ONLY_BUYS_ABSTENTION / A1_INCUMBENT_RETAINED / NO_POLICY_ADMISSION` 收口。当前里程碑不重开
threshold/classifier/XGBoost/Sky，不新增训练、数据集、cohort 或多臂比较。Grounding DINO 仍仅是 proposal，
provider score 不是 truth；P0 的既有证据和 claim ceiling 保持不变。

## P1 历史终态与重开边界

旧 P1 persistence、tracker/correspondence、keyframe/world-anchor 路线全部保持关闭，不因 P1-VF0 改写终态：

- [`P1-A1`](P1_A1_CONSERVATIVE_LOCAL_TRACK_VALIDITY_RESULT_2026-08-21.md) 到
  [`P1-A4`](P1_A4_ONLINE_STRONG_TEMPORAL_CORRESPONDENCE_RESULT_2026-08-22.md) 只保留 consumed Development
  failure evidence；最终 `STRONG_TEMPORAL_CORRESPONDENCE_NOT_SUFFICIENT`。
- [`P1-W1 Stage A`](P1_W1_STAGE_A_SINGLE_EXECUTION_RESULT_2026-08-22.md) 为
  `W1_T0_NOT_EVALUABLE_DATA_OR_INTERFACE`，没有 Stage B authority。
- [`P1-W2 single execution`](P1_W2_SINGLE_EXECUTION_RESULT_2026-08-22.md) 为
  `P1_W2_RGB_REFERENT_INTERFACE_NOT_SUPPORTED`，不允许在 sealed cohort 上降 gate、换模型、改 crop/context 或重跑。

P1-AMRM0 来自本轮用户显式改变主路径，不命名为 `P1-W3`，也不从旧 successor、W0 design 或历史 handoff 恢复权限。
P1 历史 JSON schemas、evaluator 和结果只用于追溯；新实现不得导入其 tracker、SAM propagation、DINO identity、
LoFTR、keyframe memory、SLAM、VIO 或 world-relative state。

## AMRM0 终态与 proposal successor

`P1_AMRM0_DATA_ADAPTER_AND_MATCHED_DEVELOPMENT_CANARY` 已执行并以
`P1_AMRM0_MEMORY_POISONING_FAIL` 终止。它固定复用 consumed P1-D0
15 episodes 与 P1-A4 的 exact candidate bbox stream；A4 sealed output 是 correspondence baseline，AMRM 只能在同一
candidate 上 commit/abstain。Target 与 masked-context 继承 P1-A2 unchanged DINOv2-S dense gate，不搜索阈值。
Prediction 与 contribution trace 写完后才允许 private evaluation。真实 physical viewpoint truth 不存在，固定报告
2D bearing-change proxy 并将 physical viewpoint 指标标为 `NOT_EVALUABLE`。

结果见 [`P1-AMRM0 matched Development canary`](P1_AMRM0_MATCHED_DEVELOPMENT_CANARY_RESULT_2026-08-22.md)：
AMRM0 precision `9.48% -> 10.65%`、coverage `96.87% -> 80.13%`，但 wrong-instance reacquisition
`12 -> 38`，发生 17 次 verified-bank poisoning，且 newly verified KF 对正确重捕获贡献为 0。当时唯一 successor 是
保留 outcome 的 [first-poison autopsy](P1_AMRM0_FIRST_POISON_AUTOPSY_2026-08-22.md) 已完成：17 次 admission
收敛为 9 个 first-poison episode，9/9 为 background-only single candidate，正确 candidate 全部 absent；其中 7 次
目标仍可见。主分叉是 proposal bottleneck，multi-candidate contrastive verifier 在本 cohort 中 `NOT_EVALUABLE`。
随后显式授权的 P1-PA0 已在独立 proposal-only contract 下执行，不修改 AMRM0。不得调 AMRM 阈值或启动
AMRM1/2/3、VLM、VIO/SLAM/geometry。

网络场景 3x5 与 action-responsive sanity 均已完成。P1-AMRM0 不自动进入 minimal geometry；只有真实 canary 把
主要剩余失败定位为 translation ambiguity 后，才允许另行考虑 VIO/triangulation/local parent frame。

Claim ceiling：`POST_OUTCOME_SELECTED_CONSUMED_DEVELOPMENT_FAILURE_COHORT_MECHANISM_DIAGNOSTIC_ONLY`。
默认 App：不变。
