# Goal-Driven Visual Copilot

状态：`current / PRODUCT_AND_RESEARCH_MAINLINE / DESTINATION_GOAL_GROUNDING_P0 / PRIOR_ART_ASSIMILATED / P0_CONTRACT_V1_REFERENCE_SET_ADDENDUM / GROUNDING_DINO_PROPOSAL_PATH_RUN / SILVER_B_DEVELOPMENT_COHORT_47_GOALS_43_FRAMES / SINGLE_BRAIN_MECHANICS_RUN / NO_SCIENTIFIC_VERDICT / R5_PERMANENTLY_CLOSED / NO_R6_R7 / NO_SKY / DEFAULT_APP_UNCHANGED`

完整系统蓝图见 [`V2 路线图`](BLINDASSIST_GOAL_DRIVEN_VISUAL_COPILOT_V2_ROADMAP_2026-08-21.md)；P0 冻结合同与机器合同见 [`Protocol V1`](P0_GROUNDING_PROTOCOL_V1.md) / [`JSON`](p0_grounding_protocol_v1.json)。
本页仍是动态执行状态真源；当前已完成 schema/evaluator mechanics、真实 map+geometry materialization、
47-goal/43-frame Silver-B Development cohort，以及一个 `gpt-5.6-terra / medium` 单 Brain mechanics run；
没有正式 baseline、科学性能、导航或用户效果结果。

[`Prior-Art Assimilation`](P0_PRIOR_ART_ASSIMILATION_2026-08-21.md) 已将路线对齐到 BLV last-few-meters、
embodied referring expression、interactive VLN 与 POI-goal navigation。BridgeNav 和 ABot-N1/POIBench 已覆盖
入口级 POI navigation 的关键部分；BlindAssist 不主张该任务首创，潜在缺口收窄为未预扫描开放街景中的
BLV set-valued referential grounding、证据化 abstention 与后续持续引导。Silver-B 原样继续。

## 当前主线：Destination / Goal Grounding

BlindAssist 的当前唯一 Goal Copilot successor 是 `BA_DESTINATION_GOAL_GROUNDING_R0`。它回答的不是
“画面里有没有门”，而是“哪个视觉实体真正满足用户目标”：先确认目标建筑，再找属于该建筑的入口，
最后才把锁定目标交给 persistence、approach 与 completion。首个 P0 不造模块群，只验证一个直接任务：
给定“找到目标建筑的入口”，在含多个建筑、多个门的真实图像或视频中，将目标建筑所属入口排在其他门
之前，并输出支持该排序的可审计 evidence。

允许组合的 evidence 包括 OCR 招牌、logo、building/facade semantics、entrance detection、VLM relational
reasoning 与 POI/map coarse prior；核心关系是 `entrance_of(target_building)`。首个 P0 只判定目标建筑与
入口归属排序，不提前把 `public / accessible entrance`、导航完成或安全效果混入成功主张。

本页同时是执行约束：`BA_DESTINATION_GOAL_GROUNDING_R0` 是唯一 active 产品与研究主线。D-ORACLE、SVRF、
Assistive Geometry、TARO、SATOM、DepthART、旧 GC/Sky search 与 Android/default-App promotion 只保留为
历史、关闭或暂停的支撑上下文；除非用户显式改变主线并先同步本页与算法研究入口，否则不得自行恢复、
占用执行预算或产生竞争 successor。

系统侧输入只允许真实 RGB；ADT bbox、object/device trajectory、depth、segmentation 与 visibility GT
只允许进入隔离的 mining/evaluator。ADT 是录好轨迹，因此本路线最多证明真实 RGB 能否恢复 target
visibility、bearing、tracking、reacquisition、relative nearness 与 approach evidence，以及这些 evidence
能否支撑合理的 prerecorded guidance timeline；它不能证明引导改变了用户动作或完成 closed-loop
navigation。

已完成的 BA-ADT real-evidence 工作现在是 Target Persistence 的基础与 R5 终止证据，不再是 active
tiny-object successor。ADT-0 的稳定实现入口见
[`scripts/research/ba_adt_real_evidence/README.md`](../../../scripts/research/ba_adt_real_evidence/README.md)。
当前 ADT-0、ADT-1 Development evaluation 与首个 ADT-2 prerecorded demo 已运行；Sky、GC2-C、
held-out、Android/default-App 接线、产品和安全主张均关闭。

ADT-0 sample 结果见 [`sample result`](BA_ADT_REAL_EVIDENCE_ADT0_SAMPLE_RESULT.md)；完整 sequence 与首个
RGB-only canary 见 [`selection/canary result`](BA_ADT_REAL_EVIDENCE_ADT0_SELECTION_ADT1_CANARY_RESULT.md)。
固定 miner 选择 `seq136 / Carrot_A`；修正 90° clockwise 坐标变换后，detector-only recall 为 0.4041，
5-frame sparse flow 提至 recall/mean IoU `0.5808/0.4469`，false-visible 0.0073，但最长 dropout 仍 162、
@30 重捕获 0.4。ADT-2 demo 已接冻结 GC1 winner，且保持 clearance/completion fail closed。

实例重检测 R1 结果见
[`BA_ADT_INSTANCE_REDETECTION_1_RESULT_2026-08-21.md`](BA_ADT_INSTANCE_REDETECTION_1_RESULT_2026-08-21.md)。
多模板 RGB appearance memory、弱时空先验和 2-of-3 保守确认把 recall/mean IoU 提至
`0.6203/0.4743`，13 次 instance-redetection 为 `13 correct / 0 wrong / 0 unresolved`，false-visible
保持 `0.0073`；但 @30/@90/@180 reacquisition 仍为 `0.4/0.5/0.5`，最长 dropout 只降到 159 帧。
因此仅建立 bounded Development utility，长时重捕获主瓶颈未解决。

后置 failure accounting 将 5 个失败 opportunity 全部分为 `NO_CANDIDATE`，R1 在 GT-visible LOST
search 中的 candidate recall 仅 `34/405 = 0.0840`。随后只替换 LOST proposal generator 的
YOLOE-26n visual-prompt canary 仍是 `5/0/0` failure A/B/C，candidate recall 降至 `29/423 = 0.0686`，
@30 reacquisition 降至 `0.2`、最长 dropout 增至 164，wrong-instance 仍为 0。结果见
[`BA_ADT_YOLOE_VISUAL_PROMPT_CANARY_RESULT_2026-08-21.md`](BA_ADT_YOLOE_VISUAL_PROMPT_CANARY_RESULT_2026-08-21.md)。
因此该单 visual-prompt candidate route 不获支持，DINOv2 不进入。

[`R3 observability audit`](BA_ADT_REAPPEARANCE_OBSERVABILITY_R3_RESULT_2026-08-21.md) 的独立 GT-proposal
oracle 为 `4/5`，五窗分为 2 个不可见/重遮挡和 3 个太小；640 输入上最短边最大值均低于 10 px。
[`R4 search-scale result`](BA_ADT_SMALL_TARGET_SEARCH_SCALE_R4_RESULT_2026-08-21.md) 排除两个
insufficient-evidence 窗口后，S0/S1-1280/S2-2x2 在 97 个 eligible LOST-search frames 上均为 `0/97`、
重捕获 `0/3`。S1/S2 全局 recall 虽升至 `0.6720/0.6852`，却产生 1/4 次 wrong-instance，
false-visible 升至 `0.0128/0.0182`；已测试 scale arms 不获支持，也不进入部署。

## 上位产品定义

BlindAssist is a goal-driven visual copilot for visually impaired users. Instead of only
reporting objects or hazards, the system maintains a user goal across time, searches for
relevant visual evidence, tracks progress, recovers from target loss, guides the user
toward completion, and verifies that the requested task has actually been completed.

BlindAssist 是一个目标驱动的视觉副驾。用户提出“找入口、找门、找座位、找到某个物品”等目标后，
系统持续观察环境、维护任务状态、追踪目标、判断进展、在目标丢失后重新搜索，并持续引导直到任务完成
或安全地声明无法继续。

这一定义不废弃现有模块。object detection、depth、optical flow/motion、OCR、VLM、tracking、
traversability、risk estimation 和 semantic perception 统一作为 `Evidence Providers`。`Goal Copilot
Brain` 拥有 task belief、temporal memory、target tracking state、progress estimation、action
proposal/selection、recovery、termination 与 goal-completion verification。

## GOAL-COPILOT-1

全名：`BlindAssist Goal-Copilot-1: Completion-Capable Closed-Loop Policy Synthesis`。

研究问题限定为：在冻结的 symbolic/oracle-style perception evidence 已给定时，外部搜索能否提出具有
完整 goal-completion chain 的 closed-loop policy，而不仅是局部动作 patch。V0 只完成零模型 mock
roundtrip；未启动 Sky 模型搜索、EvoX、多臂实验、真实 perception、摄像头或训练。

三个 task family：

- `FIND_AND_REACH`：search → acquisition → alignment → approach → completion；
- `TRACK_AND_REACQUIRE`：track → loss → recovery → reacquisition → continued progress；
- `FIND_ALIGN_INTERACT`：search → acquisition → fine alignment → approach → interaction readiness。

typed candidate observation 见
[`task_api.py`](../../../scripts/research/goal_copilot_bridge/task_api.py)。它只暴露 target bearing/scale/
confidence、可通行方向、relative nearness/approach rate、tracking/observation quality 和 interaction
readiness；不暴露 scenario graph、正确动作、completion truth 或 hidden labels。

## Authority 与 hard gates

BlindAssist 独占 task definition、scenario truth、evaluator、safety contract、score vector、validation
gate 和 `ACCEPT / REJECT / NOT_EVALUABLE` authority。SkyDiscover 只有 candidate proposal/search
authority；其 score 永远只是 provenance。

evaluator 输出 `goal_completion`、`normalized_progress`、`reacquisition_success`、
`tracking_continuity`、`wrong_way_actions`、`unsafe_guidance`、`premature_completion`、
`recovery_steps`、`total_actions`、`timeout`、`semantic_validity`。`unsafe_guidance > 0`、
`premature_completion > 0` 或 semantic invalid 立即硬拒绝；未完成目标时，secondary metrics 只能描述
partial progress，不能改写成 completion。V0 不用单一 scalar 产生科学 verdict。

## Bridge V0

稳定入口是 `python scripts/run_research_tool.py goal-copilot sky_bridge.py`。导出的 public
`SearchTaskBundle` 只有 typed API、初始 policy、protocol、public scenario descriptions、README、
manifest 与 checksums；BA evaluator 和 `sealed_scenarios.json` 永不导出。CandidateBundle 只允许
`candidate/policy.py` 作为候选源码，并用精确成员 allowlist、受限 AST、protocol/source digest 和逐文件
SHA-256 fail closed。完整外部合同见
[`SKYDISCOVER_INTEGRATION_CONTRACT.md`](SKYDISCOVER_INTEGRATION_CONTRACT.md)。

## 与 L10M 的边界

Existing L10M work is precursor evidence for goal-conditioned temporal control and search
behavior. GOAL-COPILOT-1 is a new protocol lineage. No previous L10M result is
retroactively reclassified, recomputed, or claimed as GOAL-COPILOT-1 evidence.

L10M 中的 progress memory、tracking、recovery、safety contract、termination 与 balanced
exploration 可以解释为 Goal Copilot Brain 的先导机制，但其封存 protocol、terminal、receipt、run root、
claim ceiling 与 evidence role 均保持原样。

## 当前结论

- 当前结论：`GOAL_COPILOT_1_SKY_SEARCH_SIGNAL_ESTABLISHED_ON_SEALED_PILOT`；
- 证据范围：small deterministic symbolic closed-loop Pilot；
- 默认 App、真实用户、安全效果和产品可用性：无新权限；

## 唯一 successor

[`ADT1_SMALL_TARGET_VISUAL_UPPER_BOUND_R5`](BA_ADT_SMALL_TARGET_VISUAL_UPPER_BOUND_R5_RESULT_2026-08-21.md)
已终止封存。DINOv-SwinL Teacher A 在冻结的 3-window/97-frame cohort 上为 `1/3 windows`、`1/97`
correct candidate，唯一命中是 W4 的 20 px 目标，且正确分数低于同帧最强错误候选。条件 Teacher B 在
固定帧 inference 前被官方接口审计判为 `SAM31_IMAGE_CROSS_IMAGE_PROMPT_NOT_EVALUABLE`：SAM 3.1
checkpoint 只接 multiplex video predictor，公开 image visual box 只作用于当前同一图，没有历史
exemplar 到独立 target image 的受支持路径。该项不是 `0/3`，R5 总结论为 inconclusive；不追加其他
teacher、R6/R7 rescue、Sky、held-out 或 default-App。

当前唯一 successor 是 `BA_DESTINATION_GOAL_GROUNDING_R0`。V1 已冻结 goal、episode/output schema、
Provider/Brain/End-to-End/P1-handoff evaluator mechanics，并用 12 个 stdlib-only unit tests 覆盖规定的
mock 归因路径。后续 [`P0-S0 canary`](P0_S0_SILVER_MATERIALIZATION_CANARY_RESULT_2026-08-21.md) 因
`MAPILLARY_ACCESS_TOKEN_MISSING / MANDATORY_CANDIDATE_GENERATOR_NOT_AUTHORIZED` 关闭为
`P0_S0_SOURCE_OR_LICENSE_BLOCKED`：0 episode、0 `SILVER_A_PRIMARY`、无科学 verdict。随后 [`P0-S0-V0`](P0_S0_VISUAL_CANDIDATE_GENERATOR_ADMISSION_RESULT_2026-08-21.md)
将固定上游 YOLO checkpoint 判为 `P0_S0_VISUAL_CANDIDATE_GENERATOR_NOT_ADMITTED`：训练 provenance、replay envelope、逐候选 lineage 不足，且上游全类别放行并丢失 provenance。
Mapillary token 现已通过只读 Graph API 鉴权。随后固定 Grounding DINO Tiny 仅作 proposal generator，在 20 张
anchor-facing 真实图上产生 177 个 bbox；自动链 nominal 产出 1 个 `SILVER_A_PRIMARY`，但结果后的可视核对显示
关键跨 sequence proposal 与其余 proposals 属于不同实体入口，暴露现有 ray-wall multiview 缺少 same-physical-region
correspondence。nominal Silver 不接受，当前仍无 cohort/baseline/科学 verdict；详见
[`S0-R1 result`](P0_S0_GROUNDING_DINO_R1_RESULT_2026-08-21.md)。训练数据 provenance 不完整仅作为 proposal-only
limitation，不再是 bbox generation 硬门；模型仍无任何 truth authority。下一步只修复 crossview identity mechanics，
不比较 detector、不换模型或调视觉阈值。map/cluster/fixture 不得升格，baseline 仍未授权。未来最小输入是用户目标与真实多建筑/多门 RGB 图像或视频；最小输出是 entrance candidates、goal-conditioned ranking 与逐候选 evidence；最小成功条件
是目标建筑所属入口排在其他建筑或无关实体的门之前。TargetMemory、flow tracking、conservative
redetection 与 failure evaluator 保留为找到目标后的 persistence 基础；VIO/SLAM/world memory 只保留为
未来 spatial-memory 增强，不为 carrot 单独开启 rescue 长线。Sky 仅在某个真实模块出现明确、可测的
headroom 后才可另立任务，不能驱动当前研究方向。

[`P0-S1 Crossview Entrance Identity`](P0_S1_CROSSVIEW_ENTRANCE_IDENTITY_RESULT_2026-08-21.md) 已在上述 consumed
20-image canary 上完成：4 个 wall-associated candidates 的 3 个同 sequence pair 外观高度相容，但 baseline 只有
`4–12 mm`、ray angle `0.08–0.16°`；另外 3 个合格视差 pair 全部跨 sequence、相隔约 4.9 年且外观不一致，只能弱支持。
终态 `P0_S1_IDENTITY_RULE_TOO_WEAK`、strong identity `0`，所以 S0-R1 不重跑。当前唯一 blocker 是同 source 内
target-anchor-facing、同 sequence、具有 `3–30 m` 真实 parallax 且逐 view 独立过 map/geometry 的 pair coverage；不得
降低 identity gates 或升级 cross-sequence evidence 来救当前 record。

从 [`P0 Silver-B Development Addendum V1`](P0_SILVER_B_DEVELOPMENT_ADDENDUM_V1.md) 起，P0 Development
不再以 Silver-A coverage 为前置条件。已通过既有 provenance/license/map/geometry/conflict gates 的 parent A/B
记录可向下导出为 `SILVER_B_MAP_GEOMETRY`。初始 4 个全 `AMBIGUOUS` episodes 已扩展为 47 个 goal
episodes / 43 个 unique frames：`UNIQUE=12 / SET_VALUED=4 / AMBIGUOUS=31`。它们只支持 pipeline、
conditioned yield、弱排序原型与 failure/abstention 开发；由于 detector
参与样本生成且 exact entrance identity 未建立，不能报告 detector recall/precision、exact Brain/E2E accuracy，
也不能回写 P0-S1 终态。P0 goal-reference truth 现显式支持 `UNIQUE / SET_VALUED / AMBIGUOUS`；set-valued
命中任一合法物理目标均正确，ambiguous 必须允许 `AMBIGUOUS` 或 fail-closed abstention。单 Brain run 的
主要结构信号是：4/4 SET 返回多个 candidates；31 个 AMBIGUOUS 中 23 个被 branding/signage/facade
association 推动为 grounding，且 0 个主动返回 `AMBIGUOUS`；正确 proposal 可用的 14 个 resolvable goal
中 top-1 命中 13 个。所有数字只属于 Silver-B conditioned
  mechanics，详见 [`Development result`](P0_SILVER_B_BRAIN_DEVELOPMENT_RESULT_2026-08-21.md)。随后两个
  consumed-development calibration canary 将 place identity 与 exact entrance relation 分开：V1 把 unsupported
  commit 从 `23/31` 降到 `12/31`，但只保留 `4/13` 原正确 grounding；V2 显式输出两级 support，将 unsupported
  commit 降到 `6/31`，却只保留 `2/13`，并拒绝 14/16 resolvable episodes。两者均属 over-refusal、不得准入，
  详见 [`P0-D1 result`](P0_D1_AMBIGUITY_CALIBRATION_CONSUMED_CANARY_RESULT_2026-08-21.md)。下一步先补 25–40
  个新 venue-parent-disjoint signage/conflict/UNIQUE Development episodes；当前缺少 Mapillary token，未启动
  parent-disjoint confirmation、Sky、detector 调参或模型比较。

`GOAL-COPILOT-1-SKY-PILOT` 已按独立冻结协议完成并封存；协议见
[`GOAL_COPILOT_1_SKY_PILOT_PROTOCOL.md`](GOAL_COPILOT_1_SKY_PILOT_PROTOCOL.md)，结果与严格 claim
ceiling 见 [`GOAL_COPILOT_1_SKY_PILOT_RESULT.md`](GOAL_COPILOT_1_SKY_PILOT_RESULT.md)。若继续，
应另立 `GOAL-COPILOT-2 / NOISY-OBSERVATION ROBUSTNESS` 协议，而不是恢复本 Pilot 或复用其 fresh。

GOAL-COPILOT-1 现已永久关闭，不再授权任何 GC1 模型搜索。下一条已冻结路线是
[`GOAL_COPILOT_2A_PROTOCOL.md`](GOAL_COPILOT_2A_PROTOCOL.md)：在不调用模型、不接触 GC1
fresh cohort 的前提下，对冻结 GC1 baseline 与 winner 做 perception-uncertainty
characterization。GC2-A 只使用已消费 dev 场景语义，因此其结果不得称为 fresh evidence。

GC2-A 已完成并确定性 replay 通过；结果见
[`GOAL_COPILOT_2A_RESULT.md`](GOAL_COPILOT_2A_RESULT.md)。冻结 winner 在 primary
`COMBINED_MODERATE` 下 completion `0/12`、eligible reacquisition `0/3`，并有 `1` 次
premature completion，因此只准入 GC2-B noise-robust Sky search 的独立协议设计。GC2-B
模型调用、正式 Sky/EvoX 搜索和多臂实验仍未授权。

GC2-B 的双仓协议设计已冻结，见
[`GOAL_COPILOT_2B_PROTOCOL.md`](GOAL_COPILOT_2B_PROTOCOL.md)。冻结设计规定 BA 独占
task/noise/evaluator/hidden/acceptance authority，Sky 仅有 proposal/search authority；其当前
claim ceiling 仅为 protocol design。正式 bundle、held-out envelope、provider/run seal 与模型调用
仍需后续单独授权。

GC2-B 随后已按独立 formal seal 完整执行并关闭；见
[`GOAL_COPILOT_2B_RESULT.md`](GOAL_COPILOT_2B_RESULT.md)。32/32 generation calls 完成，
但 BA 锁定的公开 dev winner 在 `COMBINED_MODERATE` 仍为 `0/12`，未达到 held-out
admission，因此 encrypted held-out 未进入 winner-lock 后的正式开启，也从未用于候选评估。当前结论是
`GC2B_NOISE_ROBUST_SEARCH_SIGNAL_NOT_ESTABLISHED`，不授权 rescue rerun、扩预算或同一候选面的
GC2-C。

随后完成的零模型
[`GOAL-COPILOT-2 observability and reality audit`](GOAL_COPILOT_2_OBSERVABILITY_AUDIT_RESULT.md)
只使用 consumed scenarios、冻结 simulator、candidate traces 和既有 device evidence。12 个 moderate
episode 的首次偏离分散在 stale evidence、tracking collapse、dropout 与方向/动作错误；逐项关闭任一
corruption 最多只恢复到 `1/12`。Hidden oracle、完整历史 lookup 和六函数 surface memorization 均可在
这 12 条 consumed 轨迹达到 `12/12`，但 lookup 明确是 simulator leakage diagnostic，不是可迁移
policy evidence。现有 Android trace 是公开/已消费真实世界 RGB 的 device replay，不是 real-phone
capture，且缺少 target identity、tracking、bearing、nearness 与时间映射，所以真实手机噪声校准为
`NOT_EVALUABLE`。

该审计的历史决策仍保持：停止 synthetic moderate optimization 并保持 policy search 关闭。新的
`BA-ADT-REAL-EVIDENCE` 不是 GC2 rescue；它使用独立 ADT 真实 RGB/GT lineage，先执行 ADT-0
数据适配性与 episode mining。GC2-C、held-out opening、新模型/Sky 调用、扩预算和 consumed
representation ladder 均继续禁止。
