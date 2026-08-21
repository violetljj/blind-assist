# Goal-Driven Visual Copilot

状态：`current / PRODUCT_AND_RESEARCH_MAINLINE / P1_AMRM0_ADAPTIVE_MULTI_VIEW_REFERENT_MEMORY / MATCHED_CANARY_TERMINAL=P1_AMRM0_MEMORY_POISONING_FAIL / FAILURE_AUTOPSY_ONLY / DEFAULT_APP_UNCHANGED`

完整系统蓝图见 [`V2 路线图`](BLINDASSIST_GOAL_DRIVEN_VISUAL_COPILOT_V2_ROADMAP_2026-08-21.md)。本页是
Goal Copilot 动态执行状态真源；历史协议与数字只通过链接保留，不再授予执行权限。

## 当前研究实现

用户已明确将 `P1-AMRM0 Adaptive Multi-view Referent Memory` 定为新的主实验路径。准确状态是“当前最值得优先
验证的新研究假设”，不是已经证明有效。核心假设为：在第一视角最后十米任务中，相比持续维护 2D correspondence，
积累经过身份验证的多距离、多视角 referent memory，是否能提高真实同实例重捕获，同时降低 wrong-instance
reacquisition。

当前唯一 active surface 是
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

P1-AMRM0 与已消费的 W1-T0/W2 实质不同，不修改、续跑或覆盖旧 cohort，也不继承旧 execution authority。当前未选择
RGB provider、数据 roster 或 performance experiment；VIO、SLAM、metric 3D、POMDP、主动平移、自动到达距离、
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

## 唯一 successor

`P1_AMRM0_DATA_ADAPTER_AND_MATCHED_DEVELOPMENT_CANARY` 已执行并以
`P1_AMRM0_MEMORY_POISONING_FAIL` 终止。它固定复用 consumed P1-D0
15 episodes 与 P1-A4 的 exact candidate bbox stream；A4 sealed output 是 correspondence baseline，AMRM 只能在同一
candidate 上 commit/abstain。Target 与 masked-context 继承 P1-A2 unchanged DINOv2-S dense gate，不搜索阈值。
Prediction 与 contribution trace 写完后才允许 private evaluation。真实 physical viewpoint truth 不存在，固定报告
2D bearing-change proxy 并将 physical viewpoint 指标标为 `NOT_EVALUABLE`。

结果见 [`P1-AMRM0 matched Development canary`](P1_AMRM0_MATCHED_DEVELOPMENT_CANARY_RESULT_2026-08-22.md)：
AMRM0 precision `9.48% -> 10.65%`、coverage `96.87% -> 80.13%`，但 wrong-instance reacquisition
`12 -> 38`，发生 17 次 verified-bank poisoning，且 newly verified KF 对正确重捕获贡献为 0。唯一 successor 是
保留 outcome 的 poisoning failure autopsy；不得调阈值、增加候选或启动 AMRM1/2/3、VLM、VIO/SLAM/geometry。

网络场景 3x5 与 action-responsive sanity 均已完成。P1-AMRM0 不自动进入 minimal geometry；只有真实 canary 把
主要剩余失败定位为 translation ambiguity 后，才允许另行考虑 VIO/triangulation/local parent frame。

Claim ceiling：`HIGHEST_PRIORITY_HYPOTHESIS / SYNTHETIC_AMRM_MECHANICS_ONLY / NO_REAL_DATA_UTILITY_OR_SCIENTIFIC_CONFIRMATION`。
默认 App：不变。
