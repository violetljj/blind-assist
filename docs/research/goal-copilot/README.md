# Goal-Driven Visual Copilot

状态：`current / PRODUCT_AND_RESEARCH_MAINLINE / BLINDASSIST_LAST_10M_REGROUNDING_V0 / ENGINEERING_READY / FIELD_3X5_REQUIRED / P1_CLOSED / NO_P1_W3 / NO_REFERENT_PERSISTENCE / NO_SCIENTIFIC_CONFIRMATION / DEFAULT_APP_UNCHANGED`

完整系统蓝图见 [`V2 路线图`](BLINDASSIST_GOAL_DRIVEN_VISUAL_COPILOT_V2_ROADMAP_2026-08-21.md)。本页是
Goal Copilot 动态执行状态真源；历史协议与数字只通过链接保留，不再授予执行权限。

## 当前工程里程碑

当前唯一执行面是 `BLINDASSIST_LAST_10M_REGROUNDING_V0`：复用且不修改现有 P0 named-building entrance
grounding/provider，完成“入口寻找—引导—重新观测—确认”的当前帧机械闭环。只支持清晰、相对唯一的建筑入口。

最小状态机：

```text
SCAN
-> CURRENT_CANDIDATE
-> ALIGN
-> ADVANCE_AND_REOBSERVE
-> ARRIVAL_CONFIRM
-> COMPLETE / RESCAN / ABSTAIN
```

每次转向、前进或重扫后必须提交新的 frame 并重新调用 P0 grounding。控制 state 不保存 candidate id、bbox、
图像、特征、score、handoff 或 identity，也不比较相邻帧；P0 persistence handoff 只校验当前帧绑定后丢弃。
无唯一可靠 candidate 时固定输出“没有可靠找到入口，请停下并缓慢重新扫描。”；连续三次无法确认后进入
`ABSTAIN`；即使持续有候选，12 条指令仍未完成也必须停止。两种停止都提供现场工作人员或可信任真人协助出口。
任何指令都不得输出“前方安全”。

到达不是历史 identity 延续：当前帧出现居中、近距机械 cue 后先停下，再用一个新的当前帧重新 grounding；
只有新的输出仍独立满足当前帧条件才能 `COMPLETE`。该 bbox cue 只是机械任务规则，不是距离或安全模型。

稳定实现与现场命令见 [`last_10m_regrounding_v0`](../../../scripts/research/goal_copilot_bridge/last_10m_regrounding_v0/README.md)。
当前实现专项 tests 已覆盖 fresh-frame、跨帧/stale fail-close、无 candidate 连续 abstain、二次到达确认、错误确认
优先归因和 3x5 汇总形状。Android/default App 不变。

## 现场执行与报告边界

里程碑必须在 3 个真实地点各执行 5 次，共 15 episodes。Prerecorded RGB 可用于调试 provider payload，但不能
冒充方向指令改变用户动作的真实机械任务。真实现场尚未完成时只允许
`ENGINEERING_READY / FIELD_EXECUTION_INCOMPLETE`，不得填造完成率或错误确认数。

每次 observation、candidate、direction、rescan、abstention、completion 和现场错误确认进入 append-only JSONL/
episode summary。最终报告首先单列错误入口确认数，再报告任务完成率、完成时间、首次发现时间、指令数和重扫数。
每个已 adjudicate episode 只允许以下三个归因之一：

1. `CURRENT_FRAME_GROUNDING_BOTTLENECK`
2. `INTERACTION_OR_CONTROL_BOTTLENECK`
3. `REGROUNDING_LOOP_MECHANICALLY_USEFUL`

错误入口确认无条件计入第一类。只有恰好 3 locations x 5 adjudicated episodes 才能标记
`FIELD_EXECUTION_COMPLETE`。本里程碑是机械工程结果，不是 scientific confirmation、用户安全、导航有效性或
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

## P1 正式关闭

P1 persistence、tracker/correspondence、keyframe/world-anchor 路线全部关闭，不再承担当前主线：

- [`P1-A1`](P1_A1_CONSERVATIVE_LOCAL_TRACK_VALIDITY_RESULT_2026-08-21.md) 到
  [`P1-A4`](P1_A4_ONLINE_STRONG_TEMPORAL_CORRESPONDENCE_RESULT_2026-08-22.md) 只保留 consumed Development
  failure evidence；最终 `STRONG_TEMPORAL_CORRESPONDENCE_NOT_SUFFICIENT`。
- [`P1-W1 Stage A`](P1_W1_STAGE_A_SINGLE_EXECUTION_RESULT_2026-08-22.md) 为
  `W1_T0_NOT_EVALUABLE_DATA_OR_INTERFACE`，没有 Stage B authority。
- [`P1-W2 single execution`](P1_W2_SINGLE_EXECUTION_RESULT_2026-08-22.md) 为
  `P1_W2_RGB_REFERENT_INTERFACE_NOT_SUPPORTED`，不允许在 sealed cohort 上降 gate、换模型、改 crop/context 或重跑。

明确不建立 `P1-W3`，不自动重开 referent persistence，也不从旧 successor、W0 design 或历史 handoff 恢复权限。
P1 历史 JSON schemas、evaluator 和结果只用于追溯；当前闭环不得导入其 tracker、SAM propagation、DINO identity、
LoFTR、keyframe memory、SLAM、VIO 或 world-relative state。

## 唯一 successor

`FIELD_3X5_MECHANICAL_EXECUTION` 只是本里程碑内完成 3x5 真实机械 episodes 并生成上述限定报告，不是新研究
协议。现场执行完成或因现实条件无法执行而
明确停止后，本路线不自动创建任何后继协议、P1-W3、模型 arm、数据 cohort、Android 接入或 scientific
confirmation。新的研究问题必须由用户另行明确启动。

Claim ceiling：`REAL_SITE_MECHANICAL_TASK_ONLY_NO_SCIENTIFIC_CONFIRMATION_NO_SAFETY_CLAIM`。
默认 App：不变。
