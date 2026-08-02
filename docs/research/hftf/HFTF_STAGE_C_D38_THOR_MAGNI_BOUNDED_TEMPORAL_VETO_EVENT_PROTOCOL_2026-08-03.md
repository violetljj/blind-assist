# HFTF Stage C D38 THOR-MAGNI bounded temporal veto event protocol

冻结时间：2026-08-03（Asia/Hong_Kong）

状态：

`D38_THOR_MAGNI_BOUNDED_TEMPORAL_VETO_EVENT_FROZEN_BEFORE_D38_KERNEL_REPLAY_OR_OUTCOME_JOIN`

## 1. adaptive Development hypothesis

D37 已在 outcome-open Development cohort 上观察到：

- scene-scale contradiction coverage 足够：351 anchors / 19 sessions
- 508 次逐帧 feedback suppression
- 但 negative triggered windows 仅减少 1

因此 D38 是明确的 post-D37 adaptive hypothesis，不伪装成独立验证：

> causal contradiction 若只抑制当帧，会被同一 event/window 内稍后的 trigger
> 覆盖；复用 evidence 自身 TTL 的 bounded persistence，是否能把已经成立的
> frame-level veto 转化为 event-level utility？

D38 不改 detector、scene producer、threshold、association 或 risk model；唯一
新变量是 feedback veto 的时间作用域。

## 2. frozen data and source

原样复用 D36/D37：

- 19 THOR-MAGNI sessions
- 530 proximity-eligible anchors
  - 157 positive anchors
  - 373 negative anchors
  - 107 positive events
- 7-frame source windows
- D36 truth-free `detections.tsv`
  - SHA-256：
    `5083db4c86ff20c01d12a47aa9b419d3cd1727b8f8ed1383020b33601ad6f731`
- D36 producer receipt
  - SHA-256：
    `26f2991f4f836c2611224bc9ea1c9066d00b6bf03b600155736fafcbf8ab5ade`

数据角色：

`POST_D37_ADAPTIVE_OUTCOME_OPEN_DEVELOPMENT`

任何 supported 结果都只用于选择后续独立实验候选。

## 3. single system variable

新增独立 runtime mode：

`ACTIVE_CONTRADICT_TTL`

它保持 D37 `ACTIVE_CONTRADICT_ONLY` 语义完全不变。candidate 仍由 production
`CausalSceneScaleTristateGeometryProducer` 产生 evidence；当且仅当当前 evidence
被 admission 为 `CONTRADICT_APPROACH` 时：

1. 当帧 feedback 被 veto；
2. feedback-only veto latch 激活至
   `decisionAtNs + 250,000,000 ns`；
3. 在 latch 有效期内，后续帧 feedback 继续被 veto；
4. 后续 admitted contradiction 可以把截止时间延长到新的
   `decisionAtNs + 250 ms`；
5. abstain/evidence absent 不主动延长 latch；
6. session start/reset 清空 latch。

`250 ms` 不是 outcome 搜索结果；它原样复用 scene-scale evidence 的 production
TTL。不得在 D38 上搜索 100/250/500 ms 或其他 hold duration。

latch 只能改变 feedback decision：

- 不改变 raw/stable risk
- 不改变 event identity/state
- 不创建 alert
- 不把 confirm/abstain 转成风险
- 默认 App 与既有 D37 mode 均不启用此新 mode

## 4. paired production replay and truth firewall

- baseline：全新 production `AssistDecisionKernel` + `OFF`
- candidate：全新 production `AssistDecisionKernel` +
  `ACTIVE_CONTRADICT_TTL`
- source-only Kotlin replay 先原子写出：
  - baseline/candidate triggers
  - direct contradiction suppressions
  - latch-carried suppressions
  - admitted source identity
  - raw/stable risk parity
- Kotlin replay 不读取 onset truth
- Python evaluator 在 replay 完成后才按 frozen identity join D12 truth

parser、Gradle、path、serialization 或实现错误在 truth join 前可修复重跑，不烧毁
cohort。

## 5. evaluability gates

必须全部通过：

1. complete cohort：530 anchors / 19 sessions
2. D36 input hash 与 receipt 完整匹配
3. D31 anchor detector parity 全通过
4. raw/stable risk mismatch = 0
5. admitted non-scene source observations = 0
6. baseline positive/negative alerted anchors 均 >=20
7. admitted contradiction opportunity >=10 anchors / >=5 sessions
8. latch-only suppression opportunity：
   - >=10 anchors
   - >=5 sessions

任一失败：

`D38_THOR_MAGNI_BOUNDED_TEMPORAL_VETO_EVENT_NOT_EVALUABLE`

## 6. support gates

仅 evaluable 时解释：

1. positive event losses = 0
2. positive anchor losses <=1
3. positive anchor recall delta >= -1.0 pp
4. negative alert absolute reduction >=10
5. negative alert relative reduction >=20%
6. 至少 3/5 folds 的 negative alerts 减少
7. candidate-only triggered windows = 0

全部通过：

`D38_THOR_MAGNI_BOUNDED_TEMPORAL_VETO_EVENT_SUPPORTED_DEVELOPMENT_ONLY`

否则：

`D38_THOR_MAGNI_BOUNDED_TEMPORAL_VETO_EVENT_NOT_SUPPORTED`

## 7. claim ceiling

即使 supported，也只说明 bounded temporal feedback seam 是值得带入新独立
outcome evidence 的候选系统变量。它不覆盖 D35 device gate，不改变默认 App，
不建立 independent generalization、产品或安全主张，也不直接替换主线。

若 NOT_SUPPORTED，不在同一 cohort 上调 hold duration；应停止这一 persistence
实例，重新定位 event seam 或取得新的科学变量。
