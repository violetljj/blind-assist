# HFTF Stage C D39 THOR-MAGNI confirm-release veto event protocol

冻结时间：2026-08-03（Asia/Hong_Kong）

状态：

`D39_THOR_MAGNI_CONFIRM_RELEASE_VETO_EVENT_FROZEN_BEFORE_D39_SOURCE_REPLAY_OR_OUTCOME_JOIN`

## 1. post-D38 adaptive hypothesis

D38 证明固定 250 ms persistence 会改变 event terminals，但同时损失 6 个
positive events。D39 不搜索更短时长，而检验一个可解释的解除条件：

> contradiction latch 仍以 production evidence TTL 作为硬上限；若 scene
> measurement 明确反向越过同一个 deadband，产生
> `CONFIRM_APPROACH`，则立即解除 latch。`ABSTAIN` 不承担解除责任。

这把“继续抑制多久”从单纯 wall-clock tuning 改成具有物理方向含义的双向状态。
D39 是明确的：

`POST_D38_ADAPTIVE_OUTCOME_OPEN_DEVELOPMENT`

它不被解释为 independent validation。

## 2. frozen source and cohort

原样复用 D36-D38：

- 19 THOR-MAGNI sessions
- 530 proximity-eligible anchors
  - positive anchors：157
  - negative anchors：373
  - positive events：107
- 7-frame source windows
- D36 truth-free detections SHA-256：
  `5083db4c86ff20c01d12a47aa9b419d3cd1727b8f8ed1383020b33601ad6f731`
- D36 producer receipt SHA-256：
  `26f2991f4f836c2611224bc9ea1c9066d00b6bf03b600155736fafcbf8ab5ade`
- production risk、event、feedback planner、profile 与 scenario

D37/D38 runtime modes、producer source identity 与 replay artifact 必须保持
不变。

## 3. bidirectional scene source

新增独立 source identity：

`CAUSAL_SCENE_SCALE_BIDIRECTIONAL_R1`

它复用 production scene-scale producer 的全部 association、median-rate、
minimum matches、quality、gap 与 target binding，只改变 tri-state mapping：

- median rate `<= -0.05/s`：
  `CONTRADICT_APPROACH`
- median rate `>= +0.05/s`：
  `CONFIRM_APPROACH`
- `-0.05/s < rate < +0.05/s`：
  `ABSTAIN / SCENE_RATE_IN_DEADBAND`
- 少于两个 scene matches：
  `ABSTAIN / INSUFFICIENT_SCENE_MATCHES`

正负 threshold 严格对称，未进行 outcome search。confirm evidence 仍不能创建、
升级或提前触发 alert；它只允许解除当前 candidate latch。

## 4. independent runtime mode

新增：

`ACTIVE_CONTRADICT_TTL_CONFIRM_RELEASE`

状态转换：

1. admitted `CONTRADICT_APPROACH`
   - 当帧 veto feedback
   - latch 截止设为 `decisionAtNs + 250 ms`
2. admitted `CONFIRM_APPROACH`
   - 立即清空 latch
   - 不自行触发 feedback；该帧回到 baseline planner path
3. `ABSTAIN` / evidence absent
   - 不延长也不提前清除 latch
4. latch 到达 250 ms hard cap
   - 自动失效
5. session start/reset
   - 清空 latch

该 mode 只改变 feedback seam，不改变 raw/stable risk 或 event identity/state。

## 5. truth firewall

source-only Kotlin replay 先读取 frozen detector TSV，输出：

- baseline/candidate triggered frames/windows
- admitted contradict/confirm frames
- direct、latch-only suppressions
- confirm-release frames
- raw/stable risk parity
- source identity integrity

Kotlin 不读取 onset truth。只有 replay 原子落盘后，Python evaluator 才 join
frozen D12 truth。

truth join 前的 path、parser、Gradle、serialization 或实现错误可修复重跑，不烧毁
cohort，也不产生科学终态。

## 6. evaluability gates

必须全部通过：

1. complete cohort：530 anchors / 19 sessions
2. D36 input hash / receipt 完整匹配
3. D31 anchor detector parity 全通过
4. raw/stable risk mismatch = 0
5. admitted source 若存在，必须是 bidirectional source
6. baseline positive/negative opportunities 均 >=20 anchors
7. admitted contradiction opportunity >=10 anchors / >=5 sessions
8. admitted confirm opportunity >=10 anchors / >=5 sessions
9. confirm actually releases a live latch：
   - >=10 anchors
   - >=5 sessions
10. latch-only suppression opportunity >=10 anchors / >=5 sessions

任一失败：

`D39_THOR_MAGNI_CONFIRM_RELEASE_VETO_EVENT_NOT_EVALUABLE`

## 7. support gates

仅 evaluable 时解释：

1. positive event losses = 0
2. positive anchor losses <=1
3. positive anchor recall delta >= -1.0 pp
4. negative alert absolute reduction >=10
5. negative alert relative reduction >=20%
6. 至少 3/5 folds 的 negative alerts 减少
7. candidate-only triggered windows = 0

全部通过：

`D39_THOR_MAGNI_CONFIRM_RELEASE_VETO_EVENT_SUPPORTED_DEVELOPMENT_ONLY`

否则：

`D39_THOR_MAGNI_CONFIRM_RELEASE_VETO_EVENT_NOT_SUPPORTED`

## 8. interpretation and stop rule

即使 supported，也只说明“对称方向 evidence + explicit release”值得进入新鲜独立
event evidence；它不会覆盖 D35 device gate，不改变默认 App、主线、产品或安全
主张。

若 NOT_SUPPORTED，不搜索 asymmetric thresholds、更多 hold durations 或 confirm
count。应停止当前 scene-scale persistence family，转向新的独立 evidence 或不同
event semantics。
