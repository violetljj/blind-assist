# HFTF Stage C D37 THOR-MAGNI production scene-scale veto event protocol

冻结时间：2026-08-03（Asia/Hong_Kong）

状态：

`D37_THOR_MAGNI_PRODUCTION_SCENE_SCALE_VETO_EVENT_FROZEN_BEFORE_D37_KERNEL_REPLAY_OR_OUTCOME_JOIN`

## 1. 科学问题

D36 已证明 production selected-target track contradiction 在当前真实事件 cohort
只有 `2 anchors / 2 sessions` 的 admitted opportunity，因此不能评价 event veto
utility。D37 不修改 track threshold，也不继续优化 D36；它只回答：

> production `CausalSceneScaleTristateGeometryProducer` 的 collective-receding
> contradiction，在完全相同的 detector frames、production kernel 与真实事件
> cohort 上，是否拥有足够机会，并能在不损失 positive events 的前提下降低
> negative alerts？

这是 D36 后预先指定的单变量替换，不是对 D37 outcome 的事后 rescue。

## 2. 数据角色与不可变输入

数据角色：

`OUTCOME_OPEN_DEVELOPMENT_PAIRED_EVENT_REPLAY`

它可以回答当前 THOR-MAGNI cohort 上的 event utility，不建立 independent
generalization、默认 App、产品或安全主张。

冻结复用 D36：

- 19 个 THOR-MAGNI source sessions
- 530 个 proximity-eligible anchors
  - 157 positive onset anchors
  - 373 negative anchors
  - 107 positive events
- 每个 anchor 的 7-frame source window
- D36 `detections.tsv`
  - 3,710 unique decoded source frames
  - 14,364 person detections
  - SHA-256：
    `5083db4c86ff20c01d12a47aa9b419d3cd1727b8f8ed1383020b33601ad6f731`
- D36 `producer_receipt.json`
  - SHA-256：
    `26f2991f4f836c2611224bc9ea1c9066d00b6bf03b600155736fafcbf8ab5ade`
- D31 anchor parity 必须继续为 raw count `0`、selected mask `0`、maximum
  selected-box error `0.0`

D37 不重新解码视频、不重新运行 detector，也不改变 timestamps、box order、
frame size、profile、scenario、risk engine、event tracker 或 feedback planner。

## 3. paired production kernel

每个 sample 使用两个全新且独立的 production `AssistDecisionKernel`：

- baseline：`DualLoopRuntimeMode.OFF`
- candidate：`DualLoopRuntimeMode.ACTIVE_CONTRADICT_ONLY`
  - 不传入外部 `dualLoopGeometryEvidence`
  - 由 kernel 内现有 production
    `CausalSceneScaleTristateGeometryProducer` 直接消费当前 detections

唯一允许的行为差异是 candidate 在 admitted
`CONTRADICT_APPROACH` 时抑制 feedback。每一帧都必须满足：

- baseline 与 candidate raw risk 完全一致
- baseline 与 candidate stable risk 完全一致
- admitted evidence source 若存在，必须是
  `CausalSceneScaleTristateGeometryProducer.SOURCE_ID`
- candidate 不得产生 baseline 没有的 triggered window

scene-scale producer 参数保持 production 默认值：

- `rateThresholdPerS = -0.05`
- `maximumGapNs = 500,000,000`
- minimum matches = 2
- association IoU / center-distance 与 production 实现一致

不允许搜索 threshold、gap、minimum matches、association rule 或 window length。

## 4. truth firewall

Kotlin replay 只读取 source-only `detections.tsv`，输出每个 sample 的：

- baseline/candidate triggered frames 与 window terminal
- admitted contradict/confirm counts
- feedback suppressions
- raw/stable risk parity
- admitted source integrity

Kotlin replay 不读取 onset labels 或 event groups。只有 replay 完成并原子写入
`kernel_replay.tsv` 后，Python evaluator 才按 frozen `sample_id` join D12 truth。

任何发生在 truth join 前的 parser、path、Gradle、serialization 或实现错误都属于
可修复 engineering failure，不烧毁 cohort，也不产生科学终态。

## 5. evaluability gates

以下全部通过后才允许解释 effect：

1. complete cohort：`530 anchors / 19 sessions`
2. D36 source receipt 与 detector hash 完整匹配
3. D31 anchor detector parity 全部通过
4. raw/stable risk mismatch 均为 0
5. admitted non-scene evidence 为 0
6. baseline positive alerted anchors `>=20`
7. baseline negative alerted anchors `>=20`
8. scene-scale admitted contradict opportunity：
   - `>=10 anchors`
   - `>=5 source sessions`

若第 8 项失败，终态固定为：

`D37_THOR_MAGNI_PRODUCTION_SCENE_SCALE_VETO_EVENT_NOT_EVALUABLE`

不得解释为 scene-scale 算法无效。

## 6. support gates

仅当 evaluability 全部通过时，同时满足下列条件才为 supported：

1. positive event losses = `0`
2. positive anchor losses `<=1`
3. positive anchor recall delta `>= -1.0 pp`
4. negative alert absolute reduction `>=10 anchors`
5. negative alert relative reduction `>=20%`
6. 5 folds 中至少 3 folds 的 negative alerts 减少
7. candidate-only triggered windows = `0`

全部通过：

`D37_THOR_MAGNI_PRODUCTION_SCENE_SCALE_VETO_EVENT_SUPPORTED`

可评价但任一 support gate 失败：

`D37_THOR_MAGNI_PRODUCTION_SCENE_SCALE_VETO_EVENT_NOT_SUPPORTED`

## 7. claim ceiling 与后续

即使 supported，也只建立：

`THOR_MAGNI_OUTCOME_OPEN_PRODUCTION_SCENE_SCALE_EVENT_VETO_SUPPORTED_DEVELOPMENT_ONLY`

它不会覆盖 D35 的物理设备 parity/runtime/non-interference gate，不改变
`RESEARCH_MAINLINE_UNCHANGED / DEFAULT_APP_UNCHANGED`。只有后续在独立 outcome
evidence 上直接超过主线 event utility，才有资格讨论主线替换。

若 D37 NOT_EVALUABLE 或 NOT_SUPPORTED，不在同一 outcome 上调整 scene-scale
参数。下一步必须更换科学变量或取得新的独立证据，而不是扩大控制面。
