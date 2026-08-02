# HFTF Stage C D40 THOR-MAGNI continuous track-projected risk protocol

冻结时间：2026-08-03（Asia/Hong_Kong）

状态：

`D40_THOR_MAGNI_CONTINUOUS_TRACK_PROJECTED_RISK_FROZEN_BEFORE_D40_SOURCE_REPLAY_OR_OUTCOME_JOIN`

## 1. research rebase

D36-D39 尝试把 second-loop evidence 作为现有提醒的 veto；结果已把
scene-scale persistence family 关闭。D40 不再抑制 mainline feedback，而回到
HFTF 原问题：

> causal track 的历史尺度变化，能否直接构造约一秒后的预测风险，并在连续
> production session replay 中形成相对当前帧风险的事件级 Pareto 增量？

它连接两个已有但层级不同的证据：

- D33：七帧 detector-track `log(box_height)` slope 对约一秒 future range
  direction 的 precision 为 `96.82%`
- D34：同一 estimator 在 production Kotlin 中 parity/runtime supported

D40 不使用 D36 的 strict tri-state 作为 veto。它使用 estimator 已经产生的连续
OLS slope，对当前 selected target 做显式未来投影，再由现有 production risk
kernel 重新评价。

数据角色：

`POST_D39_ADAPTIVE_OUTCOME_OPEN_DEVELOPMENT`

即使 supported，也需要新的独立 event cohort。

## 2. frozen source

原样复用 D36 truth-free detector source：

- 19 THOR-MAGNI sessions
- 530 frozen anchors
- 3,710 unique source frames
- 14,364 person detections
- 7-frame causal source windows
- detections SHA-256：
  `5083db4c86ff20c01d12a47aa9b419d3cd1727b8f8ed1383020b33601ad6f731`
- producer receipt SHA-256：
  `26f2991f4f836c2611224bc9ea1c9066d00b6bf03b600155736fafcbf8ab5ade`
- D31 anchor parity：raw count/mask/box `0 / 0 / 0.0`

同一 `(source_session_id, source_scene_frame)` 必须具有唯一 timestamp、frame
size 与 detection list；重叠 sample windows 只用于重建 unique continuous
frames，不重复送入 kernel。

## 3. continuous session replay

每个 source session 按 `captured_at_ns, source_scene_frame` 排序。

- 相邻 frame gap `<=500 ms`：同一 continuous segment
- gap `>500 ms`：baseline kernel、candidate kernel 与 track estimator 全部 reset
- 不跨 session 继承任何状态

baseline：

- 当前 detections
- production `AssistDecisionKernel`
- `DualLoopRuntimeMode.OFF`

candidate：

1. baseline current raw risk 选出的 target 进入 production
   `CausalTrackTristateGeometryProducer`
2. 只有 evidence 的 `signedApproachRatePerS` 非空、finite，且 selected target
   在 detection list 中唯一绑定时，构造 +1.0 s forecast
3. 预测尺度：

   `scale = exp(signedApproachRatePerS * 1.0 s)`

4. 预测 box：
   - 保持 current bottom-center
   - width 与 height 同比例缩放
   - clamp 到 current frame
5. candidate detections 只把 selected target 替换为 forecast box；其他 detections
   原样保留
6. candidate production kernel 对预测 detections 运行 `OFF`
7. slope 缺失、非 finite、scale 非 finite、target 非唯一或 clamped box 退化时，
   candidate fail closed 为 current detections

不使用 slope sign threshold、unanimity decision、future truth、depth、pose 或
metric TTC。projection horizon 固定为原始 HFTF/D32 的 `1.0 s`，不搜索
0.5/1/1.5/2 s。

## 4. anchor/window mapping and truth firewall

source-only Kotlin replay 先原子输出：

- 每个 unique frame 的 baseline/candidate triggered state
- forecast availability、slope、scale 与 projected target
- segment/session reset
- 每个 frozen anchor 对应七帧 window 的 baseline/candidate terminal

Kotlin 不读取 D12 onset labels。replay 完成后，Python evaluator 才按 frozen
`sample_id/source_session_id/anchor_scene_frame/fold` join：

- 157 positive anchors
- 373 negative anchors
- 107 positive events

工程故障在 truth join 前可修复重跑，不烧毁 cohort。

## 5. evaluability gates

必须全部通过：

1. exact cohort：530 anchors / 19 sessions
2. exact unique source census：3,710 frames / 14,364 detections
3. D36 input hash、receipt 与 D31 parity 全通过
4. 每个 anchor 的七帧全部映射到同一 continuous source replay
5. baseline 与 candidate source frame identity 完全一致
6. forecast opportunity：
   - >=50 anchors 的七帧 window 内至少一个 forecast
   - >=5 sessions
7. positive 与 negative baseline opportunities 均 >=20 anchors
8. candidate output 无 non-finite risk、box 或 serialization

任一失败：

`D40_THOR_MAGNI_CONTINUOUS_TRACK_PROJECTED_RISK_NOT_EVALUABLE`

## 6. support gates

D40 要求相对同一 continuous-session production baseline 的 Pareto 改善：

1. positive event hit delta `>=0`
2. positive anchor recall delta `>=-1.0 pp`
3. negative alert delta `<=0`
4. candidate-only negative windows `<=5`
5. 至少一个 meaningful strict gain：
   - positive event gains `>=5`；或
   - negative alert absolute reduction `>=20`
6. 5 folds 中至少 3 folds 在 positive event hits 或 negative alerts 上严格改善，
   且另一主指标不恶化

全部通过：

`D40_THOR_MAGNI_CONTINUOUS_TRACK_PROJECTED_RISK_SUPPORTED_DEVELOPMENT_ONLY`

否则：

`D40_THOR_MAGNI_CONTINUOUS_TRACK_PROJECTED_RISK_NOT_SUPPORTED`

## 7. claim ceiling and stop rule

supported 只建立：

`CONTINUOUS_TRACK_PROJECTED_ONE_SECOND_RISK_PARETO_SIGNAL_DEVELOPMENT_ONLY`

它不会覆盖 D35 device gate，不改变默认 App、主线、产品或安全主张。

若 NOT_EVALUABLE，保留 D33/D34 mechanism/parity，并定位 forecast opportunity。
若 NOT_SUPPORTED，不在同一 outcome 上搜索 projection horizon、slope clamp、
track history、association 或 risk threshold；停止当前 box-scale projection
实例，转向新鲜几何 teacher/field evidence。
