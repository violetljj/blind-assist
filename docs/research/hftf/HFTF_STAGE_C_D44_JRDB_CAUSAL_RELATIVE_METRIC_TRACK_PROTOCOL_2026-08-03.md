# HFTF Stage C D44 JRDB causal relative metric-track protocol

日期：2026-08-03（Asia/Hong_Kong）

状态：

`STAGE_C_D44_JRDB_CAUSAL_RELATIVE_METRIC_TRACK_FROZEN`

证据角色：Development / causal metric-measurement information ceiling

## 科学问题

D42 证明 world-frame ego+person geometry teacher 强支持；D43.1 证明十个 2D
track features 的 linear student 无法跨 sequence 恢复 metric residual。D44 不再
增加模型容量，而检验一个更直接、可部署语义更清楚的测量假设：

> 如果运行时已经有 same-target relative metric center history，那么不显式分解
> ego 与 person world motion，简单 causal relative-velocity 是否足以预测一秒后
> 的相对 geometry？

## cohort

- exact D42/D43.1 3,384 opportunities / four JRDB sequences
- current detector association 与 native identity 只定义 teacher opportunity
- history：同一 identity 连续 7 frames
- future：`+15 frames`
- packet `center_base_link_m` 作为 metric-depth-track oracle

D44 是 adaptive Development，不是独立验证。

## 两臂

1. `CURRENT_RELATIVE_STATIC`
   - prediction = current `center_base_link_m`
2. `CAUSAL_RELATIVE_METRIC_TRACK`
   - 对 7-frame `center_base_link_m` x/y/z 分别做 timestamp-aware OLS
   - 外推到 future packet 的实际 timestamp

两个 arms 都不读取 future。不得裁剪速度、平滑、改 history/horizon、删 z、
筛 identity 或 sequence。

## 指标

相对 future `center_base_link_m`：

- horizontal position error
- absolute range error
- absolute bearing error
- candidate horizontal-error better fraction

同时披露 D42 full world-decomposed teacher 的既有 aggregate reference，但不以
D42 outcome 调整 D44 gate。

## evaluability

- producer 480/480 frames
- `>=400` opportunities / `>=15` identities
- 4 sequences 各 `>=50` opportunities
- timestamp contiguous
- all predictions/metrics finite

否则：

`D44_JRDB_CAUSAL_RELATIVE_METRIC_TRACK_NOT_EVALUABLE`

## support gates

candidate 相对 current-static 必须全部满足：

1. pooled mean horizontal-error relative reduction `>=20%`
2. pooled median horizontal-error relative reduction `>=20%`
3. horizontal-error better fraction `>=60%`
4. mean absolute-range-error relative reduction `>=15%`
5. mean absolute-bearing-error relative reduction `>=10%`
6. 至少 3/4 sequences mean horizontal error 改善
7. 无 sequence mean horizontal error恶化超过 `5%`

通过：

`D44_JRDB_CAUSAL_RELATIVE_METRIC_TRACK_SUPPORTED_DEVELOPMENT_ONLY`

否则：

`D44_JRDB_CAUSAL_RELATIVE_METRIC_TRACK_NOT_SUPPORTED`

## stopping 与 claim ceiling

支持只证明 causal same-target metric history 足以形成 future-relative geometry
primitive，允许下一步选择一个端侧 metric-depth source 做 shadow canary。它不证明
当前手机已经提供该测量，也不建立 event、Android runtime、主线、产品或安全主张。

失败则停止当前 relative constant-velocity metric recipe；不得在同一 outcome 上调
窗口、速度裁剪或 smoother。

`RESEARCH_MAINLINE_UNCHANGED / DEFAULT_APP_UNCHANGED`
