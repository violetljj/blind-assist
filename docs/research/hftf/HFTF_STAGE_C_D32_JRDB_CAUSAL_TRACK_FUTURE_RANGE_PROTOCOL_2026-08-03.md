# HFTF Stage C D32：JRDB causal track future-range canary

日期：2026-08-03

证据角色：Development / native identity-bound short-future mechanism canary

研究主线：不变

默认 App：不变

## 科学问题

D27 已证明目标中存在强 history-kinematic information；D30/D31 又证明 current
person bearing 可测，但单帧 box-height distance calibration 跨 source 不稳定。D32
不再继续拟合静态距离，也不训练新模型。它检验一个更直接的问题：

> 仅使用同一 JRDB person identity 过去七帧的 stitched-2D box-height 趋势，能否
> 预测该身份在未来约一秒内相对机器人是接近还是不接近？

这是 forward-predictive estimand，不是对上一帧或当前帧 3D 变化的复述。

## 冻结输入

直接复用四个已经物化并验证的 native multisensor observation packet：

- `clark-center-2019-02-28_0`
- `gates-basement-elevators-2019-01-17_1`
- `meyer-green-2019-03-16_0`
- `stlc-111-2019-04-19_0`

每个 packet 含 120 个连续帧，以及按原生 `label_id` 精确 join 的 stitched 2D box
和 3D `center_base_link_m`。不重新下载、不重新扫描 metadata、不重新生成
manifest，也不把 packet 的既有使用视为 source burning。

## 冻结估计量

source rule 原样继承
`DUAL_LOOP_CAUSAL_TRACK_TRISTATE_R0`，不根据 D32 outcome 调参：

- 输入：同一 `label_id` 的七个连续帧；
- 特征：`log(box_height)`；
- 拟合：对实际 image timestamp 做 causal ordinary least squares；
- `CONFIRM_APPROACH`：slope `>= 0.2/s` 且六个相邻 box-height 变化全部为正；
- `CONTRADICT_APPROACH`：slope `<= -0.2/s` 且六个相邻变化全部为负；
- 其他：`ABSTAIN`。

future truth：

- horizon：当前 frame index `+15`，对应 JRDB 约一秒；
- range：`center_base_link_m` 的三维欧氏距离；
- signed future approach rate：
  `(range_now - range_future) / elapsed_seconds`；
- `APPROACHING`：rate `>= 0.1 m/s`；
- `RECEDING`：rate `<= -0.1 m/s`；
- 其他：`QUASI_STATIC`。

`CONFIRM_APPROACH` 预测 `APPROACHING`；
`CONTRADICT_APPROACH` 预测 `NOT_APPROACHING`
（`RECEDING` 或 `QUASI_STATIC`）。

## 可判定性 gate

以下任一不满足，终态为
`D32_JRDB_CAUSAL_TRACK_FUTURE_RANGE_NOT_EVALUABLE`，不构成科学负结果：

1. pooled non-abstain evidence 至少 80 rows；
2. pooled evidence 至少覆盖 20 个 sequence-bound distinct tracks；
3. 至少 3/4 sequences 各有至少 10 rows non-abstain evidence；
4. `CONFIRM_APPROACH` 与 `CONTRADICT_APPROACH` 各至少 20 rows。

## 支持 gate

在可判定前提下，全部满足才支持该 forward mechanism：

1. pooled overall precision 至少 `0.85`；
2. pooled confirm precision 至少 `0.80`；
3. pooled contradict precision 至少 `0.80`；
4. confirm precision 相对全部 opportunity 中 `APPROACHING` prevalence 的 lift
   至少 `0.10`；
5. contradict precision 相对全部 opportunity 中 `NOT_APPROACHING`
   prevalence 的 lift 至少 `0.10`；
6. 至少 3/4 sequences 在 evidence 至少 10 rows 时 precision 至少 `0.75`。

通过终态：

`D32_JRDB_CAUSAL_TRACK_FUTURE_RANGE_SUPPORTED`

可判定但未通过：

`D32_JRDB_CAUSAL_TRACK_FUTURE_RANGE_NOT_SUPPORTED`

无论总终态如何，都必须保留每个 direction、sequence、coverage、prevalence 与
lift 的实际测量结果；总 gate 不得抹除局部正信号。

## 工程失败与重跑

- 文件缺失、JSON 解析错误、字段/时间/identity 不一致、非有限数值或写出失败，
  都是 engineering failure：程序非零退出，不写科学终态；
- engineering failure 修复后允许在同一冻结协议下重跑；
- 不使用 closed-set、one-shot、source burning、fresh-cohort retirement 或
  fsync-interruption 作为本 canary 的科学有效性条件；
- 输出可由同一输入和脚本确定性重建，已有 Development 输出允许原子替换。

## 主张边界

通过只建立：

`JRDB_ANNOTATION_TRACK_SHORT_FUTURE_MECHANISM_SUPPORTED`

它不建立：

- live detector/tracker performance；
- 未插值的独立 3D sensor truth；
- event-level obstacle/collision utility；
- Android runtime 或默认 App authority；
- product effectiveness 或 human safety。

若支持，则下一步才允许冻结 detector-bound 或 sensor-supported 的独立 replication；
若不支持，则关闭这个固定 7-frame tri-state 对一秒 future range 的配方，但不否定
D27 的 kinematic information ceiling，也不否定更丰富的 identity-bound state
estimator。
