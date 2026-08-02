# HFTF Stage C D42 JRDB ego-object metric teacher protocol

日期：2026-08-03（Asia/Hong_Kong）

状态：

`STAGE_C_D42_JRDB_EGO_OBJECT_METRIC_TEACHER_FROZEN`

证据角色：Development / source-native metric-geometry information ceiling

## 科学问题

D41 的 7-frame image-box 外推获得 pooled mean IoU 增量，但 median、改善覆盖、
center gate 与 scale gate 均失败；四个 sequences 的 log-area error 全部恶化。
D42 不在 D41 outcome 上删 state 或调 horizon，而用 packet 已绑定的 metric pose
与 person 3D center，直接检验：

> 把 ego motion 与 person world motion 分开后，是否能稳定改善一秒后的相对
> metric geometry？

## 冻结 cohort

- D33 detector tracks、producer receipt 与四个 JRDB observation packets
- current association：D33 Hungarian IoU floor `0.30`
- 每个 current detector-track match 绑定 native `label_id`
- history：current 及之前连续 7 frames
- future：`+15 frames` 的同一 native identity
- 只有 native identity 在全部 7 history frames 与 future 中存在时可评价

不重跑 detector/tracker，不改变 association、history、horizon 或 sequence。

## pose/geometry authority

packet 每帧已经绑定：

- `pose`: `odom <- base_link`
- native `center_base_link_m`
- native `center_odom_m`

冻结 parity：

`R(pose) @ center_base_link_m + translation == center_odom_m`

最大绝对误差必须 `<=1e-9 m`。否则 D42 不可评价。

## 三个因果 arm

全部只读 current 及历史，不读 future truth：

1. `CURRENT_RELATIVE_STATIC`
   - prediction = current `center_base_link_m`
2. `EGO_KINEMATIC_OBJECT_STATIC`
   - 对 7-frame ego odom translation 与 unwrapped yaw 做 timestamp-aware OLS
   - object 固定在 current `center_odom_m`
   - 用预测 future ego pose 转回 future base frame
3. `EGO_OBJECT_KINEMATIC`
   - ego prediction 与 arm 2 相同
   - 对同一 identity 的 7-frame `center_odom_m` x/y/z 做 OLS
   - 用预测 future ego pose 转回 future base frame

不裁剪速度，不搜索 smoother、坐标子集、history、horizon 或 regression order。

## 冻结指标

相对 future native `center_base_link_m`，三个 arms 报告：

- horizontal position error（m）
- absolute range error（m）
- absolute bearing error（degree）

主要 effect 是 arm 3 相对 arm 1。arm 2 相对 arm 1 与 arm 3 相对 arm 2 用于
ego/person contribution attribution，不单独触发晋级。

## evaluability

必须全部满足：

- producer receipt 480/480 frames
- `>=400` paired opportunities
- `>=15` distinct native identities
- 至少 `3` sequences 各有 `>=50` opportunities
- pose/center transform parity `<=1e-9 m`
- 所有 predictions/metrics finite

失败终态：

`D42_JRDB_EGO_OBJECT_METRIC_TEACHER_NOT_EVALUABLE`

## support gates

arm 3 相对 arm 1 必须全部满足：

1. pooled mean horizontal-error relative reduction `>=20%`
2. pooled median horizontal-error relative reduction `>=20%`
3. horizontal-error better fraction `>=60%`
4. pooled mean absolute-range-error relative reduction `>=15%`
5. pooled mean absolute-bearing-error relative reduction `>=10%`
6. 至少 3 个 evaluable sequences 的 mean horizontal error 改善
7. 任一 evaluable sequence 的 mean horizontal error不得恶化超过 `5%`

通过：

`D42_JRDB_EGO_OBJECT_METRIC_TEACHER_SUPPORTED_DEVELOPMENT_ONLY`

否则：

`D42_JRDB_EGO_OBJECT_METRIC_TEACHER_NOT_SUPPORTED`

## stopping 与 claim ceiling

看到结果后不得在同一 outcome 上切换到 ego-only、删 z/yaw、改 history/horizon、
裁剪速度或排除 sequence 来救结果。

支持只证明 source-native metric teacher 能修复 D41 所暴露的几何可识别性缺口，
允许冻结 D43 轻量学生输入/蒸馏合同；不证明 RGB/IMU student 可学、不证明事件效用、
Android runtime、主线替换、产品或安全效果。

`RESEARCH_MAINLINE_UNCHANGED / DEFAULT_APP_UNCHANGED`

D42 不覆盖 D35 device gate。
