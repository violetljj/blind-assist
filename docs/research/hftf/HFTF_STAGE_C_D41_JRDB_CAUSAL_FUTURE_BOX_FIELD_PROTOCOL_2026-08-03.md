# HFTF Stage C D41 JRDB causal future-box field protocol

日期：2026-08-03（Asia/Hong_Kong）

状态：

`STAGE_C_D41_JRDB_CAUSAL_FUTURE_BOX_FIELD_FROZEN_R0_2`

证据角色：Development / future spatial representation canary

## 科学问题

D33 已证明 detector-track 的 causal log-height trend 能高精度判断约一秒后的
range direction；D40 证明只把 selected box 做尺度投影再送回现有 risk kernel
不会改变 THOR-MAGNI event terminal。D41 不再评价提醒或 veto，而直接检验：

> 对全部可评价 detector tracks，7 帧 causal image-space state 是否能比
> current-box baseline 更准确地定位同一行人约一秒后的 2D box？

## 冻结输入

- D33 `tracks.jsonl`，SHA-256：
  `efa249fdfe8114dfeb1da419ffdb359189e3d4e6b1f406fabad04a31a39a0fa1`
- D32/D33 已绑定的四个 JRDB observation packets
- D33 current-frame Hungarian association，IoU floor `0.30`
- history：连续 7 frames
- future：`+15 frames`

不得重跑 detector、改变 tracker、tile、confidence、association、history 或 horizon。

## 冻结单变量

对每条 source track 的连续 7 帧，分别对以下四个状态做 timestamp-aware OLS：

- box center x
- box center y
- log width
- log height

candidate 按 future packet 的实际 timestamp 外推到 `+15 frames`，重建有限的
raw future box。baseline 是 current detector box。forecast 生成完全不读取 annotation；
current native association 与 future same-identity box 只在 evaluation join 使用。

这与 D40 的差异是：D41 使用全部可评价 tracks，并显式预测 2D translation 与
scale；输出是 future spatial representation，不进入现有 risk/alert kernel。

R0.1 source-only repair：首次 join 在生成任何聚合 outcome 前，遇到一个完全离开
画面的 forecast，原 `clamp` 语义会产生零面积 box 而中止。纯 source census 为
`3,692` forecasts，其中 `20` fully outside、`227` partially clipped。冻结修复为
保留 raw projected box，不把中心拉回画面；IoU union、center error 与 log-area
error 都原样惩罚越界预测。该修复没有读取汇总指标，也不搜索或排除样本。

R0.2 control-plane repair：首次完整 report 错把 `tracks.jsonl` 中出现的 478 个
非空帧当作 producer source census，导致 `exact_source_frames=false`；冻结 D33
producer receipt 实际记录 480/480 frames，其中两帧允许零 tracked occurrence。
评估器改为绑定 receipt SHA 并以 `frame_count=480` 做 source census，同时另报
`frames_with_tracked_occurrences=478`。该修复不改变 opportunity、metric 或 support
gate；首次 report 已有四项 support gate 失败，修复不可能把结果救成 supported。

## 冻结指标

每个 same-identity opportunity 同时计算 baseline/candidate：

- future-box IoU
- future-box center error，以 future-box diagonal 归一化
- absolute log-area error

pooled、四 sequence 与 distinct native identities 全部报告。不得只报告
non-abstain 子集；D41 没有 evidence threshold。

## evaluability

必须全部满足：

- exact 480 source frames
- `>=400` same-identity future opportunities
- `>=15` distinct native identities
- `>=3` sequences 各有 `>=50` opportunities
- 所有 forecast 与指标 finite

否则终态：

`D41_JRDB_CAUSAL_FUTURE_BOX_FIELD_NOT_EVALUABLE`

## support gates

必须全部满足：

1. pooled mean IoU delta `>= +0.02`
2. pooled median IoU delta `>= +0.02`
3. candidate IoU better fraction `>=55%`
4. pooled mean normalized-center-error relative reduction `>=10%`
5. pooled mean absolute-log-area-error delta `<=0`
6. 至少 `3` 个 evaluable sequences 的 mean IoU delta `>0`
7. 没有 evaluable sequence 的 mean IoU delta `<-0.02`

通过：

`D41_JRDB_CAUSAL_FUTURE_BOX_FIELD_SUPPORTED_DEVELOPMENT_ONLY`

未通过：

`D41_JRDB_CAUSAL_FUTURE_BOX_FIELD_NOT_SUPPORTED`

## stopping rule

看到结果后不得在同一 outcome 上搜索 regression order、state subset、history、
horizon、clamp、association、sequence exclusion 或 gate。

若不支持，停止当前 constant-velocity image-box field recipe；下一变量必须引入新的
ego-motion/geometry teacher 或新鲜 source。若支持，只允许保留
detector-bound future spatial representation 主张，不能升级为 event utility、
Android runtime、主线、产品或安全主张。

## claim ceiling

`RESEARCH_MAINLINE_UNCHANGED / DEFAULT_APP_UNCHANGED`

D41 不覆盖 D35 device gate。
