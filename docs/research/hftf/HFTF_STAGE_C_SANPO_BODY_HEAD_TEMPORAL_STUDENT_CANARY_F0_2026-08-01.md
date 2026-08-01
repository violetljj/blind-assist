# HFTF Stage C SANPO body/head temporal-student canary F0

日期：2026-08-01

状态：`FROZEN_BEFORE_F0_SOURCE_OUTCOME`

## 1. 只检验身体/头部支路

EgoWalk 的足底 ground-continuity source route 已按 E0.2 stop rule 关闭，但这不关闭
HFTF。F0 转向 R4 已有 reference-relative 支持的 SANPO-Synthetic obstacle source，
只回答：

> 最近 0.8 秒的 RGB history，是否比同结构的单帧 RGB 更好地预测 0.4 秒后的
> body/head swept-envelope geometry-proxy field？

F0 不包含 foot layer，不把 SANPO proxy 称为人体事件或安全真值。即使成功，也只形成
SANPO-Synthetic body/head temporal signal 的 Development evidence，不能冒充完整
HFTF、主线晋级、Android 提醒或安全证据。

## 2. Outcome-blind source 固定

source pool 先排除 R4 前 56 个 burned sessions，再排除 R4 已打开 obstacle outcome
的四个 sessions，共 60 个。官方 train split generation 固定为
`1692794964120907`，文本 SHA-256 固定为
`f9c5dc4c289fa87342abc0d2cc49f112fcc78c7e02e0b6b081e296a99344173c`。

只读 inventory 按完整 session ID 字典序取前 12 个 eligible sessions；不得读取
geometry labels。rank 1–6 固定 train，7–9 固定 dev，10–12 固定 heldout。任何
geometry 或 student outcome 打开后都不得换样。

eligible source 必须提供 `camera_chest/left`、intrinsics、source-authoritative
pose-frame binding，以及从 frame 0 开始至少 50 个 RGB/mask/depth 同 index 帧。
目标 timeline 使用 `min(10, source_fps)`，固定取 25 帧。

## 3. 物理时间与因果边界

SANPO sessions 混合 5/20 FPS，因此所有 offset 从物理时间复算，禁止把固定帧差当成
固定时间：

- student history：`[-.8,-.6,-.4,-.2,0] s` 五张 RGB；
- future horizon：`.4 s`；
- future origin：只由 `anchor-.4 s -> anchor` 的 history pose 速度外推；
- field orientation：只用 anchor yaw；
- future pose 只把 future observation 变换到 world，不得选择 origin、方向或样本；
- future RGB/depth/mask/pose 不进入 student。

## 4. Teacher 与 UNKNOWN

field 固定为 `2 horizons × 2 heights × 6 theta × 6 distance`。body 为
`.35–1.35 m`，head 为 `1.35–2.05 m`，方向为 `[-45°,45°]`，距离 edges 为
`[0,1,2,3,4,6,8] m`。

训练 teacher 使用 R4 支持的 swept-envelope stride-8/offset-4 candidate；评价使用
像素 lattice 不相交的 stride-4/offset-2 dense reference。primary positive threshold
固定为 2。known 仍要求 9 个 prism probes 中至少 5 个具有 depth-front support；
UNKNOWN 永不默认 SAFE。

future label 把 anchor 与精确 `+.4 s` observation 统一变换到 causal future field。
这只产生 geometry proxy。

## 5. 训练前顺序门

任何 corpus 或 training 前，12/12 source 都必须：

- authority、对象 receipt、local hash、完整 decode 与 pose binding 全过；
- current/future、body/head known coverage 均不低于 `.10`；
- future body/head 各至少 5 positive known 与 20 negative known cells；
- train/dev/heldout 每个 role 的每个高度至少有 2 个 positive sources；
- UNKNOWN→SAFE 为 0；
- 第二遍 teacher payload byte-exact。

任一失败即
`F0_SANPO_BODY_HEAD_STUDENT_CANARY_NOT_EVALUABLE`，停止且不得换样或训练。

## 6. 同结构三臂

三个 arm 共用 MobileNetV3-Small、同参数 temporal fusion 与 field head：

1. `SF_CURRENT`：anchor RGB 重复五次，预测 current；
2. `SF_FUTURE`：anchor RGB 重复五次，预测 `.4 s`；
3. `HIST_FUTURE`：五张真实 history RGB，预测 `.4 s`。

这样 temporal effect 的主比较只改变是否提供真实 history，不改变网络参数量。
ImageNet checkpoint SHA-256 固定为
`047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`。
训练使用三个 seeds、30 epochs、固定 `.5` threshold；heldout 不做阈值或超参选择。

## 7. 成功门

主比较为 exact heldout known cells 上
`HIST_FUTURE - SF_FUTURE`：

- 三 seed median micro-F1 delta `>= .03`，且每个 seed 都为正；
- median recall delta `>= -.02`；
- median false-positive-rate delta `<= +.02`；
- body/head 各自 median F1 delta `>= 0`；
- worst source median F1 delta `>= -.02`；
- `SF_CURRENT` micro-F1 至少 `.60`，排除标签完全不可学习。

通过终态仅为
`F0_SANPO_BODY_HEAD_TEMPORAL_STUDENT_SIGNAL_SUPPORTED`；不通过则
`F0_SANPO_BODY_HEAD_TEMPORAL_STUDENT_SIGNAL_NOT_SUPPORTED_STOP`。两者都不自动改变
研究主线或 App。
