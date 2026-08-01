# HFTF Stage C causal signed-clearance transport D2

## 结论

G0-D1 已因一次性 fresh acquisition 的 Windows 长路径传输失败诚实关闭；本设计不救援
D1，也不把该传输失败解释成模型失败。HFTF 的下一条科学问题冻结为：

> 在完全不读取未来观测的条件下，仅用历史位姿估计恒速因果原点，把 current
> signed-clearance teacher field 运输到 `+0.4/+0.8 s`，是否比“不移动 current
> field”的 persistence 基线更准确地预测 future signed clearance？

这是 field dynamics/mechanics 问题，而 D1 是 current RGB 对 current clearance 的
learnability 问题。D2 不使用 D1 checkpoint、loss、prediction 或三条 fresh cohort，
不补全 partial root、不换源、不修改 D1 终态。

机器可读合同见
[D2 protocol](HFTF_STAGE_C_CAUSAL_SIGNED_CLEARANCE_TRANSPORT_D2_2026-08-01.json)。

## 当前数据边界

既有 G0 plan 只证明 12 条 official-train parent 合格：9 条已成为 outcome-open
Development，另外 3 条属于已经关闭的完整 D1 fresh cohort。后两条虽未打开，也不能
改角色供 successor 使用。3 条 official-test reservation 继续封闭。

所以当前可直接用于 D2 的新 parent 数是 **0**。这不证明数据池耗尽，只说明在任何新
媒体打开前，必须另冻一份 official-train metadata-only qualification 实现合同。扫描
按绑定 generation/SHA 的 split 内 session ID 升序，排除全部历史 burned/consumed、
9 条 D1 Development、完整 3 条 D1 fresh cohort、F0.1 consumed official-test 与
G0 reserved official-test，锁定 6 条全新 parent；不足即
`STOP_NO_ELIGIBLE_NEW_DEVELOPMENT_COHORT`。

metadata 阶段只允许读取 split、description、RGB/mask/depth object listing 和 pose
object receipt；不得读取 RGB、mask、depth 或 pose 内容，也不得打开 teacher/student
outcome。本设计本身尚不授权执行扫描。

## T0 前置门

在 D2 打开任何新媒体前，必须先让新的短路径 transport 在一条**已经 outcome-open 的
Development source**上通过：

- synthetic filesystem canary；
- final、staging 与 downloader `.tmp` 的全部内容路径 `<240` 字符；
- 每个对象绑定 generation、size、MD5；
- 新 short-path package 与既有 consumed canonical package 在 source object identity、
  selected frames 和本地 RGB/mask/depth hashes 上等价。

T0 只证明未来采集基础设施，不产生 fresh、模型效果或安全证据。失败时 D2 保持
`NOT_EVALUABLE`，不得打开新 source。

## 一次性 Development mechanics

只有 T0 通过且新的 6-parent metadata qualification 成功后，才能另冻执行合同。两臂
固定为：

1. `CURRENT_FIELD_PERSISTENCE`
2. `HISTORY_CAUSAL_ADVECTED_CURRENT_FIELD`

candidate 只读封存的 current-only teacher point-field 与 history pose。future depth
仅在两臂 prediction
耐久落盘后作为 truth；future pose 只允许重投影 truth，不能定义 candidate 的原点、
方向或 prediction。更精确地说，current-only teacher 可用 current depth、current
mask/冻结 semantic filter 与 current pose，先生成并封存**精确继承 G0 定义**的
reference point-field；两个 arm 只读该封存 field 与 history pose，不再读取 raw
current/future media。这样既保留 G0 signed-clearance 的语义，又不把 future 信息泄漏
给 candidate。

G0 的 `known` 不能从 obstacle points 或“有限 clearance”推断。current-only
preprocessor 因此在任何 future truth 打开前，用 current depth/mask、history pose 与
已经冻结的 predicted SE(2)，分别封存 persistence current-grid 以及每个 advected
anchor×horizon predicted-grid 的精确 9-probe pass counts 和 `>=5/9` known masks。
随后 raw current media 隐藏。两臂每个 cell 的输出合同固定为
`{known, clearance_m}`：`known=false` 时 `clearance_m=null`；只有 `known=true`
才允许 G0 second-order finite clearance，不能把无点时的正 sentinel/clip 偷换成
SAFE。

所有 source 统一为 5 Hz 的 13-frame timeline：原生 5 Hz 用 `0..12`，原生 20 Hz 用
`0,4,...,48`。anchor 固定为 normalized indices `2..8` 共 7 个；历史位姿固定读取
`t-0.4 s,t`，future offset 固定为 `+2/+4` normalized frames。平移使用这两个历史
位姿的 planar constant velocity；yaw 使用 `[-π,π)` 最短角差的 constant yaw-rate。
预测 origin 在 outcome 前由该 SE(2) 外推唯一确定，vertical origin/up 保持 current
ground-aligned 定义。

persistence 将 current field 原样复制到相同 `theta×distance×height` indices；
advected arm 把封存的 current obstacle points 由 current anchor 变到 world，再变到
预测 future SE(2) frame，随后严格按 G0 的 6×6×2 bins、membership、second-order
statistic、clip 与 zero tie-break 重算 field。future truth 在 predictions 耐久后，
才用 future depth/mask 与 actual future pose生成 G0 reference points，再投到**已经
冻结的 causal predicted frame**。common-known 必须是同一
`parent×anchor×height×horizon×theta×distance`，且 truth、persistence、advected
三者全部 `known=true`。两臂全预测格（不只是 common-known）均要求
UNKNOWN→SAFE `=0`。

删除/扰动 future depth、扰动 future pose 都不得改变 candidate bytes，重复运行必须
byte-deterministic。

每个 `source × body/head × 0.4/0.8 s` 必须先满足 common-known coverage `>=0.10`、
known risk/safe cells `>=5/20`、UNKNOWN→SAFE `=0`。每个分层的固定分母为
`7×6×6=252` cells；24 个分层缺一不可。任一不足立即
`D2_NOT_EVALUABLE_OPPORTUNITY_INADEQUATE_NO_SOURCE_REPLACEMENT`，不进入 effect，
不换源。

MAE 聚合也已固定：先在每个 source×height×horizon 内对 common-known cells 做
micro mean，再等权平均四个 height×horizon 得 source MAE，最后等权平均 6 个 source。
height/horizon gate 分别等权平均其余轴与 6 个 source。risk-sign 仍是 clearance
严格 `<0`；parent 内汇总所有 common-known cells 得 F1，再 6-parent 等权。F1 零分母
记 0；浮点比较容差 `1e-12`；parent 改善必须严格大于 `1e-12`。若 persistence macro
MAE `<=1e-12`，relative reduction 定义为 0，因此不能借零分母通过。

效果门全部预冻结：

- 6-source macro clearance MAE 相对 persistence 至少下降 `10%`，且绝对下降
  `0.03 m`；
- body、head、`0.4 s`、`0.8 s` 均不退化；
- 至少 `5/6` parent 的 MAE 改善；
- parent-macro risk-sign F1 delta `>=+0.03`；
- UNKNOWN→SAFE 违规为 0。

任一失败永久关闭为
`CAUSAL_SIGNED_CLEARANCE_TRANSPORT_NOT_SUPPORTED_STOP`，同 cohort 不得调整 horizon、
bin、margin、matching tolerance、source、metric 或 gate。全部通过也只授权另冻
RGB-student 合同，不改变研究主线、默认 App、Android、生产或安全权限。
