# HFTF H1 causal-advected-origin geometry teacher protocol R2

日期：2026-08-01

状态：`FROZEN_RESULT_NOT_RUN`

## 1. 唯一改动

R2 保留 R1 的 field、teacher sampling、UNKNOWN、denominator、gates 与顺序终点，只把
future field 的固定 anchor origin 改成 history-causal rolling origin：

`origin(h) = anchor_ground_origin + project_ground(v_history) * h`

其中 `v_history` 仅由最接近 `anchor-400 ms` 的严格历史 camera pose 到 anchor pose
计算，history tolerance 为 `100 ms`。若存在等距 history candidates，选择 source
frame 较大的一个。future pose 不参与 origin、方向或样本选择。

forward/right/up 全部保持 anchor basis，不外推 yaw；这隔离“平移造成的 future-view
support mismatch”，不同时引入另一项旋转模型。

## 2. Fresh authority-bound cohort

parent unit 为四个 source sessions：

- `03694304…e34a`
- `03b6dc99…e68c`
- `03c87279…e3b0`
- `03d70593…b39e1`

四者均按已提交的 source-preparation rule outcome-blind 选择，并通过
`HFTF_H0_2_SANPO_CANONICAL_PROXY_REPLICATED`。机器协议绑定 preparation contract
hash、完整 session IDs，以及各自 authority report、manifest、dataset spec、pose
SHA-256。替换、重复、增加或使用 R0/R1 burned source 均 fail closed。

## 3. Usable anchor 与 field

usable anchor `U` 必须同时具有：

- 最接近 `-400 ms` 且 tolerance `<=100 ms` 的 history；
- current；
- `+400 ms` near；
- `+800 ms` far。

nominal time 由 `source_frame_num / source fps` 复算。`U` 固定后，任何
UNKNOWN/invalid 不缩小 denominator。

field 保持：

- theta：6 bins，`[-45°,45°]`；
- distance：`[0,1,2,3,4,6,8] m`；
- horizons：`0/.4/.8 s`；
- height：`foot/body/head`；
- 9 probes，至少 5/9 通过 camera-z、image、semantic nonzero 与
  depth-front `0.20 m` 才 known；
- `risk=min(1, obstacle_point_count/8)`。

每个 horizon 的 obstacle points 与 probes 都使用该 horizon 的 rolling origin 和同一
anchor orientation。

## 4. 固定门

- 4/4 source authority；
- usable anchors 每 session `>=12`；
- single/multi consistency `<=1e-12`；
- current/near/far coverage `.15/.10/.10`；
- height disagreement fraction `>=.02`；
- future near-or-far union change fraction `>=.02`；
- 后两项均需 4/4。

denominators 分别为 `|U|×6×6×3`、`|U|×6×6` 与 `|U|×6×6×3`。

## 5. 解释边界

overall future change 可来自 rolling origin 穿过静态空间，支持的是“未来局部空间表示
不冗余”，不是“环境主体运动已预测”。dynamic semantic support 必须单列，但不把
semantic dynamic class 自动当作真实运动。

成功终点仍只到 `SYNTHETIC_CAUSAL_ADVECTED_GEOMETRY_PROXY_MECHANICS_ONLY`，且不自动
授权 H2。任何结果都不改变研究主线、Android、提醒、默认 App、生产或安全权限。

## 6. 一次性规则

runner 与 tests 必须先提交推送，再运行一次正式 R2。执行后四 sessions burned：

- 不在其上改 history lookback、origin、sector、horizon、probe 或门；
- 更早顺序门失败时，后续 fractions 只作 diagnostic；
- atlas 与 causal-advection error 只定位失败，不选择阈值；
- 缺 history、degenerate basis、非有限 velocity 或 authority mismatch 均 fail closed。
