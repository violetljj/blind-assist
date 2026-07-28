# RCLE 周期性自运动反事实 R2 预注册

日期：2026-07-28

协议：`RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2`

当前状态：`CORE_DESIGN_FROZEN_FOR_INDEPENDENT_REVIEW / EXECUTION_NOT_AUTHORIZED`

## 先给结论

Temporal Structure R1 的合法终态是
`HOLD_MIXED_OR_INSUFFICIENT_TEMPORAL_EVIDENCE / VALID`：四段 ADVIO 都有强
pose 周期，flow 方向大多连续，但 flow 与 pose 主频同步很弱，高响应也只部分伴随
blur、feature collapse 和 forward-backward failure；motion 与 quality routing
均为 `0/4`。

因此本轮不再切分 sequence13/14/15/17，不访问 sequence16，也不改算法。R2 改用
严格配对的受控 2×3 干预，同时改变“周期性 6DoF 自运动”和“测量质量退化”，回答
在受控生成器内哪条机制会使 unchanged R3 的固定分母三-pair 触发密度上升。

本文件及其机器合同只冻结设计。生成器尚未实现，blur/low-texture 的全局强度尚未用
response-blind calibration seed 选出，正式 `480+16` 条序列没有运行。

## 设计与有效样本量

| 自运动 | clean | blur | low-texture |
| --- | --- | --- | --- |
| 静止相机 | ✓ | ✓ | ✓ |
| endpoint-closed 周期性 6DoF | ✓ | ✓ | ✓ |

四个 motion block 固定为 ADVIO sequence13、14、15、17 的 pose/timestamp 波形。
只借用经过响应盲 `0.7–3.0 Hz` 提取、去线性趋势并在 SE(3) 中闭合端点的运动形状；
不读取 R1 response 来选 block、相位、幅值或最好片段，也不做幅值/时间缩放。

每个 block 使用 20 个 block-specific 新场景 seed。一个
`scene_seed × motion_block` cluster 的六臂共享场景、几何、深度、材质源、光照、
相机内参、时间戳、帧数和随机流，只允许声明的 motion 或 quality 字段变化。

因此：

- 正式主体为 `4×20×6=480` 条、每条 602 frame / 601 pair；
- 有效独立分析单位只有 `4×20=80` 个配对 cluster；
- frame、pair、周期、3×3 cell 和六个 arm 都是重复测量，不能制造大样本量。

同一 seed ordinal 在不同 block 会派生为不同数值 seed 和不同场景。如果未来改为
四个 block 复用同一批场景，cluster 必须降为 scene seed，不能仍宣称 80 个单位。

## 生成器与质量干预

生成器必须是真 3D mesh、z-buffer、metric depth、可见性和 source-known optical
flow。每个场景在 0.75–2 m、2–5 m、5–20 m 三个深度层都有实质覆盖；单平面、
stacked image plane、`warpPerspective` 或一个 homography 生成实验臂均禁止。

静止臂使用恒定 `T_ref`。周期臂使用同一 `T_ref` 上的 endpoint-closed SE(3)
残差轨迹，首尾净平移和净旋转必须接近零，避免把净接近偷渡成“周期自运动”。

quality strength 不从 RCLE output 调整：

- blur 只在 clean render 后施加一个全局 Gaussian PSF；有限 sigma grid 已冻结；
- low-texture 只在 render 前把材质 albedo 向其均值收缩；有限 alpha grid 已冻结；
- 四个 block 的独立 CAL seed、两种 motion 和固定 16 帧只计算 Laplacian、gradient、
  contrast 与 source-known edge-spread；
- 全部指标先把 uint8 sRGB 精确线性化为 Rec.709 luminance；Laplacian 使用固定
  4-neighbour 3×3 kernel，gradient 使用 sigma `{0,1,2}`、3×3 Sobel 和
  paired-clean 75th-percentile threshold，local RMS 使用固定 16×16 tile，
  edge-spread 使用 CAL plate 的 32 条 source-known step edge 和 10–90% crossing；
- source-known edge-spread 只属于 CAL scene 与解析 fixture；main scene 不放入 CAL
  plate，也不把自然边缘事后升级为 source-known edge；
- ratio 必须先逐 paired frame 计算再取 16-frame average-rank median；overall 与
  8 个 `block×motion` subgroup 都必须分别过门；
- 每种退化只能选一个覆盖全部 block/motion 的全局值，不能按 block/seed/arm 调整；
- 候选 grid 无合法值时直接 `INTERVENTION_NOT_EVALUABLE`，不能读取 RCLE 后扩 grid。

6 个 sigma、5 个 alpha 和 paired clean 共形成每个 motion 下 12 个 candidate
state；calibration 是 `6144` 次 image evaluation，不是 1536。选定全局值后，全部
main seed 还必须在任何 R3 import/run 前用同一 16 帧做 manipulation check；每个
degradation 的 8 个 `block×motion` subgroup 各至少 `18/20` sequence 通过，否则
停止。main blur 必须同时通过 Laplacian 目标与 local-RMS cross-proxy；main
low-texture 只复核 gradient-density 目标，并绑定独立审查后的 albedo-contraction
实现哈希与 alpha、声明无 PSF、通过 quality/geometry identity gate。任何缺失、
零分母、空 valid mask 或非有限指标均 fail-close，不允许插补或省略。

精确图像指标算法见
[机器合同](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_CONTRACT_2026-07-28.json)，
全部 14 项 required geometry gate 见
[geometry spec](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_GEOMETRY_VALIDATION_R0_2026-07-28.json)。

## Unchanged R3

本轮绑定的复合锁为：

- `ADVIO_WXYZ_TCAMIMU_VALIDMASK_CONTINUOUS_R3`；
- response：`compensated_expansion_median_per_s`；
- strict `>0.01/s`；
- 当前 pair 与前两 pair 必须连续可评价且均越门；
- abstention、sequence/arm 边界或 `<=0.01/s` 立即 reset；
- 每 arm 只有一个连续 `PairState`，无 lookahead。

synthetic transport 只能提供 source-native `K`、timestamp 和 camera pose。它必须先
证明与现有 R3 pair core 数值等价，不能修改 Sparse LK、support manager、局部拟合、
response、threshold、三-pair 或 reset。implementation/equivalence lock 尚未创建，
所以 formal run 仍关闭。

## 主要 endpoint 与五个对比

唯一驱动科学终态的主要 endpoint 是：

```text
Y[u,m,q] =
  unchanged R3 compensated three-pair trigger 为 true 的 scheduled pair 数
  / 该 cluster-arm sequence 固定 601 scheduled pair
```

abstention 保留在固定分母中并 reset streak；连续 response 不可评价时不得填 0。
absolute response 只在 evaluable pair 中报告分布/中位数，并同时报告 coverage，不
参与 terminal。R1 的 arm 内 top 20% high-response 在 R2 禁止，因为每个 arm 都会
机械地产生 20%。

五个 confirmatory paired contrasts 为：

```text
MOTION_CLEAN =
  Y(periodic, clean) - Y(static, clean)

BLUR_STATIC =
  Y(static, blur) - Y(static, clean)

LOW_TEXTURE_STATIC =
  Y(static, low_texture) - Y(static, clean)

MOTION_X_BLUR =
  [Y(periodic, blur) - Y(static, blur)] - MOTION_CLEAN

MOTION_X_LOW_TEXTURE =
  [Y(periodic, low_texture) - Y(static, low_texture)] - MOTION_CLEAN
```

两个 interaction 是不同 estimand。R2 不能回答 blur+low-texture 联合退化；增加该
问题需要另立 2×2×2 八臂版本。

## 统计规则

先在完整 cluster 内算六臂配对差，再在每个 block 内平均 20 seed，最后对四个固定
block 等权平均。

冻结 `20,000` 次 block-stratified paired cluster bootstrap，seed `20260728`。
每次在各 block 内重采样 20 个完整六臂 cluster。五个机制 contrast、两个
motion-plus-degradation 相对 static-clean combination check，以及两个预先固定的
`STATIC degradation − STATIC clean` quality-failure-union accompaniment，共 9 个
terminal-driving contrast，共享同一重采样权重和 max-t familywise 95%
simultaneous interval。

一个正向机制效应只有同时满足以下三项才是 `SUPPORTED`：

1. overall paired trigger-density difference `>=0.10`；
2. simultaneous 95% lower bound `>0`；
3. 至少 `3/4` block 的 paired point estimate 均为正且 `>=0.10`。

一个竞争机制只有 simultaneous 95% upper bound `<0.10`，才是
`RULED_OUT_AS_MATERIAL`。其余均为 `INCONCLUSIVE`；“没有通过支持门”不等于“没有
效应”。

20 seed/block 是冻结的最小资源预算，不是已经证明充分的 power。正式 output 前若
response-blind 精度情景或运行预算认为不足，只能在新版本增加 seed；不能先跑 480
再补 seed。

## 测量层

clean arm 的 trackability 使用已冻结的 feature/FB 条件：feature `>=60`、
FB-consistent track `>=60`、consistent fraction `>=0.50`、median FB error
`<=0.75 px`、occupied 3×3 cell `>=5`。每序列至少 70% scheduled pair 通过，
且每个 block 的两个 clean arm 各至少 `18/20` sequence 通过。

固定 3×3 局部 flow 只作为同一实验内的 measurement layer：

- normalized camera coordinates 和 source-native `dt`；
- 每 cell 独立 robust affine `v(p)=A(p-c)+b`；
- `<8` track 为 `NA`，不补零、不跨 cell 借样本；
- `divergence=A11+A22`，`curl=A21-A12`；
- radial/tangential 相对 principal point，中心半径过小为 `NA`。

quality accompaniment 预先固定为
`quality_failure_union = feature collapse OR FB failure`，使用同一 601 固定分母；
blur 和 low-texture 都只比较对应的 `STATIC degradation − STATIC clean`，并进入
同一个 9-member simultaneous family，不能在看结果后挑选 collapse 或 FB 指标。

这些量、cycle locking、feature collapse 和 FB failure 只能解释机制，不得反馈
选择 seed、质量强度、block 或算法。

## 正向护栏

主体外另用 2 个新 guard seed/block，生成：

- clean monotonic approach only；
- clean monotonic approach + periodic 6DoF。

共 16 条，不进入五个 estimand。source truth 要求至少 20% inverse-depth increase、
至少 95% persistent visible point 深度单调下降、median radial expansion
`>=0.05/s`。每个 arm 先在 block 内平均两个 seed density，再对四 block 等权；
两个 arm 必须分别满足 overall `>=0.50` 且至少 3/4 block mean `>=0.50`，一个 arm
不能救另一个，否则本生成器不能作为未来 suppression candidate 的正向护栏。

未来候选必须另行预注册，并在本 R2 中通过 `.50` 门的 block 上保留至少 90% 的
`approach + periodic 6DoF` R3 护栏密度；R2 本身不实现候选，也不做
gait-frequency notch。

## 互斥终态

执行错误和科学终态分开：

- `INVALID`：hash、pairing、algorithm lock、firewall、geometry、统计实现或 receipt
  不一致，只关闭本 evidence version；
- `EXECUTION_INCOMPLETE`：任一冻结 arm 未完成，只能按同一 identity 续跑，不能换
  seed 或分析子集。

完整有效执行按以下优先级只产生一个科学终态：

1. `INTERVENTION_NOT_EVALUABLE`：3D、质量操纵、clean tracking 或 positive
   guardrail 任一 fidelity gate 失败；
2. `MIXED`：motion 与至少一种 quality simple effect 同时 supported，或正
   interaction 及其相应组合 cell 相对 static-clean 都 supported；
3. `MOTION_SUPPORTED`：clean motion supported，两个 quality effect 与两个正
   interaction 均被排除为 material，clean tracking 良好；
4. `QUALITY_SUPPORTED`：至少一种 quality simple effect supported，并伴随相应
   FB failure 或 feature collapse 增量；motion 与两个正 interaction 均被排除。
   同时报告 `BLUR / LOW_TEXTURE / BOTH`；
5. `NO_SEPARATION_HOLD`：其余全部模式，包括一个效应 supported 但竞争机制仍
   inconclusive。

## 结果后的唯一动作

- `MOTION_SUPPORTED`：只允许另立 background flow、FOE、translation/rotation
  decomposition 候选；不直接做 gait-frequency notch；
- `QUALITY_SUPPORTED`：才允许另立 quality-gate revision；
- `MIXED`：另立 motion decomposition + quality rejection 两阶段候选；
- `HOLD`：停止这条机制路线，不切 ADVIO、不进 sequence16/Android/realtime；
- `INTERVENTION_NOT_EVALUABLE`：只修失败的 generator/calibration/transport
  evidence version，不解释机制。

CoTracker/reference-track 保持后置，只能定位真实视频中的 LK 支持或运动模型不足，
不能替代反事实因果识别。

## 权限边界

即使 R2 得到明确终态，也只支持“在受控生成器内，某种干预会改变 unchanged R3 的
内部响应”。它不能证明：

- 自然视频中的高响应是假警；
- 正常步态是现实原因；
- 受控 blur/low-texture 等于自然退化；
- 障碍、风险、提醒或安全性能改善；
- sequence16、Android、实时、产品或独立助行权限。

运行规模、observed lower-bound runtime、staging/RAM/wall-time ceiling、heartbeat、
resume 和 guarded-host activation 条件见
[run budget](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_RUN_BUDGET_R0_2026-07-28.json)。
runtime preflight 使用独立 PREFLIGHT namespace 的 6 条完整 factorial sequence 加
2 条完整 guardrail sequence，共 `8×602` frame；16-frame CAL panel 不承担持续运行
资格。
当前只有 P0 文档/静态审查允许执行，P1–P4 均为 `allowed_now=false`。
