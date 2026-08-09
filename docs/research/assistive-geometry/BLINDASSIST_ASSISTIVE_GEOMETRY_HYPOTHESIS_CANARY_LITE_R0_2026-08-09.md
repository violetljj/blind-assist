# BlindAssist Assistive Geometry hypothesis canary LITE R0

状态：`CANARY_LITE_PASS / MATH_MECHANICS_SUPPORTED / PAPER_NOVELTY_AND_LEARNABILITY_NOT_ESTABLISHED / A0_ROUTE_UNCHANGED`

日期：2026-08-09

## 决策先行

保留四个可证伪假设，优先级为 `H1 > H2 > H3 > H4`：

1. **H1：censored robust-contact survival geometry**，是最强 loss/输出分布假设；
2. **H2：profile-conditioned swept configuration clearance field**，是最强几何表示假设；
3. **H3：maximum-bottleneck corridor loss**，是有任务价值但最近邻碰撞较高的结构化 loss；
4. **H4：cluster-level one-sided conformal/CRC uncertainty**，数学上成立但当前 4 个
   calibration parents 不足，只保留为条件臂。

最有论文潜力的核心不是 survival、SDF、widest path 或 conformal 中任一零件本身，而是：

> 用可查询身体 profile 的 configuration-space geometry 表示稳健首碰距离，再以删失生存分布
> 统一 clearance/occupancy，以 corridor bottleneck 约束连通性，最后把低证据显式路由到 UNKNOWN。

本轮是 `WILD_LAB / CANARY_LITE`。它不修改冻结的 A0–A4，不激活 B2，不读取任何
TRAIN/DEVELOPMENT/CONFIRMATION outcome，也不改变正在运行的 A0 seed 17 或唯一 successor。

## 现行 B1 的结构缺口

现行 A2–A4 同时包含：

- 只在有限 intrusion truth 上生效的 clearance Huber；
- 三个 horizon 的独立 occupancy BCE；
- 无 intrusion 但 support 完整的 `censored-clear` 不进入 clearance regression；
- 固定 `0.42 m` half-width research profile 与固定 left/center/right 输出。

因此当前模型可以出现三类不是靠调 lambda 能消除的表示缺口：

1. `P(occupied within 1.0 m) > P(occupied within 1.5 m)` 一类 horizon 逻辑违例；
2. censored-clear 对连续 clearance 不产生梯度，clearance 与 occupancy 不是同一随机变量；
3. 固定 profile 输出不能回答“同一场景换一个身体宽度/余量后 clearance 如何变化”；
4. cell/band 平均相同的两个场景可以有完全不同的可通行连通性。

## H1：censored robust-contact survival geometry

### 数学对象

令 `T_q` 为同一 frozen body-swept support 上的稳健 `q-contact` 距离；当前 truth reader 的
`q=0.02` 不是严格第一点，所以不得把它写成未经修饰的 first contact。对 range bins
`0=r_0<r_1<...<r_K`，预测：

```text
h_k(x) = P(r_{k-1} < T_q <= r_k | T_q > r_{k-1}, x)
S_k(x) = product_{j<=k} (1 - h_j(x))
F_k(x) = P(T_q <= r_k | x) = 1 - S_k(x)
```

于是 `occupancy(r_k)=F_k` 由构造随 horizon 非减，clearance distribution 与 occupancy 不再是
两个可能互相冲突的 head。事件落入第 `m` 桶时：

```text
L_event = -log h_m - sum_{j<m} log(1-h_j)
```

support 已完整观测到最后 horizon 且没有 intrusion 时，才允许作为 right censor：

```text
L_right_censor = -sum_{j<=K} log(1-h_j) = -log S_K
```

deterministic invalid、深度洞、镜面失败、遮挡或不足 support 不是 censored-clear，必须 mask 并保持
`UNKNOWN`。否则 censoring 与危险共因，非信息删失假设不成立。

### 可证伪命题

在相同 encoder、参数量和输入预算下，H1 相对 `Huber clearance + independent BCE` 必须同时：

- horizon monotonicity violation 从结构上为 0；
- 降低 robust-contact NLL/CRPS 与 false-clear；
- 不通过 known coverage 塌缩或 UNKNOWN→CLEAR 获益。

任一项不成立，就否决 H1 的实际价值，不以“分布更优雅”救援。

### 新颖性边界

删失 likelihood、ordinal depth、ray termination probability 都已有成熟先例。本轮 primary-source
检索未发现同时覆盖“单目 RGB + body-swept q-contact + censored clear + UNKNOWN 分离 +
assistive false-clear”的工作；这只是 bounded search 结论，不是完整 novelty 证明。最近邻包括：

- [Survival Regression with Proper Scoring Rules, AISTATS 2022](https://proceedings.mlr.press/v151/rindt22a/rindt22a.pdf)；
- [Deep Ordinal Regression for Monocular Depth, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Fu_Deep_Ordinal_Regression_CVPR_2018_paper.html)；
- [Generating and Exploiting Probabilistic Monocular Depth Estimates, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Xia_Generating_and_Exploiting_Probabilistic_Monocular_Depth_Estimates_CVPR_2020_paper.html)；
- [DS-NeRF, 2021](https://arxiv.org/abs/2107.02791)。

新颖性风险：`MEDIUM_LOW`，四个假设中最值得优先。

## H2：profile-conditioned swept configuration clearance field

### 数学对象

令环境障碍集合为 `O`，身体 profile 为 `B_rho`，configuration obstacle 为：

```text
O_rho = O (+) (-B_rho)
C(rho, theta) = inf { t >= 0 : t * u_theta is in O_rho }
```

其中 `(+)` 是 Minkowski sum。若 `B_rho1 subset B_rho2`，则
`O_rho1 subset O_rho2`，因此必有：

```text
C(rho2, theta) <= C(rho1, theta)
```

这允许模型学习局部 obstacle/configuration field，再查询未参与训练的身体宽度、余量、band 或
horizon；fixed `3×3` head 则把 research profile 烙死在 label 中。第一轮只做 ground-aligned
2.5D observable support，不把单目输入包装成完整可观测 3D SDF。

### 可证伪命题

在 synthetic 2D/2.5D 和未来 fresh parent-disjoint Development 上，留出未训练 profile：

- profile-query arm 的 nesting violation 必须为 0；
- unseen-profile clearance/false-clear 必须优于参数匹配 fixed-head；
- observed profile 上不得显著回退。

否则 H2 只是更昂贵的重参数化。

### 数学陷阱和新颖性边界

普通 SDF、configuration-space field 与沿路径 sweep body-SDF 均不是新贡献：

- [Representing Robot Geometry as Distance Fields, ICRA 2024](https://calinon.ch/papers/Li-ICRA2024.pdf)；
- [Configuration Space Distance Fields, RSS 2024](https://www.roboticsproceedings.org/rss20/p131.pdf)；
- [Differentiable Composite Neural SDFs, 2025](https://arxiv.org/abs/2502.02664)。

新颖性只能落在单目 assistive perception、profile-queryable geometry、unseen-profile
generalization、q-contact/censoring 与 UNKNOWN 的组合。另一个必须避免的错误是 normalized
softmin：

```text
softmin_normalized(d) = -tau log((1/N) sum_i exp(-d_i/tau))
min(d) <= softmin_normalized(d) <= min(d) + tau log N
```

它可能高估最小 clearance 并制造 false-clear。canary 使用 hard min；后续训练若需 smooth
relaxation，必须给出保守方向或显式 `tau log N` 误差门。

新颖性风险：`MEDIUM`。

## H3：maximum-bottleneck corridor loss

### 数学对象

对局部 clearance/capacity field `c_v`，以及从近端边界到 horizon 边界的允许路径集合 `Gamma`：

```text
W(c) = max_{gamma in Gamma} min_{v in gamma} c_v
```

`W` 是 widest/maximin path capacity。它直接回答“是否存在身体宽度足够的连通 corridor”，而不是
只回答像素或 band 的平均误差。可加入 cellwise loss 之外的 bottleneck regression 或不对称
corridor false-clear hinge。

### 可证伪命题

在 cell/band BCE、MAE 匹配的场景中，加入 H3 必须降低：

- predicted corridor exists / truth corridor absent 的 path-level false-clear；
- bottleneck capacity error；
- thin barrier 与绕行 gap 的拓扑混淆。

### 限制和新颖性边界

本轮 DP 只允许 forward-monotone 邻接路径；它是 geometry-existence，不是用户意图路线。UNKNOWN
cell 不能静默当 clear 或 blocked。log-sum-exp 的 nested softmax/softmin 会高估 max capacity，
所以 smooth value 不能直接作为保守 clear 证据。

拓扑 loss、differentiable planning 与 max-min reachability 已高度拥挤：

- [Neural A*, ICML 2021](https://proceedings.mlr.press/v139/yonetani21a/yonetani21a.pdf)；
- [clDice, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html)；
- [DataSP, UAI 2024](https://proceedings.mlr.press/v244/lahoud24a.html)；
- [Widest-Path Reachability Fields, 2026](https://arxiv.org/abs/2607.07123)。

因此 H3 不能单独宣称“首次 differentiable widest path”。只有 body-profile bottleneck、
geometry false-clear 和 censored configuration field 的结合仍可能形成论文贡献。

新颖性风险：`MEDIUM_HIGH`；保留为结构化 loss 臂，不作为单独主论文命题。

## H4：cluster-level one-sided conformal/CRC uncertainty

### 数学对象

最小版本以 clearance overestimation residual `s_i = C_hat_i - C_i` 做 split conformal，取有限样本
order statistic `q_(1-alpha)`，并输出：

```text
L(x) = C_hat(x) - q_(1-alpha)
CLEAR at horizon h only if L(x) > h
```

在 exchangeability 下，`P(C < L) <= alpha`。因为 false-clear 事件
`{C<=h and L>h}` 是 `{C<L}` 的子集，所以可以控制固定 known denominator 上的**边际**
false-clear；它不保证 `P(error | predicted CLEAR)`，也不是每场景/每 parent 的 conditional safety。

更一般的 Conformal Risk Control 选择：

```text
lambda_hat = inf { lambda : n/(n+1) * R_hat_n(lambda) + B/(n+1) <= alpha }
```

参见 [Conformal Risk Control, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html)。

### 当前数据为何不够

parent/session 才是 exchangeable unit，不能把同一 parent 的帧伪装成独立 `n`。取 `B=1`、最好
情形 `R_hat=0`、当前 false-clear 目标 `alpha=0.08`，仍需：

```text
1/(n+1) <= 0.08  =>  n >= 12 independent calibration parents
```

当前只有 4 个 calibration parents，最好有限样本项也是 `1/5=0.20`，不可能签署 8% CRC；若
`alpha=0.05`，至少需 19 个独立 parents。并且 `false-clear/all-known` 若 denominator 随 UNKNOWN
threshold 改变，不保证关于 conservativeness 单调，不能直接套标准 CRC；必须固定 eligible
truth denominator，把 coverage 作为独立 gate，或使用经证明的多风险版本。

### 可证伪命题

只有在新 roster 提供足够 parent/session 单位后，H4 才允许检验：

- fixed eligible denominator 上的 parent-cluster expected false-clear 是否受控；
- coverage、false-block 与 worst-parent 风险是否仍满足独立门；
- shift/OOD 时是否正确进入 UNKNOWN，而不是继续引用 iid 保证。

vanilla conformal application 的新颖性风险：`HIGH`。只有 geometry lattice 上的
cluster-structured set、多个风险的可证控制或新的 shift-aware abstention 才可能降到 `MEDIUM`。

## CPU canary

实现：`scripts/research/assistive_geometry/run_hypothesis_canary_lite.py`

测试：`scripts/research/assistive_geometry/test_run_hypothesis_canary_lite.py`

运行：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.assistive_geometry.run_hypothesis_canary_lite `
  --output artifacts.local/evidence/assistive-geometry/hypothesis-canary-lite-r0/result.json
```

evidence：`artifacts.local/evidence/assistive-geometry/hypothesis-canary-lite-r0/result.json`

结果 SHA-256：`FDE9C9D800681F907466CE377DD512EA8E5B6C5893BB52813C596709D27683BD`

### 结果

| 检查 | 结果 | 解释 |
|---|---:|---|
| H1 hazard-derived occupancy | `0.5987 / 0.7793 / 0.9316 / 0.9500` | 由构造 monotone |
| H1 independent occupancy counterexample | `2` violations | 独立 logits 可违反 nesting |
| H1 right-censored NLL, low/high hazard | `0.19435 / 3.89631` | censored-clear 提供正确方向梯度 |
| H2 clearance at half-width `0.10/0.25/0.60/0.80 m` | `3.0(censored) / 1.6 / 0.8 / 0.6 m` | profile 变宽时 clearance 非增 |
| H3 two scenes, each three band means | `0.8333 / 0.8333 / 0.8333` | band aggregate 完全相同 |
| H3 blocked/routed widest capacity | `0.0 / 1.0` | bottleneck 正确分离连通性 |
| H3 soft relaxation error, `tau=.20/.05/.01` | `.15342 / .03774 / .00755` | 收敛但当前形式高估 capacity |
| H4 point/conformal false-clear all-known | `.13775 / .00130` | iid synthetic observation only |
| H4 conformal miscoverage / clear fraction | `.05275 / .31690` | 接近 5% 边际目标但付出 coverage |
| H4 shifted-test miscoverage | `.31995` | 分布漂移明确击穿 iid observation |
| H4 current 4-parent best finite term | `.20 > .08` | 当前真实 calibration 规模不可行 |

focused tests：`4/4 PASS`。runner 全部数学不变量通过；一次最终运行约几十毫秒 CPU 时间，
不是稳定性能 benchmark。

终态：

```text
MATH_MECHANICS_SUPPORTED_PAPER_NOVELTY_AND_LEARNABILITY_NOT_ESTABLISHED
```

## 明确否决的 standalone 主张

以下可以作为 control 或实现不变量，但不应再单独申报论文核心：

- crop/resize/K equivariance；
- pseudo-spherical/camera-ray representation；
- 普通 SDF sweep；
- vanilla conformal thresholding；
- 未与 body profile、censoring 或 false-clear 绑定的普通 topology loss。

前两者已被 [ICCV 2023 depth/normal equivariance](https://openaccess.thecvf.com/content/ICCV2023/html/Zhong_Improving_Equivariance_in_State-of-the-Art_Supervised_Depth_and_Normal_Predictors_ICCV_2023_paper.html)、
[UniDepth, CVPR 2024](https://avi.ethz.ch/publications/2024/UniDepth/UniDepth_CVPR2024.pdf) 和
[UniK3D, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Piccinelli_UniK3D_Universal_Camera_Monocular_3D_Estimation_CVPR_2025_paper.html)
强覆盖。

## 下一步与停止门

本轮完成后停止，不在 A0 训练期间修改或运行 A1–A4。A0 三 seed 与真实 Development 终态齐备后，
若用户继续本方向，最有信息增益的下一轮是独立的 synthetic/Development learnability 设计：

1. 首先只比较 current direct-head 与 `H1 survival`，验证同一 q-contact truth/censor mask；
2. 再加入 H2 的 held-out profile query；
3. H3 只作为额外 loss ablation，不把 widest path 本身写成创新；
4. H4 在获得至少 12 个独立 calibration parents 前不得签署 `alpha=0.08` CRC。

任何平均改善若来自 coverage collapse、UNKNOWN→CLEAR、将 invalid 当 right-censored clear、将帧当
exchangeable parent、soft relaxation 高估 clearance 或复用已消费 outcome，立即 FAIL。

## Claim ceiling

本轮只证明四类数学机制在 deterministic synthetic CPU counterexample 上按实现成立，并识别了
若干必要反例。它不证明网络可学、真实数据有效、论文 novelty 完整、Development/Confirmation
通过、设备可运行、产品可用或真实助盲安全。
