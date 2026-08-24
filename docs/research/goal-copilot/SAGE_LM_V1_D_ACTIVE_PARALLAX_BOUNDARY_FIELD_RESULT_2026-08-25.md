# SAGE-LM V1-D Active Parallax Boundary Field

日期：2026-08-25（Asia/Hong_Kong）

状态：`DEVELOPMENT / ACTIVE_PARALLAX_BELOW_R3 / R3_MISSING_RESCUE_0_OF_9 / PARALLAX_ROUTE_CLOSED / V1_E_PRIVILEGED_GEOMETRY_NEXT / R6_NOT_RUN / B2_NOT_RUN`

## 问题与实现

V1-D 不再从单帧 RGB 学习边界，而问：冻结的主动横移是否能把弱 aperture boundary 显现为 residual parallax
discontinuity。实验完全复用 V1-B-R2 的 24 episodes、exact anchor、source pose、9 px localization、R2 triangulation、
confidence、arrival 与 policy；不训练、不重采样 cohort，也不运行 R6/B2。

两张 RGB 使用冻结的 torchvision RAFT-Small `C_T_V2` 权重做双向 dense flow。相对 source pose 经
`H = K R K^-1` 转为 rotation-only pixel flow，并从观测 flow 中相减；forward/backward consistency 去掉明显错误对应。
随后对 horizontal residual-flow 与 residual-magnitude 的 x 梯度沿竖直方向累积，遮掉 composited anchor，在既有
aperture span/anchor-edge corridor 内形成 LEFT/RIGHT 各 top-8 x candidates，再原样进入 R2 oracle association 与
triangulation。

首个工程 r1 把 anchor 两侧错误当成 LEFT/RIGHT 语义；cohort 中 anchor 并不保证位于 aperture 内，因此该输出在主张前
作废。r2 改为由有序 aperture-pair hypotheses 赋予 LEFT/RIGHT role，固定后只运行一次完整 24 条评估。

## 结果

| 指标 | R3 DeepLSD | V1-D parallax | 差值/裁决 |
|---|---:|---:|---|
| 四边界 Recall@8 | n/a | **4/24** | 低覆盖 |
| true boundary pair available | **15/24** | **4/24** | -11 |
| geometry output | **13/24** | **4/24** | -9 |
| confident geometry | 0/24 | **0/24** | 不作主裁决 |
| missing | 9/24 | **20/24** | +11 |
| R3 missing rescued | n/a | **0/9** | 无互补信号 |
| R3 available lost | n/a | **11/15** | 明显 collateral |

四个 V1-D geometry 的 median center error=`0.0266 m`、median range error=`0.4489 m`。少数完整命中仍能进入冻结
triangulation，但主问题仍是候选 coverage。最关键的互补指标为 `0/9`：V1-D 没有救回任何 R3 missing，因此不启动
R3 + parallax 双通道。

## 裁决与边界

在当前 frozen 24-episode endpoint 条件下，RAFT-Small residual-parallax field 没有提供 R3 缺失边界的新信息，路线按
预定停止条件关闭。不能用 flow model、top-k、FB threshold、gradient、pair score、endpoint 或 9 px gate sweep 抢救本臂。
V1-C 同时正式关闭：held-out proxy recall 与真实 collapse 已说明现有标签生成器没有覆盖目标弱边界；不再训练 C0/C1、
换 backbone/loss 或重采样 336 个 proxy。

该负结果不泛化否定所有 active parallax。冻结 cohort 只约束 lateral/forward translation；endpoint 相对旋转 median
约 `20.1 deg`，且 `6/24` 超过 `30 deg`，部分 pair 的 forward/backward overlap 很低。因此结论仅覆盖当前 endpoint、
pose compensation、RAFT-Small 与候选构造。对本项目的实际决策仍然充分：当前运行时实现既未超过 R3，也没有救回任何
R3 missing，不值得继续融合或调参。

唯一 successor 为 V1-E privileged geometry supervision：从 ARKitScenes source-native mesh/depth（优先 Faro mesh
投影的高分辨率 depth）产生与 RGB line strength 独立的 weak-aperture boundary teacher，再训练 RGB(+anchor) student。
V1-E 必须先证明 teacher 的弱边界分母确实独立覆盖目标，不得把现有 opening proxy 换名复用。Android/P1/default App
保持不变。

本机证据：

- `artifacts.local/evidence/sage-lm-v1d/active-parallax-boundary-field-b1-r2/report.json`
  (SHA-256 `558BA8A5D382D2158D4247A41424E7689C3608F89BA2A83AC2A32F83671D9B8F`)；
- RAFT-Small checkpoint SHA-256 `01064C6DBA73B0FC9FC8EDF772248560A00A3ACFD62AC6677E9EEEBAD9680E27`。
