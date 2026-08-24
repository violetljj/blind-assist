# SAGE-LM V1-B Source-Pose Two-View Boundary Geometry Result

状态：`DEVELOPMENT / NOT_EVALUABLE_SOURCE_POSE_PAIR_CONTRACT_INVALID / NO_BOUNDARY_ROUTE_ADJUDICATION`

## 冻结问题与实现

V1-B 在原 24 episode、baseline、frozen `_sage_lm()`、movement、`near <= 0.82 m`、alignment、两帧 completion、
LOST stop、6 controls 与原八条 criteria 上实现三个同跑 arm：

- B0：evaluator boundary pixels + ARKitScenes source-native pose；
- B1：RGB LSD line candidates + evaluator association + source-native pose；
- B2：RGB LSD line candidates + automatic pose-constrained association + source-native pose。

两帧独立检测 line feature；image line 形成 interpretation plane，两平面相交恢复 3D boundary line，再输出 aperture
center/width/range/confidence。核心 observation path 完全不运行 LK，也不使用 Depth Anything metric range。B2 provider
不接收 evaluator truth，exact anchor 只限定搜索邻域，不创建或重绑定 identity。

## 前置 truth-authority 失败

V1-B 用仓库已有的 ARKitScenes 官方 pose reader 复核原始 `lowres_wide.traj`，发现 V1 materializer 把列约定解释反了。
官方格式是 `timestamp + world-to-camera rotation vector + translation`；原 materializer 却把 rotation-vector 三列直接写成
`camera_positions_m`，并把 translation 三列送入 `Rodrigues`。因此 V1 原报告的 active-pair 实测横向 baseline
`0.186–0.295 m` 主张无效。

同一冻结 frame pair 用官方 camera-to-world pose 重算后：

- 只有 `2/24` 满足原定 `0.18–0.30 m lateral AND forward <= 0.45 m`；
- 实际 lateral 范围 `0.015–0.899 m`，mean `0.144 m`；
- 最大 forward delta 为 `2.009 m`；
- 只在 `13/24` 既有 16-frame window 内存在任何合格替代 frame B。

所以不能在“不改 24 episode/window”的约束下通过 metadata repair 恢复完整有效分母，也不能挑 13 条改写成 24 条结果。

## Raw diagnostics（不作 B1/B2 裁决）

| arm | geometry output | confidence pass | target-front arrival | median lateral error | adjudication |
|---|---:|---:|---:|---:|---|
| B0 oracle pixels + pose | 23/24 | 23/24 | 23/24 | ~0.000 m | geometry implementation diagnostic pass |
| B1 RGB + oracle association | 1/24 | 0/24 | 0/24 | 0.343 m | **not adjudicated** |
| B2 RGB + automatic association | 18/24 | 3/24 | 2/24 | 0.462 m | **not adjudicated** |

B0 的八条 raw criteria 全过，支持 line-plane triangulation 与官方 pose 坐标实现本身可工作。B1/B2 的 raw 数值混入
无/弱视差、过大 forward motion、越过 aperture 或视野不重叠，不能解释成 boundary extraction 或 association 失败。

## 裁决与停止规则

正式裁决是：`NOT_EVALUABLE_SOURCE_POSE_PAIR_CONTRACT_INVALID`。这不关闭 source-pose two-view boundary 路线，也不建立
V1-B uplift。当前约束下无可执行 successor；只有显式授权一个在 outcome 前用正确官方 pose 物化、冻结 motion gate 的新
24-episode cohort，才能重新运行同一 B0/B1/B2。不得用本 raw outcome 调 detector、association、confidence、policy、criteria 或挑帧。

future materializer 已改为官方 world-to-camera inversion，防止再次把 rotation vector 当 camera position；现有 consumed cohort
与结果没有被覆盖或重写。

本机证据：`artifacts.local/evidence/sage-lm-v1b/source-pose-two-view-r1/report.json` 与 `observation_ladder.png`。
