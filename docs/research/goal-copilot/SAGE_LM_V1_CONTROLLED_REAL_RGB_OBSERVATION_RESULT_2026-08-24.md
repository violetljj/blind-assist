# SAGE-LM V1 Controlled Real-RGB Observation Result

状态：`DEVELOPMENT / CONTROLLED_REAL_RGB_OBSERVATION_IN_SIMULATED_GEOMETRY_LOOP / FAIL_OBSERVATION_UPLIFT`

## 问题与单变量

固定 SAGE-LM V0 的 exact identity firewall、bbox-center/scale baseline、运动步长、`near <= 0.82 m`、geometry confidence
门、LOST 停步以及连续两帧 completion，只把 synthetic noisy-bearing provider 替换为真实短单目 RGB 的
boundary + LK flow + frozen Depth Anything V2 metric-depth observation adapter。

24 个 episode 均衡覆盖 `ROOM_SIGN / QR_ENTRANCE / EXACT_SHELF_TARGET`，每类 6 个明显 anchor-aperture offset 与 2 个
control。真实部分是 ARKitScenes 室内场景、纹理、边界、相机运动与深度现象；exact QR/OCR-style anchor 是
controlled composited。source depth 仅用于自动选择 opening proxy 与 evaluator truth，`RgbObservationProvider` 只接收 RGB、
内参、0.24 m commanded baseline、active-pair index 和 exact-anchor observations。本 cohort 是 curated Development，不是自然分布或 Confirmation。
ARKitScenes trajectory truth 筛选出的实测横向 baseline 范围为 `0.186–0.295 m`（mean `0.234 m`），覆盖 14 个 sequence。

## 结果

| 指标 | bbox center + scale | SAGE-LM V1 RGB |
|---|---:|---:|
| target-front arrival | 7/24 (29.2%) | 2/24 (8.3%) |
| median lateral error | 0.219 m | 0.261 m |
| completion precision | 7/24 (29.2%) | 2/2 (100%) |
| premature arrival | 17 | 0 |
| controls retained | — | 1/6 |
| movement while LOST | 1 | 0 |

预设 8 条最低线只有 completion precision、premature arrival 与 `movement while LOST = 0` 成立；但前两项来自
强 abstention（仅 2 次 completion），整体为 fail，不能声称
真实 RGB 保留了 synthetic uplift。

## 失败分解

- 0/24 的 active-pair reciprocal LK flow confidence >= 0.5；flow survival 是首要失效层。
- 24/24 均输出 boundary pair，但 aperture-center absolute error 为 mean `0.289 m`、median `0.151 m`；association 仍不稳定。
- 17/24 的 depth consistency >= 0.5，但 metric range absolute error 为 mean `1.038 m`、median `0.801 m`；range 不能单独作为 authority。
- 只有 2/24 通过冻结的 `geometry_confidence >= 0.35`；失败发生在 observation 门前，不能通过改 policy、arrival 或 cohort 门槛救回。

## 结论与下一步

结论严格为：`CONTROLLED_REAL_RGB_OBSERVATION_IN_SIMULATED_GEOMETRY_LOOP / OBSERVATION_UPLIFT_NOT_PRESERVED`。
唯一后续是继续在同一 Development 输入上分解 observation：先检查 active-pair flow reciprocal survival，再修正 boundary pair
association，最后检查 monocular metric range。不得修改 SAGE-LM policy、baseline、阈值、semantic anchor、Android/default App，也不得写成真实导航。

本机可复现输出位于 `artifacts.local/evidence/sage-lm-v1/controlled-real-rgb-r1/`：`report.json`、
`observation_overlay.mp4`、`baseline_vs_sage_lm.mp4` 与 `trajectory_demo.png`。
