# SAGE-LM V1-A All-Oracle Observation Ceiling Result

状态：`DEVELOPMENT / EVALUATOR_ONLY / ALL_ORACLE_CEILING_PASS / DOWNSTREAM_POLICY_CEILING_ESTABLISHED`

## 问题与冻结项

在 V1 相同的 24 个 curated ARKitScenes episode 上，`OracleApertureObservationProvider` 仅在显式 diagnostic arm
读取 evaluator truth：aperture center、aperture width、start range、source camera positions，并固定
`geometry confidence = 1`。输出原封不动进入既有 `_sage_lm()`。

movement、`near <= 0.82 m`、alignment tolerance、两帧 completion、baseline、cohort 与 V1 预设八条成功标准均未修改。
本臂不读取 RGB observation 结果，不运行 Hough、LK 或 Depth Anything，也不构成可部署 provider。

## 结果

| 指标 | bbox center + scale | SAGE-LM all-oracle |
|---|---:|---:|
| target-front arrival | 7/24 (29.2%) | **24/24 (100%)** |
| direction accuracy | 48.6% | **100%** |
| median lateral error | 0.219 m | **0.000 m** |
| completion precision | 7/24 (29.2%) | **24/24 (100%)** |
| premature arrival | 17 | **0** |
| controls retained | — | **6/6** |
| movement while LOST | 0 | **0** |

八条既有标准全部通过：arrival `>=18/24`、net gain `>=8/24`、median error `<=0.20 m`、error reduction
`>=50%`、completion precision `>=85%`、premature arrival `<=3`、controls `>=5/6`、LOST movement `=0`。

## 裁决与边界

Oracle 明显通过，因此 frozen SAGE-LM downstream policy 在这 24 个真实场景 truth geometry 上存在充分 ceiling。
V1 的失败可以定位于 observation adapter，而不是 downstream policy：当前 Hough boundary + edge-point reciprocal LK + single-frame metric-depth
组合没有保留 V0 synthetic mechanism signal。该结果不证明现有 RGB 中已经恢复了 aperture geometry，也不证明真实导航、
可通行性、安全或产品效果。

随后实现的 [`V1-B source-pose two-view`](SAGE_LM_V1_B_SOURCE_POSE_TWO_VIEW_BOUNDARY_GEOMETRY_RESULT_2026-08-24.md)
发现原 V1 materializer 把 ARKitScenes rotation-vector 列误当 camera positions；冻结 pair 仅 `2/24` 满足原 motion gate，故
V1-B 为 `NOT_EVALUABLE`，不能裁决 boundary/parallax observation 路线。新运行须先显式授权正确 source-pose 物化的 cohort；
不得用 raw B1/B2 outcome 调 detector、association 或 policy。

后续新授权的 [`V1-B-R2`](SAGE_LM_V1_B_R2_CORRECT_POSE_BOUNDARY_RESULT_2026-08-24.md) 已在 24/24 正确 pose pair 上执行：
B0=`24/24`，B1=`0/24` arrival，正式把当前主导失败层定位为 RGB boundary candidate extraction；association 仍被上游
candidate recall 混杂。

本机可复现输出：`artifacts.local/evidence/sage-lm-v1a/all-oracle-ceiling-r1/report.json`。
