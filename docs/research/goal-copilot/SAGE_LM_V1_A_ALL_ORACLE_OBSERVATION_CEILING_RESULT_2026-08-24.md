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

唯一 successor 是固定 source-native pose 的 two-view aperture geometry：两帧独立 boundary、纹理带 support、左右 association、
triangulation 与 reprojection residual；Depth Anything 只作一致性支持。若该臂失败，关闭当前 boundary/parallax observation 路线；
若通过，才做 3–6 段明确标注 `real RGB + exact semantic anchor + source-pose-assisted` 的展示 canary。

本机可复现输出：`artifacts.local/evidence/sage-lm-v1a/all-oracle-ceiling-r1/report.json`。
