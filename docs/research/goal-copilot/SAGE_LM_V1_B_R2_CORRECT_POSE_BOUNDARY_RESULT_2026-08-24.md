# SAGE-LM V1-B-R2 Correct-Pose Boundary Result

状态：`DEVELOPMENT / EVALUABLE / B0_GEOMETRY_PASS / B1_RGB_BOUNDARY_EXTRACTION_FAIL / B2_ASSOCIATION_NOT_SEPARATELY_ADJUDICATED`

## 冻结 cohort

在读取任何 B0/B1/B2 outcome 前，R2 用 ARKitScenes 官方 `world-to-camera` trajectory 反演得到
`camera-to-world`，从 source 重新枚举 frame pair。24 个 episode 继续保持三类各 8、6 controls、同一 exact semantic
anchor、baseline、SAGE-LM policy、movement、八条 criteria 与 B0/B1/B2 定义；不运行 LK，也不使用 Depth Anything
metric range。

pair gate 固定为 lateral `0.18–0.30 m`、absolute forward `<=0.45 m`，并要求第一帧 aperture truth 投影到第二视图后，
两条 boundary 均位于 3% image margin 内且 aperture span 至少为图宽 10%。枚举 stride 为 3 frames，每个 source sequence
最多 4 条。最终 24 条来自 11 个 sequence，实际 lateral `0.1889–0.2847 m`，mean `0.2253 m`，最大 forward
`0.4101 m`；24/24 motion gate 通过，stored camera positions 与官方 pose 最大误差 `0.0 m`。冻结 manifest SHA-256：
`AB085770CB8EBD539B35DCD8C7F0BE8E4288BFAC96D2ABED26815484FF165CAB`。

## 结果

| arm | geometry output | confidence pass | target-front arrival | median endpoint error | controls | 八条 criteria |
|---|---:|---:|---:|---:|---:|---:|
| bbox center + scale | — | — | 5/24 | 0.290 m | — | — |
| B0 oracle pixels + pose | **24/24** | **24/24** | **24/24** | **~0.000 m** | **6/6** | **8/8** |
| B1 RGB + oracle association | 2/24 | 0/24 | 0/24 | 0.319 m | 0/6 | 2/8 |
| B2 RGB + automatic association | 14/24 | 5/24 | 3/24 | 0.196 m | 2/6 | 3/8 |

B1 有 21/24 episode 明确记录 `BOUNDARY_CANDIDATE_MISSING`。两帧都不是完全无 line：24/24 的 A/B frame
均检测到至少一条 vertical-ish line，但在 oracle boundary 的 9 px 门内通常没有对应 candidate。B2 在 10/24 为
`NO_POSE_CONSISTENT_PAIR`；其余 output 中包含错误边界组合，median aperture-center error `0.155 m`、median range error
`0.497 m`。B0、B1、B2 均确认 `uses_lk=false / uses_metric_depth=false`。

## 裁决

R2 的输入合同成立，故本轮可以裁决。B0 再次全过，排除 source pose、line-plane triangulation、3D boundary line 与
aperture center/width/range 实现为当前主导失败层。B1 在 evaluator association 下仍只有 2/24 geometry output、0/24
confidence pass，因此当前 real-RGB 主导瓶颈是 **boundary candidate extraction**。

B2 的 3/24 不能单独解释为 automatic association 机制失败：association 接收的候选集合已经被 B1 证明不具备足够
boundary recall，B2 的额外 output 主要来自 pose-consistent 但非真实 aperture boundary 的组合。只有先建立更高 recall、
仍不读取 evaluator identity/metric truth 的 RGB boundary representation，再在独立或明确 Development 数据上重做 B1/B2，
才可能单独裁决 association。

本结果只属于 controlled-composited exact anchor + curated ARKitScenes real scene/boundary/motion 的 Development evidence；
不建立自然 identity、可通行性、导航、安全、Android、P1 或默认 App authority。

本机证据：`artifacts.local/evidence/sage-lm-v1b/correct-pose-cohort-r2/cohort.json`、
`artifacts.local/evidence/sage-lm-v1b/source-pose-two-view-r2/report.json` 与 `observation_ladder.png`。

后续 [`V1-B-R3/R4`](SAGE_LM_V1_B_R3_R4_DENSE_BOUNDARY_RESULT_2026-08-25.md) 已执行：DeepLSD dense field 带来
partial recall rescue 但未过门；naive pose-conditioned joint-support accumulation 退化，B2 未运行。
