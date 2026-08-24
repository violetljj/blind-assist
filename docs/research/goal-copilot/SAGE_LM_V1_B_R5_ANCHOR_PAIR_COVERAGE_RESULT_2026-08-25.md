# SAGE-LM V1-B-R5 Anchor-Conditioned Aperture Pair Coverage

日期：2026-08-25（Asia/Hong_Kong）

状态：`DEVELOPMENT / R5_DIVERSITY_AWARE_PAIR_COVERAGE_BELOW_R3 / R5S_SEQUENCE_DISJOINT_BOUNDARY_HEAD_REJECTED / R6_NOT_RUN / B2_NOT_RUN`

## 问题与冻结面

R5 复用 R3 官方 DeepLSD MegaDepth distance/orientation field、R2 正确 source pose、24 episode Development cohort、
9 px oracle localization、interpretation-plane triangulation、confidence、arrival 与 policy。唯一变化是 proposal 单位：
从 R4 的单条 3D boundary global top-96 改为完整的 `LEFT x RIGHT` aperture pair。pair score 以
`min(left coverage, right coverage)` 为主项，再结合 two-view field residual、anchor bracketing/adjacency、3D span 与
support continuity；96 个名额在 `(left_A, right_A, left_B, right_B)` 投影空间做 weighted farthest-point retention。
R5 不运行 B2，也不修改 support、localization 或 confidence threshold。

## R5 training-free 结果

| 指标 | R3 fragment fusion | R4 single-boundary top-96 | R5 pair top-96 | R5 目标 |
|---|---:|---:|---:|---:|
| true boundary pair available | 15/24 | 9/24 | **12/24** | >=18/24 |
| B1 geometry output | 13/24 | 8/24 | **12/24** | >=18/24 |
| B1 confident geometry | 0/24 | 1/24 | **0/24** | 非本轮调参目标 |
| missing | 9/24 | 15/24 | **12/24** | <=6/24 |

R5 相对 R4 恢复了 3 条 pair/4 条 geometry，但没有恢复 R3，更没有越过 18/24 gate。仅作 failure localization 的
512-pair budget diagnostic 达到 true pair/geometry=`15/24`、missing=`9/24`；它说明现有 parametric generator 内可找到的
pair 约回到 R3 ceiling，同时证明 96-pair learned-free selector 仍不能把 task-relevant boundary 稳定压缩到预算内。512 臂
不是正式成功臂，不改变 R5 裁决。

## R5S 小型 task-conditioned head

按 R5 失败前指定的 fallback，实现 8-channel 1D field/anchor feature、`24-24-2` Conv1D left/right head，直接以 source-native
left/right x 作分类监督。为防止 24 episode 记忆被写成监督收益，评估使用 leave-one-source-sequence-out：11 folds，
每 fold 训练 40–46 个 frame examples（另含 horizontal flip），held-out sequence 不进入训练。预测每个 role 的 top-8 x，
并把四个 role probability 加入同一 96-pair proposer；geometry/confidence 继续冻结。

| 指标 | R5 training-free | R5S sequence-disjoint head | R5S 目标 |
|---|---:|---:|---:|
| head top-8 四边界同时覆盖 | n/a | **5/24** | diagnostic |
| true boundary pair available | 12/24 | **11/24** | >=18/24 |
| B1 geometry output | 12/24 | **11/24** | >=18/24 |
| B1 confident geometry | 0/24 | **0/24** | 非本轮调参目标 |
| missing | 12/24 | **13/24** | <=6/24 |

11 folds 的 median train loss 从 `5.5458` 降至 `2.5753`，但 held-out top-8 四边界覆盖只有 `5/24`，最终 pair coverage
也从 12 退到 11。因此拒绝的是这个小样本、field-summary-only、sequence-disjoint head；它没有证明 task-specific
supervision 家族无效。不得把训练 loss 下降当成 boundary generalization，也不得在已打开的 24 episode 上改 head 宽度、
epoch、feature、top-k 或 loss 追 gate。

## Confidence contract 修正与下一边界

R3/R5 的 `support_length_px` 是 dense field 上的 supported-row count，而冻结 confidence 仍用
`min(line_support)/(0.55 * image_height)`，再与 verticality、range consistency、width score 组合，并在 B1 乘
`exp(-mean_pixel_distance/7)`。因此这些臂的 `0/24 confident` 不能被解释成所有已产生 geometry 都质量很差，也不授权降低
`0.35`。R6 dense-native boundary quality 只有在 candidate/geometry 先达到 `>=18/24` 后才有可辨识价值；本轮 coverage 未过，
故 `R6_NOT_RUN`。B2 association 同理保持 `B2_NOT_RUN / UNADJUDICATED`。

当前最窄结论是：generic salience 与 aperture-pair coverage 的 objective mismatch 仍成立，pair-level diversity 能部分救回 R4，
但当前 96-pair training-free compression 和 24-episode sequence-disjoint tiny head 都不足。下一次若继续 supervision，需要新的、
预先分离的训练 denominator 或更保真的 image/field representation；不能在本 cohort 上扫 detector/head。

本机证据：

- `artifacts.local/evidence/sage-lm-v1b-r5/anchor-conditioned-aperture-pair-b1-r3/report.json`
  (SHA-256 `DE0D954523C4E9B7E62E34CFC1F1806897F5E4E92050139C89A7FC1A685BE3E2`)；
- `artifacts.local/evidence/sage-lm-v1b-r5/diagnostic-pair-budget-512/report.json`
  (SHA-256 `9A45EFD7D2181EAF78BED11DB53CD0CC7087BC7BD4BF4CFC40D7DD2287FDD0CC`)；
- `artifacts.local/evidence/sage-lm-v1b-r5s/sequence-disjoint-boundary-head-b1-r2/report.json`
  (SHA-256 `0B9297840F3FC12274195B242208A3C089BAD7133C8741E8BC9C539D6BE6FE16`)。
