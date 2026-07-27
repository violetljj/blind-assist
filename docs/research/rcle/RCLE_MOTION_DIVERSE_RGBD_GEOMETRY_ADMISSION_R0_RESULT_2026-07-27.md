# RCLE motion-diverse RGB-D geometry admission R0

日期：2026-07-27

## 结论

正式终态：

`NOT_EVALUABLE_NO_RGB_NO_REPLACEMENT / VALID`

rank-one ETH3D SLAM `desk_3` 的七个、10 秒、outcome-blind 冻结候选窗中，
`POSITIVE_APPROACH_WINDOW=0`、`BELOW_TRIGGER_REFERENCE_WINDOW=0`、
`AMBIGUOUS_OR_INELIGIBLE=7`。因此无法冻结要求的
`2 positive + 2 below-reference` 四窗身份。

本轮到此停止：

- RGB bytes accessed：`0`；
- candidate replacement：`false`；
- post-outcome windows added：`0`；
- algorithm changed：`false`；
- 默认正式并行度：`8 workers`。

这不是算法失败结论，也不授权性能、Android、人体、安全或产品资格判断。
它只说明当前 rank-one 来源不能承担冻结的 geometry admission 角色组合。

## 冻结与执行顺序

1. metadata-only 排名先冻结为 ETH3D `desk_3`、OpenLORIS corridor、
   ICL-NUIM `lr kt3`；trajectory 和描述均无角色授权。
2. rank-one payload 前，burned Floor3_2 W3 smoke 通过；执行模板统一
   Decimal/float 比较为预冻结 `rel_tol=1e-12`、`abs_tol=1e-15`，并覆盖
   偶数样本 median。
3. 通过稳定 root adapter 只取得 `desk_3` pose、calibration、depth index
   及七个冻结 10 秒窗所需的 1895 个 depth members；未取得 RGB。
4. 在任何 geometry metric 前，冻结继承的 Floor3 门、7 个完整候选窗、
   implementation lock 和 8-worker execution claim。
5. 正式运行生成 1888 个固定分母 pair rows；独立验证器不调用 geometry
   生成器，重新计算 band、coverage、连续时长、角色、2+2 选择和终局并通过。

metadata 阶段所说的“冻结四窗”指最终只有满足门的
`2 positive + 2 below-reference` 才能冻结为四窗身份；候选身份集合在看任何
geometry outcome 前已冻结为七个完整、连续、不补加的 10 秒窗。

## 结果概览

| 窗 | coverage | positive fraction | below fraction | 最长 positive | 最长 below | 角色 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.9188 | 0.4686 | 0.4317 | 1.069 s | 1.733 s | ambiguous |
| 1 | 0.9407 | 0.4037 | 0.5037 | 1.253 s | 1.475 s | ambiguous |
| 2 | 1.0000 | 0.7889 | 0.1926 | 3.428 s | 0.774 s | ambiguous |
| 3 | 1.0000 | 0.4834 | 0.4945 | 2.396 s | 2.064 s | ambiguous |
| 4 | 0.7256 | 0.3759 | 0.3459 | 1.106 s | 1.253 s | ambiguous |
| 5 | 0.9407 | 0.2481 | 0.6815 | 1.253 s | 2.691 s | ambiguous |
| 6 | 0.9556 | 0.3185 | 0.6111 | 1.511 s | 2.912 s | ambiguous |

角色门要求 coverage、固定分母目标 band fraction 均至少 `0.8`，且对应最长
连续时长至少 `5.0 s`。窗 2 的 positive fraction 接近但仍低于 `0.8`，最长
positive 也仅 `3.428 s`；不得按接近程度提升为角色。窗 5/6 的 below
比例和连续时长同样不足。

## 可复核证据

- burned smoke receipt：
  `artifacts.local/evidence/rcle_motion_diverse_rgbd_geometry_admission_r0/burned_fixture_smoke_r0.json`
- execution claim：
  `artifacts.local/evidence/rcle_motion_diverse_rgbd_geometry_admission_r0/geometry_execution_claim_r0.json`
- fixed window identity：
  `artifacts.local/evidence/rcle_motion_diverse_rgbd_geometry_admission_r0/desk_3_window_freeze_r1_inherited_dt_correction.json`
- pair ledger：
  `artifacts.local/evidence/rcle_motion_diverse_rgbd_geometry_admission_r0/formal_geometry_r0/geometry_pair_ledger.jsonl`,
  SHA-256 `151a6cc7646f7438d7038442e409a7d85a0de0e9d73f67ab378fd77afffa4454`
- selection：
  `artifacts.local/evidence/rcle_motion_diverse_rgbd_geometry_admission_r0/formal_geometry_r0/geometry_selection.json`,
  SHA-256 `67900ee697d1e16fb6923b1b9ef1d9bc47847d0efaa6105554ff70b7520c312d`
- terminal result：
  `artifacts.local/evidence/rcle_motion_diverse_rgbd_geometry_admission_r0/formal_geometry_r0/result.json`,
  SHA-256 `b46b0fbf1ef44cbd6f62d632a6a6fbd50828c24515c69fdc9b54e1e8ff08d020`

独立验证命令返回：
`PASS / NOT_EVALUABLE_NO_RGB_NO_REPLACEMENT / selected_window_indices=[]`。
模板与 transport 单测共 5 项通过；project-structure 自身 smoke suite 通过。
仓库全量 structure check 仍报告 9 个既有的 USTRF/Floor3 私有 Implementation
路径引用；本轮新增 root transport test 已进入 reviewed allowlist，本轮未新增
structure failure。

## 下一阶段边界

本 protocol 已消费并终止。不得在本轮改门、补窗、换候选或下载 RGB。
若未来另立来源 discovery，必须是新的、预注册的证据版本；只有某个新来源
先通过 geometry 并冻结四窗，才允许取得对应 RGB。cross-source holdout、
性能、Android 和产品资格继续分别立项。
