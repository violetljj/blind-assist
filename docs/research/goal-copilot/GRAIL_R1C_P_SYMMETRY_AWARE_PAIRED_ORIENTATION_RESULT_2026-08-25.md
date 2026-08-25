# GRAIL-R1C-P Symmetry-Aware Paired Relative Owner Orientation Result

日期：2026-08-25（Asia/Hong_Kong）

状态：`FRESH_HOUSE_DISJOINT / FIXED_ZERO_SHOT_OA_V2 / PAIRED_RELATIVE_FINAL_FAILED / STOP_BEFORE_DEPTH_GEOMETRY / FORMAL_TEST_UNOPENED / DEFAULT_APP_UNCHANGED`

终态：`GRAIL_R1C_P_PAIRED_RGB_OWNER_ORIENTATION_NOT_ESTABLISHED_STOP_BEFORE_DEPTH_GEOMETRY`

## 结论

R1C-P 没有建立可用的 RGB owner orientation。唯一正式 arm `OA_V2_PAIRED_RELATIVE_FINAL` 在 fresh、source-disjoint 的 12 个 ProcTHOR val houses、78 个 positives 上得到：

| 指标 | 结果 | 冻结门 | 通过 |
|---|---:|---:|---:|
| cross-view canonical slot agreement | 37/78 | >=70/78 | 否 |
| referent top-1 | 36/78 | >=70/78 | 否 |
| complete pose | 29/78 | >=50/78 | 否 |
| wrong-target | 2/43 | <=1/43 | 否 |
| absence false commit | 0/78 | <=1/78 | 是 |
| permutation consistency | 156/156 | 156/156 | 是 |
| selector collateral vs privileged owner-local | 38 | 0 | 否 |
| complete collateral vs privileged owner-local | 26 | 0 | 否 |

同一 fresh cohort 上，evaluator-only native owner-local 坐标仍得到 referent=`74/78`、complete=`55/78`。因此任务、candidate set、冻结 M1 selector/pose head 与 owner-local mechanism 在 fresh houses 上仍有足够上界；失败集中在固定 OA-V2 RGB orientation evidence 到 canonical sibling slot 的转换，而不是下游匹配器或 pose head。

## 诊断边界

冻结 OA-V2 reference symmetry 为 `alpha=0:26 / alpha=1:4 / alpha=2:37 / alpha=4:11`。按协议，`alpha=0` 没有合法 canonical mode，只能 `UNKNOWN`；多 mode 必须一致，否则同样弃权。最终 arm 只有 `39/78` mode consensus、`39/78` mode unknown。即使不看下游，这已使正式 arm 无法达到 70/78 的 referent 门。

预注册的 independent-absolute arm仅作诊断：slot=`44/78`、referent=`41/78`、complete=`31/78`、wrong-target=`9/43`、absence=`0/78`。它也失败，且不得因数值高于 paired arm 而在 outcome 后晋升为正式路线。

本结果只说明：一个固定的 zero-shot Orient Anything V2 checkpoint，在一个 fresh、house-disjoint、synthetic ProcTHOR cohort 上，没有把 masked owner-group RGB 稳定变成冻结下游所需的跨视角 canonical sibling coordinate。它不否定 OA-V2 的通用物体姿态能力，也不建立自然图像、设备、产品或安全结论。

## 冻结身份

- ProcTHOR revision：`439193522244720b86d8c81cde2e51e3a4d150cf`
- val SHA-256：`d808540514e26b6726cd2790490e669b572eeb94febb5188a2f403591dd21721`
- collection SHA-256：`82883219f12be2afdc5077c865a41185a4bc2e68cd619d5b120985d389042d1e`
- OA-V2 code commit：`73b11c9dc83e84daeb563d0c766831f2c66b0a18`
- OA-V2 checkpoint SHA-256：`7b6b7f258d32b95123b9d023005ecca357d8ab944fb83476f532d3cf7a2295eb`
- OA-V2 predictions SHA-256：`f074c51ecf3899cbcbef5bcbb420d81747280ca90d5b3e369f17948b5b66d7d2`
- frozen M1 checkpoint SHA-256：`d838e8c1f648a771a41a32df7cbc0146b6bcebe98715fcd7f7c6c24ed7988b18`
- result artifact：`artifacts.local/evidence/grail-r1cp/result.json`
- result SHA-256：`16e67a1164f8535ea4b8b8a252f5c491b8ecbaaddaf997db23e4256172748d55`

模型接口依据固定为 [Orient Anything V2 官方代码](https://github.com/SpatialVision/Orient-Anything-V2) 与 [官方论文](https://arxiv.org/abs/2601.05573)。

## 停止条件

R1C-P 已消费。不得在该 cohort 上调整 symmetry handling、crop、mode consensus、threshold、selector、pose head、aggregation 或把 diagnostic arm 改成正式 arm；也不运行预定的 depth-geometry successor。若未来重开，必须改变独立信息源并另立 fresh、source-disjoint protocol。
