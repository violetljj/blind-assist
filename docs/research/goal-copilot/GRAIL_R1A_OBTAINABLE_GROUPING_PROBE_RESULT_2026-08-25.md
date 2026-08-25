# GRAIL-R1A Obtainable Grouping Probe Result

日期：2026-08-25（Asia/Hong_Kong）

状态：`DEVELOPMENT / TRAINING_FREE / QUERY_GROUPING_HIGH_BUT_LOW_SPECIFICITY / REFERENCE_ORDINAL_PARTIAL / FALSE_COMMIT_UNRESOLVED / R1B_REFERENCE_SIDE_GROUP_SOURCE_ONLY / FORMAL_TEST_UNOPENED / STOP_BEFORE_M2 / DEFAULT_APP_UNCHANGED`

## 问题与输入边界

R1A 只问：不读取 ProcTHOR `root_id`，能否从现有可获得输入恢复 same-root grouping、确定性 sibling ordinal，并把一部分 R0 relational uplift 接回冻结 selector/pose interface。

复用同一 consumed Development 78-case、candidate set、M1 V2b checkpoint、K=3 pose head、threshold=`0.9353410602` 与 evaluator。输入只有既有 query/reference RGB、simulator candidate bbox proposal、semantic candidate type、frozen DINO/M1 feature。collection 只保存 `mask_area`，没有保存逐像素 mask；本结果因此明确是 `RGB + bbox` probe，不冒充 mask-based grouping。

training-free grouping 对同类型 candidate 使用扩张 bbox 局部接触，并以 expanded-context DINO cosine 作弱 gate；connected component 内的 bbox centroid 直接计算 `LEFT/CENTER/RIGHT × TOP/MIDDLE/BOTTOM`。reference crop 没有完整 candidate proposals，因此以三种固定 carrier 比较目标 ordinal：冻结 appearance 对照、same-position DINO context、允许 ±2 token shift 的 DINO context。没有训练、threshold sweep 或 outcome 后 affinity rescue。

## Grouping mechanics

| 指标 | 结果 |
|---|---:|
| Same-root pair TP / FP / FN / TN | 986 / 52 / 2 / 34 |
| Same-root pair precision / recall / F1 | 95.0% / 99.8% / 97.3% |
| Different-root specificity | **39.5%** |
| Pair balanced accuracy | 69.7% |
| Exact query partition | 62/78 |
| Candidate ordinal | 372/419 |
| Query target candidate ordinal | 75/78 |

高 F1 主要由 Drawer 的大量 same-root positive pairs 主导；52 个 false-positive merges 表明当前局部接触规则不能可靠分开邻近的同类型不同 root。不能只报告 F1 并声称 grouping 已解决。

## 端到端结果

| Arm | Referent | Complete | Wrong-target | Absence false commit | R0 uplift recovery（referent / complete） |
|---|---:|---:|---:|---:|---:|
| Appearance carrier control | 45/78 | 32/78 | 30/43 | 38/78 | 3.2% / 28.6% |
| **Aligned-context DINO** | **51/78** | **38/78** | **25/43** | **35/78** | **22.6% / 45.7%** |
| Shift-2 context DINO | 51/78 | 38/78 | 25/43 | 36/78 | 22.6% / 45.7% |

aligned-context arm 超过 M1 complete=`22/78` 和 B1=`23/78`，但只恢复 R0 35 个 complete uplift 中的 16 个净数量；31 个 R0 selector rescues 仅保留 `11/31`，35 个 R0 complete rescues 保留 `18/35`，同时产生 selector collateral=`4`、complete collateral=`2`。高 wrong-target 与 absence false commit 使它不能成为干净的 obtainable selector。shift invariance 没有带来增益。

27 个 referent failure 的归因为：query grouping/ordinal error=`11`；query grouping 正确后 reference context ordinal error=`15`；两侧 ordinal 正确后 collision/appearance tiebreak=`1`。因此当前主要缺口不是确定性 rank 公式，而是两侧可对齐的 group ownership，尤其是 reference crop 缺少完整 scene/proposals。

## 裁决与下一步

```text
GRAIL_R1A_QUERY_GROUPING_HIGH_REFERENCE_ORDINAL_PARTIAL_FALSE_COMMIT_UNRESOLVED
```

R1A 建立了一个窄的 partial signal：空间对齐 context 比 appearance carrier 多 `+6` referent、`+6` complete，但没有达到无 collateral 的 R0 机制，也没有达到建议的 60% complete-uplift recovery。当前 artifact 上继续调 bbox dilation、DINO shift、affinity、threshold 或 fusion 只是在缺失 reference-side group observation 时救 matcher，故关闭。

唯一 successor 是 R1B reference-side group source：新版本必须保存 full-scene reference RGB 与 candidate masks/proposals，或引入独立 part-owner/whole-object signal；随后对 reference/query 两侧运行同一 grouping 与确定性 ordinal，并继续冻结 pose head、threshold 和 evaluator。nearest stable type、M2、formal test、新 pose head、Android/default-App 仍不启动。

## Evidence identity 与 claim ceiling

- artifact SHA-256：`a9428da7356815b1d1bcb9f05833243ebb06bdce0d6d2794aac7ebba25ac2300`
- schema：`blindassist_grail_r1a_obtainable_grouping_probe_v1`
- selected arm：`ALIGNED_CONTEXT_DINO`
- candidate permutation：三个 arm 均 `156/156`

本结果仅为 `PROJECT_CONSUMED_DEVELOPMENT_TRAINING_FREE` synthetic ProcTHOR diagnostic，不建立自然 RGB、learned grouping、formal generalization、Android、产品或安全 authority。
