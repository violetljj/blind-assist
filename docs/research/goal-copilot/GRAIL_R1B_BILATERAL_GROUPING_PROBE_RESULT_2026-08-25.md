# GRAIL-R1B Bilateral Grouping Probe Result

日期：2026-08-25（Asia/Hong_Kong）

状态：`DEVELOPMENT / TRAINING_FREE / REFERENCE_OWNER_GROUP_74_OF_78 / VIEW_LOCAL_ORDINAL_NOT_CROSS_VIEW_ALIGNED / COMPLETE_35_BELOW_R1A_38 / FALSE_COMMIT_UNRESOLVED / FORMAL_TEST_UNOPENED / STOP_BEFORE_M2 / DEFAULT_APP_UNCHANGED`

## 问题与冻结边界

R1B 只改变 reference-side 观测合同：把 R1A 的 target crop 换成同一 reference pose 的 full-scene RGB、target proposal/mask 与全部可见 actionable proposals/masks。query grouping、expanded-context DINO affinity、确定性 ordinal、appearance collision tiebreak、M1 pose head、threshold=`0.9353410602` 与 evaluator 全部冻结。

同一 consumed Development 78-case 被 replay；reference target crop 32/78 逐像素完全一致，78/78 均满足冻结 replay 等价门。全体最大 mean absolute channel error=`0.205/255`、最大单通道误差=`2/255`；最终保存 317 proposals 与 317 masks。该数值只验证同一 renderer/pose 重放，不进入 scorer。

## Ownership 与 ordinal mechanics

| 指标 | Query R1A | Reference full scene |
|---|---:|---:|
| Same-root pair F1 | 97.3% | 99.1% |
| Different-root specificity | 39.5% | 83.6% |
| Exact partition | 62/78 | 69/78 |
| Target owner-group exact | — | **74/78** |
| Target bbox ordinal correct | 75/78 | **74/78** |
| Target mask-centroid ordinal correct | — | 70/78 |

full-scene reference 确实解决了 R1A 所缺的局部 ownership observation；mask centroid 没有进一步改善 ordinal。问题出现在下一层：R0 的 ordinal 是按每张 query 图像的 bbox 中心计算的，它不是跨视角 canonical relation label。

即使两侧都使用 privileged owner group，query/reference image-space oracle ordinal 也只一致 `54/78`；horizontal 与 vertical 分别为 `67/78`、`59/78`。obtainable bbox grouping 后 exact agreement 进一步降到 `48/78`。因此“reference 侧算出 LEFT，再在 query 侧找 LEFT”在观测合同上并不等价。

## 端到端结果

| Arm | Referent | Complete | Wrong-target | Absence false commit | R0 uplift recovery（referent / complete） |
|---|---:|---:|---:|---:|---:|
| Appearance-only M1 | 44/78 | 22/78 | 16/43 | 3/78 | 0% / 0% |
| R1A aligned crop context | 51/78 | 38/78 | 25/43 | 35/78 | 22.6% / 45.7% |
| **R1B full-scene RGB+bbox** | **47/78** | **35/78** | **11/43** | **29/78** | **9.7% / 37.1%** |
| R1B full-scene RGB+mask centroid | 45/78 | 33/78 | 12/43 | 29/78 | 3.2% / 31.4% |
| Privileged R0 ceiling | 75/78 | 57/78 | 0/43 | 0/78 | 100% / 100% |

bbox arm 保留 R0 selector rescues=`12/31`、complete rescues=`17/35`，但产生 selector collateral=`9`、complete collateral=`4`。虽然 wrong-target 与 absence false commit 均比 R1A 降低，referent/complete 也分别下降 `4`、`3`，故不能晋级为 selector。

31 个 bbox-arm referent failure 的互斥归因是：query grouping/ordinal=`3`、reference grouping/ordinal=`4`、两侧 view-local ordinal 都正确但跨视角 oracle label 不一致=`23`、ordinal 一致后的 collision/tiebreak=`1`。dominant failure 已不再是 reference ownership。

## 裁决与 successor

```text
GRAIL_R1B_REFERENCE_OWNERSHIP_HIGH_BUT_VIEW_LOCAL_ORDINAL_NOT_ALIGNABLE
```

R1B 是有信息量的负端到端结果：它验证 RGB full-scene ownership 在该 synthetic cohort 上基本可恢复，同时否证“两个视角分别计算 image-space ordinal 后可直接对齐”。当前 gap 修正为：

```text
obtainable owner group
        +
cross-view canonical / equivariant part coordinate
```

下一研究前门只能改变 relation coordinate 的信息源或合同，例如 owner-local canonical orientation、可靠 relative-pose canonicalization，或显式跨视角 part correspondence；必须另立 R1C 协议后才能执行。不得在本 artifact 调 affinity、mask rendering、DINO shift、threshold、fusion 或 pose head，也不得把 full-scene ownership 的高分写成端到端成功。nearest stable type、formal test、M2、Android/default-App 继续关闭。

## Evidence identity 与 claim ceiling

- result artifact SHA-256：`3068b322481626a7a9b923b38ee3fa8f7ea0e6752b6b4c28f68ef850afe79ac2`
- reference supplement SHA-256：`dece9861cb41f4320616c336717556325786ce154e9fbd1155ef9b8d0149e14b`
- schema：`blindassist_grail_r1b_bilateral_grouping_probe_v1`
- candidate permutation：两 arm 均 `156/156`

本结果仅为 `PROJECT_CONSUMED_DEVELOPMENT_TRAINING_FREE_REFERENCE_SOURCE_CHANGE_ONLY` synthetic ProcTHOR diagnostic，不建立自然 RGB、learned ownership、跨视角 canonical relation、formal generalization、Android、产品或安全 authority。
