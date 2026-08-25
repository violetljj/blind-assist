# GRAIL-R1 Signature Observability Ablation Result

日期：2026-08-25（Asia/Hong_Kong）

状态：`DEVELOPMENT / EVALUABLE / MINIMAL_PRIVILEGED_SIGNATURE_IDENTIFIED / OBTAINABLE_RELATION_ACQUISITION_NEXT / FORMAL_TEST_UNOPENED / STOP_BEFORE_M2 / DEFAULT_APP_UNCHANGED`

## 问题

GRAIL-R0 已证明 privileged relation 能把 referent 从 `44/78` 提至 `75/78`、complete pose 从 `22/78` 提至 `57/78`。本轮不继续优化 R0，只回答：R0 signature 中哪些字段实际承载这项增益，以及下一 student 最少需要尝试恢复什么。

沿用同一已消费 ProcTHOR Development 78-case、candidate set、M1 V2b checkpoint、K=3 pose head、threshold=`0.9353410602` 和 evaluator。未训练模型、未打开 formal test。九组字段为 semantic type、support、room type、coarse height、native root/part 内的 2D sibling ordinal，以及 nearest stable object 的 type/direction/distance/height。

## 结果

| Signature 投影 | Referent | Complete | Wrong-target | Absence false commit | R0 uplift recovery（referent / complete） |
|---|---:|---:|---:|---:|---:|
| semantic only | 45/78 | 29/78 | 16/43 | 11/78 | 3.2% / 20.0% |
| semantic + support + room + height | 47/78 | 30/78 | 15/43 | 2/78 | 9.7% / 22.9% |
| 上述 + sibling ordinal | 74/78 | 56/78 | 0/43 | 2/78 | 96.8% / 97.1% |
| 上述 + nearest stable type | 75/78 | 57/78 | 0/43 | 0/78 | 100% / 100% |
| **semantic + sibling ordinal + nearest stable type** | **75/78** | **57/78** | **0/43** | **0/78** | **100% / 100%** |
| full R0 minus sibling ordinal | 48/78 | 31/78 | 15/43 | 0/78 | 12.9% / 25.7% |

完整 signature 中分别移除 semantic、support、room、height 或任一 nearby 子字段，当前指标都不变；这是字段冗余/交互，不是这些来源在其他分布上无用。所有 23 个投影变体的 candidate permutation 检查均为 `156/156`。

## 解释与下一实现

当前 cohort 的主导区分变量是 sibling ordinal，而不是精确方向、距离或高度。最小的 full-recovery 投影是：

```text
semantic object type
+ root/part grouping 内的 horizontal/vertical sibling ordinal
+ nearest stable object type sequence
```

这把 R1 student 的首个目标收窄为离散结构恢复：先从 reference/query RGB + masks 建立 object/part grouping 与 coarse ordinal，再预测邻近稳定物体的 semantic type；无需先回归 metric 3D relation。需要注意，R0 的 sibling ordinal 使用 ProcTHOR native `root_id` 决定哪些 part 属于同一根对象，因此仍是 privileged grouping，不能把本结果写成视觉可获得性已建立。

下一步只授权一个小型 obtainable structured-relation probe，输入限于 RGB、已有 masks、公开目标文本或可信环境图，输出对齐上述三组离散字段并接回冻结 selector/pose interface。仍不授权 threshold/backbone sweep、新 pose head、formal test、M2、Android 或默认 App。

## 证据 identity

- R1 ablation artifact SHA-256：`188157d70eadd37d6c4254812f9a805b2734a49c095152d2b3dc9d95d9a64217`
- schema：`blindassist_grail_r1_signature_observability_ablation_v1`
- terminal：`GRAIL_R1_SIGNATURE_OBSERVABILITY_ABLATION_COMPLETE`

本结果是同一 `PROJECT_CONSUMED_DEVELOPMENT` 上的 privileged-metadata diagnostic；staircase 归因依赖字段加入顺序，不建立 RGB/text obtainability、自然场景、formal generalization、产品或安全声明。
