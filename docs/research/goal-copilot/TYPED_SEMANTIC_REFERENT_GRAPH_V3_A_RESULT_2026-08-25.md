# Typed Semantic-Referent Graph V3-A Result

状态：`DEVELOPMENT_STANDARD / SYNTHETIC_GRAPH_TRAIN / V2_1_RAPIDOCR_TUNED_ON_DEVELOPMENT / FULL_CORRECT_14_OF_16 / EVALUABLE_TARGET_7_OF_7 / NONE_7_OF_7 / WRONG_LOCK_0 / UNKNOWN_2_OF_2 / NO_RELATION_TARGET_0_OF_7 / V2_BELIEF_FROZEN / NATURAL_PHOTO_NOT_RUN / V3_B_NOT_RUN / V4_NOT_RUN`

## 结论

V3-A 已完成一次真正的 scorer 质变，而不是学习 V2 的四个手工权重。新模型显式建立 `TargetToken / ObservedOCR /
PhysicalCandidate` 三类节点、14 类有向 relation edge，并进行 3 层 relation-specific attention/message passing。每个 candidate
从自己的 graph state 与 target context 独立输出 identity evidence；另有 reliability 和 `NONE` head，但它们本轮只作诊断，
实际 sequential observability、novelty、belief、`NONE` construction 与 terminal threshold 全部继续使用冻结 V2。

| 同一 16 帧 V2.1 RapidOCR Development cohort | correct terminal | evaluable target recall | wrong lock | NONE | UNKNOWN | directory false binding |
|---|---:|---:|---:|---:|---:|---:|
| substring + FSM | 1/16 | 1/7 | 4 | 0/7 | 0/2 | 4 |
| V2 heuristic | 9/16 | 6/7 | **0** | 3/7 | 2/2 | **0** |
| V3 no-relation | 7/16 | 0/7 | **0** | 7/7 | 2/2 | **0** |
| **V3 full typed graph** | **14/16** | **7/7** | **0** | **7/7** | **2/2** | **0** |
| V3 full + learned belief | `NOT_RUN` | `NOT_RUN` | `NOT_RUN` | `NOT_RUN` | `NOT_RUN` | `NOT_RUN` |

两个 blur target 帧按冻结 V2 observability 正确保留 `UNKNOWN`，因此 full arm 的 14 个 correct terminal 加 2 个 expected
`UNKNOWN` 覆盖全部 16 帧。full 相对 V2 correct terminal `+5`，同时 wrong lock 维持 `0`；相对 no-relation correct terminal
`+7`、evaluable target `0/7 -> 7/7`。no-relation 的 `NONE=7/7` 不能包装成 identity 能力：它完全没有正确 target lock，说明
full 的增益来自 relation path，而不是 learned scorer 更愿意 abstain。

full arm 对 16/16 帧通过 candidate-order reverse audit，最大同 ID score 误差 `2.98e-7`；hard-neighbor wrong lock 为 0。

## 实现

节点：

- `TARGET`：目标 token text feature；
- `OCR`：原始/适配后的 text、confidence、polygon-derived box、token height；
- `CANDIDATE`：candidate geometry。

edge family：

- target↔OCR：edit、exact、partial-prefix、`O/0 I/1 B/8` confusion、distinctiveness；
- OCR↔OCR：same-line、left/right 和 distance；
- OCR↔candidate：above、inside、near、overlap、alignment 和 distance；
- candidate↔candidate：left/right competition；
- self edge。

每层用 relation-specific transform、edge projection、attention weight 和 permutation-invariant `index_add` aggregation；full 为
3 层、hidden width 64。训练使用 1,600 个 domain-randomized synthetic graph，独立 validation 320 个，30 epochs，seed 302；
覆盖 `ROOM/GATE/OFFICE/AREA`、相邻编号、suffix、directory、partial、confusion、absent、detached target text、token split/merge、
polygon/geometry/quality jitter。full validation present top-1=`1.00`，best loss=`0.08`；no-relation top-1=`0.36`，best loss=`1.89`。

V2 source SHA-256 保持：`c6ecb7625c996982b04134c2824c8efe6026dab7da49b4c13c3a4893a28b95cb`。

## 失败学习与数据角色

第一版训练目标在 candidate-negative 多数类下塌缩成全低分，V3 full/no-relation 都得到 target lock `0/9`。修复依据是 synthetic
validation 的 majority-class collapse：candidate positive-weighted BCE 加 present-graph ranking loss，没有修改 V2 belief 或测试阈值。

后续 full 在 synthetic validation 已到 top-1 `0.99`，但打开 V2.1 Development rows 后仍在 directory case 把 A、B 同时打高并
错锁 A。生成器随后补入“directory 与整组门空间分离”的 candidate-layout randomization，最终形成上表结果。因此：

- gradient training 仍只使用 synthetic graph；
- 但 V2.1 rows 已影响 generator coverage；
- 最终 `14/16` 必须标为 `TUNED_ON_DEVELOPMENT`，不是 held-out、source-disjoint 或 Confirmation evidence。

这个过程证明当前实现能够学到目标 relation，也明确说明下一轮不能再用这 16 帧签更高 claim。

## 复现

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest `
  scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1.test_typed_semantic_referent_graph_v3 `
  scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1.test_semantic_anchor_graph_and_belief_v2_1_real_ocr `
  scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1.test_semantic_anchor_graph_and_belief_v2

E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1.typed_semantic_referent_graph_v3 `
  --run-dir artifacts.local/evidence/typed-semantic-referent-graph-v3/run-20260825T001500+0800-final `
  --v2-1-raw artifacts.local/evidence/semantic-anchor-graph-belief-v2-1-real-ocr/run-20260824T223500+0800/raw-decisions.json `
  --train-count 1600 --validation-count 320 --epochs 30 --batch-size 64 --learning-rate 0.002 --seed 302
```

Evidence：`artifacts.local/evidence/typed-semantic-referent-graph-v3/run-20260825T001500+0800-final/`。

| artifact | SHA-256 |
|---|---|
| `raw-decisions.json` | `8b0707715e39136565eae909d90fd8f32e3fac1487a978c734c9aca32948eb67` |
| `final-report.json` | `94bfb479750cd6c1af193e04fc53d4906de082df21c5c7956ecf80b63cf717c2` |
| `v3_no_relation.pt` | `4fb110b8518dec4800e9a7b113e84aaa56a5a6a065685105742a7c985c8cf5a2` |
| `v3_full.pt` | `926cc835b9a33499fac210272723b2c986aab19aabfc0b08e52f154f47a8b454` |

## 下一动作与 claim ceiling

唯一 successor 是冻结 V3-A full 后，建立未参与训练和 generator 修订的自然照片 source-disjoint Development cohort：至少覆盖
真实 door/directory/office/gate/exit/area，并让 target type 与训练类型隔离。只有自然照片仍保留 relation uplift、wrong lock=0
且 no-relation 消融成立，才进入 V3-B learned evidential belief。V4 active acquisition 继续不启动。

Claim ceiling：

`SYNTHETIC_GRAPH_TRAIN_GENERATED_PIXEL_RAPIDOCR_TUNED_ON_DEVELOPMENT_NO_NATURAL_PHOTO_GENERALIZATION_LEARNED_BELIEF_ACTIVE_PERCEPTION_ANDROID_NAVIGATION_SAFETY_OR_PRODUCT_CLAIM`
