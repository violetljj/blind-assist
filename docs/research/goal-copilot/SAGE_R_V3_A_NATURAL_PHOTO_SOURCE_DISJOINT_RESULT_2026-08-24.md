# SAGE-R V3-A Natural-Photo Source-Disjoint Result

状态：`DEVELOPMENT_STANDARD / 9_INDEPENDENT_NATURAL_SOURCES / 12_PREDECLARED_QUERIES / 36_DERIVED_OBSERVATIONS / INPUTS_SEALED_BEFORE_OCR_AND_MODEL / FROZEN_V3_A / V3_FULL_CORRECT_3_OF_36 / V2_CORRECT_4_OF_36 / V3_FULL_WRONG_LOCK_9 / V3_NO_RELATION_TARGET_0_OF_27 / PERMUTATION_36_OF_36 / DO_NOT_ENTER_V3_B / V4_NOT_RUN`

## 结论

Frozen V3-A 没有通过自然照片 source-disjoint Development。full typed relation arm 仍比 no-relation 多得到 3 个 evaluable
target lock，说明 relation path 在真实图上不是完全失效；但它没有超过 V2，而且把 relation evidence 多次分配给错误 physical
region：full correct terminal=`3/36`，低于 V2 的 `4/36`，wrong lock=`9`，NONE=`0/6`。因此当前证据不授权 V3-B。

| 36 个 natural-photo-derived observations | correct terminal | evaluable target recall | NONE | wrong lock | directory false binding | final query correct |
|---|---:|---:|---:|---:|---:|---:|
| substring FSM | 0/36 | 0/27 | 0/6 | 1 | 0 | 0/12 |
| V2 heuristic | **4/36** | 2/27 | **2/6** | **1** | **0** | **3/12** |
| V3 no-relation | 0/36 | 0/27 | 0/6 | **0** | **0** | 0/12 |
| **Frozen V3 full** | 3/36 | **3/27** | 0/6 | 9 | 2 | 1/12 |

no-relation 再次没有获得 target identity（`0/27`），而 full 为 `3/27`；但这次 relation uplift 与 9 个 wrong lock 同时出现，
不能再解释为“relational scorer 初步成立”。它只证明网络会使用 relation path，不证明学到的 relation vocabulary 能在自然 scene-text
statistics 上正确分配 identity authority。

full 与 no-relation 都在 candidate order reverse audit 上通过 `36/36`；full 最大同 ID score 误差为
`5.96e-7`。因此失败不是 candidate enumeration order，而是 representation / association 本身。

## 预冻结 cohort

cohort 在任何 OCR 或模型调用前封存：

- 9 张独立 Wikimedia Commons 自然照片；
- 12 个预声明 query；同一照片上的不同 query 明确视为 correlated；
- 每个 query 固定 `far_blur -> oblique -> native` 三个派生 view，共 36 observations；
- query、target tokens、truth、candidate region、scenario、变换、source/image SHA 全部先冻结；
- freeze receipt：OCR calls=`0`，model calls=`0`；
- 输入 manifest SHA-256：`0f972002bbe20d841afcac09801851f994d9204b0aed1363c65ef6fb1e69bdee`。

测试语义类型为 `PLATFORM / EXIT / EMERGENCY EXIT / LAB / CLASSROOM`，没有使用 synthetic generator 的
`ROOM / GATE / OFFICE / AREA` target type。candidate 是预标注 physical destination region，不是 detector proposal；本轮隔离检验
semantic evidence 到 candidate authority assignment，不检验 proposal generation。

这些分母不能包装成 36 张独立照片：只有 9 个独立 source，12 个 query 与 36 个派生 observation。近景 door-sign 照片也不提供
完整的多门 proposal 场景，因此它们主要攻击真实 OCR grouping 与 open-set behavior，不单独承担完整 physical-door grounding claim。

## 失败机制

1. **自然 OCR 缺 anchor 时 full 仍会锁候选。** Q01 只读到 `2`，Q04 连远处 `2` 也没有读到；full 却分别锁到 B，形成 wrong
   lock。当前 learned head 没有把 missing decisive token 可靠映射到 UNKNOWN/NONE。
2. **directional board 不在训练 relation vocabulary 内。** Q03 的 `EXIT / West` 位于柱面指示牌上，真实 destination 是箭头指向的
   A 区；full 最终锁 C，产生 2 个 directory false bindings。现有 14 类 edge 有 near/above/inside/left/right，但没有 arrow-to-
   destination 或 sign-to-region authority。
3. **真实 grouping/topology 与 synthetic prefix-number layout 不同。** RapidOCR 将 `2A 2B` 合并成一个 token；`323` 位于
   `COMPUTER LAB` 上方，`223` 位于 `Classroom` 上方，而非 generator 的同一行 prefix-number。Q09/Q11 最终均保持 UNCERTAIN。
4. **generic semantic token 在 absent case 产生虚假 identity。** Q10/Q12 请求 `LAB 406`，画面只有 `LAB 323` 或
   `LABORATORY`；full 最终错误锁 C/A，而 V2 没有 wrong lock。这表明 text feature + relation message 在 unseen semantic type 上会把
   partial lexical compatibility外推成 candidate authority。
5. **multiple-sign association 不稳定。** Q07 的 `EXIT B` 是 full 唯一 final correct query；同图 Q08 的
   `EMERGENCY EXIT` 被 full 判 NONE，V2/substring 则错指 B。网络尚未覆盖自然多牌、不同语序和目的地区域之间的 association。

因此下一算法动作不是增加 GNN 层数，也不是训练 belief。若另立 successor，应先改变 graph input representation：

- OCR line/token grouping，特别是上下布局、merge/split 与 decisive-token missingness；
- sign/candidate association，显式区分“文字所在牌”与“牌所指 destination”；
- relation geometry normalization，覆盖远距、透视、箭头方向与 candidate-region尺度；
- open-set scorer 对 unseen semantic type 和 missing identity token 的 abstention behavior。

本 sealed cohort 不因结果失败而换图、改 truth/candidate、调阈值或重跑。修复后的模型必须使用另一个预冻结 cohort；不能把本 36 条
重新包装成 held-out test。

## 冻结与复现

V3-A full/no-relation 权重、V2 scorer/belief/threshold 和 generator 都没有修改。冻结 hash：

| frozen artifact | SHA-256 |
|---|---|
| V3 no-relation | `4fb110b8518dec4800e9a7b113e84aaa56a5a6a065685105742a7c985c8cf5a2` |
| V3 full | `926cc835b9a33499fac210272723b2c986aab19aabfc0b08e52f154f47a8b454` |
| V2 source | `c6ecb7625c996982b04134c2824c8efe6026dab7da49b4c13c3a4893a28b95cb` |
| cohort spec | `41f0729c9bc2084bf90afcf5db81558cf6560eed62c906add8d41968bd8a3945` |
| freeze/evaluate module | `62ec9fa0133e690c64c76e3e4edf64473ecf7b089d1680e19dbc25deb968cd05` |

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1.natural_photo_v3_source_disjoint freeze `
  --spec scripts/research/goal_copilot_bridge/public_identifiable_referent_contract_v1/natural_photo_v3_source_disjoint_cohort.json `
  --source-dir artifacts.local/tmp/natural-v3-selection2 `
  --run-dir artifacts.local/evidence/sage-r-v3-a-natural-photo-source-disjoint/<new-cohort>

E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1.natural_photo_v3_source_disjoint evaluate `
  --cohort-dir artifacts.local/evidence/sage-r-v3-a-natural-photo-source-disjoint/cohort-20260824T204655+0800 `
  --run-dir artifacts.local/evidence/sage-r-v3-a-natural-photo-source-disjoint/<new-eval> `
  --runtime-root F:\ba-data\blindassist-artifacts-20260805\runtime\semantic-anchor-v1 `
  --no-relation-model artifacts.local/evidence/typed-semantic-referent-graph-v3/run-20260825T001500+0800-final/v3_no_relation.pt `
  --full-model artifacts.local/evidence/typed-semantic-referent-graph-v3/run-20260825T001500+0800-final/v3_full.pt
```

Evidence：

- `artifacts.local/evidence/sage-r-v3-a-natural-photo-source-disjoint/cohort-20260824T204655+0800/`
- `artifacts.local/evidence/sage-r-v3-a-natural-photo-source-disjoint/eval-20260824T204800+0800/`

| artifact | SHA-256 |
|---|---|
| `cohort-manifest.json` | `0f972002bbe20d841afcac09801851f994d9204b0aed1363c65ef6fb1e69bdee` |
| `cohort-seal.json` | `e5a05faade3b0705447bcc08bc4a26cf5e40ac29e6bd49553c7777682e7682aa` |
| `raw-decisions.json` | `3d48652bc50a2bc4bcd86aa6b1ce26858e1115ad11187d1688616c94670ae79f` |
| `final-report.json` | `5f599735dad169c2c457611bf7cf35a22d975a5522345f6f0e88cb4a87241bd3` |

## 决策与 claim ceiling

预声明 promotion checks 六项只有 permutation invariance 通过；状态为 `DO_NOT_ENTER_V3_B`。V3-B learned evidential belief 与
V4 active acquisition 都不启动。当前 claim ceiling：

`NINE_NATURAL_COMMONS_SOURCES_TWELVE_CORRELATED_QUERIES_THIRTY_SIX_DERIVED_OBSERVATIONS_SOURCE_DISJOINT_DEVELOPMENT_NO_POPULATION_GENERALIZATION_LEARNED_BELIEF_ACTIVE_PERCEPTION_ANDROID_NAVIGATION_SAFETY_OR_PRODUCT_CLAIM`
