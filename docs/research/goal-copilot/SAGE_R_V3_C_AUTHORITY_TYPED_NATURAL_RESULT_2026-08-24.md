# SAGE-R V3-C Authority-Typed Natural Result

状态：`DEVELOPMENT_STANDARD / AUTHORITY_TYPED_SIGN_DESTINATION_GRAPH / CONSUMED_DIAGNOSTIC_11_OF_36_WRONG_2 / FRESH_6_SOURCES_11_QUERIES_33_OBSERVATIONS / V2_CORRECT_11_WRONG_1 / V3_A_CORRECT_14_WRONG_3 / V3_C_FULL_CORRECT_13_WRONG_5 / DIRECTIONAL_0_OF_3 / DIRECTORY_FALSE_BINDING_5 / CLOSE_NATURAL_SAGE_R / V3_B_NOT_RUN / V4_NOT_RUN`

## 结论

V3-C 完成了附件提出的 representation/input-contract pivot，但没有通过新的 source-disjoint natural cohort。它把图改为
`TargetToken / ObservedOCRToken / SignCarrier / DirectionCue / PhysicalCandidate`，先预测
`LABELS / POINTS / LISTS / NEAR / UNRELATED`，只有 `LABELS / POINTS` 可以产生 identity evidence；decisive token 缺失时
不得进入 identity head。V2 belief、NONE、novelty 和 terminal threshold 均未修改。

旧 9-source cohort 仅作 consumed representation diagnostics。最终 diagnostic 上 V3-C full 从 V2 的 `4/36` 提到
`11/36`，wrong lock 为 2（V3-A 为 9），directory false binding 为 0，证明新节点和 decisive gate 有开发信号；但这批数据
已参与表示修订，不能承担 fresh 结果。

新的 6-source / 11-query / 33-observation cohort 在任何 OCR 或模型调用前冻结，四臂结果为：

| fresh 33 observations | correct terminal | target correct | NONE correct | wrong lock | directional correct | directory false binding |
|---|---:|---:|---:|---:|---:|---:|
| V2 heuristic | 11 | 2/15 | 9/18 | **1** | **2/3** | 1 |
| frozen V3-A | **14** | **6/15** | 8/18 | 3 | 0/3 | 1 |
| V3-C no authority typing | 8 | 2/15 | 6/18 | 0 | 0/3 | 0 |
| **V3-C full** | 13 | **6/15** | 7/18 | **5** | 0/3 | **5** |

V3-C full 的 correct 虽比 V2 多 2，但 wrong lock 从 1 增到 5；方向牌没有得到任何正确 lock，directory exact mention 反而
造成 5 次错误绑定。candidate permutation 为 `33/33`，最大误差 0。因此失败仍是 sign-role/authority generalization，不是候选顺序。

按预先声明的最低条件，状态是 `CLOSE_NATURAL_SAGE_R`。不在 fresh cohort 上修改 carrier/cue proposal、truth、模型、阈值、
loss 或 denominator，也不进入 V3-B learned belief、V4 active acquisition、Android、P1 或默认 App。

## 失败机制

1. **relation type 没有从 synthetic feature distribution 迁移。** HKU 与 elevator directory 在自然 OCR 密度、透视和 carrier
   尺度下被 full head 多次判成可授予身份的 relation；authority vocabulary 正确，不等于 classifier 学会了自然 sign function。
2. **显式 cue 节点仍不足以产生 direction authority。** Way-out 样本的 direction cue proposal 与 RIGHT orientation 在冻结输入中
   明确存在，但 full 保持 uncertain；新节点存在不等于 relation-to-region evidence 足够强。
3. **decisive completeness 解决了部分 absent case，但不能单独解决 sign role。** `GATE 3`、`EXIT WEST`、`OFFICE 4` 的缺失决定性
   token 没有形成 wrong lock；然而目录中完整 decisive token 恰恰让错误 relation 更自信。
4. **无 authority typing 的 arm 不产生 wrong lock，不是成功。** 它仅有 `8/33` correct、target `2/15`，主要依靠弱/均匀证据与
   abstention；full 增加 recall 的同时把不可授权关系转成 candidate identity。

这轮直接否定的不是“是否可以列出更细关系名”，而是：仅靠小型 synthetic relation classifier、OCR grouping、预声明 carrier/cue
proposal 与 generic geometry，能否学到可迁移的 natural sign-function authority。答案是否定的。

## 冻结 cohort 与来源

新来源与旧 V3-A cohort、synthetic graph train 和 generated-pixel V2.1 均 source-disjoint：

- Skewen railway station Way-out direction sign，CC BY-SA 2.0；
- Gate Two airport fence sign，CC0 1.0；
- HKU K.K. Leung Building directory，CC0 1.0；
- New Academic Building elevator directory，CC BY-SA 2.0；
- Exit-route Braille sign，CC BY-SA 3.0；
- numbered modern-office hallway，CC BY 2.0。

candidate、carrier、direction cue proposal 与 cue orientation 是预声明输入，本实验不评价 proposal detection 或 cue recognition。33 条
observation 是 6 张来源的确定性变换，不是 33 张独立照片。

## 复现与 evidence

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1.authority_graph_v3_c_source_disjoint freeze `
  --spec scripts/research/goal_copilot_bridge/public_identifiable_referent_contract_v1/authority_graph_v3_c_source_disjoint_cohort.json `
  --source-dir artifacts.local/evidence/sage-r-v3-c/new-cohort-sources `
  --run-dir artifacts.local/evidence/sage-r-v3-c/fresh-cohort-r1

E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1.authority_graph_v3_c_source_disjoint evaluate `
  --cohort-dir artifacts.local/evidence/sage-r-v3-c/fresh-cohort-r1 `
  --run-dir artifacts.local/evidence/sage-r-v3-c/fresh-eval-r1 `
  --runtime-root F:\ba-data\blindassist-artifacts-20260805\runtime\semantic-anchor-v1 `
  --v3-a-model artifacts.local/evidence/typed-semantic-referent-graph-v3/run-20260825T001500+0800-final/v3_full.pt `
  --v3-c-model artifacts.local/evidence/sage-r-v3-c/consumed-diagnostic-r3/authority-relation-v3-c.pt
```

Freeze receipt 为 OCR calls=`0`、model calls=`0`。关键 hash：

| artifact | SHA-256 |
|---|---|
| V3-C model | `c2a8678f2d6cafb5d83125ba1a04a64f89d6d1611abbbb08fe50765073d7a060` |
| cohort manifest | `7fcf1cd4320b3c90acc94fb7b9c253be4faa24a3f0c974dbde3dae29f3db7296` |
| cohort seal | `cb74af2a181f651e209a6e1a07f625dcd37af69ee99653a515eb51e2a20848a9` |
| raw decisions | `8bdcf2b2c7cabc429192bdc81374e5be3a91f4324a288ecf139540400ff078d7` |
| final report | `0fe9bede75e177904d2f433952ce6666a146f5a34b591c96180d3cb862cc862d` |

## 当前边界

自然开放世界 SAGE-R learned graph 到此关闭。保留两个可用资产：

- QR/OCR exact-anchor 与 controlled Semantic Anchor V1，作为受控 demo 组件；
- V2/V3/V3-C harness、失败 cohort 和 typed relation diagnostics，作为论文中的机制演进与负结果。

下一算法主线不得继续修 natural SAGE-R；应从 Goal Copilot 的其他正信号能力重新选题。当前没有自动授权的新算法 lane，默认 App
保持不变。
