# P0-A2 Compact Ambiguity Policy Discovery result

状态：`DETERMINISTIC_COMPACT_POLICY_SEARCH_COMPLETE / COMPLEXITY_ONLY_BUYS_ABSTENTION / A1_INCUMBENT_RETAINED / NO_POLICY_ADMISSION / NO_SCIENTIFIC_VERDICT`

日期：2026-08-21

## 结论先行

P0-A2 没有找到在 A1 有用性约束内更好的 compact policy。确定性穷举的 3,237 个 hard-feasible unique commit
behaviors 全部满足 `20/20` resolvable coverage 与 committed correctness `>=85%`；其中最小 ambiguous
venue-parent macro 仍为 A1 的 `19.61%`，最优 hard-feasible policy 就是 A1 原两谓词 conjunction：

```text
brain_confidence >= 0.85
AND
candidate_center_dispersion <= 0.2423407460503519
```

A2 对 A1 的 parent-macro 绝对增益为 `0.00` percentage points，未达到预冻结的 `5pp` meaningful gate。

更复杂规则确实能继续降低 false commitment，但只能通过拒绝 resolvable episodes。最佳 relaxed diagnostic
使用 3 个 predicates：

```text
(candidate_center_dispersion <= 0.1897791937
 AND selected_score_margin >= -0.0406584144)
OR detector_top1_score >= 0.4348066449
```

它把 ambiguous parent macro 降到 `1.47%`、episode micro 降到 `2/51 = 3.92%`，但 resolvable coverage
只有 `13/20 = 65%`。其 committed correctness `12/13 = 92.31%` 只是更容易子集上的条件值，不能补偿丢失的
7 个 resolvable commitments。因此冻结终态为：

```text
COMPLEXITY_ONLY_BUYS_ABSTENTION
```

不接受 relaxed policy，不替换 A1，不启动 Sky，也不购买 fresh confirmation。

## 冻结搜索面

A2 hash-bind A1 protocol、result、两组 cohort、Terra decisions 与 frozen evaluator，并原样复用 A1 的 8 个
runtime features、feature direction 和 outcome-blind threshold grids。没有新增 parent、feature、模型调用或
threshold resolution。

DSL 只允许最多 3 个不同 feature predicates、boolean depth `<=2` 的单调形式：single predicate、AND/OR、
三项 AND/OR、`(A AND B) OR C` 与 `(A OR B) AND C`。同一 policy 禁止重复 feature。确定性穷举规模为：

| item | count |
|---|---:|
| syntactic policies | 518,570 |
| unique commit behaviors | 55,346 |
| hard-feasible behaviors | 3,237 |
| relaxed `65% <= coverage < 100%` behaviors | 20,587 |

主目标只有 ambiguous venue-parent macro false commit；episode micro、worst-parent rate、predicate count 与
canonical expression 只作依次 tie-break。搜索未拟合连续参数，也没有 stochasticity 或 LLM/Sky provenance。

## Worst-parent behavior

A1 incumbent 在 17 个 ambiguous parents 上的分布为：

| false-commit rate bin | parents |
|---|---:|
| `0%` | 10 |
| `(0%, 25%]` | 2 |
| `(25%, 50%]` | 4 |
| `>50%` | 1 |

worst-rate parent 是 `30CC Minnepoort`：`1/1 = 100%`。它只有一个 ambiguous episode，因此是有价值的
counterexample，不是稳定的 100% 风险估计。按错误数量看，最大的剩余负担是 `NTGent Café` 的 `4/8 = 50%`；
其次包括 `Maki Maki 2/8 = 25%`、`Au Coin Gourmand 1/2 = 50%`、`Bruxelles Accueil Porte Ouverte
1/2 = 50%` 与 `SuPe 1/3 = 33.33%`。

这些 residuals 说明单帧 confidence + proposal geometry 对某些 facade/venue 仍不足，但 A2 没有证明第三个
同类 threshold 能在不丢 coverage 时解决它们。后续若研究这些 parents，应把它们当 consumed counterexamples，
优先检查 active perception、persistence 或 multi-frame observation，而不是继续扩 threshold DSL。

## 终止与 claim boundary

- A1 继续作为唯一 retained Development incumbent；它仍未被准入 App、产品 prototype default 或 scientific baseline。
- P0-A2 不重跑、不加 feature、不放宽 100% coverage hard gate。
- relaxed 65%-coverage policy 只解释“复杂度可以换来拒绝”，不得称为更聪明或更安全的 winner。
- 当前 71 episodes / 21 parents 仍是 consumed Development；D2/D3 missing runtime evidence 不补值，不能外推到
  全部 92 episodes、其他城市、设备、天气或用户。
- 没有 A2 winner，因此本轮不自动授权 fresh scientific confirmation；Sky 继续待后续独立、明确的结构性问题。

## Evidence

- frozen protocol：[`P0_A2_COMPACT_AMBIGUITY_POLICY_DISCOVERY_PROTOCOL_V1.json`](P0_A2_COMPACT_AMBIGUITY_POLICY_DISCOVERY_PROTOCOL_V1.json)
- protocol file SHA-256：`57bbd97ff9822d4d9511c4c4a3c8c27ec8792828a7f3840987876d8d3eccd7a0`
- protocol content SHA-256：`1583af5c20489d18fe99aea538ecb6a115374be002414303668bc33b04a7aae0`
- result content SHA-256：`d155f885777322f5c0cb130b42b4675293d0d588218523713939e91776968601`
- result file SHA-256：`559490f2782bb4c5517812bd74a3b4a2b51132a36f00472dc42219d827d1e579`
- protocol freeze commit：`83371cd2`
- result artifact：`artifacts.local/evidence/p0-s0/2026-08-21-p0-a2-compact-ambiguity-policy-v1/result.json`

Claim ceiling：`CONSUMED_DEVELOPMENT_COMPACT_SYMBOLIC_POLICY_DISCOVERY_ONLY_NO_POLICY_ADMISSION_OR_SCIENTIFIC_VERDICT`。
