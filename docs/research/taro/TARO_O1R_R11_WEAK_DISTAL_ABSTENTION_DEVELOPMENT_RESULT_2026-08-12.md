# TARO O1R R11 weak-distal abstention development result

R11 只读重放已消费 R10 的 sealed source/label evidence，形成一个 truth-blind、development-only
候选。它不改变 R10 的正式 `NOT_EVALUABLE` 终态，也不产生 fresh confirmation 权限。

R7 base positive 是 2 个 confidence-2 连通像素、component height 至少 0.08 m、component
minimum-forward 不超过 2.0 m。R11 仍要求该 base cell，并要求以下相邻强度 margin 至少一个成立：

- 连通像素至少 16；
- component height 至少 0.15 m；
- component minimum-forward 不超过 1.5 m。

三项都不成立的 weak/low/distal positive 只转为 `UNKNOWN`，从不转为 `CLEAR`。该 predicate 不含
parent、video、frame、query、FARO、label 或 outcome 字段。

260 frames / 2,340 queries 的 R10 replay 中，候选只抑制两个 base positive：一个 definite
`CLEAR` 误报和一个 definite `OCCUPIED` 真阳性。候选得到 1,768 true positives、0 false positives、
occupied recall `0.989922`、precision `1.0`；query-level clear specificity 为 `13/13`，其单侧
95% Wilson 下界为 `0.827733`。occupied micro recall 相对 base 只下降 `0.000560`，8-parent macro
下降 `0.000698`。

但 R10 definite clear 仍只有 4 个 physical frames / 3 parents，frame-level Wilson 下界只有
`0.596521`；因此这些数字只是候选形成证据，不能写成 confirmation PASS。下一步必须使用冻结算法、
全新 parent-disjoint cohort 和 frame/parent-aware dual-class gates。新网络、source、模型和 FARO
读取仍未授权。
