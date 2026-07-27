# RCLE unseen external confirmation R0 metrics

状态：`PURE_DATA_LAYER / NOT_A_FORMAL_RUNNER / RGB_SELECTION_OUTCOME_FORBIDDEN`

本包只实现预注册后的确定性数据层：

- 输入是同一次底层 pair evaluation 产生的有序连续 expansion ledger；
- 用 `(source_id, sequence_id, window_id)` 作为窗口隔离身份；
- 在一次遍历中同时派生 strict `> 0.01/s` 的 old trigger 和三连续 pair 的 R1
  trigger；
- 窗口边界、弃权以及 `<= 0.01/s` 都会 reset；
- 固定分母保留全部 pair，包括弃权；
- 每个 source-consecutive pair 必须满足 `0 < dt <= 0.1s`；
- 恰好接收两个来源、每个来源恰好一个 positive 窗和一个 below-reference 窗；
- 逐窗计算 reduction、retention、first-trigger delay，逐来源计算 role direction，
  最终只对这些局部门求 AND；
- pooled 指标只放在 `pooled_diagnostics`，不会挽救任何局部失败。

缺失的 source-consecutive pair 必须以 `evaluable=false` 的 ledger row 表示。
`pair_index` 缺口、时间不连续或一个复合窗口在输入中分成多个 block 都是身份/顺序
错误，不能静默改变分母。

本包不读取图像、不运行 RCLE/LK/support manager、不选择来源或窗口、不写 claim，
也不实现正式 runner。它不能被用来改变冻结阈值、三 pair 长度、底层算法或在失败后
换窗补救。
