# RCLE natural-session expansion Discovery R0

本模块执行
[`RCLE_NATURAL_SESSION_EXPANSION_DISCOVERY_R0`](../../../../docs/research/rcle/RCLE_NATURAL_SESSION_EXPANSION_DISCOVERY_R0_CONTRACT_2026-07-28.json)。

它只把现有 `ADVIO_WXYZ_TCAMIMU_VALIDMASK_CONTINUOUS_R3` 绑定到 metadata-only
冻结的 ADVIO sequence13、14、15、17。算法的 `0.01/s`、三连续 pair、连续
`PairState`、标定、去畸变和 resize 均不改变。sequence16 在所有入口都
fail-closed 为 `SEALED_UNSEEN_SESSION_ACCESS_FORBIDDEN`。

稳定顺序：

1. 用 `prepare_sources.py` 分别准备 13、14、17；15 用
   `--existing-archive` 与 `--existing-source-root` 绑定既有来源；
2. 用 `runner.py` 对每个 session 单独运行固定 frames `0..601`；
3. 四条全部结束后才用 `analyze.py` 生成 session-level result；
4. 用 `validate.py` 从四份 ledger 精确复算，并检查 sealed artifact 和禁用指标。

输出只进入
`artifacts.local/evidence/rcle_natural_session_expansion_discovery_r0/`。pair ledger
是 session 内纵向证据，不是独立样本；不得计算或声称 AUROC/F1、performance、
generalization、Android、产品或安全结论。
