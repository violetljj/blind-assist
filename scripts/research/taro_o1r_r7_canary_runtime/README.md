# TARO O1R R7 canary runtime

状态：`current / TARO_RESEARCH_MODULE / R7_FIT_LOPO_CANARY_PASS / R7_FRESH_CONFIRMATION_NOT_EVALUABLE_DUAL_CLASS_COVERAGE / HISTORICAL_EVIDENCE_READ_ONLY / NO_ACTIVE_EXECUTION`

## 稳定 Interface

- `positive_occupancy_factor.py` 与 `r7_canary.py`：实现冻结的 fail-safe positive-occupancy factor、LOPO canary 和 `UNKNOWN` retention。
- `fresh_confirmation_cohort.py`：只按 outcome-blind exclusion/identity 规则冻结 fresh parent cohort。
- `run_fresh_*.py`、`run_locked_fit_canary.py` 与 `validate_*.py`：执行并验证 HEAD、source inventory、Phase A/Phase B 和 evidence replay；动态授权仍回到 TARO current。

## 输出

正式 evidence 只写入 `artifacts.local/evidence/taro/` 下由各 execution lock 指定的 exclusive root；例如 R7 fresh confirmation 使用 `artifacts.local/evidence/taro/o1r-r7-fresh-confirmation-*/`。这些本地 evidence、模型输入和下载 payload 不进入 Git。

## 安全边界

source phase 不得读取 FARO truth、knownness 或 task outcome；`UNKNOWN` 不能当 negative。该 Module 不修改默认 App、不接 Android/HTP、不产生部署、产品或安全证明，也不能用已消费 parent 回救或晋级。

## 停止条件

任一 execution lock、parent/visit/frame identity、source/FARO firewall、hash、预算或 exclusive-root 条件不满足即 fail closed。R7 fresh confirmation 已消费并以 dual-class coverage `NOT_EVALUABLE` 终止；不得覆盖、续跑或事后更改门槛。动态状态以 [`docs/research/taro/README.md`](../../../docs/research/taro/README.md) 为唯一真源。
