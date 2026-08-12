# TARO O1R R8 clear runtime

状态：`current / TARO_RESEARCH_MODULE / R8_SELECTED_PHASE_B_COMPLETE / R8_SPARSE_RAY_INTERFACE_FAIL / R8_DENSE_TRUTH_OWNED_FALLBACK_EXECUTION_AUTHORIZED / DEFAULT_APP_UNCHANGED`

## 稳定 Interface

- `clear_enrichment.py` 与 `pool_cohort.py`：构造 outcome-blind clear-negative-control pool、source inventory 和 fixed cohort receipts。
- `run_top8_selection.py` 与 `run_selected_phase_b.py`：保持 source-only selection 与 FARO Phase B 的两阶段防泄漏边界。
- `ray_space_clear.py`：保留已失败的 sparse ray-space V1 作为不可改写的 compatibility 负证据。
- `truth_owned_fallback.py` 与 `run_truth_owned_fallback_canary.py`：对已有 source query 完全保留 prior dense label，只在 source query 缺失时构造 FARO-owned query。

## 输出

正式 evidence 只写入 `artifacts.local/evidence/taro/` 下 execution lock 指定的 exclusive root。当前 dense fallback root 为 `artifacts.local/evidence/taro/o1r-r8-dense-truth-owned-fallback-canary-r0/`；本地 FARO/source payload 与 evidence 不进入 Git。

## 安全边界

只能读取已选择的 8 parents / 133 FARO frames；禁止读取未选 FARO、重选 source、拟合 selector/threshold、训练或把 `UNKNOWN` 当 negative。该 Module 不修改默认 App、不接设备，也不支持 R8 promotion、部署、产品或安全主张。

## 停止条件

任一 lock/binding/hash、selected-cohort identity、source/result firewall、资源预算或 absent-root 条件漂移即停止。dense fallback public lock 当前为 `AUTHORIZED_UNCONSUMED`，只能消费一次且不得覆盖、resume、原地 repair 或重跑；动态状态以 [`docs/research/taro/README.md`](../../../docs/research/taro/README.md) 为唯一真源。
