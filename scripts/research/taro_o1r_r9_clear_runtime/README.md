# TARO O1R R9 clear-enrichment runtime

状态：`current / TARO_RESEARCH_MODULE / R9_DEVELOPMENT_TRUTH_COMPLETE / R9_SELECTOR_FROZEN / NO_ACTIVE_EXECUTION / DEFAULT_APP_UNCHANGED`

## 稳定 Interface

- `clear_enrichment_fit.py`：只用 source-side feature 拟合并封存 clear-enrichment cohort selector；它不输出路径状态。
- `run_development_truth.py`：在已消费的 R8 remaining-16 development parents 上物化 FARO label，并冻结 selector development evidence。

## 输出

正式 evidence 只写入 execution lock 指定的 `artifacts.local/evidence/taro/` exclusive root。本 Module 的 R9 development root 已消费，不得覆盖、resume、原地 repair 或重跑。

## 安全边界

R9 只建立 fresh-cohort selection hypothesis。`UNKNOWN` 永远不是 negative；selector 不能读取 future cohort FARO、不能输出 `CLEAR/OCCUPIED`，也不授权训练、部署、产品或安全主张。

## 停止条件

任一 role、parent identity、source/truth firewall、selector seal、manifest/hash 或 one-shot 条件漂移即停止。动态状态以 [`docs/research/taro/README.md`](../../../docs/research/taro/README.md) 为唯一真源。
