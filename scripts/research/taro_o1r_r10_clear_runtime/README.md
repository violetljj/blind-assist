# TARO O1R R10 fresh-pool runtime

状态：`current / TARO_RESEARCH_MODULE / R10_FRESH_32_PARENT_POOL_FROZEN / R10_ZERO_BODY_HEAD_96_OF_96_PASS / R10_SOURCE_DOWNLOAD_96_OF_96_PASS / R10_INVENTORY_AUTHORIZED / SOURCE_ONLY_BEFORE_FARO / DEFAULT_APP_UNCHANGED`

## 稳定 Interface

- `fresh_pool.py`：从绑定 metadata 与 exclusion snapshot 重算 exact 32-parent / 96-asset fresh Training pool。
- `run_pool_head.py`：执行一次性 zero-body HEAD preflight，并封存 availability 与 Content-Length receipts。
- `run_pool_download.py`：只在 admitted HEAD receipt 与独立 execution lock 下下载并校验 exact source assets。
- `run_pool_inventory.py`：校验 ZIP 容器 CRC、trajectory 和 exact pose-bounded frame plan；不解码像素。

## 输出

source 与 evidence 仅写入 lock 指定的 `artifacts.local/` exclusive roots；Git 只保存协议、实现和小型结果，不保存 dataset payload。

## 安全边界

当前 transport 层不解码 source frame、不运行模型、不读 FARO、不做 truth scoring 或训练。后续必须先完成全部 32 parents 的 source-only Phase A 并封存 top eight，之后才可读取 selected FARO；`UNKNOWN` 永远不是 negative。

## 停止条件

任一 metadata/exclusion/request-plan、user authority、HEAD receipt、binding/hash、资源预算、root collision 或 one-shot 条件漂移即停止。动态状态以 [`docs/research/taro/README.md`](../../../docs/research/taro/README.md) 为唯一真源。
