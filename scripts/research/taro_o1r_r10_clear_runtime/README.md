# TARO O1R R10 fresh-pool runtime

状态：`current / TARO_RESEARCH_MODULE / R10_FRESH_32_PARENT_POOL_FROZEN / R10_ZERO_BODY_HEAD_96_OF_96_PASS / R10_SOURCE_DOWNLOAD_96_OF_96_PASS / R10_INVENTORY_32_PARENT_710_FRAME_PASS / R10_PHASE_A_R0_DEPENDENCY_STOP_NO_CANDIDATES / R10_SOURCE_ONLY_PHASE_A_R1_32_PARENT_710_FRAME_PASS / R10_TOP8_R0_CANONICAL_FLOAT_VALIDATOR_STOP_PRE_SELECTION / R10_TOP8_R1_REPAIR_TESTED / SOURCE_ONLY_BEFORE_FARO / DEFAULT_APP_UNCHANGED`

## 稳定 Interface

- `fresh_pool.py`：从绑定 metadata 与 exclusion snapshot 重算 exact 32-parent / 96-asset fresh Training pool。
- `run_pool_head.py`：执行一次性 zero-body HEAD preflight，并封存 availability 与 Content-Length receipts。
- `run_pool_download.py`：只在 admitted HEAD receipt 与独立 execution lock 下下载并校验 exact source assets。
- `run_pool_inventory.py`：校验 ZIP 容器 CRC、trajectory 和 exact pose-bounded frame plan；不解码像素。
- `run_pool_phase_a.py`：先完成 710 次注册 RGB/K DepthART inference，再读取 Apple depth/confidence 构造 source-only features；全程 FARO=0。
- `run_pool_phase_a_r1.py`：仅接纳 R0 的 pre-inference `timm` 依赖停止，在新 root 进行完整重跑；不 resume 或采用任何 R0 candidate。
- `run_top8_selection.py`：重验完整 Phase A 与冻结的 R9 selector evidence，先封存全部 32 个 source-only scores，再确定性封存 top eight；全程 FARO=0。
- `run_top8_selection_r1.py`：仅接纳 R0 在任何 score/selection 输出前的 canonical-float validator stop，在新 root 用 round-12 parity 修复完整重算；不 resume 或采用部分选择输出。
- `run_selected_phase_b.py`：仅在 top eight 已封存后读取 selected `highres_depth`，构造 FARO labels，并调用 `phase_b_metrics.py` 的冻结双类门；不读取 unselected FARO。

## 输出

source 与 evidence 仅写入 lock 指定的 `artifacts.local/` exclusive roots；Git 只保存协议、实现和小型结果，不保存 dataset payload。

## 安全边界

当前 transport 层不解码 source frame、不运行模型、不读 FARO、不做 truth scoring 或训练。后续必须先完成全部 32 parents 的 source-only Phase A 并封存 top eight，之后才可读取 selected FARO；`UNKNOWN` 永远不是 negative。

## 停止条件

任一 metadata/exclusion/request-plan、user authority、HEAD receipt、binding/hash、资源预算、root collision 或 one-shot 条件漂移即停止。动态状态以 [`docs/research/taro/README.md`](../../../docs/research/taro/README.md) 为唯一真源。
