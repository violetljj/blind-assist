# TARO O0R source-adapter contract validator

状态：`SOURCE_AND_ADAPTER_CONTRACT_LOCK_PASS / SCIENTIFIC_STATUS_NOT_RUN / EXECUTION_NOT_AUTHORIZED`

本 Module 只验证
[`TARO_O0R_ARKITSCENES_SOURCE_AND_ADAPTER_CONTRACT_LOCK`](../../../docs/research/taro/TARO_O0R_ARKITSCENES_SOURCE_AND_ADAPTER_CONTRACT_LOCK_2026-08-10.md)。
动态状态、唯一 successor 与权限只读取
[`docs/research/taro/README.md`](../../../docs/research/taro/README.md)。

## 稳定 Interface

- `validate_taro_o0r_source_adapter_contract.py`：重算绑定文件、pinned Git exclusion snapshot、
  ARKitScenes metadata roster、roles、truth/injection 语义、gates、权限与 future-root absence，并冻结
  model-free SCALE truth-only、right-bracket watermark、9 query-bound receipts、source-specific receipt
  claim ceiling 与新 TARO reducer seam；
- `test_validate_taro_o0r_source_adapter_contract.py`：21 项 mutation tests，覆盖身份复用、角色漂移、
  model-before-truth、pose 因果水位、query cardinality、P0 overclaim、legacy reducer 误接、constant
  uncertainty、registration/boundary 漂移、factorial/K 混淆、gate 降级、artifact collision 与扩权。

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest `
  scripts.research.taro_o0r_source_adapter.test_validate_taro_o0r_source_adapter_contract

E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/taro_o0r_source_adapter/validate_taro_o0r_source_adapter_contract.py
```

## 输出

validator 只读本地绑定文件并向 stdout 输出短 JSON，不创建 artifact。冻结的未来 source、work、
truth evidence 与 O0R evidence 只能位于契约声明的 `artifacts.local/` 隔离根；这些根在本协议锁阶段
必须全部不存在。

## 安全边界

- 这里没有 downloader、materializer、truth adapter、DepthART runner、factorial evaluator、模型或
  scientific artifact；
- static `VALID` 只证明契约内部一致，不证明 source truth、模型输出、factor headroom 或真实 O0R；
- source-specific receipt 不等于完整 P0 `TaroFrameReceipt`，也不建立真实 camera-body mount；
- 24 个 roster parent 只由 metadata 分配，当前不得下载或打开其 source body；
- historical O0M evidence 只读且不得覆盖、删除或重跑；
- 默认 App、产品与 safety authority 始终为 false。

## 停止条件

任一绑定/hash/roster/role/truth/injection/gate/authority 漂移，任一 future root 已存在，或 validator
无法重算 pinned exclusion snapshot 时立即 fail-closed。当前唯一 successor 只允许另锁 adapter
implementation；独立 truth-only one-shot lock 提交前不得下载 source、物化 truth、运行 DepthART 或
执行 O0R。
