# dual_loop_production_temporal_ab

状态：development contract preflight

## 研究问题与版本

`DUAL_LOOP_PRODUCTION_TEMPORAL_GEOMETRY_FACTORIAL_AB_R0` 检查既有生产
`TemporalRiskTracker` 的 object-detector temporal geometry contribution，相对同构的
neutralized 分支是否改善冻结 CrowdBot decision session 上的 replay alert decision。
当前只建立 outcome-blind 输入身份；候选输出、truth join 与 Confirmation 尚未执行。

## 稳定 Interface

仓库根入口：

```powershell
pwsh -File scripts/run_dual_loop_production_temporal_ab_input_preflight.ps1
```

内部 Python 实现只接受 repo root 与 receipt 输出路径。它没有 truth ledger 或候选
output 参数。

## 输出

默认只写：

`artifacts.local/evidence/dual-loop/production-temporal-geometry-factorial-ab-r0/input-preflight/`

第一份 receipt 记录两个 frame ledger、4,422 个实际 PNG、逐文件 SHA-256、严格时间
顺序、IHDR 尺寸和 canonical inventory hash，不复制 RGB payload。第二份
truth-membership receipt 只读取已烧毁的 F-1A truth，把 17 项精确映射到 source
nanosecond/frame，冻结 8 个可评分正例、7 个负窗、两个零帧正例以及无跨项帧重叠。

## 安全边界

RGB identity preflight 不读取 `combined_event_window_ledger.jsonl`。独立的
truth-membership preflight 只读取该已烧毁 truth 与 frame ledger，不读取候选 trace
或 Confirmation。两者都不运行 detector、不产生 A/B 结果、不更改 Android/App
生产行为，也不形成算法、产品或安全结论。

## 停止条件

任一 ledger/hash/path/timestamp/PNG 失败即 `INVALID` 并停止；不得跳过坏帧或缩小
分母。只有完整 `4422/4422` 才能把 input identity 写为 `VALID`。

## 假设与规则质疑

该 receipt 只证明冻结设备输入 payload 的身份与可解码结构，不证明 QNN、A/B、
endpoint 或用户效果。若 storage/decode policy 与设备正式 runner 不一致，必须在
候选输出前版本化修订合同。

## 失败资产复用

失败 receipt 只能作为输入诊断与数据完整性反例；不得包装为候选算法负结果或 unseen
Confirmation。
