# Causal Event-Preserving Semantic Refresh Scheduling Q0

状态：`IMPLEMENTED_OFFLINE_R0 / DEVELOPMENT_ONLY / HOLD_LAST_SEMANTIC_SNAPSHOT / NO_ANDROID_AUTHORITY`

本模块把第二环重定义为语义刷新调度器，而不是危险判断器。R0 只验证一个严格的离线反事实骨架：每个策略拥有独立 cache、feedback/cooldown state 和 event-state counter；刷新时读取该帧的全频固定模型参考，跳过时只保持该 arm 自己最近一次语义 snapshot。

R0 的传播模式不是生产 tracker，也不是光流实现。它的作用是先验证反事实状态隔离、真实时间周期 baseline、三级 divergence 和 event-level Pareto 输出是否正确。

## 稳定 Interface

调用入口是 `python -m scripts.research.dual_loop_semantic_refresh_q0.cli`。输入必须是
hash-bound 的全频 detector dump、生产参考 trace 及已冻结的 baseline evaluation；可选
fast-feature JSONL 必须逐行声明 current-frame-only 因果来源。输入身份、帧序、时间戳、
detector hash 或 receipt 不一致时 fail closed；缺少 fast-feature trace 时，feature-rule
与 learned arms 返回 `NOT_EVALUABLE`。

## 运行

从仓库根目录执行：

```powershell
python -m scripts.research.dual_loop_semantic_refresh_q0.cli `
  --dump artifacts.local/evidence/dual-loop/multitrack-counterfactual-r0/device-dump/trace.jsonl `
  --dump-receipt artifacts.local/evidence/dual-loop/multitrack-counterfactual-r0/device-dump/producer_receipt.json `
  --baseline artifacts.local/evidence/dual-loop/production-temporal-geometry-factorial-ab-r0/device-producer/trace.jsonl `
  --baseline-evaluation artifacts.local/evidence/dual-loop/production-temporal-geometry-factorial-ab-r0/evaluation/result.json `
  --output-dir artifacts.local/evidence/dual-loop/semantic-refresh-q0-r0
```

默认 fixed-time arms 为 `33/66/100/167/267 ms`。策略按 `source_capture_timestamp_ns` 决定刷新，不把请求 FPS 当成实际输入频率。

## 输出

- `result.json`：输入 hash、各 arm 指标、三级 divergence、truth-item metrics、session metrics 与 Pareto front；
- `report.md`：人可读摘要；
- `traces/*.jsonl`：每个可运行 arm 的逐帧调度、cache age、candidate feedback 和 divergence。

所有输出只写入 `artifacts.local/evidence/dual-loop/` 下由调用方明确指定且不存在的目录。

## Fast-feature 合同

feature-rule / learned arms 只有在提供独立 JSONL 时才运行。每行必须声明：

```json
{
  "schema_version": "blindassist.dual_loop_semantic_refresh_fast_features.v1",
  "causal_source": "CURRENT_FRAME_FAST_LOOP_ONLY",
  "uses_full_detector_output": false,
  "session_id": "...",
  "frame_id": "...",
  "source_capture_timestamp_ns": 0,
  "features": {}
}
```

缺少该 trace 时，这些 arm 输出 `NOT_EVALUABLE`，不会从全频 detector dump 反推“快环特征”。

## 安全边界

本模块只消费 Development 输入并生成离线反事实诊断；不读取 truth 参与调度，不接 Android、
A568、默认生产路径或设备运行时，不把 event-window 保持当作逐帧语义正确率、能效、热量、
助行、产品或安全证据。

## 停止条件

任一全频 parity、输入 hash/时间身份、状态隔离或因果来源检查失败即停止；若所有低调用策略
被 fixed-period baseline 支配、漏掉关键 reference event，或 scheduler 开销接近节省的 detector
成本，则不扩展到 learned policy 或 Android。旧 R2/R3 不重跑，失败只关闭本 Q0 evidence instance。

## 证据边界

当前 4,422 帧、两 session 的输入只能支撑 Development-only 的固定模型参考保持筛查；不能训练或验证跨来源 learned scheduler，也不能支持真实语义正确率、能效、热量、助行或安全结论。正式协议见 [`DUAL_LOOP_SEMANTIC_REFRESH_Q0_PROTOCOL_2026-07-31.json`](../../../docs/research/dual-loop/DUAL_LOOP_SEMANTIC_REFRESH_Q0_PROTOCOL_2026-07-31.json)。

本模块不改 Android、不接 A568、不修改默认生产行为，也不重跑或修复旧 R2/R3。
