# dual_loop_r1_event_failure_decomposition_r0

状态：`development`

## 研究问题与版本

`DUAL_LOOP_R1_EVENT_FAILURE_DECOMPOSITION_R0` 是对已关闭的 active R1
Development trace、truth ledger 和 receipt 的 post-terminal failure analysis。它解释
为什么 CrowdBot、Matoaka、Shiraz 的 candidate 可以减少 feedback rows，却没有消除完整
negative window；它不重跑 detector、Android、candidate 或任何阈值选择。

本 Module 固定消费三类已关闭证据：

- CrowdBot 的 4,422 帧 active trace、完整 detection dump、production baseline trace、
  receipts/evaluation 和 `combined_event_window_ledger.jsonl`；
- Matoaka 的 10,724 帧 active trace、receipt/evaluation 和同一 frozen ledger 的
  `DEVELOPMENT` 条目；该 trace 没有完整 detection dump，因此 target association reset
  不可观测；
- Shiraz rank-2 的 baseline/candidate trace、receipt、effect/terminal receipt 和
  `truth-freeze-r2` ledger。

## 稳定 Interface

从仓库根目录运行：

```powershell
python -m scripts.research.dual_loop_r1_event_failure_decomposition_r0.decompose `
  --repo-root E:\linnan\linnan `
  --output-dir artifacts.local/evidence/dual-loop/r1-event-failure-decomposition-r0
```

默认路径只指向上述已经关闭的本机 evidence namespace。所有输入 receipt 的 trace/hash、
frame identity、truth item identity、baseline/candidate pairing 和既有 terminal 都会
先验证；验证失败时不发布任何结果。输出目录必须位于 `artifacts.local/`，且拒绝覆盖。

## 输出

一次成功运行生成 LF 编码的：

- `event_failure_decomposition.csv`：每个正例事件与负例窗口一行；包含用户要求的时刻、
  contradiction run、两个 scale-rate summary、association reset、row count、outcome
  与 retained-false 分类；
- `event_failure_decomposition.json`：逐窗口结果、输入 hash、聚合、upper-bound audit、
  terminal 与 R2 决策；
- `event_failure_decomposition.md`：结论优先的 Development-only 报告；
- `upper_bound_audit.json`：从同一 JSON 拆出的机器可读 upper-bound 结果。

## 安全边界

不修改 `CausalSceneScaleTristateGeometryProducer`、`AssistDecisionKernel`、
`RiskEventTracker` 或任何阈值；不新增 hold/latch/事件状态、IMU/depth/flow/模型；不读取
未来帧。Matoaka 缺失完整 detection dump 时，selected target rate 只作已记录 selected
risk box 的诊断，scene median 与 target reset 不可观测，禁止补造 0。

upper-bound audit 只在内存中评估一个因果、有限 duration 的 contradiction-triggered
抑制策略族：它使用既有 R1 contradiction 与 baseline/candidate feedback rows，truth 只在
策略模拟之后作评分，且不写新的 trace。即使找到 witness，也不实现 R2、不改变当前
terminal 或产生新的效果主张。

## 停止条件

允许的最终 terminal 只有：
`POLICY_GRANULARITY_MISMATCH_SUPPORTED`、`SIGNAL_ABSENT_OR_IRRELEVANT`、
`TARGET_ASSOCIATION_LIMITATION_SUPPORTED`、`MIXED_NO_CLEAR_SUCCESSOR`。

## 失败资产复用

结果仅可作为既有 R1 的 failure decomposition、counterexample、回归夹具和研究关闭
记录；不得重新包装为 unseen Confirmation、默认生产改善、真人助行、产品或安全证据。
