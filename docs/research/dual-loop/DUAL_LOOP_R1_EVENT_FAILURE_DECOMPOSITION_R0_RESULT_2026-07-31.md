# DUAL_LOOP_R1_EVENT_FAILURE_DECOMPOSITION_R0

状态：`COMPLETE / DEVELOPMENT_POST_TERMINAL_ANALYSIS`
Terminal：`POLICY_GRANULARITY_MISMATCH_SUPPORTED`
日期：2026-07-31（Asia/Hong_Kong）

## 结论

active R1 的实际权限是当前 feedback opportunity 的 frame-level veto。它不写入
`RiskEventTracker` 的 event identity/lifecycle，也没有 hold、latch 或事件状态。因此
它可以在某一帧减少一行 feedback，但只要之后仍有 candidate feedback opportunity，
同一 truth window 仍然是 false。这个结果只消费已关闭 Development evidence，不形成
新的事件效果、Confirmation、默认生产、产品或安全主张。

当前冻结主线仍为：

`ISOLATED_ACTIVE_MECHANISM_LANDED / DEFAULT_OFF / FIRST_UNSEEN_SOURCE_NO_EVENT_LEVEL_EFFECT / DENSITY_SIGNAL_ONLY`

## 证据范围与不变量

本任务只读取 CrowdBot、Matoaka、Shiraz 的关闭 trace、truth ledger、receipt 和既有
evaluation；先验证 trace/receipt hash、帧 identity、truth identity、baseline/candidate
配对和既有 terminal，再发布分析结果。没有修改
`CausalSceneScaleTristateGeometryProducer.kt`、`AssistDecisionKernel.kt`、
`RiskEventTracker.kt` 或阈值；没有新增 hold/latch/事件状态、IMU、depth、flow 或模型；
没有重跑或重调 R1 candidate，也没有读取未来帧。

## 三源汇总

| source | ledger positives | closed-scored positives | baseline/candidate positive hit | baseline/candidate false windows | baseline/candidate feedback rows |
| --- | ---: | ---: | --- | --- | ---: |
| CrowdBot | 10 | 8 | 8/8 | 7/7 | 38/36 |
| Matoaka | 7 | 7 | 3/3 | 7/7 | 57/55 |
| Shiraz | 7 | 7 | 7/7 | 5/5 | 111/111 |

总计为 49 个 ledger 窗口（24 个正例、25 个负窗）；47 个 closed-scored，CrowdBot
的 `F1A-P-007` 与 `F1A-P-009` 沿用既有协议标记为
`TEMPORAL_SCORING_NOT_EVALUABLE`。负窗没有任何一个被完整消除，induced negative
window 为 0；逐窗口评分范围内的 feedback rows 为 `206 -> 202`（不是三源全序列行数）。

## Retained-false 分类

| 分类 | 数量 | 解释 |
| --- | ---: | --- |
| `A_SIGNAL_ABSENT` | 1 | 评分窗内没有 contradiction evidence。 |
| `B_SIGNAL_LATE` | 2 | 首个 contradiction/veto 晚于 baseline 首次 feedback。 |
| `C_FRAME_VETO_THEN_RETRY` | 10 | 发生实际同帧 veto，随后在同一评分窗又出现 candidate feedback。 |
| `D_TARGET_OR_ASSOCIATION_MISMATCH` | 4 | 具有 full detection trace 的来源中可观测到 target association reset，且不能仅由及时 veto 解释。 |
| `E_SCALE_SIGNAL_TASK_MISMATCH` | 2 | signal 存在但未落在可 veto 的 feedback opportunity，或与 scene-scale task 语义不匹配。 |
| `MIXED_OR_UNRESOLVED` | 0 | 当前字段不足以唯一归因的窗口。 |

典型的 `C_FRAME_VETO_THEN_RETRY` 是 CrowdBot `F1A-N-001`：首次 contradiction 早于
baseline alert，实际 veto 后 `211.213 ms` 又出现 candidate feedback；Matoaka
`F1A-N-009` 同样在 veto 后 `100 ms` 重试。Shiraz 的 retained-false 窗口没有反馈行
下降，且其中四个由 full detection trace 暴露 association reset，归入 D；这解释了
为什么同一 scene-scale evidence 在新来源上不能形成稳定的事件级继承。

每个正例事件与负例窗口均保留以下字段：
`baseline_first_feedback_ns`、`first_contradiction_ns`、`first_actual_veto_ns`、
`next_candidate_feedback_after_veto_ns`、`contradiction_before_first_alert`、
`contradiction_lead_ms`、`retry_after_veto_ms`、`contradiction_row_count`、
`longest_contradiction_run_ms`、`selected_target_scale_rate_summary`、
`scene_median_scale_rate_summary`、`target_association_reset_count`、
baseline/candidate feedback row count、`final_event_outcome` 与分类。完整逐窗口值见
下方 CSV/JSON/Markdown 产物。

## Development-only upper-bound audit

审计只在内存中评估一族“已有 contradiction 触发有限 duration 抑制”的因果策略，
使用已记录的 baseline/candidate feedback rows；不写 candidate trace、不改变 R1
阈值、不读未来帧。它要求保留全部 baseline-hit 正例、induced negative window 为 0，
并满足各来源已预冻结的正例新增时延上限。

| source | witness | eliminated window | max positive added delay | induced negative windows | pre-frozen delay limit |
| --- | ---: | --- | ---: | ---: | ---: |
| CrowdBot | 49.241 ms | `F1A-N-001` | 0 ms | 0 | 0 ms |
| Matoaka | 900 ms | `F1A-N-009` | 0 ms | 0 | 0 ms |
| Shiraz | none | none | — | — | 250 ms |

因此 top-level terminal 选择 `POLICY_GRANULARITY_MISMATCH_SUPPORTED`。这两个 witness
只说明“若另加事件级有限状态，已有 signal 在个别 Development replay 上具有政策粒度
空间”；它们不是 R1 新效果结果，不产生 R2 实现授权。由于 witness 仅来自单来源已见
trace、需要新增 runtime state、且跨来源 Shiraz 没有安全 witness，本任务的工程决策仍
是：不值得设计单变量 R2，关闭 scene-scale active 路线。

## R2 决策

`worth_designing_single_variable_r2 = false`
`decision = CLOSE_SCENE_SCALE_ACTIVE_ROUTE`
`r2_implemented = false`

保留默认关闭的机制、receipt、回归夹具、失败分类与 row-density diagnostic；不自动
实现 hold/latch、事件状态、阈值调整或任何 R2。

## 产物与复核

- [逐窗口 CSV](../../../artifacts.local/evidence/dual-loop/r1-event-failure-decomposition-r0-final6/event_failure_decomposition.csv)
- [逐窗口 JSON](../../../artifacts.local/evidence/dual-loop/r1-event-failure-decomposition-r0-final6/event_failure_decomposition.json)
- [逐窗口 Markdown](../../../artifacts.local/evidence/dual-loop/r1-event-failure-decomposition-r0-final6/event_failure_decomposition.md)
- [upper-bound JSON](../../../artifacts.local/evidence/dual-loop/r1-event-failure-decomposition-r0-final6/upper_bound_audit.json)
- [只读分析 Module](../../../scripts/research/dual_loop_r1_event_failure_decomposition_r0/README.md)

已通过：模块 `py_compile`、5 项确定性单元测试、输出 LF 字节测试；分析器运行时再次
验证所有输入 receipt/hash/parity。Matoaka 没有完整 detection dump，因此 selected
target rate 仅来自 trace 中记录的 selected risk box；scene median 与 target reset
不可观测，结果明确保留为 `null`，没有用 0 填充。
