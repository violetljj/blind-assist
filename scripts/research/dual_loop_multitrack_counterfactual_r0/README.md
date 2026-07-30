# Dual-loop multi-track counterfactual R0

状态：`development`

## 研究问题与版本

`DUAL_LOOP_MULTITRACK_COUNTERFACTUAL_R0` 检验：production-selected 单目标三态源
在语义目标切换时丢失历史，是否是它无法纠正生产误报的主要原因。R0 只把同一帧的
完整 production QNN detections 做确定性多目标关联，仍使用冻结的 7-observation
框高三态数学规则。R0 失败后，同一 evaluator 还记录一个更小的 R1 Discovery：
至少两个匹配目标的中位 `log(box height)` 速率显著为负时，只在当前帧提出 veto。

## 稳定 Interface

```powershell
python -m scripts.research.dual_loop_multitrack_counterfactual_r0.evaluate `
  --dump <full-detection-trace.jsonl> `
  --baseline <production-temporal-trace.jsonl> `
  --baseline-evaluation <result.json> `
  --active-replay <optional-kotlin-active-replay.jsonl> `
  --output <artifacts.local/.../evaluation.json>
```

输入必须是 4,422 个唯一帧；每帧 detector hash 必须与冻结的
`CURRENT_FULL_PRODUCTION_TEMPORAL_GEOMETRY` 分支完全一致。evaluator 的无 veto
重放必须逐帧复现该真实生产 baseline 的 373 个 feedback 决策，否则失败；不得用
恰好同为 373 次提醒的 temporal-geometry-neutralized 消融分支替代。
提供 `--active-replay` 时，还要求 Kotlin 实现逐帧复现 scene decision、baseline feedback
和 candidate feedback，并保持 `event_mutation_allowed=false`；任一不一致都拒绝输出。

## 输出

只写显式指定的 `artifacts.local/` JSON；拒绝覆盖。

## 安全边界

这是已烧毁 CrowdBot 数据上的自适应 Development。它可以拒绝候选、选择后续
Discovery 或形成回归夹具，不能升级为 Confirmation、产品效果或安全结论。R1
scene-scale 只可在独立隔离 build 中继续，不得默认启用。

## 停止条件

R0 若无法在任何已评分负窗 baseline trigger 上给出 `CONTRADICT`，立即拒绝，不实现
active multi-track source。任何 R1 candidate 若降低正例事件召回，或无 veto 重放
无法逐帧复现 baseline，立即停止。

## 假设与规则质疑

R0 的 causal difference 仅是“保留所有检测目标的历史”，成本是一条完整检测导出
和轻量贪心关联。falsifier 是负窗触发点仍无反证。R1 质疑目标归因是否必要：当多个
目标共同缩小时，整体远离本身足以作为低权限反证，不解释是谁在运动。

## 失败资产复用

完整 detection dump、R0 零收益和逐帧 feedback simulator 可作为关联、冷却与 active
非干预回归夹具；不得重新包装为未见数据。
