# dual_loop_scene_scale_cross_source_r1

状态：development

## 研究问题与版本

`DUAL_LOOP_SCENE_SCALE_VETO_R1_CROSS_SOURCE_MATOAKA` 检验冻结的最小
scene-scale contradiction 在第二公开视频来源上能否复现，同时保持 baseline 已命中
正例不劣。它不改变 R1 的阈值、关联、active、risk、event 或 feedback 规则。
Matoaka 标签已用于此前 Development 调试，所以允许的 claim 仅为 locked
cross-source Development replication，不是独立 Confirmation。

## 稳定 Interface

准备固定 10 Hz、640×480 letterbox 输入：

```powershell
python -m scripts.research.dual_loop_scene_scale_cross_source_r1.prepare_input `
  --video <source-video> `
  --output-dir <artifacts.local/.../matoaka-input>
```

联结设备输出与冻结 truth：

```powershell
python -m scripts.research.dual_loop_scene_scale_cross_source_r1.evaluate `
  --trace <device-trace.jsonl> `
  --producer-receipt <producer_receipt.json> `
  --ledger <frozen-truth-ledger.jsonl> `
  --output <artifacts.local/.../matoaka-evaluation.json>
```

输入必须严格闭合 10,724 个连续 frame id 与 100 ms 时间戳；producer receipt 必须
`COMPLETE`、`truth_read=false`、`risk_mutation_count=0`，且 trace hash 完全匹配。

## 输出

只向显式指定的 `artifacts.local/` 目录原子写入准备后的帧、manifest、receipt 或
evaluation；已存在的输出会被拒绝，不覆盖正式 namespace。

## 安全边界

设备 producer 只读取 RGB、时间戳和冻结 detector/runtime，实现期间不能读取 truth。
host evaluator 才连接既有 Development ledger。该模块不授权默认 Android 行为、
真人助行、产品或安全结论。

## 停止条件

若 candidate 正例召回低于 baseline，立即以 positive guardrail failure 停止。若负例
窗口和负例触发行均不下降，则以 no-effect 停止。即使只有触发行下降，也只能记录
row-level Development signal，不能升级为事件级效果。

## 假设与规则质疑

候选只质疑“完整运动归因是否是提醒纠错的必要条件”。causal difference 是多个近期
检测框共同缩小的离散反证；falsifier 是跨来源不复现或伤害正例；额外成本只有固定
视频采样、production detector 回放和 truth-late join。

## 失败资产复用

固定帧 manifest、设备 trace、truth-late evaluator 和无事件级收益结果可作为
detector/runtime 回归、active 非干预检查和未来候选的 counterexample。它们不得重新
包装为 unseen Confirmation。
