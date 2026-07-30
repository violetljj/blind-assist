# dual_loop_unseen_natural_event_r0

状态：development

## 研究问题与版本

`DUAL_LOOP_R1_UNSEEN_NATURAL_EVENT_R0` 在未参与 R1 设计或调参的连续自然步行视频上，
检验冻结提交 `039757b2da41c051373f8ee3189c4b06028f5295` 是否能减少完整误提醒窗口，
同时保持同一批 baseline 已命中正例及其逐事件时延。首次来源只产生 event-level
canary，不作总体外推、Confirmation、产品或安全结论。

## 稳定 Interface

来源选择必须先于 payload 与 baseline/candidate 输出访问：

```powershell
python -m scripts.research.dual_loop_unseen_natural_event_r0.select_source `
  --output-dir artifacts.local/evidence/dual-loop-r1-unseen-natural-event-r0/source-selection
```

selector 只读取 Wikimedia Commons category/API metadata，按协议固定的 eligibility、
精确已使用标题排除和 Unicode title 升序输出 registry 与选择 receipt。已存在输出会
拒绝覆盖。后继 truth ledger、baseline adequacy、candidate replay 和 evaluator 必须
绑定该 receipt、source bytes SHA-256 与各自实现哈希。

## 输出

只写入显式 `artifacts.local/` 目录：

- `source_registry.json`：完整 metadata snapshot；
- `source_selection_receipt.json`：冻结规则、eligible 顺序、rank-1 source 与 derivative；
- 后继输入、truth、trace、评价与 learning record 各使用独立子目录。

## 安全边界

选择阶段禁止读取视频 payload、baseline、R1 candidate 或 truth outcome。事件真值必须在
baseline/candidate 前从冻结 RGB 形成，并标记为 model-reviewed evidence。active 仍只在
隔离设备回放中否决 simulated feedback-controller acceptance；不改 raw/stable risk、
目标选择、事件 identity/lifecycle 规则，也不声称物理播放或用户感知。

## 停止条件

单来源依次终止为：`FIRST_UNSEEN_SOURCE_NOT_EVALUABLE`、
`FIRST_UNSEEN_SOURCE_GUARDRAIL_FAILED`、
`FIRST_UNSEEN_SOURCE_NO_EVENT_LEVEL_EFFECT` 或
`FIRST_UNSEEN_SOURCE_EVENT_SIGNAL / SECOND_INDEPENDENT_SESSION_REQUIRED`。
窗口不得围绕输出裁切；candidate 打开后不得换来源、阈值、延迟容差或分母。若 rank-1
不可评价，只能完整披露该终点后按已冻结顺序启动新的独立 source instance。

## 假设与规则质疑

causal difference 是多框共同缩小对当前反馈的保守反证；expected information gain 是
完整负窗是否消失而非少数 row 被抑制；falsifier 是同 ID 正例丢失、超时或新增负窗；
成本是一次公开视频下载、模型复核和固定设备回放。该路线明确质疑“必须恢复完整运动
机制才能纠错”，但不降低证据隔离或把弱信号包装成效果。

## 失败资产复用

不可评价来源保留为 source-characterization；guardrail 失败保留为 counterexample；
无事件效果 trace 保留为 regression fixture。它们均不得重新包装成 unseen
Confirmation，也不得用于回调 R1。
