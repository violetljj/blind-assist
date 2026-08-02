# Judge Audit R0 — 暂缓状态记录（2026-08-03）

当前终态：`DEFERRED_HOLD`

这表示裁判审计方向保留、正式评价权限关闭，工作暂缓等待下一次独立 RGB primitive review。它不是模型失败、不是 YOLO 胜出，也不是裁判构造通过。

## 当前已冻结

- `primitive_observability_v4`：visibility、route anchor、path relation、route certainty、evidence quality 与 phase 的操作定义和字段级证据窗口已写入正式合同。
- `visibility` 只使用当前 RGB 帧；几何字段不得使用 temporal 视图；causal phase 只能使用当前帧和过去前缀；retrospective 不得改写 current-only 几何字段。
- v5 burned calibration packet 使用未消费的 `evr0-screen-041`–`evr0-screen-048` 八个事件；RGB-only staging、opaque packet 和临时 contact sheets 已生成。

## 尚未成立的证据

- v5 尚无三份有效的、逐帧且非模板化的 primitive review；因此没有 review seal、event ledger、YOLO selection-only candidate ledger、counterfactual pair manifest、oracle trace 或四项审计报告。
- v4 的常数标签提交被判无效，不得 seal、统计或解释为稳定性结果；v3 未形成有效 review，也不得复用。
- source mask 仍仅是 discovery 输入，不是真值；depth、geometry、trajectory native sidecar 未生成，缺失不构成负证据。

## 权限边界

- `formal_review_access=false`
- `formal_denominator_inclusion=false`
- 不允许模型排名、模型选型、YOLO 优劣结论、Android/产品/安全结论。
- 恢复时必须按顺序完成：三份独立 RGB primitive review → 机器校验 → seal → deterministic selection-only pair 构造 → native/system-chain oracle（有机会条件才评价）→ 四项 judge audit。
- pair 不足保持 `NOT_EVALUABLE/HOLD`；不得降低 YOLO 相似度门槛、按标签回挑样本或把缺失 oracle 当作失败。

验证记录：相关 49 项单元测试、`compileall -q scripts/research/eval_validity_r0` 与正式/校准合同结构校验已通过；这些检查只证明实现/合同完整性，不证明裁判有效性。
