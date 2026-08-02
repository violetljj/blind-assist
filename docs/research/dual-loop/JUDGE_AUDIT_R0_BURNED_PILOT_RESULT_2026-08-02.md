# JUDGE_AUDIT_R0 burned calibration pilot：第一轮结果

机器可读报告：[judge-audit-report-v4.json](../../../artifacts.local/evidence/eval-validity-r0/judge-burned-pilot-v1/judge-audit-report-v4.json)。

## 结论

本轮 pilot 的顶层终态为：

`STOP_JUDGE_AUDIT_FAILED`

这不是模型失败，也不开放正式 cohort；它表示 primitive review construct 在第一次真实考试中没有通过。失败来自 test 4 的 `visibility` primitive 一致性，不应靠改模型、调阈值或把 `UNKNOWN` 改成不提醒来掩盖。

## 输入边界

- 8 个 output-blind、8 个 source-session 独立的 burned event；来源 arm 为 `source_mask`，但 coverage 仍保持未分类，不能填充正式八类。
- 两名 causal 与一名 retrospective reviewer 均提交完整的 8 × 60 primitive observations；review bundle 已封存，hash 为 `f9a6d83b52fc9e3aa7cac8a538a7fa4b27bf3546e228a1886b6d8fa1ea505a41`。
- YOLO 只在 review seal 之后读取 pilot RGB，生成了 865 个 selection-only candidate universe 项；未读取 primitive 或 derived label。
- native/system-chain oracle 没有生成；以显式不可评价 receipt 表示，不构成 oracle 负证据。

## 四项测试

| 测试 | 结果 | 诊断 |
|---|---|---|
| test 1：YOLO-free primitive truth | `PASS` | provenance 表明 causal primitive review 不依赖 YOLO；这不等于 ontology 已稳定。 |
| test 2：物理反事实 | `NOT_EVALUABLE` | 865 个候选经 `IoU≥0.90`、尺度、位置、可见性门后，pre-label eligible pair 为 0；没有回挑 pair。 |
| test 3：oracle 双路径 | `NOT_EVALUABLE` | pair 未达到 pilot 最低数量，且 native/system-chain trace 未生成；缺失不当作 oracle 无价值。 |
| test 4：盲审稳定性 | `FAIL` | `visibility` primitive 一致率为 `0.0`；其余五个 primitive 字段为 `1.0`。 |

## test 4 细节

- `visibility` 在 causal A/B 的 480 个 frame comparisons 上全部分歧；classwise one-vs-rest agreement 对 `EVALUABLE` 与 `NOT_EVALUABLE` 均为 `0.0`。
- event sequence、boundary timing、derived actionability 和 causal/retrospective actionability consistency 均为 `1.0`。
- causal UNKNOWN union/intersection rate 均为 `1.0`；因此这轮不能把“派生动作一致”解释为 primitive 已稳定。
- primitive disagreement 没有传播为 derived actionability disagreement，但 `primitive_to_derived_determinism` 无共同 primitive 样本，不能通过该门。

## 当前权限

- 正式 `formal_review_access=false`。
- 不允许模型排名、YOLO 优劣结论、Model Matrix、Android/default App 或安全结论。
- 不允许根据标签重选 pair，也不允许为不同 oracle 调阈值。

下一步只应保留这些失败 packet/review 作为 calibration counterexample，先检查 `visibility` primitive 的可操作定义与 RGB packet 的证据边界；修订后必须重新烧录独立 pilot。正式 cohort 仍保持关闭。
