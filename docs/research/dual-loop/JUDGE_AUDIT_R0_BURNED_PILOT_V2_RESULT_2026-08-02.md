# Judge Audit R0：visibility v2 burned calibration pilot 结果

日期：2026-08-02
模式：`CALIBRATION_BURNED`  事件：8 个（`evr0-screen-009`–`016`，未消费 deterministic slice）

## 结论

visibility primitive 的操作定义与证据边界修复有效，但裁判整体仍未通过：

```text
visibility construct: PASS（pilot 内）
overall judge audit: STOP_JUDGE_AUDIT_FAILED
formal_review_access: false
model ranking / selection authority: none
```

这次 pilot 不能证明模型质量，也不能开放正式 50–100 event cohort。它证明了一个更窄、但很重要的结果：上一轮 visibility 的 0.0 一致率确实来自字段语义/证据边界问题；修复后 visibility 不再是当前的主要分歧源。但新的 output-blind review 同时暴露出 `path_relation`、`route_certainty` 和部分 `evidence_quality` 的独立操作定义问题，不能用 visibility 修复掩盖。

## 修复后的合同

新 packet 使用：

- `judge_primitive_packet.v2`；
- `judge_review.v4`；
- `judge_pilot_freeze.v2`；
- `visibility_observability_v2`；
- visibility 在 causal 和 retrospective 两条视图均为 `CURRENT_RGB_FRAME_ONLY`；
- causal temporal primitive 为 `CURRENT_PLUS_PAST_PREFIX`；retrospective temporal primitive 为 `FULL_EVENT_RGB`。

`EVALUABLE` 现在只表示当前 RGB 帧能定位前方路线/场景区域；不要求目标类别、不要求存在障碍、也不要求已经判断是否挡路。模糊、旋转、path/phase/route 不确定分别留在对应 primitive，不能反推 visibility。

## 执行完整性

| 项目 | 结果 |
| --- | --- |
| 新事件 | 8，`009`–`016` |
| packet | 2 causal + 1 retrospective，RGB-only |
| primitive review | 3 份，8 × 60 帧，schema/词表/禁用字段通过 |
| review seal | `d096ac82f478437db5931b452abbf998c091011310e15a5a884c96bb7ffdb519` |
| YOLO role | `SELECTION_ONLY`，review seal 后才读取 |
| selection candidate universe | 1,541 条，完整枚举 |
| eligible counterfactual pair | 0；未按标签回挑 |
| oracle traces | 未生成，`NOT_EVALUABLE` |

主要产物：

- [v2 pilot freeze](../../../artifacts.local/evidence/eval-validity-r0/judge-burned-pilot-v2/custodian/pilot-event-freeze.json)
- [v2 review seal](../../../artifacts.local/evidence/eval-validity-r0/judge-burned-pilot-v2/custodian/review-bundle-seal-v2.json)
- [v2 pair manifest](../../../artifacts.local/evidence/eval-validity-r0/judge-burned-pilot-v2/custodian/counterfactual-pairs-v3.json)
- [v2 audit report](../../../artifacts.local/evidence/eval-validity-r0/judge-burned-pilot-v2/judge-audit-report-v5.json)

## 四项测试

| 测试 | 结果 | 解释 |
| --- | --- | --- |
| YOLO-free truth | `PASS` | primitive provenance 不依赖 YOLO；动作标签仍由冻结规则派生 |
| matched physical counterfactual | `NOT_EVALUABLE` | 1,541 条候选中没有满足全部冻结阈值的 eligible pair |
| oracle native/system-chain | `NOT_EVALUABLE` | 没有 pair opportunity，因此没有伪造 native 或 system-chain 负证据 |
| blind review stability | `FAIL` | causal primitive 一致性与未知率未达门槛 |

## 分层稳定性诊断

### Visibility

```text
causal A vs B visibility agreement: 1.000000
causal visibility classwise EVALUABLE agreement: 1.000000
causal vs retrospective visibility agreement: 1.000000
visibility disagreement source: 0
```

上一轮同一指标为 0.0；因此可以把 visibility 的原始问题归因到操作定义/证据边界，而不是把它继续解释成模型或图像信息问题。

### 其他 primitive

```text
causal A vs B path_relation agreement: 0.000000
causal A vs B route_certainty agreement: 0.000000
causal A vs B evidence_quality agreement: 0.879167
causal primitive consistency: 0.646528
causal unknown event rate: 1.000000
primitive_to_derived_determinism: NOT_ESTABLISHED
```

causal A/B 对 visibility 已一致，但对 path/route 的解释仍明显不同；由于 actionability 在两边都被 UNKNOWN 封住，derived actionability 的表面一致率 1.0 不能替代 primitive construct stability。按合同，不能引入多数票、第三个仲裁 reviewer 或临时改写 actionability 规则来救援该结果。

## 终态与下一步

当前终态固定为：

```text
STOP_JUDGE_AUDIT_FAILED
```

只允许把 visibility v2 作为一个局部 pilot 修复记录。正式 cohort、模型排名、YOLO 优劣结论、oracle ceiling 结论和默认 App 权限继续关闭。下一次工作若要继续，必须单独冻结 `path_relation`/`route_certainty` 的操作定义与证据窗口；不能把它们混入本次 visibility 修复的成功结论。
