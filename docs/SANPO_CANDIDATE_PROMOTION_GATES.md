# SANPO 候选模型评价与晋级门

本流程把“模型在离线 dev 上是否学到了正确语义”“Keras 转 full-INT8 后是否保持语义”和“真机连续场景是否安全”拆成三个独立门。任何单门变绿都不能授权替换 App 生产模型。

## 三段门禁

| 门 | 输入 | 核心指标 | 通过后的权限 |
|---|---|---|---|
| `offline_training_quality` | canonical dev + Keras 权重 | 全局 mIoU、逐 session/scene 四类 IoU、session 宏平均/最差 mIoU、boundary precision/recall/IoU、unknown abstain/coverage | 允许进入 INT8 保真检查 |
| `int8_fidelity` | 同一 dev 上的 Keras 与 INT8 输出 | argmax agreement、逐类预测 IoU、逐类真值 IoU 退化、平均 mIoU 退化 | 允许成为真机 benchmark 候选 |
| `device_event` | 与 TFLite SHA256 绑定的独立真机连续序列报告 | event recall、critical miss、false alerts/min、事件后清除率、重复提醒率、P95 延迟 | 仅成为 benchmark promotion 候选，仍不自动替换生产模型 |

当前预注册默认阈值写在 `scripts/sanpo_candidate_quality_gate.py` 的三个不可变 dataclass 中。阈值不从待评报告读取，防止运行时通过修改报告放宽门禁。尤其不能只看像素 accuracy：boundary 像素很少，即使量化后整类消失，整体 agreement 仍可能看似很高，因此必须同时满足逐类预测 IoU 和逐类真值 IoU 退化门。

`unknown_nonwalkable` 被视为显式 abstain：报告同时给出 abstain rate、known coverage、unknown precision/recall/IoU 和 covered pixel accuracy。coverage 是诊断量，不单独追求越高越好；晋级硬门先约束 unknown precision/recall，避免模型用“全 unknown”或“从不 unknown”刷分。

## 离线与 INT8 复跑

评价入口只消费已经通过来源总门的 canonical dataset、已经通过 Torch↔TensorFlow 等价门的权重，以及 benchmark-only TFLite。它拒绝把 `app/src/main/assets` 当作候选路径，也不读取 blind holdout 标签。调用方必须显式保持训练时的 `--backbone-alpha`、`--decoder-channels` 和 `--input-size`（仅 256/384/512）；质量报告会保存完整 `model_config` 及其 canonical JSON SHA256，且等价报告 consumer 会在 TensorFlow 模型构建前拒绝配置错配。TFLite 合同同样按该尺寸验证输入 `[1,H,W,3]` 与输出 `[1,H,W,4]`。

在导出前可省略 `--tflite`，先生成 Keras 离线质量审计；此时 INT8 与设备门必然保持 `not_evaluated`。只有离线质量全绿后才值得导出 benchmark-only INT8，并带 `--tflite` 重跑完整保真门。

```powershell
& .\.venv-export312\Scripts\python.exe scripts\sanpo_candidate_quality_gate.py `
  --dataset-root test-artifacts.local\datasets\<canonical> `
  --training-gate-report qa\training_gate_report.json `
  --weights test-artifacts.local\segmentation-candidate\<run>\mobilenetv3_lraspp.weights.h5 `
  --backend-equivalence-report test-artifacts.local\segmentation-candidate\<run>\backend_equivalence.json `
  --backbone-alpha 0.75 `
  --decoder-channels 96 `
  --input-size 512 `
  --tflite test-artifacts.local\segmentation-candidate\<run>\mobilenetv3_lraspp_int8_256.tflite `
  --report test-artifacts.local\segmentation-candidate\<run>\candidate_quality_gate.json
```

无 `--device-event-report` 时，设备门必须是 `not_evaluated`，即使前两门全绿，`benchmark_promotion_eligible` 也必须为 `false`。这正是训练质量门与设备事件门分离的安全边界。

## 真机事件报告合同

真机 harness 生成的输入必须使用以下 schema，并把 `model_sha256` 绑定到本轮 INT8 文件；路径、文件名或“latest”标签不能替代哈希绑定。

```json
{
  "schema": "blindassist_sanpo_device_event_gate_input_v1",
  "report_id": "same-device-continuous-sequence-<run>",
  "model_sha256": "<64 hex>",
  "metrics": {
    "event_recall": 0.95,
    "critical_miss_rate": 0.01,
    "false_alerts_per_minute": 0.20,
    "post_event_clearance_rate": 0.95,
    "repeated_alert_rate": 0.05,
    "p95_latency_ms": 60.0
  }
}
```

将该文件通过 `--device-event-report` 传入同一评价入口即可得到独立的 `device_event.gate`。最终报告固定写入 `production_model_replacement_authorized: false`；真正替换仍需人工复核、Android 集中验证和发布决策。
