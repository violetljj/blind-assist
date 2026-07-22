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

不得手写或从非连续基准拼接该报告。先由同设备 `DetectorAbDeviceBenchmarkTest` 写出含 `schema=blindassist_detector_ab_device_benchmark_v2`、`decision_kernel_contract_id=blindassist_shared_decision_kernel_v1`、`model_asset_sha256`、事件/关键事件分母、有效 `sequenceDurationMs` 与 `falseAlertsPerMinute` 的 `benchmark.json`，再运行 `scripts/extract_sanpo_device_event_report.py`。转换器会把基准 SHA256、shared-kernel contract 和反馈 adapter 写入 provenance，并拒绝旧版手工 feedback 语义、非真机标记、模型哈希不一致或缺分母/时长的输入；转换成功仍只满足设备事件门输入，不能覆盖离线质量、INT8 fidelity 或默认模型替换授权。

当前 benchmark 的 `feedback_adapter=planner_accept_all_v1` 表示确定性的“计划可接受”模拟，只用于让生产与 benchmark 共享 stabilization、event 和 feedback-gating 顺序；其中 `deliveredAlertCount` 等历史字段不构成 TTS/震动在物理设备上实际送达的证据。真实反馈接受、冷却和 lifecycle 仍须由 `FeedbackController` 回归及独立真机验证覆盖。

历史报告不可补签：2026-07-11 的最近一次 SANPO oracle 真机 `benchmark.json` 没有嵌入 `model_asset_sha256` 或 `sequenceDurationMs`，转换器已按预期拒绝它。该结果只保留为旧版诊断证据，不能回填为候选设备门输入。

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

将该文件通过 `--device-event-report` 传入同一评价入口即可得到独立的 `device_event.gate`。全部 benchmark、INT8、同机事件和 Android 证据通过后，再通过 `--ai-release-review` 提交隔离的 GPT 发布审查与 Codex 证据审查收据；一致或第三模型仲裁为 `accept` 时，报告才写入 `production_model_replacement_authorized: true`。缺失、低置信度、abstain 或未解决分歧都会失败关闭，不回退到人工复核。
