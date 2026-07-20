# 公开视频 GPT 银标、暂定训练与端侧对照协议 v2

## 目的

当人工暂不可参与时，允许 GPT/VLM 在已取得许可、已机器脱敏的公共连续第一视角视频上完成候选筛选、风险判断和生命周期草案，并将其与 BlindAssist 端侧输出逐 episode 对照。

它解决的是“先自动获得可检验信号并开展暂定训练”，不是把模型输出误称为人工真值。旧 `v1` 银标保持仅对照；新的 `v2` 在 CC-BY/CC0 来源、哈希绑定、多帧时序审阅与机器隐私处理记录齐全时，可授权 `provisional_model_supervision` 训练。无论版本，银标都不得单独用于校准、blind 评测或替换默认模型。

## 可用来源与前置条件

1. 仅使用有可核验许可证、来源 receipt 和本地 SHA256 的公开素材；优先 CC0 GND、CC0 uB-VisioGeoloc 与 CC-BY SANPO。
2. 对含人物的公开 RGB，先保留机器脱敏 receipt；即使如此，`privacy_audit_required=true` 仍不得删除。
3. 每次 GPT 审阅必须至少提供两个时间不同的帧，禁止由单帧臆造运动、接近或已通过。

## 银标字段

每个 episode 的 `silver_should_alert` 只能是：

- `candidate_alert`：模型认为该时段可能需要提醒；
- `candidate_no_alert`：模型认为不应提醒；
- `abstain`：视角、遮挡、采样密度或不确定性不足，明确弃权。

非弃权标签还要有风险轮廓；所有标签必须绑定证据帧 SHA256、模型/提示词版本、置信度和不确定性理由。`validate_public_video_silver_labels.py` 会拒绝未知帧、单帧“时序”判断，以及任何放开校准、blind 评测或生产替换的授权。`v2` 额外要求 CC-BY-4.0 或 CC0-1.0 来源 receipt 和 `provisional_training_authorized=true`，才可训练。

## 与端侧模型对照

对每段连续视频，将端侧模型的告警区间、首告警时间、清除时间、重复提醒和弃权逐一与银标 episode 对齐，输出四类结果：

1. 银标 `candidate_alert` 而端侧未告警：候选漏报；
2. 银标 `candidate_no_alert` 而端侧告警：候选误报；
3. 两者同意：候选一致，不称为准确率；
4. 银标 `abstain`：从分子分母移出，但要单列比例，防止靠弃权虚增一致性。

独立人类真值仍是把候选一致性升级为安全质量指标的唯一依据；这不阻止将 v2 银标用于带来源权重、可撤销的暂定训练。

执行时使用：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts\compare_public_silver_to_edge_events.py `
  --silver-manifest artifacts.local\evidence\public-video-silver\<batch>\silver_labels.json `
  --source-manifest artifacts.local\evidence\datasets\<source>\machine_redacted_rgb\machine_redaction_receipt.json `
  --edge-report artifacts.local\evidence\public-video-edge\<batch>\edge_events.json `
  --output artifacts.local\evidence\public-video-silver\<batch>\edge_comparison.json
```

## 真机仅推理闭环

`run_public_video_edge_inference.ps1` 会依次构建只含银标证据帧的资产集、在真机调用默认端侧模型与生产风险/反馈路径、拉回 `edge_events.json`，最后执行上述对照。设备测试从不读取 GPT verdict；它只知道 frame、episode 与顺序。

若手机已有同包名但不同签名的 BlindAssist，Android 会报 `INSTALL_FAILED_UPDATE_INCOMPATIBLE` 并且脚本会停下。不要把它当作模型失败。只有明确接受删除该设备上现有 BlindAssist 及其应用数据时，才附加 `-RemoveConflictingInstall`：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_public_video_edge_inference.ps1 `
  -SilverManifest artifacts.local\evidence\public-video-silver\<batch>\silver_labels.json `
  -SourceManifest artifacts.local\evidence\datasets\<source>\machine_redacted_rgb\machine_redaction_receipt.json `
  -SourceImagesDir artifacts.local\evidence\datasets\<source>\machine_redacted_rgb\images `
  -OutputRoot artifacts.local\evidence\public-video-device-run\<batch> `
  -RemoveConflictingInstall
```

## v2 Negative-label fail-closed gate

`candidate_no_alert` must include `negative_decision_quality`. Its
`corridor_heading_stability`, `near_field_visibility`, and
`corridor_clearance`, and `near_field_lateral_intrusion_absent` must each be
at least `0.70`. A panning camera, an occluded near field, or a close
seated/lateral person therefore becomes `abstain`, rather than a negative
candidate in the agreement denominator.

## v3 因果可行动性标签

`v3` 不再把 episode 最终是否安全通过直接当成告警 target。每个 episode 除兼容字段 `silver_should_alert` 外，必须提供：

- `silver_actionability=no_attention`：当前及过去证据均不需要环境注意；只能对应 `candidate_no_alert`；
- `context_only`：需要无方向的环境/施工注意，但没有路线阻挡证据；
- `intervention_then_route_clear`：保持当前运动趋势会进入障碍，随后路线关系清除；
- `persistent_intervention`：干预状态持续到 marker episode 结束；
- `uncertain`：证据不足，只能对应 `abstain`。

后三个非 `no_attention` 状态都不得因为后续 `safe_pass` 或 `route_changed` 被反写为 no-alert。`eventual_outcome` 仅记录 `unknown / safe_pass / route_changed / contact_or_blocked / not_applicable`，作为生命周期或响应属性；`causal_evidence_basis` 必须为 `past_or_current_only`。验证器会拒绝使用未来路线结果重定义早期标签的 manifest。

r7.85 的三事件语义审计表明：Bangkok 原 safe-lateral episode 实际属于 `intervention_then_route_clear`，而 Düsseldorf 属于 `context_only`。这只支持修改标签合同，不代表模型准确率或部署门已经通过。

## v3 Experimental taxonomy probe

`animal_aware_candidate` is an explicit inference-only risk configuration.
It opts `horse` into the risk-target list while leaving `RiskAnalyzerConfig.Default`
unchanged. It exists to distinguish a detector feature miss from a downstream
taxonomy or policy omission when a public RGB sequence already contains a
hash-bound GPT/VLM silver candidate.

The resulting report must retain `risk_config=animal_aware_candidate` and be
summarized separately from `current`. Agreement in this probe is only candidate
agreement with model-produced silver labels. A v2 source may feed provisional
training, but this agreement itself does not authorize calibration, blind
evaluation, or changing the shipped default risk policy.

The comparison and campaign-summary scripts enforce this boundary: a report
without a non-empty `risk_config` is rejected, and one campaign summary cannot
mix two risk configurations.

## v4 Experimental lifecycle probe

`approaching_center_person_candidate` is inference-only. It leaves the default
feedback rules intact and adds one narrow branch: a detector-backed `person`
may receive MID-range feedback only when it is CENTER, MEDIUM-or-higher, and
the temporal tracker independently reports `APPROACHING`. It must be summarized
separately from `current`; it cannot promote a default policy, training run,
calibration, blind evaluation, or production model.

For every compared episode, the report records total triggers, unique tracked
event IDs, duplicate tracked-event triggers, and untracked triggers. A repeated
tracked event is a lifecycle failure; an untracked trigger is diagnostic only,
because existing static-obstacle routes intentionally predate the event layer.
Neither category may be silently treated as an event-level pass.
