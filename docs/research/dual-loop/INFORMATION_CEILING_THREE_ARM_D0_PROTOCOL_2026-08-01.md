# BlindAssist 信息上限三臂审计 D0 协议

状态：`DEVELOPMENT / PROTOCOL_FROZEN_BEFORE_NEW_THREE_ARM_DEVICE_OUTCOME /
NO_TRAINING / NO_MODEL_SELECTION_AUTHORITY`

## 研究问题

`INFORMATION_CEILING_THREE_ARM_D0` 只回答：在相同 parent natural events 与现有
`AssistDecisionKernel` 下，当前 YOLO、真值风险类别框以及真值可通行性/风险 mask
分别能够支持多少事件级提醒信息。

这里的 Arm C 不是“dense mask 直接进入 RiskAnalyzer”：现有
`TraversabilitySegmentationAnalyzer` 会先用像素形状执行 corridor/连通域/边界筛选，
再只转发一个 `Detection`。此外，当前链对 `OBJECT_DETECTOR` 与 `SEGMENTATION` source
使用不同 temporal/event policy。因此本轮能比较的是三个**当前端到端输入路径**，
不能把 Arm C 的差异纯粹归因于 bbox 几何。

本轮是可重复的 `DEVELOPMENT_STANDARD` 信息诊断。它不训练模型，不恢复已关闭的
segmentation R1/R2、USTRF 或 RCLE 权限，不更改 Android 默认模型、风险规则或反馈行为，
也不产生真人助行、产品或安全结论。

## 固定 cohort

- 本地 canonical root：
  `artifacts.local/evidence/datasets/blindassist-sanpo-v2-event-labeled-20260711`
- `manifest.jsonl` SHA-256：
  `3d7168ac975aed57ac6b437ecfa0e668c13dc5d509c7b6584353383eed19d217`
- `dataset_spec.json` SHA-256：
  `6815d8b613eca34d840255e66f02bd751196cb55e1b3b74f8ff0f659babf07bb`
- 90 帧、3 个 session、3 个 `risk_event_id`；每帧均有 source mask 与至少一个
  `source_region`，9 帧另有原 `objects`，因此 Arm B 不使用覆盖不足的 `objects`。
- 数据角色：既有 consumed Development / regression；只能支持当前 cohort 的机制诊断，
  不能称 fresh、unseen、independent validation 或 Confirmation。

parent natural event 固定为 `risk_event_id`。帧、重复 app run、mask component 或
detection 均不是独立样本。

## 三臂

所有臂固定 `riskConfig=current`、`AlertProfile.STANDARD`、
`AssistDecisionKernel.CONTRACT_ID`、每序列独立 reset、`frame_index × 100 ms`
因果时钟，以及相同 `RiskAnalyzer -> temporal/event tracking -> feedback planning`
决策链。

1. `A_CURRENT_YOLO`
   - 输入：默认 `yolo11n_fp16_320.tflite`，SHA-256
     `00edb41a528b0a7e709c4af8ce3e685491492c4539274804e5cfc17a1a867cd2`；
   - 风险输入只含当前 YOLO detections。
2. `B_ORACLE_RISK_BOX`
   - 输入：每帧 `source_regions` 中能由
     `BlindAssistSanpoTaxonomy.riskLabelFor` 映射的真值风险类别与其 source-native
     `bbox_xyxy`；
   - 每个框以 `confidence=1`、`DetectionSource.OBJECT_DETECTOR` 进入同一决策链；
   - 不把当前 YOLO detections 混入风险输入。
3. `C_ORACLE_RISK_MASK`
   - 输入：同帧 source-native semantic mask；
   - 只经现有 `TraversabilitySegmentationAnalyzer` 的固定 taxonomy、连通域、中心走廊
     与边界规则转换为 `DetectionSource.SEGMENTATION`；
   - 不把当前 YOLO detections 混入风险输入。

Arm B 与 Arm C 都是 truth oracle，不是可部署模型。其 adapter/YOLO 占位执行时间不得
用于模型速度、能耗或部署排序；本轮只比较信息进入同一现有决策链后的事件结果。

## 指标

主指标全部按 parent event 计算：

- `event_alert_recall`：至少一个 expected-alert frame 上实际交付提醒的正事件比例；
- `critical_event_miss_count`：含 expected critical alert、但未交付提醒的事件数；
- `false_alert_event_count` 与 `false_alerts_per_minute`；
- `first_effective_response_delay_frames`：每个正事件从首个 expected-alert frame 到
  首个实际提醒 frame；未提醒记为 miss，不用 0 代替；
- `post_event_clearance_rate`：含 `PASSED` phase 的事件中，passed window 无实际提醒的比例。

同时保留逐帧 risk、feedback reason、runtime event ID、source label/box 与已有 frame-level
指标作为诊断，不用逐帧样本数虚增独立性。

## 成对判定

对两个 arm `candidate` 与 `reference`，`MATERIAL_EVENT_GAIN` 要求：

1. 至少改善一个 parent-event 计数：多命中一个正事件、少一个 critical miss、少一个
   false-alert event，或多清除一个 passed event；
2. 上述其余 parent-event 计数均不恶化；
3. 若主计数完全相同，只有在全部共同命中的正事件均不更晚、且至少一个事件提前
   `>=2` 帧时，才记为 `TIMING_ONLY_GAIN`；它单独不足以宣称表征路线胜出。

终态：

- 仅 `B > A` 达到 `MATERIAL_EVENT_GAIN`：
  `DETECTOR_MODEL_OR_TAXONOMY_GAP_SUPPORTED`；
- `B` 不 materially 胜过 `A`，但 `C > B`：
  只有当 `C > A` 也同时 materially 成立时，才记
  `CURRENT_MASK_ADAPTER_AND_SOURCE_POLICY_GAIN_SUPPORTED`；该信号与 bbox 表征限制一致，
  但不单独确认 bbox 表征上限；
- `B > A` 与 `C > B`、`C > A` 同时 materially 成立，或出现不可支配 trade-off：
  `MIXED_DETECTOR_AND_REPRESENTATION_GAPS`；
- 两个比较均无 material/timing 增量：
  `DECISION_OR_MONOCULAR_OBSERVABILITY_BOTTLENECK_SUPPORTED`；
- 只有 timing 改善、没有 parent-event 计数改善：
  `TIMING_ONLY_GAIN_NO_ROUTE_CHANGE`；
- cohort、臂 membership、真值或执行合同不完整：
  `NOT_EVALUABLE`。

三事件结果只支持 Development 机制选择，不计算或声称总体显著性。
若未来要独立判断 bbox 几何上限，B/C 必须使用共同中性 source policy，并让 mask
像素形状特征进入共同风险表示；那是后继实验，不得在本轮 outcome 上补做。

## 完整性与停止条件

- 90 个 manifest ID 必须在三臂逐帧账本各出现一次；三臂必须共享相同
  `decision_kernel_contract_id`、risk config、alert profile、时钟和 reset 语义。
- 任一 mask、source region、event label、frame index、sequence membership 或臂输出缺失，
  validator fail closed。
- 独立 validator 必须从逐帧结果复算事件指标与终态，不能直接信任报告 aggregate。
- 得到一个有效终态后停止；不得在该 90 帧 outcome 上加阈值、改 taxonomy、改事件规则
  或挑子集救援。后继训练或新表征必须另立 Development 问题。
