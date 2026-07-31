# DUAL_LOOP_SEGMENTATION_CANDIDATE_UTILITY_R0

状态：COMPLETE / VALID / CURRENT_SEGMENTATION_REFERENCE_REJECTED / DEVELOPMENT_ONLY

这是一个 host-only、Development-only 的 source-native pixel/component utility
评估器。它回答一个窄问题：固定 segmentation reference 在 YOLO box union
之外是否增加可验证的 obstacle 或 boundary_step_curb 像素/组件，以及增量是否
值得 false activation、时间抖动和计算成本。

## 冻结输入

- source-native truth 使用 SANPO-Real v0 canonical R3：
  test-artifacts.local/datasets/sanpo-v4-real-canonical-r3-20260713。
- calibration 只使用 training_manifest.jsonl 的 dev split。
- formal 只使用 blind_holdout/manifest.jsonl 的 blind split。blind mask 只能
  在正式执行时读取，不能用于阈值选择或训练。
- 主 hazard 是 semantic class id 1 boundary_step_curb 与 id 2 obstacle；
  unknown_nonwalkable 是独立 ablation，不进入默认 utility。
- YOLO trace 必须使用 source_id、frame_id、image_sha256 精确配对；
  detector source 必须是 OBJECT_DETECTOR。
- segmentation model 必须是固定 256x256x3 int8 -> 256x256x4 int8 contract。

## 稳定 Interface

输入是 protocol JSON、canonical 或 normalized JSONL manifest、严格 identity-paired
YOLO trace、固定 INT8 TFLite model，以及可选的 2x3 previous-to-current motion
sidecar。evaluator 的 phase 只有 calibration、formal、temporal；validator 只接受
与 protocol、report、frames、components 同身份的结果。

## 三臂和输出

- A：YOLO-only，D_t 为所有 projected YOLO rectangles 的 union。
- B：segmentation-only，H_t 为 class id 1/2 的 union。
- C：YOLO + segmentation，D_t union H_t。
- candidate component 是 H_t minus D_t；truth counterpart 是 source-native
  hazard minus D_t。

## 输出

evaluator 生成 report.json、frames.jsonl、components.jsonl 和 progress.json。
frames 保存 A/B/C confusion、candidate pixel/component metric、unknown ablation、
runtime 和 packed candidate masks；components 保存 class、area、bbox、confidence、
top1-top2 margin、truth intersection、nearest YOLO distance 以及 temporal track
字段。

validator 不重新信任 evaluator 的 aggregate，而是从 frame confusion 和
component rows 独立复算。formal 只允许两个结果：

- CANDIDATE_UTILITY_SUPPORTED
- CURRENT_SEGMENTATION_REFERENCE_REJECTED

输入/身份/SHA/时序契约失败时为 CANDIDATE_UTILITY_NOT_EVALUABLE。校准只产生
CALIBRATION_VALID，不产生正式终态。

## 运行示例

在仓库根目录执行，输出目录应位于 artifacts.local：

    python -m scripts.research.dual_loop_segmentation_candidate_utility.evaluate_candidate_utility --phase calibration --split dev --protocol docs/research/dual-loop/DUAL_LOOP_SEGMENTATION_CANDIDATE_UTILITY_R0_PROTOCOL.json --manifest test-artifacts.local/datasets/sanpo-v4-real-canonical-r3-20260713/training_manifest.jsonl --dataset-root test-artifacts.local/datasets/sanpo-v4-real-canonical-r3-20260713 --trace <normalized-yolo-trace> --model artifacts.local/evidence/segmentation-candidate/sanpo-v3-pretrained-weighted-best-int8-20260713.tflite --report <calibration-dir>/report.json

formal 命令只替换为 blind_holdout/manifest.jsonl、blind split 和正式 trace；
不能用 calibration 结果重新调 gate。若没有 motion sidecar，报告仍保留
motion-warped_iou 字段并明确 motion_warp_available=false；不能用 identity
变换冒充 motion compensation。

## 安全边界

本模块不读取 central obstruction Agent labels、risk、event 或 feedback，
不接 Android、QNN、热功耗、主动提醒或安全效果。通过本模块只代表
source-native pixel/component utility Development evidence，不代表真实世界
可通行性、辅助产品安全性或设备端性能。

## 停止条件

formal validator 只允许 CANDIDATE_UTILITY_SUPPORTED 或
CURRENT_SEGMENTATION_REFERENCE_REJECTED；输入、身份、SHA、truth、时间序列或
tensor contract 失败时为 CANDIDATE_UTILITY_NOT_EVALUABLE。当前 reference 已因
false activation 与 incremental host cost gate 失败而关闭，任何新模型或融合算子
必须另立 protocol，不得在本模块内调门救援。
