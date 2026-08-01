# Objective image-space candidate increment D0

状态：
`COMPLETE / VALID / STOP_FIXED_PIDNET_OBJECTIVE_CANDIDATE_NO_ROBUST_INCREMENT`

权威结果见
[`OBJECTIVE_IMAGE_SPACE_CANDIDATE_INCREMENT_D0_RESULT_2026-08-02.md`](../../../docs/research/dual-loop/OBJECTIVE_IMAGE_SPACE_CANDIDATE_INCREMENT_D0_RESULT_2026-08-02.md)。
固定 raw PIDNet candidate operator 已关闭；以下命令仅保留为可复算接口，不得在已消费
cohort 上改门、换 seed 或重跑以选择结果。

本包实现
`OBJECTIVE_IMAGE_SPACE_CANDIDATE_INCREMENT_D0` 的 objective-only 数据视图、固定
PIDNet/Yolo 三臂 evaluator 和独立 aggregate/terminal validator。

顺序固定为：

1. `prepare_objective_view` 从已消费 device view 中剥离所有 event/action 字段；
2. `prepare_yolo_manifest` 生成 exact host-YOLO 输入；
3. 提交并推送协议和实现；
4. `dual_loop_segmentation_complementarity.produce_host_trace` 运行一次冻结 YOLO；
5. `evaluate` 运行一次固定 seed-20260801 PIDNet；
6. `validate` 从 `mask_ledger.npz` 独立复算。

示例：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.dual_loop_segmentation_complementarity.produce_host_trace `
  --manifest artifacts.local/evidence/objective-image-space-candidate-increment-d0/input-v1/yolo_manifest.jsonl `
  --model app/src/main/assets/yolo11n_fp16_320.tflite `
  --labels app/src/main/assets/coco_labels.txt `
  --output artifacts.local/evidence/objective-image-space-candidate-increment-d0/host-yolo-v1/trace.jsonl `
  --receipt artifacts.local/evidence/objective-image-space-candidate-increment-d0/host-yolo-v1/receipt.json `
  --progress artifacts.local/evidence/objective-image-space-candidate-increment-d0/host-yolo-v1/progress.json

E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.objective_image_space_candidate_increment_d0.evaluate `
  --protocol docs/research/dual-loop/OBJECTIVE_IMAGE_SPACE_CANDIDATE_INCREMENT_D0_PROTOCOL_2026-08-02.json `
  --manifest artifacts.local/evidence/objective-image-space-candidate-increment-d0/input-v1/objective_manifest.jsonl `
  --trace artifacts.local/evidence/objective-image-space-candidate-increment-d0/host-yolo-v1/trace.jsonl `
  --trace-receipt artifacts.local/evidence/objective-image-space-candidate-increment-d0/host-yolo-v1/receipt.json `
  --model artifacts.local/evidence/riskseg-r0/trained-export-v1/seed-20260801/tflite_export_locked_v2/pidnet_s_512x288_4class_seed_20260801_simplified_full_integer_quant.tflite `
  --output-dir artifacts.local/evidence/objective-image-space-candidate-increment-d0/d0-v1
```

本包禁止读取 `positive/bucket/alertable/passed/risk/event/feedback`，也禁止把冻结
梯形 ROI 称为真实路线。当前数据不满足 onset 门，timing 永久为 `NOT_EVALUABLE`。
