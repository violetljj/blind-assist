# SANPO 四类分割候选（benchmark-only）

执行范围：本页定义一个具体 benchmark-only implementation，不是论文 Development 的
唯一候选或强制晋级链。默认 R4 Development 可先用其他模型、host 输出或 synthetic
decoder canary 比较 utility；只有默认模型替换/生产晋级才必须进入完整 INT8 与同机事件门。

`scripts/train_export_sanpo_segmentation.py` 是 MobileNetV3Small + Lite R-ASPP 的训练与全 INT8 TFLite 导出工具。它是 benchmark-only 工具：默认模型只写入 Git 忽略的 `device-benchmark/benchmark-assets.local/segmentation/`，并且脚本明确拒绝 `app/src/main/assets/` 下的任何输出路径。生产 APK 继续只使用 `yolo11n_fp16_320.tflite`。

## 固定模型契约

- 输入：`[1,256,256,3]` NHWC RGB，TFLite `int8`。调用方按 input tensor 的 scale/zero-point 量化 0–255 RGB；模型内完成 `/255`。
- 输出：`[1,256,256,4]` NHWC `int8` logits；按 `argmax` 得到类别 ID。
- 类别顺序固定：`0 walkable`、`1 boundary_step_curb`、`2 obstacle`、`3 unknown_nonwalkable`。
- 导出强制 `TFLITE_BUILTINS_INT8`、INT8 输入/输出和代表性数据集；导出后重新打开模型验证 shape、dtype、quantization scale 和 size。

`boundary_step_curb` 只是边界证据，`unknown_nonwalkable` 只是诊断证据；模型或离线
mIoU 不构成上线条件。只有进入 `PRODUCTION_PROMOTION` 时，其结果才必须经过现有
`DetectionSource.SEGMENTATION` 规则和同机 A/B 门禁；普通 Development 可以在不接入
App 的前提下先报告 utility 和 runtime。

## 训练数据访问契约

工具只接受显式的 `--manifest` JSONL，不接受 dataset root，也不会递归发现文件。传入的文件必须是**单独的 train/dev manifest**；任何 `blind`、`blind_holdout`、`holdout` 或 `test` 行会在打开图片/掩码前失败。这样盲测集标签没有进入训练、量化代表数据或阈值选择过程。

每行至少需要以下字段（路径相对 manifest 所在目录）。v3 canonical 格式优先使用单张 `semantic_mask_path` 灰度 PNG，其中像素 ID 固定为 `0..3`；训练工具会校验尺寸和 ID 范围：

```json
{
  "id": "session_a_0001",
  "segmentation_split": "train",
  "source_session_id": "session_a",
  "image_path": "images/session_a_0001.jpg",
  "scene_bucket": "center_obstacle",
  "semantic_mask_path": "semantic_masks/session_a_0001.png"
}
```

为兼容已存在的标注工具，也接受以下四张二值 mask 格式：

```json
{
  "id": "session_a_0001",
  "segmentation_split": "train",
  "source_session_id": "session_a",
  "image_path": "images/session_a_0001.jpg",
  "scene_bucket": "center_obstacle",
  "semantic_mask_paths": {
    "walkable": "masks/session_a_0001_walkable.png",
    "boundary_step_curb": "masks/session_a_0001_boundary.png",
    "obstacle": "masks/session_a_0001_obstacle.png",
    "unknown_nonwalkable": "masks/session_a_0001_unknown.png"
  }
}
```

四张二值 mask 必须均为与图片同尺寸、互不重叠且完整覆盖每个像素；不确定区域必须标成 `unknown_nonwalkable`。同一个 `source_session_id` 出现在 train/dev 两个 split 会被拒绝，防止连续帧泄漏。脚本报告会写入 manifest SHA256、session/scene/class 像素统计和 `blind_holdout_access=not_accessed` 证明。

## 运行与验证

先由数据流水线生成不包含盲测标签的明确 manifest，再运行：

```powershell
& E:\codex-tools\bin\blindassist-python.cmd scripts\train_export_sanpo_segmentation.py `
  --manifest artifacts.local\evidence\datasets\sanpo-v3\manifests\train_dev.jsonl `
  --epochs 40 --batch-size 8
```

会生成：

- `device-benchmark/benchmark-assets.local/segmentation/mobilenetv3_lraspp_int8_256.tflite`
- `test-artifacts.local/segmentation-candidate/latest/segmentation_candidate_report.json`（逐类 IoU、混淆矩阵、TFLite 合同、量化和训练元数据）

基础 Python 合同测试不要求 TensorFlow：

```powershell
& E:\codex-tools\bin\blindassist-python.cmd -m unittest scripts\test_train_export_sanpo_segmentation.py
& E:\codex-tools\bin\blindassist-python.cmd scripts\train_export_sanpo_segmentation.py --help
```

完成训练不表示可进入 shadow mode。只有全量开发集和完全隔离盲测集均通过事件召回、关键漏报、平行路沿零提醒、误提醒和 SM-S9280 P95 门槛，才可另行讨论 shadow mode。
