# BlindAssist 专用真实助行评测集

本文档记录 BlindAssist 下一轮算法优化使用的小型真实图片评测集流程。该评测集用于离线对比检测器、风险方向、相对距离层级和提醒决策，不是训练集，也不能作为助盲安全效果的完整证明。

## 目标

- 先建立 `150` 张真实图片/帧的本地评测集。
- 每张图片保留标准检测框，并额外标注 BlindAssist 关心的风险字段：
  - `expected_risk_direction`: `NONE | LEFT | CENTER | RIGHT`
  - `expected_distance_band`: `FAR | MID | NEAR | CRITICAL`
  - `expected_should_alert`: `true | false`
  - `expected_risk_level`: `NONE | LOW | MEDIUM | HIGH`
  - `assist_scenario`: `GENERAL | INDOOR | CORRIDOR | CROWDED | OUTDOOR_SLOW`
- 场景覆盖正前方近距离目标、侧向经过目标、远处大物体、近处小障碍、弱光或遮挡、走廊或户外慢行。

## 生成方式

默认使用 COCO 2017 validation 图片和实例标注作为第一版公开真实数据源：

```powershell
.\.venv-export312\Scripts\python.exe scripts\build_blindassist_evalset.py
```

脚本会优先复用本机已有缓存：

```text
.downloads/detector-lab/datasets/coco100/
```

缺少的 COCO val2017 图片会按需从 `http://images.cocodataset.org/val2017/` 下载。输出目录形如：

```text
test-artifacts.local/datasets/blindassist-evalset-YYYYMMDD-HHMMSS/
```

该目录被 `.gitignore` 中的 `test-artifacts*/` 忽略，原图仅用于本机内部评测，不提交 Git。

## 输出结构

```text
test-artifacts.local/datasets/blindassist-evalset-*/
  dataset_spec.json
  generation_records.jsonl
  manifest.jsonl
  source_licenses.md
  images/test/
  labels_yolo/test/
  annotations/instances_test.json
  qa/
    preview.html
    report.json
    yolo_validation.json
    blindassist_manifest_validation.json
    manual_review_checklist.csv
    *_boxed.png
```

`manifest.jsonl` 是唯一真源。YOLO 和 COCO 文件只作为工具兼容导出；COCO 导出保持标准 `images/categories/annotations` 结构，不写入 BlindAssist 专用风险字段。

## 标注口径

- `expected_risk_direction` 根据主要风险目标的框中心位置分为左、中、右。
- `expected_distance_band` 使用目标框底部位置和面积比例生成预标注，字段值与 `RiskAnalyzer` 的 `ProximityBand` 对齐。
- `expected_should_alert` 按标准提醒档位下的人类助行预期预标，不等同于当前算法一定会触发提醒。
- `primary_object_id` 指向当前样本中用于风险预标的 COCO annotation id。
- `review_status` 当前为 `accepted_auto_prelabel_needs_human_visual_review`，表示已通过自动格式校验并进入评测导出，但仍建议人工逐张确认视觉语义。

## 质量检查

生成后运行：

```powershell
.\.venv-export312\Scripts\python.exe C:\Users\26442\.codex\skills\synthetic-vision-dataset\scripts\validate_yolo.py --dataset <dataset_dir>
.\.venv-export312\Scripts\python.exe C:\Users\26442\.codex\skills\synthetic-vision-dataset\scripts\coco_from_manifest.py --dataset <dataset_dir> --splits test
.\.venv-export312\Scripts\python.exe C:\Users\26442\.codex\skills\synthetic-vision-dataset\scripts\make_preview.py --dataset <dataset_dir> --limit 150
```

人工复核时打开：

```text
<dataset_dir>/qa/preview.html
<dataset_dir>/qa/manual_review_checklist.csv
```

重点检查：

- 框是否覆盖目标，是否有明显类别错配。
- 主要风险目标是否选对。
- 方向、距离层级、是否应提醒是否符合 BlindAssist 助行预期。
- 是否存在重复图、低质量图、许可证不清或不适合内部保留的隐私风险。

## 来源与限制

第一版生成脚本使用 COCO 2017 validation 图片和实例标注。当前首版本地目录为：

```text
test-artifacts.local/datasets/blindassist-evalset-20260527-impl/
```

Open Images、LOCO、GND 和 LAVN 仍适合作为后续扩展来源，但需要独立确认下载方式、许可证和标注格式后再合入。

原图只保存在本地忽略目录，不随仓库分发。若未来要公开评测集，只应公开脚本、图片来源 id、派生标注和复现说明，并重新核对每个数据源的许可要求。
