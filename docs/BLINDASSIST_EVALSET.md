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
- 后续连续帧/逼近风险评测可选字段：
  - `sequence_id`: 同一短序列的稳定 ID。
  - `frame_index`: 序列内帧序号，`0` 是合法值。
  - `expected_approach_state`: `UNKNOWN | STABLE | APPROACHING | RECEDING`
  - `expected_approach_alert`: `true | false`
  - `expected_time_to_alert_frames`: 从期望逼近开始到应提醒的帧数。
- 场景覆盖正前方近距离目标、侧向经过目标、远处大物体、近处小障碍、弱光或遮挡、走廊或户外慢行。

## 生成方式

默认使用 COCO 2017 validation 图片和实例标注作为第一版公开真实数据源：

```powershell
& E:\codex-tools\bin\blindassist-python.cmd scripts\build_blindassist_evalset.py
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
    model_review_checklist.csv
    *_boxed.png
```

`manifest.jsonl` 是唯一真源。YOLO 和 COCO 文件只作为工具兼容导出；COCO 导出保持标准 `images/categories/annotations` 结构，不写入 BlindAssist 专用风险字段。

## 标注口径

- `expected_risk_direction` 根据主要风险目标的框中心位置分为左、中、右。
- `expected_distance_band` 使用目标框底部位置和面积比例生成预标注，字段值与 `RiskAnalyzer` 的 `ProximityBand` 对齐。
- `expected_should_alert` 按标准提醒档位生成预标，不等同于当前算法一定会触发提醒。
- `primary_object_id` 指向当前样本中用于风险预标的 COCO annotation id。
- `review_status=pending_model_review`，`status=pending_review`；只有 GPT 多模态与 Codex 证据复核形成独立共识后才能进入评测。

## 质量检查

生成后运行：

```powershell
& E:\codex-tools\bin\blindassist-python.cmd C:\Users\26442\.codex\skills\synthetic-vision-dataset\scripts\validate_yolo.py --dataset <dataset_dir>
& E:\codex-tools\bin\blindassist-python.cmd C:\Users\26442\.codex\skills\synthetic-vision-dataset\scripts\coco_from_manifest.py --dataset <dataset_dir> --splits test
& E:\codex-tools\bin\blindassist-python.cmd C:\Users\26442\.codex\skills\synthetic-vision-dataset\scripts\make_preview.py --dataset <dataset_dir> --limit 150
```

当前 Codex/GPT 会话自动读取：

```text
<dataset_dir>/qa/preview.html
<dataset_dir>/qa/model_review_checklist.csv
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

连续场景扩展已于 2026-07-11 启动，首批使用官方 SANPO-Real CC BY 4.0 数据，流程见 [SANPO 连续场景试验集](SANPO_SEQUENCE_EVALSET.md)。该工作流将 15 FPS 原始序列重采样到现有 benchmark 对齐的 10 FPS，保留全部 SANPO 分割区域，但 GPT/Codex 共识复核前不生成 canonical `manifest.jsonl`。

Open Images、VIP-Mobility360、GND 和 LAVN 仍适合作为后续扩展来源，但需要独立确认下载体积、相机视角、许可证和标注格式后再合入。PEDESTRIAN 论文给出的 Zenodo DOI 与 GitHub 仓库在 2026-07-11 均不存在，暂不接入，也不把论文开放状态推断成数据集许可证。

原图只保存在本地忽略目录，不随仓库分发。若未来要公开评测集，只应公开脚本、图片来源 id、派生标注和复现说明，并重新核对每个数据源的许可要求。
