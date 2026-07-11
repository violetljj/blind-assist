# BlindAssist SANPO 连续场景试验集

本流程从官方 SANPO-Real 公开存储桶选择少量真实步行序列，生成 BlindAssist 连续场景候选集。它用于补足静态 COCO 图片无法覆盖的连续运动、可通行区域、低矮障碍和场景变化证据。

## v2 公开序列扩展工具

- `scripts/discover_sanpo_sequence_candidates.py`：只扫描 SANPO-Real 官方 session 的稀疏 mask，输出许可、official split、类别命中和推荐起始帧；发现阶段不下载 RGB。
- `scripts/create_sanpo_v2_review_decisions.py`：仅为已经视觉复核的平行边界、正前方台阶、中心通道障碍生成 provenance-marked review 决策。
- `scripts/clone_sanpo_sequence_evalset.py`：canonical manifest 不可改写；若需要修正 review 时序，复制为新 draft 后重新 review/finalize。
- `scripts/merge_sanpo_sequence_evalsets.py`：只合并已 finalize 的 SANPO sequence，校验 ID/hash 唯一并复制本地忽略的图片/mask。

公开序列扩展不能把普通通道占用写成“盲道占用”。若来源没有明确连续、许可和语义证据，应保留为缺口而不是推断标签。

## 安全边界

- 导入结果默认是 `pending_review`，不是可直接运行 benchmark 的人工真值。
- SANPO 分割区域可以生成候选框，但不能自动决定 BlindAssist 的主要风险目标、方向、距离、提醒等级或逼近状态。
- 人工复核完成前，所有 `expected_*` 风险字段保持 `null`；不能把自动预标描述为助盲安全结论。
- 原始帧和掩码只写入被 Git 忽略的 `test-artifacts.local/datasets/`，不提交仓库。
- 每个下载对象都必须通过 SANPO GCS 官方 MD5；同时记录本地 SHA256。RGB、掩码和 session description 的尺寸不一致时立即失败。

## 首批官方来源

- 数据集：[SANPO](https://google-research-datasets.github.io/sanpo_dataset/)
- 代码与下载说明：[google-research-datasets/sanpo_dataset](https://github.com/google-research-datasets/sanpo_dataset)
- 许可证：Creative Commons Attribution 4.0 International
- 默认试验 session：`-5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG`
- 默认视角：`camera_chest/left`
- session 属性：城市道路交叉口、步行、晴天、高可见度、中等障碍、较高车辆流量。
- 原始帧率：15 FPS；导入时重采样为 10 FPS，与现有连续帧 benchmark 的 100ms 时间步一致。

此前调研的 PEDESTRIAN 数据集暂不接入。其论文所列 Zenodo DOI `10.5281/zenodo.10907945` 和 GitHub 仓库在 2026-07-11 均不可访问，许可证与文件哈希无法核验。

## 运行

```powershell
.\.venv-export312\Scripts\python.exe scripts\build_sanpo_sequence_evalset.py
```

默认下载 30 张 RGB 帧及对应分割掩码，约覆盖 3 秒连续场景。可显式调整：

```powershell
.\.venv-export312\Scripts\python.exe scripts\build_sanpo_sequence_evalset.py `
  --session-id=-5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG `
  --camera camera_chest `
  --lens left `
  --start-frame 0 `
  --target-fps 10 `
  --max-frames 30
```

输出目录：

```text
test-artifacts.local/datasets/blindassist-sanpo-pilot-*/
  dataset_spec.json
  manifest.draft.jsonl
  source_licenses.md
  source_session_description.json
  source_labelmap.json
  source_annotation_types.json
  images/test/
  source_masks/test/
  qa/
    preview.html
    boxed/
    manual_review_checklist.csv
    manifest_validation.json
    download_inventory.json
```

## 预标映射

SANPO 掩码的红通道保存类别 ID，后两通道保存实例 ID。脚本保留全部 `source_regions`，但只把语义完全一致的类别映射到当前 COCO `objects`：

- `pedestrian -> person`
- `traffic light -> traffic light`

`vehicle`、`animal`、`traffic sign` 和通用 `obstacle` 不会强行映射为 `car`、`dog`、`stop sign` 或其他假类别。它们保留在 `source_regions`，等待未来的未知障碍或可通行区域指标使用。

只有 SANPO 标为 `HUMAN_ANNOTATED` 的帧才会把上述精确映射写入 detection GT `objects`；机器分割帧只保留 `source_mapped_objects` 候选，不能直接进入检测框指标。非 COCO 主风险填写 `source_primary_region_id`，不得写入 `primary_object_id`，避免污染 `primaryObjectHitRate`。

## 复核与晋级

打开 `<dataset>/qa/preview.html`，逐帧填写 `manual_review_checklist.csv`：

1. 确认画面中真正影响行走的主要风险区域。
2. 补 `primary_object_id`；非 COCO 障碍应明确保留来源类别，不能伪造 COCO 类别。
3. 标注方向、距离、是否提醒和风险等级。
4. 按整个序列标注 `APPROACHING / STABLE / RECEDING`、期望提醒与首次提醒帧数。
5. 检查残留人脸、车牌、住址等隐私信息。
6. 对缺帧、坏框、重复帧、风险不确定和敏感内容使用统一 issue tag。
7. 存在 detection GT `objects` 的帧还必须让 `objects_review_status` 与复核方式一致；任何 `issue_tags` 未清空都会阻止 finalize。

人工复核仍是安全语义的首选。全部字段由人工确认时，使用 `review_status=accepted_manual_review`，然后生成 canonical `manifest.jsonl`：

```powershell
.\.venv-export312\Scripts\python.exe scripts\finalize_sanpo_sequence_evalset.py `
  --dataset-root <dataset-root>
```

finalize 会重新校验草稿/复核表哈希、图片和掩码 SHA256、路径范围、尺寸、bbox、COCO 类别、重复 ID/图片、官方 split、连续 `frame_index`、检测框复核状态和全部风险枚举。任何一行未复核都会拒绝生成 `manifest.jsonl`；canonical manifest 使用临时文件原子发布，且发布后的 dataset root 被视为不可变，避免旧或半写入 manifest 被 Gradle 误打包。

若当前阶段明确采用多轮 AI 工程复核，可使用独立决策文件和显式门禁：

```powershell
.\.venv-export312\Scripts\python.exe scripts\apply_sanpo_review_decisions.py `
  --dataset-root <dataset-root> `
  --decisions <review-decisions.json>

.\.venv-export312\Scripts\python.exe scripts\finalize_sanpo_sequence_evalset.py `
  --dataset-root <dataset-root> `
  --allow-ai-review
```

AI 路径要求每行 `review_status=accepted_ai_review`、`reviewer_type=ai_assistant`、非空 reviewer ID、置信度不低于 0.65，且至少两次独立复核；这些字段会原样写入 `review_provenance`。不传 `--allow-ai-review` 时一律拒绝。AI 复核只代表工程数据门禁通过，不能写成“人工已复核”，也不能替代盲人用户测试、受控路线验证或安全认证。

## 首批真机结果

2026-07-11 在 Samsung `SM-S9280` / Android 16 上以 `current` 风险配置运行 30 帧 Detector A/B benchmark：

- YOLO11n：FP/img `0.233`，FN/img `0.1`，错误提醒率 `0.033`，approach recall `0`，total P50/P95 `60/68ms`。
- YOLO26n：FP/img `0.433`，FN/img `0.1`，错误提醒率 `0.033`，approach recall `0`，total P50/P95 `47/48ms`。
- 两款模型都漏掉第 24、28 帧共 3 个 person GT，并且都未跟踪到 `sanpo_20_2` 通用障碍的逼近趋势；各自出现一次不应触发的提醒。
- 结论为 `do_not_replace_default_model`。YOLO26n 的速度优势不足以抵消更多误检，默认模型继续使用 YOLO11n。
- 随后的默认模型 90 秒 CameraX 真机回归通过。

这组序列的价值不是证明现有模型有效，而是明确暴露 COCO 类别检测器对通用障碍/可通行区域的结构性盲区。下一轮应优先比较可通行区域、语义分割或深度几何候选，而不是继续只替换 COCO detector。
