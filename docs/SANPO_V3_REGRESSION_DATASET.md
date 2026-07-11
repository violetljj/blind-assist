# SANPO v3 回归基线与分割数据门禁

这套契约把现有三条已复核的 30 帧 SANPO 连续序列冻结为 `sanpo-v3-regression-90f`，并规定后续四类轻量分割候选的数据边界。所有 RGB、来源掩码、四类语义掩码和设备报告仍只留在 Git 忽略的 `test-artifacts.local/`；仓库只保存脚本、schema 和门禁规则。

## 90 帧不可变基线

冻结前应先把同机 A/B 的命令参数写为 JSON，例如 `runs_per_frame`、设备型号、数据集路径和比较模式。冻结命令复制这 90 个帧/来源掩码，并写入 source/frozen manifest SHA256、每个 RGB/掩码 SHA256、当前 Git revision、配置 SHA256 和同机报告引用。输出目录一旦存在即拒绝覆盖。

```powershell
.\.venv-export312\Scripts\python.exe scripts\freeze_sanpo_v3_regression.py `
  --source-root test-artifacts.local\datasets\blindassist-sanpo-v2-event-labeled-20260711 `
  --output-root test-artifacts.local\datasets\sanpo-v3-regression-90f `
  --benchmark-config test-artifacts.local\detector-ab-device-benchmark\<report>\benchmark-config.json `
  --device-report test-artifacts.local\detector-ab-device-benchmark\<report>

.\.venv-export312\Scripts\python.exe scripts\freeze_sanpo_v3_regression.py `
  --output-root test-artifacts.local\datasets\sanpo-v3-regression-90f --verify
```

锁定本身不是候选晋级：现有 oracle 的 90 帧误提醒失败记录保留在同机报告中，继续仅作为语义上限参考；默认 YOLO 几何路径仍是安全下限。

## v3 四类语义数据结构

每个 RGB 一张单通道 `semantic_mask_path` PNG，类别 ID 固定为：`0=walkable`、`1=boundary_step_curb`、`2=obstacle`、`3=unknown_nonwalkable`。每行必须有 `image_sha256`、`semantic_mask_sha256`、`scene_bucket`、`session_id`、`sequence_id`、连续 `frame_index`、`risk_event_id`、`expected_event_phase` 和布尔 `expected_should_alert`。来源字段必须记录 `dataset`、`license`、`license_url`、`privacy_review_status`。

`scene_bucket` 只能为：

- `parallel_boundary`
- `step_curb`
- `center_obstacle`
- `lateral_pedestrian_or_ebike`
- `low_light`
- `tactile_paving_occupied`

同一个 `session_id` 不得跨 `train`、`dev`、`blind`。同一连续序列也不得混 split 或场景桶。

## 模型复核（替代人工初筛）

每个 SANPO 连续序列草稿先运行 `review_sanpo_sequence_with_model.py`，它会在草稿的 `qa/` 目录生成带 SHA256 的三帧证据请求（首、中、末帧）。大模型必须返回其型号/版本、场景桶、走廊事件判断、`alert` 或 `no_alert`、置信度、全部证据帧及局限性；脚本会校验这些字段并写入不可篡改引用的结果记录。

```powershell
python scripts\review_sanpo_sequence_with_model.py `
  --draft-root test-artifacts.local\datasets\<draft>

python scripts\review_sanpo_sequence_with_model.py `
  --draft-root test-artifacts.local\datasets\<draft> `
  --response test-artifacts.local\datasets\<draft>\qa\model_review_response.json
```

模型复核的 `accept_for_dense_annotation` 只允许进入**四类像素级掩码标注队列**；它不会直接产生掩码、事件标签或 v3 训练/盲测数据。`reject` 与 `needs_recapture` 一律保持草稿状态。尤其是平行路沿和侧向目标可以是有效的 `no_alert` 负例，不能因为不在走廊内而被迫改写为障碍正例。任何 v3 训练或 benchmark 晋级仍必须通过下列 420 帧、哈希、会话隔离与盲测锁门禁。

## 420 帧覆盖与盲测锁

已复核的源 manifest 必须先以明确 split 写出，然后只生成两个可消费视图：

```powershell
.\.venv-export312\Scripts\python.exe scripts\prepare_sanpo_v3_dataset_views.py `
  --source-manifest <reviewed-source-manifest.jsonl> `
  --dataset-root test-artifacts.local\datasets\blindassist-sanpo-v3

.\.venv-export312\Scripts\python.exe scripts\validate_sanpo_v3_dataset.py `
  --dataset-root test-artifacts.local\datasets\blindassist-sanpo-v3 `
  --require-v3-coverage `
  --report test-artifacts.local\datasets\blindassist-sanpo-v3\qa\v3-gate-report.json
```

严格覆盖门槛是六条 train/dev 连续序列各 50 帧（总计 300 帧，六类场景各一条，且 train 与 dev 都存在）以及两条独立 blind 连续序列各 60 帧（总计 120 帧）。因此总数必须是 420。

训练唯一允许的输入是根目录 `training_manifest.jsonl`，其中只能包含 train/dev 行。`blind_holdout/manifest.jsonl` 保存盲测标签，`access_policy.json` 把它显式标记为 `benchmark_only` 和训练禁止路径。训练工具必须接收明确 `--manifest training_manifest.jsonl`，不得递归扫描数据根目录或读取 `blind_holdout/`；验证器会拒绝任何 blind 行进入训练视图、训练/盲测样本重叠、会话泄漏、哈希变化、缺失四类掩码或覆盖不足。

该逻辑锁不是 Windows 同一用户下的安全边界：拥有整个数据根目录写权限的人仍可绕过它。因此盲测目录应只在 benchmark 执行机保管，训练执行机只复制 `training_manifest.jsonl` 和其引用的非盲素材。
