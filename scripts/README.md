# BlindAssist 脚本索引

脚本暂时保持在 `scripts/` 根目录，避免一次性搬迁破坏已有命令、CI 和文档引用。新增脚本应放入下列职责之一；后续按类别分批迁移，并为旧入口保留过渡。

## 仓库、发布与环境

- `check_repo_hygiene.ps1` / `test_repo_hygiene.ps1`：仓库卫生门禁与测试。
- `check_docs_index.ps1` / `test_check_docs_index.ps1`：顶层文档索引与本地链接门禁。
- `archive_apk.ps1`、`verify_release_apk.ps1`、`verify_apk_16kb.ps1`：APK 校验与归档。
- `restore_codex_skills.ps1`：恢复仓库内 skills 快照。

## 模型导出与静态检查

- `export_yolo11n_tflite.py`、`inspect_tflite.py`
- `export_depth_anything_v2_tflite.py`、`inspect_depth_model.py`
- `smoke_depth_model.py`、`smoke_depth_anything_v2_pytorch.py`
- `fetch_remote_zip_members.py`

模型源文件、下载和导出结果应写入 `artifacts.local/models/` 或 `artifacts.local/downloads/`。

## 检测器与设备 benchmark

- `prepare_coco100.py`、`detector_lab.py`、`benchmark_tflite_detectors.py`
- `run_yolo26n_device_benchmark.ps1`
- `run_detector_ab_device_benchmark.ps1`
- `run_depth_fusion_benchmark.ps1`
- `run_device_regression.ps1`

设备脚本可能调用 Gradle、ADB 并覆盖手机上的 debug 包；运行前确认设备和目标 module。证据统一写入 `artifacts.local/evidence/`。

## SANPO 数据集构建与治理

- 发现与构建：`discover_sanpo_sequence_candidates.py`、`build_sanpo_sequence_evalset.py`、`build_blindassist_evalset.py`
- 克隆与合并：`clone_sanpo_sequence_evalset.py`、`clone_sanpo_event_phase_evalset.py`、`merge_sanpo_sequence_evalsets.py`
- 选择与定稿：`select_sanpo_sequence_by_geometry.py`、`finalize_sanpo_sequence_evalset.py`
- 复核：`create_sanpo_v2_review_decisions.py`、`apply_sanpo_review_decisions.py`、`review_sanpo_sequence_with_model.py`、`screen_sanpo_mask_windows.py`
- v3 治理：`build_sanpo_v3_annotation_queue.py`、`prepare_sanpo_v3_dataset_views.py`、`validate_sanpo_v3_dataset.py`、`freeze_sanpo_v3_regression.py`
- 公开与程序化数据：`build_public_v3_canonical_dataset.py`、`build_procedural_tactile_sequences.py`
- session 拆分：`plan_sanpo_p3_session_split.py`

改变 canonical 数据、冻结回归集或读取 blind 数据的脚本必须遵守 [SANPO 训练协议](../docs/SANPO_TRAINING_PROTOCOL.md)。

## SANPO 训练、导出与门禁

- 模型与训练：`sanpo_segmentation_model.py`、`train_sanpo_segmentation_keras_torch.py`
- 导出：`train_export_sanpo_segmentation.py`
- 门禁：`sanpo_training_gate.py`、`sanpo_candidate_quality_gate.py`
- 后端一致性：`sanpo_backend_equivalence.py`
- 辅助损失与诊断：`sanpo_boundary_distance_aux.py`、`sanpo_deterministic_linear_probe.py`

对应的 `test_*.py` 与实现保持同名配对。候选只有在离线、INT8 和设备门全部通过后才允许进入正式 App。

## SANPO benchmark

- `benchmark_sanpo_traversability.py`
- `benchmark_sanpo_gpu_throughput.py`

## 运行约定

- 从仓库根目录执行脚本。
- 本机工具来自 `E:\codex-tools`；通用 Python 检查优先使用 `E:\codex-tools\bin\blindassist-python.cmd`，旧 `.jdk/.android-sdk/.venv*` 入口仅用于迁移兼容。
- 所有下载、数据集、benchmark、训练输出和临时文件进入 `artifacts.local/`。
- 脚本若需要联网、GPU 或 ADB，应在命令或文档中显式说明。
- 新增脚本时同时补充配套测试、文档链接和默认输出目录。
- 文档治理脚本不产生项目输出；文档变更完成前运行 `scripts/check_docs_index.ps1`。
