# BlindAssist 脚本索引

`scripts/` 根目录只保留跨领域稳定 Interface、当前共享 Implementation 和兼容入口；完成或冻结的研究 campaign 下沉到 `scripts/research/<domain>/`。调用方不应依赖研究子目录的内部布局。

## 稳定入口

- `run_research_contract_tests.py`：CI 与本地共用的无 GPU、无设备研究合同回归。
- `run_research_tool.py public-video <tool.py> [args...]`：通过统一 Adapter 运行已归档的公开视频研究工具。
- `run_public_video_campaign_tests.py`：发现并运行 `scripts/research/public_video/` 的完整测试集。
- `run_public_video_edge_inference.ps1`：已冻结 campaign 真机闭环的稳定 Adapter；调用方不依赖研究目录内部路径。
- `check_repo_hygiene.ps1` / `test_repo_hygiene.ps1`：仓库卫生门禁与测试。
- `check_project_structure.ps1` / `test_check_project_structure.ps1`：脚本根 allowlist、开发日志预算、研究 Module 合同、内部路径和跨 Module import 门禁；仓库卫生检查会自动调用它。
- `check_docs_index.ps1` / `test_check_docs_index.ps1`：顶层文档索引与本地链接门禁。
- `archive_apk.ps1`、`verify_release_apk.ps1`、`verify_apk_16kb.ps1`：APK 校验与归档。

## 领域模块

- [`research/public_video/`](research/public_video/)：已冻结的公开视频 / public-silver 历史 campaign。细粒度语义索引和迁移说明保留在该目录，不再向根目录增加实验轮次脚本。
- [`research/common/`](research/common/)：至少两个研究域真实复用的共享 Implementation；领域规则和授权不得进入该 Module。
- [研究 Module 模板](research/README.md)：新路线必须声明稳定 Interface、输出、安全边界与停止条件。
- 根目录 USTRF-SC、SANPO 数据治理、训练与 benchmark 脚本：当前仍共享少量 SANPO 模型/证据 helper，待形成独立稳定 Interface 后再按域下沉，禁止为追求目录外观一次性拆断依赖网。

## 模型、设备与数据约定

- 模型导出/检查：`export_yolo11n_tflite.py`、`inspect_tflite.py`、`export_depth_anything_v2_tflite.py`、`inspect_depth_model.py`。
- detector/device benchmark：`detector_lab.py`、`benchmark_tflite_detectors.py`、`run_yolo26n_device_benchmark.ps1`、`run_detector_ab_device_benchmark.ps1`、`run_device_regression.ps1`。
- SANPO 训练与门禁：`train_sanpo_segmentation_keras_torch.py`、`train_export_sanpo_segmentation.py`、`sanpo_training_gate.py`、`sanpo_candidate_quality_gate.py`。
- USTRF-SC 当前主线必须遵守 [窗口交接](../docs/research/ustrf-sc/USTRF_SC_WINDOW_HANDOFF_2026-07-20.md)：route-conditioned、object-agnostic risk field 优先；真实事件真值与设备米制几何是硬门；detector 只能进入独立 crop-view FP 抑制实验。

## 运行约定

- 从仓库根目录执行；通用 Python 优先使用 `E:\codex-tools\bin\blindassist-python.cmd`。
- 下载、数据集、benchmark、训练输出和临时文件进入忽略的 `artifacts.local/`。
- 需要联网、GPU 或 ADB 的脚本必须显式说明；设备脚本运行前确认目标设备与 module。
- 新增 runnable script 必须归入明确领域 Module，并同时更新本索引或该领域 README；不要重新把研究轮次平铺回根目录。
- `scripts/policy/root-files.txt` 是根目录精确清单；只有真正稳定的 Interface 或跨域共享 Implementation 才能经评审加入。
- `DEVELOPMENT_LOG.md` 上限为 1500 行、300000 bytes、最老条目 28 天；超限时按月原文归档并在根日志保留链接。
- 改变 canonical 数据、冻结回归集或读取 blind 数据时遵守 [SANPO 训练协议](../docs/SANPO_TRAINING_PROTOCOL.md)。
- 文档变更完成前运行 `scripts/check_docs_index.ps1`。
