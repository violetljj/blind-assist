# BlindAssist 脚本索引

`scripts/` 根目录只保留跨领域稳定 Interface、当前共享 Implementation 和兼容入口；完成或冻结的研究 campaign 下沉到 `scripts/research/<domain>/`。调用方不应依赖研究子目录的内部布局。

## 稳定入口

- `run_research_contract_tests.py`：CI 与本地共用的无 GPU、无设备研究合同回归。
- `evaluate_ustrf_sc_u0_teacher_upper_bound.py`：USTRF U0 六臂（四个正式比较臂 + uniform/shuffled route 负控）LOSO 事件评价；会重算完整 route-conditioned GPT/Codex 共识真值门并拒绝 blind/future/hash 漂移。
- `run_ustrf_sc_u0_candidate_bundle.py`：U0 六臂统一 subprocess runner；只向 adapter 暴露去标签 inference manifest，冻结 500ms cadence、真实 route control、逐 fold LOSO artifact/训练收据及 kernel backend，再组装可 admission 的 v2 bundle。
- `run_ustrf_sc_u0_android_baseline_adapter.py`：`baseline_yolo_geometry` 真实 Android adapter；只搬运 hash-bound 去标签 request/video/ledger/artifact/config，最终 output 由真机 instrumentation 内的 shipped TFLite YOLO + shared Kotlin kernel 生成。需先用稳定 `.android-home` debug key 安装 app 与 `device-benchmark` APK，且只能作 U0 评测 adapter，不授权 App/模型替换。
- `run_ustrf_sc_u0_android_bbox_route_adapter.py`：`detector_bbox_explicit_route` 真实 Android adapter；设备按 500ms frame 因果选取最新显式路线 sample，用冻结路线走廊筛选未改写的 YOLO bbox 后调用同一 shared kernel，并输出逐 sample/bbox 门控回执；host 独立重算几何。unknown route 空 gate，不能用于 App 运行时或模型替换。
- `generate_ustrf_sc_u0_dense_teacher_loso_artifact.py`：第三臂 WIP 的 label-free、fold-local dense teacher 校准入口；只接受训练 fold 的许可视频与帧清单并输出 deterministic auxiliary-only artifact/receipt，不读取 GPT/Codex 共识事件真值、blind、future 或 held-out 输入，也不授权 Android/App/生产使用。dense field v2 使用百万分之一 fixed-point、分离 source/route-interaction hash，并由 admission 从序列化 cell 重算摘要；四个真实 dense/control adapter 与正式 evidence 仍缺失，不代表第三臂完成。
- `generate_sanpo_counterfactual_capture_plan.py --pilot`：生成固定 1 session × 5 scene × 1 matched pair 的 10-episode 空试采槽位；只描述采集责任，不生成证据或标签。
- `validate_ustrf_sc_route_conditioned_event_pilot.py`：试采链路审计；重算 video/clock/frame ledger/route 投影因果绑定及独立双审/裁决，只能输出 pipeline audit，永不产生 truth、U0、训练或运行时授权。
- `validate_ustrf_sc_model_proxy_event_pilot.py`：大模型生成的 5 场景/10 episode 代理 pilot 稳定审计入口；重算 contact sheet、1000 个解码帧、生成 lineage、两次隔离模型 review 及原始输出绑定。通过只允许扩正式代理矩阵，`human_truth/U0/training/runtime/production` 始终为 false。
- `validate_ustrf_sc_arcore_frame_bound_canary.py`：独占 ARCore `Session` 单 `Frame` 设备 canary 的 host 重算入口；从逐帧 JSONL 重算 Camera2 时间戳、camera image、raw depth/confidence、tracking 与持久 Anchor，门关闭以 exit 2 和 `FREEZE_FRAME_BOUND_METRIC_GEOMETRY` 收口。
- `validate_ustrf_sc_u0_prediction_bundle.py`：U0 预测证据 admission；重算 runner/registry、去标签输入、逐 fold LOSO provenance、route control、adapter request/output 与逐帧 shared-kernel trace，只允许从 feedback receipt 派生提醒；Android dense/control 臂另强制 teacher 许可证/权重/实现、fold、field、unknown 与归一化算术 receipt，当前缺真实 adapter 时 fail closed。
- `validate_ustrf_sc_capture_frame_ledger.py`、`validate_ai_review_receipt.py`：正式 full-matrix truth 与 pilot 共用的帧证据和 GPT/Codex receipt 验证 Implementation；由上述稳定入口调用。
- `validate_ustrf_sc_device_metric_geometry.py`：同设备米制几何总门；完整包要求五类 typed artifact 与设备/mount/calibration/metrics 精确绑定并继续 hash-bind raw/gate source，`blocked/in_progress` 包也会审计已有收据；通过只授权 isolated geometry shadow。
- `run_research_tool.py <domain> <tool.py> [args...]`：统一研究 Adapter；当前支持 `public-video` 历史归档、`ustrf-crosscam-codex` 代理评测、`ustrf-sensor-replay` 多来源 RGB-D+pose 回放和 `ustrf-route-target-evidence-closure` route-target 证据闭环域。
- `run_public_video_campaign_tests.py`：发现并运行 `scripts/research/public_video/` 的完整测试集。
- `run_public_video_edge_inference.ps1`：已冻结 campaign 真机闭环的稳定 Adapter；调用方不依赖研究目录内部路径。
- `check_repo_hygiene.ps1` / `test_repo_hygiene.ps1`：仓库卫生门禁与测试。
- `check_project_structure.ps1` / `test_check_project_structure.ps1`：脚本根 allowlist、开发日志预算、研究 Module 合同、内部路径和跨 Module import 门禁；仓库卫生检查会自动调用它。
- `check_docs_index.ps1` / `test_check_docs_index.ps1`：顶层文档索引与本地链接门禁。
- `archive_apk.ps1`、`verify_release_apk.ps1`、`verify_apk_16kb.ps1`：APK 校验与归档。

## 领域模块

- [`research/public_video/`](research/public_video/)：已冻结的公开视频 / public-silver 历史 campaign。细粒度语义索引和迁移说明保留在该目录，不再向根目录增加实验轮次脚本。
- [`research/ustrf_crosscam_codex/`](research/ustrf_crosscam_codex/)：公开头戴视角视频上的 Codex provisional silver / causal comparator，以及显式 route-projection receipt、polygon bottom-center 三档不确定性审计；不产生客观传感器事实、真人用户效果、设备米制几何或 U0/生产授权。
- [`research/ustrf_route_target_evidence_closure/`](research/ustrf_route_target_evidence_closure/)：route-target 候选盲真值、指标资格、receipt-aware replay、L2 fresh-selection 预注册和 L3 non-executable lockbox 合同；R2-L1X-L2P 最终因 materialization/6 GiB guard 以 `FAIL_CLOSED_EXECUTION_ABORTED` 闭合，C1–C3 未运行。
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
