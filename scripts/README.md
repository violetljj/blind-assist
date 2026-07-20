# BlindAssist 脚本索引

脚本暂时保持在 `scripts/` 根目录，避免一次性搬迁破坏已有命令、CI 和文档引用。新增脚本应放入下列职责之一；后续按类别分批迁移，并为旧入口保留过渡。

## 仓库、发布与环境

- `check_repo_hygiene.ps1` / `test_repo_hygiene.ps1`：仓库卫生门禁与测试。
- `check_docs_index.ps1` / `test_check_docs_index.ps1`：顶层文档索引与本地链接门禁。
- `run_research_contract_tests.py`：本地与 CI 共用的无 GPU、无设备、无可选科学依赖研究合同回归入口。
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

## USTRF-SC 独立研究与回放

- 合成与 SANPO-Synthetic 回放：`acquire_sanpo_synthetic_replay.py`、`audit_sanpo_synthetic_replay.py`、`audit_sanpo_synthetic_metric_replay.py`、`generate_ustrf_synthetic_temporal_geometry_benchmark.py`、`generate_ustrf_synthetic_dynamic_ttc_benchmark.py`、`generate_ustrf_synthetic_corridor_safety_benchmark.py`
- 公开几何/动态来源审计：`audit_tartanair_slice.py`、`estimate_tartanair_ground_plane.py`、`audit_tartanair_temporal_reprojection.py`、`audit_carla_ped_rgbd_slice.py`、`audit_bonn_rgbd_dynamic_source.py`、`audit_bonn_rgbd_dynamic_reprojection.py`、`audit_vkitti2_dynamic_tracks.py`、`audit_argoverse_av1_timestamped_ttc.py`
- REveL 来源与 detector 诊断：`audit_revel_dynamic_rgb_labels.py`、`audit_revel_dynamic_bag_inventory.py`、`audit_revel_dynamic_vicon_trajectories.py`、`audit_revel_rgb_vicon_reprojection.py`、`benchmark_revel_yolo_person_detector.py`、`run_guarded_revel_yolo_smoke.ps1`、`analyze_revel_detector_failures.py`、`compare_revel_detector_sensitivity.py`、`compare_revel_detector_tiling.py`、`align_revel_detector_failures_with_vicon.py`
- 路线事件与设备硬门：`validate_sanpo_counterfactual_episodes.py` 在 route-conditioned 配置下额外验证 capture clock、非未来 route trace、criticality 与 hashed adjudication；`validate_ustrf_sc_device_metric_geometry.py` 验证独立标定、稳定 pose、source-aligned depth、body-local ground、完整路线事件真值和同机性能收据。
- 汇总门禁：`report_ustrf_sc_research_benchmark.py`；默认继续关闭 device gate，只有原始设备 evidence bundle 经上述 validator 完整通过才可标为 `device-geometry-shadow-only`。对应 `test_*.py` 与实现同名配对。

这些脚本只产出隔离研究证据；公开来源的 source-native 距离、轨迹或 TTC proxy 不能替代 body-local assistive event truth，也不授权默认 App 或模型替换。GPU 调度按 [USTRF-SC 窗口交接](../docs/research/ustrf-sc/USTRF_SC_WINDOW_HANDOFF_2026-07-20.md) 的风险分级边界执行。

## SANPO 数据集构建与治理

- 发现与构建：`discover_sanpo_sequence_candidates.py`、`build_sanpo_sequence_evalset.py`、`build_blindassist_evalset.py`
- 克隆与合并：`clone_sanpo_sequence_evalset.py`、`clone_sanpo_event_phase_evalset.py`、`merge_sanpo_sequence_evalsets.py`
- 选择与定稿：`select_sanpo_sequence_by_geometry.py`、`finalize_sanpo_sequence_evalset.py`
- 复核：`create_sanpo_v2_review_decisions.py`、`apply_sanpo_review_decisions.py`、`review_sanpo_sequence_with_model.py`、`screen_sanpo_mask_windows.py`
- v3 治理：`build_sanpo_v3_annotation_queue.py`、`prepare_sanpo_v3_dataset_views.py`、`validate_sanpo_v3_dataset.py`、`freeze_sanpo_v3_regression.py`
- 公开与程序化数据：`build_public_v3_canonical_dataset.py`、`build_procedural_tactile_sequences.py`
- session 拆分：`plan_sanpo_p3_session_split.py`（经同意手机来源只接受 `blindassist_4class_mask_v1` 的人工像素 mask，并逐帧重验 RGB/mask 哈希、原始尺寸、camera/lens 与同意回执）
- 全量候选发现：`run_sanpo_p3_discovery_batches.py`（按参数与官方 session 顺序绑定 checkpoint，支持 fail-closed 断点续跑；完整发现不等于候选接纳、人工真值或训练授权）
- 反事实事件：`validate_sanpo_counterfactual_episodes.py`（字段模板：`configs/sanpo_counterfactual_episode_manifest_template_v1.json`；只验证人工复核的本地 manifest；每条均需两名独立人工复核的本地 SHA256 证据，不下载、不生成标签，且永不授权替换生产模型）
- 风险轮廓/生命周期 target：`build_sanpo_risk_lifecycle_targets.py`（只从已完整验证的 episode manifest 生成主监督 target；像素监督标注为 `auxiliary_only`，不执行训练或模型升级）
- 风险轮廓/生命周期原型：`sanpo_risk_lifecycle_prototype.py`（读取 hash-attested 的人工双审 target，或显式授权的 `hash_bound_model_silver_provisional` 暂定监督；将半开生命周期区间映射为时序标签，像素监督保持 `auxiliary_only`，不授权校准、blind 评测或默认模型替换）
- 走廊熟悉度诊断：`run_sanpo_corridor_anomaly_probe.py`（冻结 DINO 仅拟合 canonical-train 的 source-semantic walkable 特征，并在 dev 报告 source-class outlier AUROC；高分只能表示 `unknown_motion_or_surface` 候选，永不等价于提醒）
- unknown 蒸馏可行性：`run_sanpo_mobile_unknown_distill_probe.py`（以冻结 DINO 熟悉度分数为 teacher，测试冻结 MobileNet 特征能否在 dev 上重现该非安全分数；不训练学生网络、不保存权重）
- 反事实采集清单：`generate_sanpo_counterfactual_capture_plan.py`（从冻结合同生成空的 96-slot 采集计划；不生成证据、标签或训练许可）
- 稀疏 SANPO 长时间线：`acquire_sanpo_rgb_timeline_candidate.py`（只下载公开 CC-BY RGB 帧，默认 1 FPS × 10 秒；不下载 mask、不生成事件标签，只供大模型决定是否值得后续审查）
- SANPO 边界辅助候选：`plan_sanpo_boundary_aux_candidates.py`、`redact_sanpo_auxiliary_candidate.py`（从完整公开 discovery 记录中排除 canonical session，只规划 `auxiliary_pixel_geometry_only` 的 source-mask 候选；SANPO RGB 草稿必须先整人/车脱敏且仍需隐私审核，永不从此路径构造事件或风险真值。）

改变 canonical 数据、冻结回归集或读取 blind 数据的脚本必须遵守 [SANPO 训练协议](../docs/SANPO_TRAINING_PROTOCOL.md)。

## SANPO 训练、导出与门禁

- 模型与训练：`sanpo_segmentation_model.py`、`train_sanpo_segmentation_keras_torch.py`
- 导出：`train_export_sanpo_segmentation.py`
- 门禁：`sanpo_training_gate.py`、`sanpo_candidate_quality_gate.py`、`extract_sanpo_device_event_report.py`（仅将模型 SHA256、事件分母和序列时长完整的真机 `benchmark.json` 转为设备事件门输入；任一绑定缺失即拒绝）
- 后端一致性：`sanpo_backend_equivalence.py`
- 辅助损失与诊断：`sanpo_boundary_distance_aux.py`、`run_sanpo_balanced_distance_ablation.py`、`sanpo_deterministic_linear_probe.py`、`sanpo_depth_anything_linear_probe.py`。`run_sanpo_balanced_distance_ablation.py` 只从 canonical train 中按完整 session 隔离评测，并要求 train/eval 的边界像素覆盖比落在预设区间；它不读取 canonical dev/blind、不保存权重，且无论结果均不授权事件真值、训练升级或默认模型替换。后者以冻结官方 Depth Anything V2 特征做 train/dev-only ridge 诊断；可选拼接既有 raw OS8/OS32，但不训练、不读取 blind、不产生可部署权重。

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
