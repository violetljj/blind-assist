# BlindAssist 代码地图

状态：`current / navigation-only`

本页用于编码任务冷启动，只描述稳定职责，不复制动态研究状态。

| 任务 | 首读位置 | 主要实现 |
|---|---|---|
| 正式 Android App、manifest、依赖装配 | `app/build.gradle.kts` | `app/src/main/` |
| CameraX、会话协调、检测到反馈链 | `feature/assist/` | `feature/assist/src/main/` |
| 风险、事件、稳定和提醒策略 | `core/assist/` | `core/assist/src/main/` |
| TFLite、图像预处理、native vision | `core/vision/` | Kotlin 与 `src/main/cpp/` |
| 设备、语音、震动和硬件 adapter | `core/device/` | `core/device/src/main/` |
| Compose UI 模型 | `core/ui/` | `core/ui/src/main/` |
| USTRF 共享 kernel | `core/ustrf/` | `core/ustrf/src/main/` |
| benchmark、canary、demo、候选 App | [`apps/README.md`](../apps/README.md) | `apps/<role>/<module>/` |
| 稳定脚本 Interface | [`scripts/README.md`](../scripts/README.md) | `scripts/` 根 allowlist |
| 研究实现 | [`MODULE_INDEX.md`](../scripts/research/MODULE_INDEX.md) | `scripts/research/<module>/` |
| Assistive Geometry 训练、Development/failure-anatomy、R2 reducer/F1-P、deterministic FactorTensorAdapter、跨传感器 factor Confirmation executor/calibration-control/source-manifest、teacher/mobile/temporal mechanics | [`assistive_geometry/README.md`](../scripts/research/assistive_geometry/README.md) | `train_b1_a0_formal.py`、`train_b1_additive_arm.py`、`materialize_b1_development_targets.py`、`observe_b1_a0_development.py`、`evaluate_b1_a0_development.py`、`analyze_b1_a0_failure_anatomy.py`、`geometry_r2_reducer.py`、`run_geometry_r2_f0_canary.py`、`validate_geometry_r2_f1_protocol.py`、`audit_geometry_r2_f1_adapter_gap.py`、`factor_tensor_adapter.py`、`run_factor_tensor_adapter_canary.py`、`ag_r2_cross_sensor_confirmation/{control_format,calibration_control,validate_calibration_control,control_format_r1,calibration_control_r1,validate_calibration_control_r1,validate_depthart_source_manifest}.py`、`run_ag_r2_cross_sensor_factor_accuracy_confirmation.py`、`evaluate_teacher_complementarity.py`、`export_assistive_geometry_onnx.py`、`temporal_geometry_ablation.py`、`run_hypothesis_canary_lite.py` |
| TARO 独立并行路线的 P0 schema/protocol/analytic-fixture 静态锁 | [`taro/README.md`](../scripts/research/taro/README.md) | `validate_taro_p0_protocol.py`、`test_validate_taro_p0_protocol.py`；当前无 solver、runner、materializer、模型或 scientific artifact |
| TARO O0M execution-family 协议锁 | [`taro_o0m/README.md`](../scripts/research/taro_o0m/README.md) | `validate_taro_o0m_protocol.py`、`test_validate_taro_o0m_protocol.py` |
| TARO O0M 独立 analytic runtime | [`taro_o0m_runtime/README.md`](../scripts/research/taro_o0m_runtime/README.md) | `o0m_mechanics.py`、`run_o0m_canary.py`、`test_o0m_mechanics.py`；执行仅由 exact one-shot lock 管理 |
| TARO O0R ARKitScenes source-and-adapter 契约验证 | [`taro_o0r_source_adapter/README.md`](../scripts/research/taro_o0r_source_adapter/README.md) | `validate_taro_o0r_source_adapter_contract.py`、`test_validate_taro_o0r_source_adapter_contract.py`；当前无 downloader、materializer、truth adapter、DepthART runner 或 scientific artifact |
| TARO O0R ARKitScenes 纯内存 source adapter | [`taro_o0r_source_adapter_runtime/README.md`](../scripts/research/taro_o0r_source_adapter_runtime/README.md) | `source_adapter.py`、`test_source_adapter.py`；roster/asset-bound timestamp/pose、source+9-query receipt、内部 residual uncertainty、FARO/candidate 双 extractor、immutable base、sparse boundary、TARO reducer 与 8×2 injection；无 source I/O、model runner、materializer 或 artifact writer |
| TARO O0R ARKitScenes HEAD/source/truth materializer | [`taro_o0r_truth_materializer_runtime/README.md`](../scripts/research/taro_o0r_truth_materializer_runtime/README.md) | `materializer.py`、`run_head_preflight.py`、`run_truth_only.py` 与 tests/validator；exact 72-URL HEAD、bounded GET/archive、all-exact denominator、fit-before-eval、per-query FARO/confidence lookup、original-member envelope、content-addressed ndarray reload 与 atomic truth one-shot；当前无 execution lock，HEAD/source/truth 均未运行 |
| AG-QSF 并行路线、H1 survival mechanics/TRAIN canary、共享资源 manifest 与输出隔离门 | [`assistive_geometry_qsf/README.md`](../scripts/research/assistive_geometry_qsf/README.md) | `h1_survival.py`、`run_h1_train_canary.py`、`validate_h1_train_canary.py`、`validate_qsf_preparation.py`；未来 H2 profile-query 只进入该 Module |
| AG-CBF 并行路线、TRAIN-only ground-grid 数据支撑审计与后续 corridor bottleneck oracle | [`assistive_geometry_cbf/README.md`](../scripts/research/assistive_geometry_cbf/README.md) | `audit_grid_support.py`；oracle、representation-value 与模型/训练当前未授权 |
| AG-DCA 数据能力 atlas 与 hypothesis admission checker | [`assistive_geometry_data_capability/README.md`](../scripts/research/assistive_geometry_data_capability/README.md) | `build_capability_atlas.py`、`check_hypothesis_requirements.py`；只读 TRAIN truth/source，不授予算法执行权限 |
| AG-DUE gap-driven source prescreen、claim-bound evidence、SANPO manifest/result/source-audit lock、exact metadata preflight 与数据角色防火墙 | [`assistive_geometry_data_upgrade/README.md`](../scripts/research/assistive_geometry_data_upgrade/README.md) | `validate_due_r0.py`、`validate_due_sanpo_manifest_lock.py`、`validate_due_sanpo_prescreen_result.py`、`validate_due_sanpo_synthetic_r1_protocol.py`、`run_due_sanpo_synthetic_r1_metadata_preflight.py`、`validate_due_sanpo_synthetic_r1_preflight_result.py` 及 tests；metadata/source inventory 不建立 source data support、DCA/F1 PASS 或 body-canary 权限 |
| 配置与冻结合同 | `configs/` | JSON/YAML；先看调用方和协议 |
| 机器 schema | `schemas/` | JSON Schema 与 validator |
| 当前文档与研究真源 | [`docs/PROJECT_STATE.md`](PROJECT_STATE.md) | `docs/`、`docs/research/` |

## 默认跳过

- `build/`、`.cxx/`、`artifacts.local/` 和其他生成物。
- 与任务无关的研究 Module、archive、snapshot 和完整历史日志。
- 隔离 `apps/` 不等于正式 `app/`；研究脚本不等于默认产品路径。

## 定位顺序

1. 从本表选一个实现域。
2. 用 `rg` 搜索具体 symbol、Gradle module 或稳定 Adapter。
3. 只读取直接调用方、实现和相关测试；需要权限结论时再回到 current 文档。
