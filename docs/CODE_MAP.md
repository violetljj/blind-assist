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
| Assistive Geometry 训练、Development/teacher 评估、移动/时序 mechanics 与数学 canary | [`assistive_geometry/README.md`](../scripts/research/assistive_geometry/README.md) | `train_b1_a0_formal.py`、`train_b1_additive_arm.py`、`materialize_b1_development_targets.py`、`observe_b1_a0_development.py`、`evaluate_b1_a0_development.py`、`evaluate_teacher_complementarity.py`、`export_assistive_geometry_onnx.py`、`temporal_geometry_ablation.py`、`run_hypothesis_canary_lite.py` |
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
