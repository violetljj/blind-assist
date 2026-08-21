# BlindAssist 代码地图

状态：`current / navigation-only`

本页只描述稳定职责，不复制动态研究状态、轮次或脚本清单。

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
| 研究 Module 角色与稳定入口 | 各 Module `README.md`；HFTF 另见 [`INDEX.md`](../scripts/research/hftf/INDEX.md) | 从 `MODULE_INDEX.md` 选择一个 Module |
| Goal Copilot P0/P1 历史桥与当前帧重定位机械 runner | [`goal_copilot_bridge/README.md`](../scripts/research/goal_copilot_bridge/README.md) | `scripts/research/goal_copilot_bridge/` |
| 配置与冻结合同 | `configs/` | JSON/YAML；先看调用方和协议 |
| 机器 schema | `schemas/` | JSON Schema 与 validator |
| 当前文档与研究真源 | [`docs/PROJECT_STATE.md`](PROJECT_STATE.md) | `docs/`、`docs/research/` |

## 默认跳过

- `build/`、`.cxx/`、`artifacts.local/` 和其他生成物；
- 与任务无关的研究 Module、archive、snapshot 和完整历史日志；
- 隔离 `apps/` 不等于正式 `app/`，研究脚本不等于默认产品路径。

## 定位顺序

1. 从本表选一个实现域；
2. 用 `rg` 搜索具体 symbol、Gradle module 或稳定 Adapter；
3. 只读直接调用方、实现和相关测试；需要权限结论时再回到 current 文档。
