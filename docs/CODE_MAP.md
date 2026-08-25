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
| Goal handoff/completion 产品接口 | `core/assist/.../goal/GoalHandoffContract.kt` | 纯状态机与 receipt schema 在 `core:assist`；可注入 owner 在 `feature:assist`；无状态确认卡片在 `core:ui` |
| USTRF 共享 kernel | `core/ustrf/` | `core/ustrf/src/main/` |
| benchmark、canary、demo、候选 App | [`apps/README.md`](../apps/README.md) | `apps/<role>/<module>/` |
| 稳定脚本 Interface | [`scripts/README.md`](../scripts/README.md) | `scripts/` 根 allowlist |
| 研究实现 | [`MODULE_INDEX.md`](../scripts/research/MODULE_INDEX.md) | `scripts/research/<module>/` |
| 研究 Module 角色与稳定入口 | 各 Module `README.md`；HFTF 另见 [`INDEX.md`](../scripts/research/hftf/INDEX.md) | 从 `MODULE_INDEX.md` 选择一个 Module |
| Goal Copilot public-real episode mining、selective guidance 与 authority-separated last-mile geometry | [`goal_copilot_bridge/README.md`](../scripts/research/goal_copilot_bridge/README.md) | `scripts/research/goal_copilot_bridge/`；`semantic_authority_last_mile_v0/` owns synthetic regression plus controlled real-RGB observation diagnostics |
| Research-only exact semantic anchor + QR planar pose live demo | [`SAGE-LM V2 marker-pose implementation`](research/goal-copilot/SAGE_LM_V2_MARKER_POSE_LIVE_SEAM_IMPLEMENTATION_2026-08-25.md) | `apps/demos/semantic-anchor-demo-app/`；独立 application ID，不修改默认 `:app` |
| GRAIL set-valued interaction-pose teacher 与 GRAIL-R relational oracle | [`GRAIL-R1A grouping result`](research/goal-copilot/GRAIL_R1A_OBTAINABLE_GROUPING_PROBE_RESULT_2026-08-25.md) | `scripts/research/grail/`；M0 teacher、frozen M1、privileged R0、signature ablation 与 training-free RGB+bbox grouping probe，均不修改默认 `:app` |
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

## Goal handoff 集成边界

`GoalHandoffStateOwner` 只接受显式 `FOUND -> APPROACH -> HANDOFF_READY`，再由用户事件
进入 `COMPLETED_BY_USER`；`HANDOFF_READY` 不会自动完成，完成 receipt 必须来自按钮或
上游语音端口传入的精确“找到了”短语。语音端口不包含麦克风权限或 ASR runtime。
默认 App 不创建该 owner，`BlindAssistAppState.goalHandoff = null` 时界面完全不激活；
未来 Goal Copilot runtime 需注入 owner、持久 receipt sink，并把状态与按钮事件接到现有
App contract。不得从 detector 输出推导或默认展示 `FOUND`。
