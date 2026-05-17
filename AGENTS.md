# AGENTS.md

本文件为本项目的 Codex/Agent 开发约定。所有后续自动化代理、AI 助手或人工协作者在修改本仓库前，都应先阅读并遵守这些规则。

## 项目概况

- 项目类型：原生 Android Kotlin App。
- 应用名称：BlindAssist。
- 主要功能：使用 CameraX 获取实时摄像头画面，通过 TFLite YOLO11n 模型做本地目标检测，并用规则层生成助盲避障提醒。
- 核心模块：
  - `app/src/main/java/com/linnan/blindassist/MainActivity.kt`：应用入口、相机流、UI 控制、推理调度。
  - `app/src/main/java/com/linnan/blindassist/vision/`：图像预处理、TFLite YOLO 检测。
  - `app/src/main/java/com/linnan/blindassist/risk/`：风险规则、方向和风险等级判断。
  - `app/src/main/java/com/linnan/blindassist/feedback/`：语音和震动提醒。
  - `app/src/main/java/com/linnan/blindassist/ui/`：检测框覆盖层。
  - `scripts/`：YOLO/TFLite 模型导出与检查脚本。

## 开发日志要求

每次更新、修改、修复、重构、配置变更、模型资产变更或验证结果，都必须详细写入 `DEVELOPMENT_LOG.md`。

日志必须包含：

- 日期和时间。
- 修改人或执行者。
- 修改类型，例如：功能、修复、重构、文档、构建、模型、测试、分析。
- 修改范围，列出涉及的关键文件。
- 修改内容，说明做了什么。
- 修改原因，说明为什么要做。
- 验证方式，记录运行过的命令、结果、失败原因或未验证原因。
- 后续事项，如遗留问题、风险和建议。

如果一次任务只做了分析，没有改代码，也要记录分析结论和验证命令。

## README 与版本号要求

每次会影响项目状态、使用方式、功能行为、构建方式、模型资产、测试结论或重要技术决策的更新，都必须同步写入 `README.md`，让 README 保持对当前项目状态、用法、版本和重要变更的准确描述。

纯文字微调、措辞调整、错别字修复、格式整理或协作规则的轻量说明补充，如果不改变项目功能、使用方式、构建方式、模型资产或重要决策，不计为版本更新，也不强制写入 README；但仍应按开发日志要求记录必要背景。

版本号调整由 Codex/Agent 根据本次变更的影响范围协助判断：

- 小更新：版本号增加 `v0.1`，适用于小功能、局部优化、会影响使用理解的重要文档补充、测试补充、轻量修复等。
- 大更新：版本号增加 `v0.5`，适用于重要功能、明显体验升级、较大范围重构、模型或核心链路调整等。
- 质变更新：版本号增加 `v1.0`，适用于产品形态、核心能力、架构或可用性发生阶段性跃迁的更新。

每次任务结束前，必须在最终说明中写明本次版本判断及理由；如果未实际修改版本号，也必须说明原因。

## 构建与验证

常用验证命令：

```powershell
$env:JAVA_HOME='C:\Program Files\Android\Android Studio\jbr'
$env:PATH="$env:JAVA_HOME\bin;$env:PATH"
.\gradlew.bat :app:testDebugUnitTest :app:assembleDebug --no-daemon
```

APK 输出位置：

```text
app/build/outputs/apk/debug/app-debug.apk
```

模型检查命令：

```powershell
.\.venv-export312\Scripts\python.exe scripts\inspect_tflite.py
```

期望模型形状：

```text
input shape=[1, 320, 320, 3] dtype=float32
output shape=[1, 84, 2100] dtype=float32
```

## 修改原则

- 优先保持当前原生 Android/Kotlin 架构，不随意引入大型框架。
- 涉及 CameraX、TFLite、坐标映射、风险规则和语音提醒的修改，应尽量补充或更新测试。
- 不要提交本地 SDK、Gradle 缓存、虚拟环境、下载目录等机器相关文件。
- 不要随意替换或删除模型资产；模型变更必须在开发日志中记录来源、导出参数和检查结果。
- 对助盲避障相关提示要保持谨慎表述，避免把原型描述为可完全替代人工判断的安全设备。

## 协作注意事项

- 每次执行任务前，优先检查当前请求是否有合适的 Codex skills 可用；如有，应先阅读并遵循对应 `SKILL.md` 的工作流。
- 修改前先查看 `git status --short`，确认是否存在他人未提交改动。
- 不要回滚自己不理解或不是自己产生的改动。
- 修改后尽量运行相关测试或构建；如果无法运行，必须在开发日志和最终说明中写清原因。
- 每次任务结束前，检查 `DEVELOPMENT_LOG.md` 是否已经补充本次记录。
