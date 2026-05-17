# Development Log

本文件记录 BlindAssist 项目的每次分析、更新、修改、验证和遗留事项。后续所有协作者和自动化代理在完成任务前，都必须把本次工作详细写入此文件。

## 2026-05-17

### Codex skills 优先检查规则补充

- 时间：2026-05-17 15:03:55 +08:00
- 执行者：Codex
- 类型：文档 / 协作规范
- 修改范围：
  - `AGENTS.md`
  - `DEVELOPMENT_LOG.md`
- 修改内容：
  - 在 `AGENTS.md` 的“协作注意事项”中新增规则：每次执行任务前，优先检查当前请求是否有合适的 Codex skills 可用；如有，应先阅读并遵循对应 `SKILL.md` 的工作流。
  - 在 `DEVELOPMENT_LOG.md` 记录本次协作规则补充。
- 修改原因：
  - 用户要求为项目协作规范补充 skills 优先使用要求，确保后续任务能主动匹配并使用合适技能。
- 验证方式：
  - 已运行 `git status --short`，确认存在一个既有未跟踪 PPT 文件，本次未处理该文件。
  - 本次仅修改协作规则文档和开发日志，未涉及 Android 代码、构建脚本、模型资产或运行逻辑，因此未运行 Gradle 构建和单元测试。
- 版本判断：
  - 本次属于协作规则的轻量说明补充，不改变项目功能、使用方式、构建方式、模型资产或重要技术决策，不计为版本更新。
  - 未修改 README 和应用版本号。
- 后续事项：
  - 后续任务开始时应先判断是否有适用 skills，并在适用时读取对应 `SKILL.md`。

### 版本更新规则校正

- 时间：2026-05-17 02:08:06 +08:00
- 执行者：Codex
- 类型：文档 / 协作规范 / 配置
- 修改范围：
  - `AGENTS.md`
  - `README.md`
  - `app/build.gradle.kts`
  - `DEVELOPMENT_LOG.md`
- 修改内容：
  - 明确纯文字微调、措辞调整、错别字修复、格式整理或协作规则的轻量说明补充，如果不改变项目功能、使用方式、构建方式、模型资产或重要决策，不计为版本更新。
  - 将 README 同步要求收窄为影响项目状态、使用方式、功能行为、构建方式、模型资产、测试结论或重要技术决策的更新。
  - 将项目版本从 `v0.2.0` 恢复为 `v0.1.0`，并将 `versionCode` 从 `2` 恢复为 `1`。
- 修改原因：
  - 用户指出小级别文字改动不应算作版本更新；此前将协作规则文字调整计入 `v0.2.0` 偏激进，需要校正版本策略和当前版本。
- 验证方式：
  - 未运行构建；本次仅校正文档规则和版本号配置，不涉及 Android 代码、模型资产或构建逻辑。
  - 已检查 `git status --short`，确认除既有未跟踪 PPT 外，本次只产生文档与版本号配置改动。
- 版本判断：
  - 本次属于版本策略校正，不作为版本更新计数。
  - 项目版本保持 `v0.1.0`。
- 后续事项：
  - 后续由 Codex/Agent 判断版本级别时，应先排除纯文字微调、错别字、格式整理和轻量协作规则说明。

### README 与版本号协作规范更新

- 时间：2026-05-17 02:03:07 +08:00
- 执行者：Codex
- 类型：文档 / 协作规范
- 修改范围：
  - `AGENTS.md`
  - `README.md`
  - `app/build.gradle.kts`
  - `DEVELOPMENT_LOG.md`
- 修改内容：
  - 在 `AGENTS.md` 中新增 README 同步与版本号要求。
  - 明确每次更新、修改、修复、重构、配置变更、模型资产变更或验证结果都必须同步写入 `README.md`。
  - 明确版本号判断规则：小更新增加 `v0.1`，大更新增加 `v0.5`，质变更新增加 `v1.0`，由 Codex/Agent 根据影响范围协助判断。
  - 在 `README.md` 中新增版本说明和近期更新记录。
  - 将应用版本从 `0.1.0` 提升到 `0.2.0`，并将 `versionCode` 从 `1` 提升到 `2`。
- 修改原因：
  - 用户要求把 README 更新规则和版本号判断规则写入 `AGENTS.md`，并由 Codex/Agent 后续协助判断版本级别。
- 验证方式：
  - 已运行构建验证：`$env:JAVA_HOME='C:\Program Files\Android\Android Studio\jbr'; $env:PATH="$env:JAVA_HOME\bin;$env:PATH"; .\gradlew.bat :app:assembleDebug --no-daemon`。
  - 首次在沙箱内运行 Gradle 构建时失败，错误为 `java.net.SocketException: Permission denied: getsockopt`，原因是 Gradle wrapper 需要网络下载权限。
  - 已按权限要求提权重跑，构建结果为 `BUILD SUCCESSFUL in 24s`。
  - 已检查 `git status --short`，确认除既有未跟踪 PPT 外，本次只产生文档与版本号配置改动。
- 版本判断：
  - 本次属于小更新，原因是仅调整协作规范、README 记录和版本号，不影响应用功能、模型或核心链路。
  - 按小更新规则，项目版本从 `v0.1.0` 提升到 `v0.2.0`。
- 后续事项：
  - 后续每次任务结束前，需要同时检查 `README.md` 和 `DEVELOPMENT_LOG.md` 是否已补充。
  - 后续最终说明需要写明版本判断及理由。

### 性能剖面与前处理稳妥优化

- 时间：2026-05-17 01:46:17 +08:00
- 执行者：Codex
- 类型：性能 / 重构 / 测试
- 修改范围：
  - `app/src/main/java/com/linnan/blindassist/MainActivity.kt`
  - `app/src/main/java/com/linnan/blindassist/vision/ImagePreprocessor.kt`
  - `app/src/main/java/com/linnan/blindassist/vision/TfliteYoloDetector.kt`
  - `app/src/test/java/com/linnan/blindassist/vision/ImagePreprocessorTest.kt`
  - `DEVELOPMENT_LOG.md`
- 修改内容：
  - 将 CameraX 分析链路固定为 `640x480` 目标分辨率，保留 `KEEP_ONLY_LATEST` 和 RGBA 输出格式。
  - 在界面状态栏展示 `total/pre/infer/post/FPS` 性能指标，并每秒通过 `BlindAssistPerf` 写入一次 Logcat 性能摘要。
  - 将 `ImagePreprocessor` 改为可复用实例，复用 letterbox bitmap、canvas、paint、目标矩形、像素数组和模型输入 `ByteBuffer`。
  - 移除每帧 `createScaledBitmap`，改为直接把原始帧绘制到复用的 letterbox bitmap。
  - 在 `TfliteYoloDetector` 中复用输出 `ByteBuffer` 和 `FloatArray`，并新增 `lastPreprocessMs`、`lastInferenceMs`、`lastPostprocessMs`、`lastTotalDetectMs` 只读性能字段。
  - 新增前处理单元测试，覆盖横向/纵向 letterbox 计算，以及连续写入 buffer 时 rewind 和覆盖旧内容的行为。
- 修改原因：
  - 降低实时检测每帧分配和前处理开销，提升真机连续运行时的稳定性和可观测性。
  - 为后续进一步优化 YUV 直采样、模型量化或风险平滑提供明确的性能基线。
- 验证方式：
  - 已运行模型检查：`.\.venv-export312\Scripts\python.exe scripts\inspect_tflite.py`。
  - 模型检查结果：输入 `images` 为 `[1, 320, 320, 3] float32`，输出 `Identity` 为 `[1, 84, 2100] float32`。
  - 首次在沙箱内运行 Gradle 构建时失败，错误为 `java.net.SocketException: Permission denied: getsockopt`，原因是 Gradle wrapper 需要网络下载权限。
  - 已按权限要求提权重跑：`$env:JAVA_HOME='C:\Program Files\Android\Android Studio\jbr'; $env:PATH="$env:JAVA_HOME\bin;$env:PATH"; .\gradlew.bat :app:testDebugUnitTest :app:assembleDebug --no-daemon`。
  - 构建验证结果：命令退出码为 `0`，`app/build/outputs/apk/debug/app-debug.apk` 已生成。
  - 单元测试结果：`RiskAnalyzerTest` 共 4 个用例通过，`ImagePreprocessorTest` 共 3 个用例通过，失败和错误均为 0。
- 后续事项：
  - 本轮未更换模型资产，未调整 `RiskAnalyzer` 风险阈值，未改变语音或震动提醒策略。
  - 仍需真机连续运行 3-5 分钟观察状态栏性能指标、Logcat 输出、发热、卡顿和内存增长情况。
  - 如果真机前处理仍是瓶颈，下一步可评估 YUV_420_888 直接采样到模型输入，或进一步降低分析分辨率。

### 创建项目协作规范和开发日志

- 执行者：Codex
- 类型：文档 / 协作规范
- 修改范围：
  - `AGENTS.md`
  - `DEVELOPMENT_LOG.md`
- 修改内容：
  - 新增 `AGENTS.md`，说明项目类型、核心模块、构建验证命令、修改原则和协作注意事项。
  - 新增 `DEVELOPMENT_LOG.md`，作为后续所有项目更新、修改、分析和验证的统一记录文件。
  - 明确要求每次更新、修改、修复、重构、配置变更、模型资产变更或分析结果都必须写入开发日志。
- 修改原因：
  - 用户要求为本项目创建 `AGENTS.md` 和开发日志，并要求后续每次更新与修改都详细记录。
- 验证方式：
  - 已检查根目录，未发现已有 `AGENTS.md` 或开发日志文件。
  - 已检查 `git status --short`，确认当前只有一个既有未跟踪 PPT 文件，不属于本次修改。
- 后续事项：
  - 后续任何代码、配置、模型、文档或测试变更，都应先阅读 `AGENTS.md`，并在任务结束前更新本文件。
