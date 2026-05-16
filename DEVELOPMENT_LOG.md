# Development Log

本文件记录 BlindAssist 项目的每次分析、更新、修改、验证和遗留事项。后续所有协作者和自动化代理在完成任务前，都必须把本次工作详细写入此文件。

## 2026-05-17

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
