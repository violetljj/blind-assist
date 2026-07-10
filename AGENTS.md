# AGENTS.md

本文件为本项目的 Codex/Agent 开发约定。所有后续自动化代理、AI 助手或人工协作者在修改本仓库前，都应先阅读并遵守这些规则。

## 项目概况

- 项目类型：原生 Android Kotlin 多模块 App。
- 应用名称：BlindAssist。
- 主要功能：使用 CameraX 获取实时摄像头画面，通过 TFLite YOLO11n 模型在本地完成目标检测，再由规则层生成助盲避障提醒，并通过语音、震动和 Compose 界面反馈给用户。
- 当前模块边界：
  - `:app`：启动壳层、`BlindAssistApplication`、`MainActivity`、Manifest、资源、模型资产和 APK 配置。
  - `:feature:assist`：Hilt ViewModel、运行时状态机、CameraX/TFLite 协调、配置同步、渲染和性能日志边界。
  - `:core:assist`：纯 Kotlin 助盲领域模型、风险分析、提醒策略、会话统计、本地化和偏好映射。
  - `:core:vision`：TFLite YOLO 检测器、图像预处理、YOLO 输出解析和视觉帧处理。
  - `:core:device`：CameraX 帧源、Android 语音/震动反馈、SharedPreferences 用户偏好和设备侧适配。
  - `:core:ui`：Compose/UI 状态模型、检测框覆盖层、相机引导和现场测试摘要映射。
  - `scripts/`：模型导出/检查、APK 归档、仓库卫生检查、真机回归和技能恢复脚本。

## 开发日志要求

每次更新、修改、修复、重构、配置变更、模型资产变更或验证结果，都必须详细写入 `DEVELOPMENT_LOG.md`。

本项目作为研究生阶段小毕设/课程设计展示项目，开发日志不仅用于协作追踪，也用于向指导老师展示持续工作量、阶段进展、技术路线和验证过程。因此日志应尽量详实、具体、可复盘，避免只写一句“已修改”或“已优化”。每次记录应体现本次工作的背景、思路、改动拆解、验证证据和下一步计划，让读者能看出实际投入和迭代过程。

日志必须包含：

- 日期和时间。
- 修改人或执行者。
- 修改类型，例如：功能、修复、重构、文档、构建、模型、测试、分析。
- 修改范围，列出涉及的关键文件。
- 修改内容，说明做了什么；建议按模块、文件或功能点拆开写，尽量写清楚具体实现、参数调整、交互变化、测试补充和文档同步。
- 修改原因，说明为什么要做；尽量补充问题背景、毕设展示价值、用户体验或技术风险，而不只写“按要求修改”。
- 验证方式，记录运行过的命令、结果、失败原因或未验证原因。
- 后续事项，如遗留问题、风险和建议。

如果一次任务只做了分析，没有改代码，也要记录分析结论和验证命令。

项目执行人命名统一为 `violjjet`。后续写入 `DEVELOPMENT_LOG.md` 的“执行者”字段时，应使用 `violjjet`。

## 版本 APK 留存要求

每次构建出可用于演示、测试或提交给老师查看的 APK 后，除了保留 Gradle 默认输出 `app/build/outputs/apk/debug/app-debug.apk`，还必须复制一份按版本命名的 APK 到完整本地归档目录，方便后续课堂展示、答辩汇报和不同版本效果对比。

完整本地归档目录：

```text
E:\linnan\blind-assist-apk-archive\apks
```

Git 里程碑归档目录：

```text
releases/apk/
```

推荐命名格式：

```text
BlindAssist-v版本号-构建类型-日期时间.apk
```

示例：

```text
releases/apk/BlindAssist-v2.6.0-debug-20260518-012726.apk
```

归档要求：

- 每次版本号变化后，必须先保存对应版本 APK 到完整本地归档目录。
- 重要功能阶段、真机验证阶段或需要给老师展示的阶段，即使版本号不变，也应先保存一份带日期时间的 APK 到完整本地归档目录。
- 只有累计 `versionName` 差值达到 `>= 0.5`、或用户明确要求提交里程碑 APK 时，才把 APK 同步到 `releases/apk/` 并提交到 Git。
- 小版本、临时演示、普通测试 APK 默认只保存在完整本地归档目录，不提交到 Git。
- `DEVELOPMENT_LOG.md` 中必须记录归档 APK 的路径、来源 APK、文件大小、构建时间或复制时间；如果同步到 `releases/apk/`，还必须记录原因。
- 如果因为构建失败、没有生成 APK、磁盘空间或其他原因无法归档，必须在 `DEVELOPMENT_LOG.md` 和最终说明中写清楚原因。
- 不要删除旧版本 APK，除非用户明确要求清理；旧版本用于展示项目演进和对比效果。

## README 与版本号要求

每次会影响项目状态、使用方式、功能行为、构建方式、模型资产、测试结论或重要技术决策的更新，都必须同步写入 `README.md`，让 README 保持对当前项目状态、用法、版本和重要变更的准确描述。

纯文字微调、措辞调整、错别字修复、格式整理或协作规则的轻量说明补充，如果不改变项目功能、使用方式、构建方式、模型资产或重要决策，不计为版本更新，也不强制写入 README；但仍应按开发日志要求记录必要背景。

版本号调整由 Codex/Agent 根据本次变更的影响范围协助判断：

- 小更新：版本号增加 `v0.1`，适用于小功能、局部优化、会影响使用理解的重要文档补充、测试补充、轻量修复等。
- 大更新：版本号增加 `v0.5`，适用于重要功能、明显体验升级、较大范围重构、模型或核心链路调整等。
- 质变更新：版本号增加 `v1.0`，适用于产品形态、核心能力、架构或可用性发生阶段性跃迁的更新。

每次任务结束前，必须在最终说明中写明本次版本判断及理由；如果未实际修改版本号，也必须说明原因。

## 构建与验证

优先使用当前仓库的本地 JDK 17、Android SDK 和 Gradle 缓存，避免系统 Java 版本漂移导致误判：

```powershell
$env:JAVA_HOME=(Resolve-Path '.\.jdk\jdk17.0.19_10').Path
$env:PATH="$env:JAVA_HOME\bin;$((Resolve-Path '.\.android-sdk\platform-tools').Path);$env:PATH"
$env:GRADLE_USER_HOME=(Resolve-Path '.\.gradle-local').Path
.\gradlew.bat :core:assist:test :core:vision:testDebugUnitTest :core:device:testDebugUnitTest :core:ui:testDebugUnitTest :feature:assist:testDebugUnitTest :app:testDebugUnitTest --no-daemon --console=plain
.\gradlew.bat :app:lintDebug :core:vision:lintDebug :core:device:lintDebug :core:ui:lintDebug :feature:assist:lintDebug --no-daemon --console=plain
.\gradlew.bat :app:assembleDebug :app:assembleDebugAndroidTest --no-daemon --console=plain
```

如果本地 Gradle 缓存或临时目录卡住验证流程，例如 `.gradle-local\.tmp` 删除失败、wrapper zip 解压异常、或临时 `GRADLE_USER_HOME` 下生成的缓存不可用，且确认没有正在运行的 Gradle/Android Studio 进程依赖这些缓存，可以清理对应的本地 Gradle 缓存目录后重试。清理前应确认目标路径位于当前仓库或明确的本地缓存目录内，避免误删用户数据；此类缓存目录不应提交到 Git。

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

有真机在线时，可用仓库脚本做边界清晰的设备回归：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_device_regression.ps1 -SampleSeconds 90
```

该脚本输出到 `test-artifacts.local-device-regression-*`，属于本机验证证据，默认不提交到 Git。

## 修改原则

- 优先保持当前原生 Android/Kotlin 架构，不随意引入大型框架。
- 涉及 CameraX、TFLite、坐标映射、风险规则和语音提醒的修改，应尽量补充或更新测试。
- 不要提交本地 SDK、Gradle 缓存、虚拟环境、下载目录等机器相关文件。
- 不要随意替换或删除模型资产；模型变更必须在开发日志中记录来源、导出参数和检查结果。
- 对助盲避障相关提示要保持谨慎表述，避免把原型描述为可完全替代人工判断的安全设备。

## 协作注意事项

- 每次执行任务前，优先检查当前请求是否有合适的 Codex skills 可用；如有，应先阅读并遵循对应 `SKILL.md` 的工作流。
- 在用户主动提出前，后续计划和实现暂不以答辩、课堂展示、演示话术、证据包装或展示材料润色为优先驱动；普通工程文档、测试记录和开发日志仍按仓库规则执行。
- 修改前先查看 `git status --short`，确认是否存在他人未提交改动。
- 不要回滚自己不理解或不是自己产生的改动。
- 后续默认将正式推送目标理解为 `master` 分支；如当前工作在功能分支或临时分支，推送前应先明确是否需要合并/切换到 `master`，并确认不会夹带无关本地改动。
- 修改后尽量运行相关测试或构建；如果无法运行，必须在开发日志和最终说明中写清原因。
- 每次任务结束前，检查 `DEVELOPMENT_LOG.md` 是否已经补充本次记录。
- 后续出现有可能性、创造性、产品路线价值但不一定立即实施的想法时，应记录到仓库根目录 `idea.md`，并在真正实施时再同步更新 README、开发日志、版本号、测试结果和 APK 归档。若 `idea.md` 中某个想法已经实现，应把对应标题明显标注为 `【已完成】`；若只实现了一部分，应标注为 `【部分完成】` 并写清剩余范围。
- ESP32 / 旧眼镜工程相关任务的可信源码路径是 `E:\linnan\glassses`，参考资料路径是 `E:\linnan\esp32参考资料`；不要再依赖此前截断的 `glassses-main.zip`、失败克隆残留目录或临时恢复目录。当前 Android App 中 `连接眼镜设备` 仍是占位入口，不扫描蓝牙、不联网、不申请额外权限，也不代表已经接入真实 ESP32/眼镜硬件。分析旧链路时优先读取 `E:\linnan\glassses\services\audio_service.py`、`E:\linnan\glassses\services\camera_service.py`、`E:\linnan\glassses\services\microphone_service.py`、`E:\linnan\glassses\stm32code\esp32_firmware_mic.ino` 和 `E:\linnan\glassses\stm32code\speaker.cpp`；真正迁移到 Android 前，应先设计 `GlassesConnectionRepository`、`GlassesControlChannel`、`GlassesFrameSource` 等边界和无外设降级路径，再决定是否复用 TCP PCM / MJPEG 实验协议。

## 文本编码

- 本仓库的 Markdown、Kotlin、Gradle 和脚本文件默认按 UTF-8 读取，除非文件本身明确声明了其他编码。
- 在 Windows PowerShell 中读取中文项目文件时，不要依赖 `Get-Content` 的默认解码；读取 `AGENTS.md`、`README.md`、`DEVELOPMENT_LOG.md`、`idea.md` 等中文文本文件时，应使用 `Get-Content -Raw -Encoding UTF8 <path>`。
- 如果终端输出出现类似 `鏈枃` 的乱码，应先用显式 UTF-8 解码重新读取，再判断文件内容是否真的损坏，避免误修复正常文件。

## 已知沙箱权限问题与处理方式

本仓库在 Codex 沙箱环境中反复遇到过一些不是代码本身导致的失败。对已经确认会稳定复现的命令，后续代理应直接按权限规则提权执行，不需要先在普通沙箱中重复制造一次失败；但仍应说明提权原因，并在结果中记录命令和输出。遇到新的、未确认的失败类型时，才需要先保留原始错误再判断是否提权重跑。

- Gradle wrapper 下载或构建验证：`.\gradlew.bat :app:testDebugUnitTest :app:assembleDebug --no-daemon` 在普通沙箱中已多次因 `java.net.SocketException: Permission denied: getsockopt` 失败。需要运行完整验证时，直接使用 `require_escalated` 执行该命令。
- Git 推送：普通沙箱推送已多次因凭据或 SSH 权限失败。历史上 HTTPS 推送曾在 GitHub 返回 `HTTP/2 401` 后卡在 `git credential-manager get`；当前 SSH 远端也会出现 `Identity file ... not accessible: Permission denied` 或 `ssh: connect to host github.com port 22: Permission denied`。执行 `git push` 前仍要确认远端和分支，但推送本身应直接使用 `require_escalated`。
- 手机安装：优先使用仓库内的 `.\.android-sdk\platform-tools\adb.exe`。安装前可先运行 `adb devices` 确认设备为 `device` 状态；安装命令为 `.\.android-sdk\platform-tools\adb.exe install -r app\build\outputs\apk\debug\app-debug.apk`。如果后续也稳定出现权限限制，再按同样规则改为直接提权安装。
- 对直接提权执行的已知命令，最终说明和 `DEVELOPMENT_LOG.md` 中应写明：这是基于本仓库已知沙箱限制直接提权、实际执行的命令和结果；不必再记录一次人为复现的普通沙箱失败。

## GitHub CLI 状态

- 本机已安装 GitHub CLI：`E:\linnan\tools\gh\bin\gh.exe`。
- 已通过 `gh auth login --hostname github.com --git-protocol ssh --web` 登录 GitHub。
- 当前 GitHub CLI 登录账号：`violetljj`。
- GitHub CLI 已将 `github.com` 的 Git 协议配置为 `ssh`。
- 如果当前终端或 Codex 会话尚未刷新用户 PATH，请使用完整路径 `E:\linnan\tools\gh\bin\gh.exe`；新开的 PowerShell 通常可以直接使用 `gh`。
- 后续需要查看认证状态时，运行 `E:\linnan\tools\gh\bin\gh.exe auth status`。
- 后续需要创建 Pull Request 时，可在目标分支推送后运行 `E:\linnan\tools\gh\bin\gh.exe pr create`。
