# Development Log

本文件记录 BlindAssist 项目的每次分析、更新、修改、验证和遗留事项。后续所有协作者和自动化代理在完成任务前，都必须把本次工作详细写入此文件。

## 2026-05-17

### 前端关怀模式提交推送与真机安装

- 时间：2026-05-17 18:22:33 +08:00
- 执行者：violjjet
- 类型：发布 / 验证 / 安装
- 修改范围：
  - `DEVELOPMENT_LOG.md`
- 修改内容：
  - 将“前端与关怀模式大幅升级”相关改动提交到本地 Git，并推送到 GitHub `origin/master`。
  - 使用仓库内 debug APK 覆盖安装到已连接 Android 手机。
- 修改原因：
  - 用户要求提交推送，并下载到手机，需要记录实际提交、推送和安装结果，保持项目开发日志完整。
- 验证方式：
  - 已运行 `git status --short`，确认提交前仅 stage 了本次 UI 升级相关 5 个文件，未跟踪文件 `多模态智能助盲系统1.4.pptx` 未纳入提交。
  - 已提交：`e29b99a Upgrade assistive camera UI`。
  - 已根据仓库已知 SSH/凭据沙箱限制直接提权运行 `git push origin master`，结果成功：`1082ab3..e29b99a  master -> master`。
  - 已运行 `.\.android-sdk\platform-tools\adb.exe devices`，确认设备 `R5CX10M8Y8X` 状态为 `device`。
  - 已运行 `.\.android-sdk\platform-tools\adb.exe install -r app\build\outputs\apk\debug\app-debug.apk`，结果为 `Success`。
- 版本判断：
  - 本次属于提交、推送和真机安装记录，不改变应用功能、构建方式、模型资产或重要技术决策，不新增版本号。
  - 应用版本保持上一条功能更新确定的 `v1.3.0`。
- 后续事项：
  - 建议在真机上打开应用，观察关怀模式、底部面板遮挡比例、风险状态动效和语音/震动开关是否符合实际使用预期。

### 前端与关怀模式大幅升级

- 时间：2026-05-17 18:16:58 +08:00
- 执行者：violjjet
- 类型：功能 / 交互 / 无障碍 / 文档 / 构建
- 修改范围：
  - `app/src/main/java/com/linnan/blindassist/MainActivity.kt`
  - `app/src/main/java/com/linnan/blindassist/ui/DetectionOverlayView.kt`
  - `app/build.gradle.kts`
  - `README.md`
  - `DEVELOPMENT_LOG.md`
- 修改内容：
  - 保留原生 Android View 架构，参考相机、导航和实时工具类 App 的思路，把底部区域升级为“品牌/场景说明 + 状态徽标 + 主风险指令 + 辅助说明 + 两行控制区”的实时工作台。
  - 新增 `关怀` 开关：开启后主指令、说明和目标行字号增大，面板对比度提高，开发调试入口隐藏，并在 overlay 中显示中心引导线。
  - 将检测、语音、震动、关怀四个开关拆成两行布局，减轻横向拥挤，保留 52dp 最小触达高度和 TalkBack 状态描述。
  - 新增状态徽标，按平稳、观察、需留意、高风险、迫近提醒等状态动态调整颜色和读屏说明。
  - 为主标题变化加入 180ms 的轻量进入动效，为面板初次出现加入 260ms 的上滑淡入动效，让界面反馈更流畅但不干扰摄像头预览。
  - 优化关怀模式下的文案：使用更直接的行动建议，例如“立刻注意：正前”“建议减速，先确认左前方向”，避免把原型表述成可替代人工判断的安全设备。
  - 将应用版本从 `v0.8.0` 提升到 `v1.3.0`，`versionCode` 从 `4` 提升到 `5`，并在 README 同步记录界面行为。
- 修改原因：
  - 用户要求大幅优化前端和交互界面，目标是精美、流畅、易用，并加入关怀模式；当前界面虽然已有基础分层，但仍偏调试工具，缺少面向助盲使用场景的情绪稳定感、清晰行动指令和弱视/紧张场景下的可读性。
  - 本次更新把优秀 App 常见的清晰层级、少而明确的状态、克制动效、单手可触达控制和高对比辅助模式落到当前原生 Android 实现中，同时不改 CameraX、TFLite、风险算法、语音和震动反馈策略。
- 验证方式：
  - 已运行 `git status --short`，确认存在既有未跟踪文件 `多模态智能助盲系统1.4.pptx`，本次未处理该文件。
  - 根据仓库已知沙箱限制，已直接提权运行完整 Gradle 验证：`$env:JAVA_HOME='C:\Program Files\Android\Android Studio\jbr'; $env:PATH="$env:JAVA_HOME\bin;$env:PATH"; .\gradlew.bat :app:testDebugUnitTest :app:assembleDebug --no-daemon`。
  - 验证结果：`BUILD SUCCESSFUL in 38s`，共 41 个 actionable tasks，其中 14 个 executed、27 个 up-to-date。
  - 编译期间仅保留既有 CameraX `ImageAnalysis.Builder.setTargetResolution(Size)` deprecated warning，本次未调整分析分辨率策略。
  - APK 已生成：`app/build/outputs/apk/debug/app-debug.apk`，大小约 32.2 MB。
- 版本判断：
  - 本次属于大更新，原因是实时前端形态、交互层级、关怀模式和可访问性体验发生明显升级，但未改变核心检测能力、模型资产、风险算法、反馈策略或整体架构。
  - 按大更新规则，项目版本从 `v0.8.0` 提升到 `v1.3.0`。
- 后续事项：
  - 建议真机观察关怀模式下底部面板在不同屏幕尺寸、横竖屏和强光/弱光摄像头画面上的遮挡比例与可读性。
  - 后续如继续打磨，可考虑增加用户偏好持久化，让关怀模式、语音和震动开关在下次启动时保持上次选择。

### 直接提权执行已知失败命令规范收紧

- 时间：2026-05-17 17:03:40 +08:00
- 执行者：violjjet
- 类型：文档 / 协作规范
- 修改范围：
  - `AGENTS.md`
  - `DEVELOPMENT_LOG.md`
- 修改内容：
  - 将“已知沙箱权限问题与处理方式”从“先保留失败再提权重跑”收紧为：对已经确认会稳定复现的命令，后续代理应直接按权限规则提权执行。
  - 明确完整 Gradle 验证命令 `.\gradlew.bat :app:testDebugUnitTest :app:assembleDebug --no-daemon` 已多次在普通沙箱中因 `java.net.SocketException: Permission denied: getsockopt` 失败，需要运行完整验证时可直接提权。
  - 明确 `git push` 已多次在普通沙箱中因 HTTPS Credential Manager 或 SSH key/22 端口权限失败，推送前仍需确认远端和分支，但推送本身可直接提权。
  - 保留对 ADB 安装的谨慎规则：先用仓库内 `adb.exe` 检查设备和安装；如果后续也稳定出现权限限制，再改为直接提权安装。
  - 明确直接提权执行时，最终说明和开发日志只需记录这是基于已知沙箱限制直接提权、实际执行命令和结果，不必人为复现一次普通沙箱失败。
- 修改原因：
  - 用户指出这类失败已经百分百复现，继续每次先普通执行再失败会浪费时间；规范应反映当前真实环境，避免重复踩坑。
- 验证方式：
  - 已运行 `git status --short`，确认当前除本次规范文档变更外，仍存在既有未跟踪文件 `多模态智能助盲系统1.4.pptx`，本次未处理该文件。
  - 本次仅修改协作规范和开发日志，未改 Android 源码、构建脚本、模型资产或运行逻辑，因此未运行 Gradle 构建。
- 版本判断：
  - 本次属于协作规范文档补充，不改变项目功能、使用方式、构建方式、模型资产、测试结论或重要技术决策，不计为版本更新。
  - 未修改 README、`versionName` 或 `versionCode`。
- 后续事项：
  - 后续执行完整 Gradle 验证或 `git push` 时，应直接按本节规范提权执行，并在最终说明中写明结果。

### 沙箱权限问题处理规范补充

- 时间：2026-05-17 17:00:17 +08:00
- 执行者：violjjet
- 类型：文档 / 协作规范
- 修改范围：
  - `AGENTS.md`
  - `DEVELOPMENT_LOG.md`
- 修改内容：
  - 在 `AGENTS.md` 新增“已知沙箱权限问题与处理方式”章节。
  - 记录 Gradle wrapper 下载或构建验证在普通沙箱中可能因 `java.net.SocketException: Permission denied: getsockopt` 失败，应保留错误并用 `require_escalated` 提权重跑同一命令。
  - 记录 Git 推送在普通沙箱中可能因 HTTPS Git Credential Manager 卡住、SSH key 不可访问或 GitHub 22 端口被拒绝而失败，应确认远端和分支，必要时 trace 定位，再提权执行 `git push`。
  - 记录手机安装优先使用仓库内 `.\.android-sdk\platform-tools\adb.exe`，先用 `adb devices` 确认设备在线，再执行 `adb install -r`，如遇权限限制则记录错误并提权重试。
  - 明确所有提权重跑都要在最终说明和开发日志中写明首次失败命令、原始错误、提权后的命令和结果。
- 修改原因：
  - 用户指出此前推送、构建验证和安装等操作遇到的失败多为沙箱权限或凭据访问限制，后续可以通过提权成功解决；这些经验应写入规范，避免重复踩同一类坑。
- 验证方式：
  - 已运行 `git status --short`，确认当前除本次规范文档变更外，仍存在既有未跟踪文件 `多模态智能助盲系统1.4.pptx`，本次未处理该文件。
  - 本次仅修改协作规范和开发日志，未改 Android 源码、构建脚本、模型资产或运行逻辑，因此未运行 Gradle 构建。
- 版本判断：
  - 本次属于协作规范文档补充，不改变项目功能、使用方式、构建方式、模型资产、测试结论或重要技术决策，不计为版本更新。
  - 未修改 README、`versionName` 或 `versionCode`。
- 后续事项：
  - 后续遇到同类沙箱权限失败时，应优先按本节规范记录失败并提权重跑，避免把环境权限问题误判为代码问题。

### 前端与交互实用升级

- 时间：2026-05-17 16:49:41 +08:00
- 执行者：violjjet
- 类型：功能 / 交互 / 无障碍 / 文档 / 构建
- 修改范围：
  - `app/src/main/java/com/linnan/blindassist/MainActivity.kt`
  - `app/src/main/java/com/linnan/blindassist/ui/DetectionOverlayView.kt`
  - `app/build.gradle.kts`
  - `README.md`
  - `DEVELOPMENT_LOG.md`
- 修改内容：
  - 保留原生 Android View 架构，未引入 Jetpack Compose 或新 UI 框架。
  - 将实时检测界面底部区域从单行长状态文本升级为主风险状态、控制开关和默认折叠的调试信息三层结构。
  - 主状态区突出显示风险等级、相对距离、方向、目标、目标数和紧急度；FPS、total/pre/infer/post 耗时与模型状态移入可折叠调试区。
  - 检测、语音、震动开关保留不低于 48dp 的触控高度，并补充用于 TalkBack 的 `contentDescription` 状态说明。
  - 检测关闭时清空 overlay、重置风险稳定器，并显示暂停状态；重新开启后进入等待画面和稳定风险结果状态。
  - 优化检测覆盖层：风险源目标增加高亮 halo 和更粗边框，普通检测框更克制；危险区域改为轻量填充加描边；标签横向贴边时自动收进屏幕内。
  - 将应用版本从 `v0.7.0` 提升到 `v0.8.0`，`versionCode` 从 `3` 提升到 `4`，并在 README 记录界面行为。
- 修改原因：
  - 原界面把风险、目标数量、性能耗时、FPS 和模型状态压在一行长文本中，行走场景下难以快速读取，也不利于区分用户信息与开发调试信息。
  - 本次按“实用升级 + 调试信息可折叠”目标优化可读性、交互层级和无障碍语义，同时保持 CameraX、TFLite、风险规则、语音和震动反馈策略不变。
- 验证方式：
  - 首次在沙箱内运行 `$env:JAVA_HOME='C:\Program Files\Android\Android Studio\jbr'; $env:PATH="$env:JAVA_HOME\bin;$env:PATH"; .\gradlew.bat :app:testDebugUnitTest :app:assembleDebug --no-daemon` 失败，错误为 `java.net.SocketException: Permission denied: getsockopt`，原因是 Gradle wrapper 下载网络访问受限。
  - 已按权限要求提权重跑同一命令，结果为 `BUILD SUCCESSFUL in 34s`；随后补充 overlay 标签截断逻辑后再次验证，沙箱内仍因同一网络权限错误失败，提权重跑后结果为 `BUILD SUCCESSFUL in 26s`。
  - 构建输出 `app/build/outputs/apk/debug/app-debug.apk` 已生成，大小约 32.2 MB。
  - 编译期间仅保留既有 CameraX `setTargetResolution` deprecated warning，本次未改分析分辨率策略。
- 版本判断：
  - 本次属于小更新，原因是明显改善前端信息层级、交互和无障碍语义，但未改变产品核心检测能力、模型资产、风险算法、反馈策略或构建方式。
  - 按小更新规则，项目版本从 `v0.7.0` 提升到 `v0.8.0`。
- 后续事项：
  - 建议真机观察底部面板在不同屏幕尺寸上的遮挡比例，以及风险标题颜色在室外/暗光摄像头画面上的可读性。
  - 如后续需要进一步压缩界面占用，可考虑把调试入口改为更小的图标按钮，或按手势展开高级信息。

### 距离化风险提醒大更新

- 时间：2026-05-17 16:07:01 +08:00
- 执行者：violjjet
- 类型：功能 / 重构 / 测试 / 文档 / 构建
- 修改范围：
  - `app/src/main/java/com/linnan/blindassist/risk/RiskModels.kt`
  - `app/src/main/java/com/linnan/blindassist/risk/RiskAnalyzer.kt`
  - `app/src/main/java/com/linnan/blindassist/risk/RiskStabilizer.kt`
  - `app/src/main/java/com/linnan/blindassist/feedback/FeedbackController.kt`
  - `app/src/main/java/com/linnan/blindassist/MainActivity.kt`
  - `app/src/main/java/com/linnan/blindassist/ui/DetectionOverlayView.kt`
  - `app/src/test/java/com/linnan/blindassist/risk/RiskAnalyzerTest.kt`
  - `app/src/test/java/com/linnan/blindassist/risk/RiskStabilizerTest.kt`
  - `app/src/test/java/com/linnan/blindassist/feedback/FeedbackControllerTest.kt`
  - `app/build.gradle.kts`
  - `README.md`
  - `DEVELOPMENT_LOG.md`
- 修改内容：
  - 新增 `ProximityBand`，包含 `FAR`、`MID`、`NEAR`、`CRITICAL`，并在 `RiskResult` 中记录相对距离等级和 `urgencyScore`。
  - 升级 `RiskAnalyzer`，综合检测框底部位置、面积比例、中心偏置、类别权重和置信度计算紧急分数，按相对距离输出分层风险。
  - 明确不估算真实米数，仅使用远处、中距、近处、迫近等相对等级，避免单目视觉误导。
  - 调整 `RiskStabilizer`，稳定键纳入相对距离等级；距离升级时可更快确认，短暂丢帧仍保留已确认提醒。
  - 将 `FeedbackController` 的反馈计划抽为可测试逻辑：`CRITICAL` 使用 850ms 冷却和 420ms 震动，`NEAR` 使用 1500ms 冷却和 160ms 震动，`MID/FAR` 不触发语音或震动。
  - 状态栏新增距离等级、方向、目标标签和紧急分数；检测框颜色按迫近、近处、中距风险源区分。
  - 将应用版本从 `v0.2.0` 提升到 `v0.7.0`，`versionCode` 从 `2` 提升到 `3`。
- 修改原因：
  - 当前应用已经具备本地检测、风险规则、性能埋点和提醒稳定化能力；下一步核心体验瓶颈是只知道“有风险”，但不能表达相对接近程度。
  - 本次大更新让提醒从“方向 + 风险等级”升级为“方向 + 距离等级 + 紧急程度 + 分层反馈”，更符合助盲避障原型的使用语义。
- 验证方式：
  - 首次在沙箱内运行 `$env:JAVA_HOME='C:\Program Files\Android\Android Studio\jbr'; $env:PATH="$env:JAVA_HOME\bin;$env:PATH"; .\gradlew.bat :app:testDebugUnitTest --no-daemon` 失败，错误为 `java.net.SocketException: Permission denied: getsockopt`，原因是 Gradle wrapper 网络访问受限。
  - 已按权限要求提权重跑单元测试；第一次测试失败 1 个用例，原因是测试误把 `MID` 视觉低风险期望为 `NONE`，与本次规划不一致。
  - 已修正测试后重新运行同一单元测试命令，结果为 `BUILD SUCCESSFUL in 19s`，共 22 个测试通过。
- 版本判断：
  - 本次属于大更新，原因是核心提醒体验和风险结果模型发生明显升级，但未更换模型资产、未引入新架构、未改变构建方式。
  - 按大更新规则，项目版本从 `v0.2.0` 提升到 `v0.7.0`。
- 后续事项：
  - 仍需真机观察 `NEAR` 与 `CRITICAL` 阈值是否适合室内步行场景，重点关注误报、漏报、语音频率和震动强度。
  - 如真机提醒过于频繁，可继续按类别或方向微调冷却、确认帧数和阈值。

### 项目执行人命名规则更新

- 时间：2026-05-17 15:43:22 +08:00
- 执行者：violjjet
- 类型：文档 / 协作规范
- 修改范围：
  - `AGENTS.md`
  - `DEVELOPMENT_LOG.md`
- 修改内容：
  - 在 `AGENTS.md` 的开发日志要求中新增规则：项目执行人命名统一为 `violjjet`。
  - 明确后续写入 `DEVELOPMENT_LOG.md` 的“执行者”字段时，应使用 `violjjet`。
- 修改原因：
  - 用户要求将项目执行人命名写入 `AGENTS.md`，统一后续开发日志中的执行者名称。
- 验证方式：
  - 已运行 `git status --short`，确认当前存在上一轮功能更新的未提交改动和一个既有未跟踪 PPT 文件；本次未回滚或处理这些既有改动。
  - 本次仅修改协作规范和开发日志，未涉及 Android 代码、构建脚本、模型资产或运行逻辑，因此未运行 Gradle 构建和单元测试。
- 版本判断：
  - 本次属于协作规则的轻量说明补充，不改变项目功能、使用方式、构建方式、模型资产、测试结论或重要技术决策，不计为版本更新。
  - 未修改 README 和应用版本号。
- 后续事项：
  - 后续任务写入开发日志时，“执行者”字段应使用 `violjjet`。

### 风险提醒稳定化更新

- 时间：2026-05-17 15:38:21 +08:00
- 执行者：Codex
- 类型：功能 / 测试 / 文档 / 构建
- 修改范围：
  - `app/src/main/java/com/linnan/blindassist/risk/RiskStabilizer.kt`
  - `app/src/main/java/com/linnan/blindassist/MainActivity.kt`
  - `app/src/test/java/com/linnan/blindassist/risk/RiskStabilizerTest.kt`
  - `app/build.gradle.kts`
  - `README.md`
  - `DEVELOPMENT_LOG.md`
- 修改内容：
  - 重新检测并读取了新安装的 app 开发相关 skills：`android-architecture`、`android-testing`、`kotlin-specialist`、`kotlin-concurrency-expert`、`android-accessibility`。
  - 新增纯 Kotlin `RiskStabilizer`，位于 `risk` 模块，不引入 Android framework 依赖、协程、Hilt 或多模块改造。
  - 将稳定器接入 `MainActivity`：`RiskAnalyzer.analyze()` 输出原始风险，`RiskStabilizer.update()` 输出稳定风险，overlay、状态栏、语音和震动反馈均使用稳定后的风险结果。
  - HIGH 风险单帧立即确认；MEDIUM 风险需要同方向、同消息连续 2 帧确认；已确认的中高风险在短暂 LOW/NONE 帧中最多保持 600ms。
  - 检测开关关闭时会清空 overlay 并重置稳定器状态。
  - 按 accessibility skill 检查底部开关，并补充 48dp 最小触达高度。
  - 新增 `RiskStabilizerTest`，覆盖高风险立即确认、中风险两帧确认、单帧中风险消失不触发、方向切换重置、短暂无风险保持后清空。
  - 将项目版本从 `v0.1.0` 提升到 `v0.2.0`，`versionCode` 从 `1` 提升到 `2`，并在 README 记录提醒稳定化行为和谨慎安全表述。
- 修改原因：
  - 上一轮性能优化后，后续事项中仍保留风险阈值和提醒策略未调整的问题；本次通过轻量稳定层降低帧间检测抖动对语音/震动提醒的影响。
  - 本次更新直接改善助盲避障提醒体验，同时保持现有 CameraX、TFLite、YOLO 解析、NMS 和模型资产不变，风险边界可通过 JVM 单元测试覆盖。
- 验证方式：
  - 初次运行 `$env:JAVA_HOME='C:\Program Files\Android\Android Studio\jbr'; $env:PATH="$env:JAVA_HOME\bin;$env:PATH"; .\gradlew.bat :app:testDebugUnitTest :app:assembleDebug --no-daemon` 时，沙箱内 Gradle wrapper 下载失败，错误为 `java.net.SocketException: Permission denied: getsockopt`。
  - 已按权限要求提权重跑同一命令；第一次提权验证编译和 APK 构建通过，但 `RiskStabilizerTest` 有 2 个用例失败，原因是中风险未确认回退时误清空了 pending 状态。
  - 已修复稳定器 pending/hold 分离逻辑后再次提权运行同一命令，结果为 `BUILD SUCCESSFUL in 23s`。
  - 最终单元测试结果：共 12 个测试通过，失败和错误均为 0；`app/build/outputs/apk/debug/app-debug.apk` 已生成。
- 版本判断：
  - 本次属于小更新，原因是新增局部体验功能和测试，未改变产品形态、核心架构、模型资产或构建方式。
  - 按规则项目版本从 `v0.1.0` 提升到 `v0.2.0`。
- 后续事项：
  - 需要真机观察稳定层是否降低语音/震动抖动，以及 600ms 保持时长和 2 帧中风险确认是否适合实际步行场景。
  - 如真机仍频繁误报，可继续评估按目标类别、距离估计或风险等级变化设置更细的确认阈值。

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
