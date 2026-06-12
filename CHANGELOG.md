# BlindAssist 更新记录

本文件按真实版本记录 BlindAssist 的功能演进、验证证据和可展示 APK 归档。它用于课程汇报、答辩材料整理和版本对比，不替代 `DEVELOPMENT_LOG.md` 的逐次工作记录。

## v8.8.0 - 综合风险函数与连续帧逼近风险
- 状态：已完成本地单测、lint、构建、同设备 BlindAssist EvalSet benchmark、90 秒真机回归和里程碑 APK 归档，`versionCode=32`，`versionName=8.8.0`。
- 主要变化：
  - `RiskAnalyzer` 从单帧硬阈值升级为可解释综合风险函数，输出 `riskScore` 与 `RiskScoreBreakdown`，保留 `RiskLevel`、`RiskDirection`、`ProximityBand` 和兼容用 `urgencyScore`。
  - 新增纯 Kotlin `TemporalRiskTracker`，在 `AssistEngine` 中位于 `RiskAnalyzer` 与 `RiskStabilizer` 之间，使用最近 5 帧、约 900ms 的同目标轨迹判断 `APPROACHING/STABLE/RECEDING`。
  - 逼近趋势默认使用框底部位置、面积增长、中心连续性和可选距离证据，不新增或替换模型资产；侧向目标最多保持中风险，避免把横向经过误升为高风险。
  - `DetectorAbDeviceBenchmarkTest` 兼容可选序列字段，并新增 `approachRiskRecall`、`approachFalsePositiveRate`、`approachDirectionAccuracy`、`approachCriticalMissCount`、`meanTimeToAlertFrames` 和 `approachLabeledSequenceCount`。
- 验证：
  - `.\.venv-export312\Scripts\python.exe scripts\inspect_tflite.py`：通过，默认 yolo11n 输入 `[1, 320, 320, 3] float32`，输出 `[1, 84, 2100] float32`，`assertions=passed`。
  - `:core:assist:test :core:vision:testDebugUnitTest :core:device:testDebugUnitTest :core:ui:testDebugUnitTest :feature:assist:testDebugUnitTest :app:testDebugUnitTest`：通过。
  - `:app:lintDebug :core:vision:lintDebug :core:device:lintDebug :core:ui:lintDebug :feature:assist:lintDebug`：通过。
  - `:app:assembleDebug :app:assembleDebugAndroidTest`：通过。
  - `powershell -ExecutionPolicy Bypass -File .\scripts\run_detector_ab_device_benchmark.ps1 -DatasetKind BlindAssistEvalSet -DatasetRoot test-artifacts.local\datasets\blindassist-evalset-20260527-impl`：通过，设备 `R5CX10M8Y8X`，证据目录 `test-artifacts.local/detector-ab-device-benchmark/20260612-135829`；当前旧 evalset 无序列字段，新增 approach 指标按预期为 `0`。
  - 默认 90 秒真机回归通过，证据目录 `test-artifacts.local/device-regression/20260612-140055`，冷启动 `TotalTime=836` / `WaitTime=840`。
- APK：
  - 本地归档 `E:\linnan\blind-assist-apk-archive\apks\BlindAssist-v8.8.0-debug-20260612-140333.apk`，大小 `47,288,840` bytes，SHA256 `017CF063FFD651270335DDD3033E5018C64A86A8BD78351D2F4B9C1B16D23364`。
  - 里程碑同步 `releases/apk/BlindAssist-v8.8.0-debug-20260612-140333.apk`。

## v8.3.0 - 生命周期串行化、隐私备份与英文无障碍一致性

- 状态：已完成本地单测、lint、构建、Compose 真机用例和 90 秒真机回归验证，`versionCode=31`，`versionName=8.3.0`。
- 主要变化：
  - `feature:assist` 新增 `AssistRuntimeLifecycleGate`，统一串行化相机会话、帧处理和反馈关闭边界；停止/关闭时先拒收新帧，再等待在途帧完成，最后 reset processor、关闭 detector 和 shutdown feedback。
  - `AssistFrameProcessor` 在处理前登记在途帧，stop/shutdown 后的新帧会立即关闭且不进入 detector；`AssistRuntimeEffectExecutor` 在 Start/Stop/Reset/Close 路径中按固定顺序进入 gate。
  - `CameraXFrameSource` 增加 session generation 与 lifecycle lock，late provider callback 在 stop/close/shutdown 后不会重新 bind 或触发过期 `onStarted`；`shutdown()` 继续保持幂等与 analyzer executor 有界等待。
  - `FeedbackController` 用生命周期锁串行 `applySettings`、`notify` 和 `shutdown`，shutdown 后不再触达 speech/haptic output。
  - Manifest 保留 `allowBackup=true`，新增 `dataExtractionRules` 与 `fullBackupContent`，系统备份/设备迁移只 include `sharedpref/blindassist_user_preferences.xml`，不备份帧、日志、缓存、测试证据或其他本地文件。
  - 英文模式下补齐相机返回 `Back to features`、底部导航、Profile 状态、权限/拒绝/眼镜占位弹窗，以及 Compose 测试定位的中英文兼容。
- 验证：
  - `.\.venv-export312\Scripts\python.exe scripts\inspect_tflite.py`：通过，默认 yolo11n 输入 `[1, 320, 320, 3] float32`，输出 `[1, 84, 2100] float32`，`assertions=passed`。
  - `:core:assist:test :core:vision:testDebugUnitTest :core:device:testDebugUnitTest :core:ui:testDebugUnitTest :feature:assist:testDebugUnitTest :app:testDebugUnitTest`：通过。
  - `:app:lintDebug :core:vision:lintDebug :core:device:lintDebug :core:ui:lintDebug :feature:assist:lintDebug`：通过。
  - `:app:assembleDebug :app:assembleDebugAndroidTest`：通过；期间修正 androidTest 中 `assertDoesNotExist` API 兼容、设置页测试标签和 Android 16 debug app 兼容性弹窗处理。
  - `:app:connectedDebugAndroidTest` 完整矩阵未作为通过证据：设备端历史 `DetectorAbDeviceBenchmarkTest` 在约 40 秒后进程被 signal 9 kill，UTP 报 `Process crashed`。随后按类过滤运行本轮相关 Compose instrumentation：`BlindAssistComposeTest` 与 `CameraControlPanelStandaloneTest` 共 11 个用例通过。
  - `powershell -ExecutionPolicy Bypass -File .\scripts\run_device_regression.ps1 -SampleSeconds 90`：通过，证据目录 `test-artifacts.local/device-regression/20260611-173421`。
- APK：
  - 完整本地归档 `E:\linnan\blind-assist-apk-archive\apks\BlindAssist-v8.3.0-debug-20260611-174127.apk`，大小 `47,490,277` bytes，SHA256 `26217859834CE2907288BA38939E50AEDEFFCE9A59735D53C6ACA641C414BE25`。
  - 本轮不默认提交 `releases/apk/` 里程碑 APK。

## v8.2.0 - 相机可靠性与无障碍修复

- 状态：已完成本地与真机验证，`versionCode=30`，`versionName=8.2.0`。
- 主要变化：
  - 相机关闭时清空运行时的 view-ready 状态和旧 `PreviewView` / overlay 引用，重新打开必须等待新的 `AndroidView` 创建后再绑定 CameraX，避免旧预览 surface 导致黑屏或预览不更新。
  - `CameraXFrameSource.shutdown()` 对 analyzer executor 做有界等待；`TfliteYoloDetector.detectFrame()` 与 `close()` 使用同一把生命周期锁，避免推理与 interpreter/delegate close 交叉。
  - TFLite 初始化把 labels 加载纳入同一失败路径，模型校验失败时会关闭已创建的 interpreter/delegate。
  - 相机页 `CompactAction` 不再给普通动作统一声明“已启用/未启用”；真正的 toggle 和 debug 展开态保留本地化 `stateDescription`。
  - 相机底部面板增加导航栏避让、最大高度和内部滚动，大字体与 debug 展开时仍可访问关键控件。
  - `scripts/inspect_tflite.py` 从只打印升级为断言模型存在、输入 `[1,320,320,3] float32`、输出 `[1,84,2100] float32`，CI 增加模型检查和 `:app:assembleDebugAndroidTest`。
- 验证：
  - `.\.venv-export312\Scripts\python.exe scripts\inspect_tflite.py`：通过，使用 `ai-edge-litert` 后端，输入 `[1, 320, 320, 3] float32`，输出 `[1, 84, 2100] float32`，`assertions=passed`。
  - `:core:assist:test :core:vision:testDebugUnitTest :core:device:testDebugUnitTest :core:ui:testDebugUnitTest :feature:assist:testDebugUnitTest :app:testDebugUnitTest`：通过。
  - `:app:lintDebug :core:vision:lintDebug :core:device:lintDebug :core:ui:lintDebug :feature:assist:lintDebug`：通过。
  - `:app:assembleDebug :app:assembleDebugAndroidTest`：首次因新增大字体 androidTest 缺少 Compose foundation 测试依赖失败；补充 `androidTestImplementation(libs.androidx.compose.foundation)` 后通过。
  - `:app:connectedDebugAndroidTest`：在 Samsung `SM-S9280 - 16` / serial `R5CX10M8Y8X` 上通过，7 个 Compose 仪器测试 0 failures。期间新增大字体组件测试先后暴露 `MainActivity` 已设置 content 与 `FPS` 文本匹配不唯一两个测试写法问题，修正后重跑通过。
  - `powershell -ExecutionPolicy Bypass -File .\scripts\run_device_regression.ps1 -SampleSeconds 90`：通过，证据目录 `test-artifacts.local-device-regression-20260526-231417`。
- APK：
  - 完整本地归档 `E:\linnan\blind-assist-apk-archive\apks\BlindAssist-v8.2.0-debug-20260526-215736.apk`，大小 `47,238,655` bytes，SHA256 `068F2515954F8D96D3BBE92B9D788725ED54872FBB1E54CDCE1DFEA9FC877027`。
  - 本次为 `+0.1` 小版本，未提交 Git 里程碑 APK。

## v8.1.0 - 真实使用体验 UI 升级

- 状态：已完成，`versionCode=29`，`versionName=8.1.0`。
- 主要变化：
  - 功能页改为日常辅助启动台，先展示当前行走任务、场景、提醒档位和 Care Mode 状态，再突出 `使用手机摄像头` 主入口。
  - 日常使用向导继续保留五个既有预设，不新增偏好 key、不改变场景/档位/语音/震动映射。
  - 相机页底部面板重排为主要风险状态、行动建议、场景/模式、检测/语音/震动核心开关优先；快捷调安静、调敏感、场景切换和调试信息降级到更多调整区域。
  - Care Mode 成为低干扰相机体验：放大主要指导语、提高面板对比度、隐藏快捷调节和调试入口，只保留必要控制。
  - 设置页按“界面与辅助、提醒方式、行走场景、调试与记录”分组，保留既有中英文 TalkBack 文案和 Compose testTag 契约。
  - 本轮不修改 CameraX/TFLite/风险规则/反馈策略，不新增联网、蓝牙、定位、存储或新的显示模式偏好。
- 验证：
  - `.\gradlew.bat :core:ui:testDebugUnitTest :feature:assist:testDebugUnitTest :app:testDebugUnitTest --no-daemon --console=plain`：通过。
  - `.\gradlew.bat :app:lintDebug :core:ui:lintDebug :feature:assist:lintDebug :app:assembleDebug :app:assembleDebugAndroidTest --no-daemon --console=plain`：通过。
  - `.\gradlew.bat :app:connectedDebugAndroidTest --no-daemon --console=plain`：首次因设备端旧签名包安装冲突失败，确认包状态清理后重跑通过，`SM-S9280 - 16` 上 6 个 Compose 仪器测试通过。
  - `powershell -ExecutionPolicy Bypass -File .\scripts\run_device_regression.ps1 -SampleSeconds 90`：通过，证据目录 `test-artifacts.local-device-regression-20260525-222502`，冷启动 `TotalTime: 761` / `WaitTime: 762`，设备端包信息 `versionName=8.1.0` / `versionCode=29`。
- APK：
  - `releases/apk/BlindAssist-v8.1.0-debug-20260525-222724.apk`，大小 `47,238,663` bytes，SHA256 `A48769A7A9F5233526DDD936AF40A6DB9321DBFD163E648D098D1F16576F94D8`。

## v7.6.0 - 运行时管线与发布卫生更新
- 状态：已完成，`versionCode=28`，`versionName=7.6.0`。
- 主要变化：
  - `AssistRuntimeController` 收敛为较薄的入口；帧处理、运行时效果执行、相机生命周期适配、渲染、设置/配置同步和帧管线统计拆分到更小的协作者。
  - 实时 CameraX 路径改为传递可关闭的 `VisionFrame` / `RgbaVisionFrame`，避免 analyzer 路径逐帧分配 `Bitmap` 和旋转后的 `Bitmap`。
  - `ImagePreprocessor` 可在采样旋转时直接把 RGBA 帧缓冲写入模型输入 buffer；原有 Bitmap 路径继续保留用于兼容。
  - 运行时性能日志和调试指标新增 dropped-frame rate 以及 P50/P95 推理耗时。
  - 新增真机回归、APK 归档和 APK 校验脚本；release 构建启用 R8/resource shrink，并要求本地未跟踪签名配置。
  - 历史已跟踪 `test-artifacts/` 证据复制到本地归档，并从后续 Git 跟踪中移除。
- 验证：
  - `.\.venv-export312\Scripts\python.exe scripts\inspect_tflite.py`：通过，输入 `[1, 320, 320, 3] float32`，输出 `[1, 84, 2100] float32`。
  - 完整多模块单元测试矩阵通过。
  - 完整多模块 lint 矩阵通过。
  - `:app:assembleDebug :app:assembleDebugAndroidTest` 通过。
  - 缺少 `keystore.properties` 时运行 `:app:assembleRelease`，按预期失败并给出明确的本地签名配置提示。
- APK：
  - `releases/apk/BlindAssist-v7.6.0-debug-20260525-004833.apk`, size `47,222,231` bytes, SHA256 `2DE5CE894D0C46A8D099000B6E2624DA4B102E61A0E31BE8F501252D102520DC`.

## v7.1.0 - 反馈触达与 CameraX 可靠性更新

- 状态：已完成，`versionCode=27`，`versionName=7.1.0`。
- 主要变化：
  - 反馈链路改为真实触达语义：只有 TTS 或震动输出被系统 API 接受后才返回 `TRIGGERED`，否则返回 `FEEDBACK_UNAVAILABLE` 且不写入冷却/疲劳记录。
  - `core:device` 新增 library manifest 声明 `VIBRATE`，震动实现固定使用 minSdk 26+ 的 `VibrationEffect` 路径，修复多模块 lint 对权限契约的漏检/误报。
  - CameraX analyzer 和 frame processing 异常统一走 `CameraSourceFailed`，进入错误态后停止相机、清空 overlay 并重置会话；runtime 配置改为 atomic 快照，单帧处理使用同一份配置。
  - TTS、震动、CameraX 和 TFLite 可恢复路径会重新抛出 `VirtualMachineError`、`ThreadDeath` 和 `LinkageError` 等 fatal 错误，避免把严重运行时问题吞掉。
  - TFLite YOLO 输出解析拆出可测 helper，并新增 golden JVM 测试覆盖 confidence threshold、label mapping、letterbox 坐标回映射和 same-class NMS。
  - README、CI lint 矩阵和 APK 归档材料同步到多模块验证方式。
- 验证：
  - `.\gradlew.bat :core:assist:test :core:device:testDebugUnitTest :core:vision:testDebugUnitTest :feature:assist:testDebugUnitTest --no-daemon --offline --console=plain`：通过。
  - 完整显式多模块单测、显式多模块 lint、debug/androidTest APK 构建和模型检查均通过；设备端 `connectedDebugAndroidTest` 未运行，因为 `adb devices` 当前无在线设备。
- APK：
  - `releases/apk/BlindAssist-v7.1.0-debug-20260524-162936.apk`，大小 `47,205,843` bytes，SHA256 `5ADA5DC82A71AABDA3438C76CA7E7AA9341C15FDD11308C2861E9695AD75F323`。

## v6.9.0 - 完整多模块架构迁移

- 状态：已完成，`versionCode=25`，`versionName=6.9.0`。
- 主要变化：
  - 将项目从单 `:app` 模块拆分为 `:app`、`:feature:assist`、`:core:assist`、`:core:vision`、`:core:device` 和 `:core:ui`。
  - `:core:assist` 保持纯 Kotlin，承载模型、提醒档位、风险规则、会话编排、定位文案、日常模式和反馈规划契约。
  - `:core:vision` 承载 TFLite YOLO 检测器和图像预处理；`:core:device` 承载 CameraX、ImageProxy 转换、SharedPreferences 偏好和 Android TTS/震动反馈；`:core:ui` 承载 Compose 页面、UI 状态、guidance/summary mapper 和 overlay。
  - `:feature:assist` 承载 Hilt ViewModel、runtime 状态机/controller/config applier 和运行期依赖模块；`:app` 保留启动入口、manifest、资源、模型 assets 和 APK 配置。
  - 本轮为零用户可见行为迁移：UI 流程、权限、YOLO 模型路径、CameraX/TFLite 链路、风险阈值、SharedPreferences key、隐私边界、Room/DataStore、联网、蓝牙和存储行为均保持不变。
- 验证：
  - `.\gradlew.bat :core:assist:test --no-daemon`：通过。
  - `.\gradlew.bat :core:vision:testDebugUnitTest :core:device:testDebugUnitTest :core:ui:testDebugUnitTest :feature:assist:testDebugUnitTest --no-daemon`：通过。
  - `.\gradlew.bat :app:testDebugUnitTest :app:lintDebug :app:assembleDebug --no-daemon`：通过，生成 debug APK。
  - 首次新增 Android library plugin marker 和 AndroidX transitive dependency 时，普通沙箱因网络权限受限失败；随后按仓库已知沙箱限制提权解析依赖后继续验证。
- APK：
  - `releases/apk/BlindAssist-v6.9.0-debug-20260522-204908.apk`，大小 `47,205,843` bytes。

## v6.4.0 - Runtime 状态机与 Hilt 依赖拆分

- 状态：已完成，`versionCode=24`，`versionName=6.4.0`。
- 主要变化：
  - 接入 Hilt：新增 `@HiltAndroidApp` Application，`MainActivity` 改为 `@AndroidEntryPoint`，`BlindAssistViewModel` 改为 `@HiltViewModel`，由 Hilt 提供偏好、反馈、检测器、会话编排和相机源工厂依赖。
  - 新增纯 Kotlin `AssistRuntimeStateMachine`，集中管理 Idle、权限说明、权限请求、启动中、运行中、检测暂停、权限拒绝和错误状态；`AssistRuntimeController` 退为 effect executor。
  - 新增 `AssistRuntimeConfig` 和 `RuntimeConfigApplier`，把反馈开关、Care Mode、提醒档位、场景、语音风格、震动强度、语言和日常模式收敛成单一 runtime 配置快照。
  - 相机页补强状态反馈：相机启动中、模型不可用、检测已暂停、权限拒绝和启动失败都有明确 guidance；主视觉结构、模型、风险阈值、权限集合、SharedPreferences key、Room/DataStore、联网、蓝牙、存储和单模块结构保持不变。
  - Hilt 使用 `2.55` 而不是原计划的 `2.59.2`，因为 `2.59.2` Gradle 插件要求 AGP 9+，当前项目仍保持 AGP `8.7.3` 以避免扩大迁移范围。
- 验证：
  - 已运行 `.\gradlew.bat :app:testDebugUnitTest --no-daemon`，JVM 单元测试通过。
  - 已运行 `.\gradlew.bat :app:lintDebug :app:assembleDebug --no-daemon`，构建成功，lint 结果为 `0 errors, 15 warnings`。
  - 新增 JVM 测试覆盖 runtime 状态机的权限、视图晚到、检测暂停/恢复、启动失败、关闭相机和模型不可用路径，以及 runtime 配置的日常模式、快捷模式、自定义回退和语言/反馈开关同步。
- APK：
  - `app/build/outputs/apk/debug/app-debug.apk`，大小 `47,205,607` bytes。

## v5.9.0 - 测试稳定性与相机调试控件修复

- 状态：已完成，`versionCode=23`，`versionName=5.9.0`。
- 主要变化：
  - 修复 v5.8.0 真机完整测试中发现的相机页 `展开调试信息` ADB 点击不稳定问题：调试开关改用与其他相机控制一致的 `CompactAction` 按钮，位置上移到 Care Mode 和场景切换之前，并暴露明确的 `展开相机调试信息` / `收起相机调试信息` content description。
  - 修复设备端 Compose 测试在相机路径上的等待条件：测试现在显式授予相机权限，识别文本和 content description，打开相机前重置为 通用日常，避免 Care Mode 隐藏 Debug 控件，并在相机路径用例结束后关闭相机页。
  - 本轮不修改 YOLO 模型、CameraX/TFLite 推理链路、风险阈值、权限、联网、蓝牙、存储、Hilt、多模块、Room 或 DataStore。
- 验证：
  - 已按本仓库已知 Gradle 沙箱限制直接提权运行 `:app:testDebugUnitTest :app:assembleDebug --no-daemon`，构建成功。
  - JVM 单元测试结果：105 tests，0 failures，0 errors，0 skipped。
  - 已按本仓库已知 Gradle/设备测试沙箱限制直接提权运行 `:app:connectedDebugAndroidTest --no-daemon`，在 `SM-S9280 - 16` 上完成 6 个 Compose 仪器测试，0 failures，0 errors，0 skipped。
  - 已安装 v5.9.0 debug APK 到 `SM-S9280`，包信息核对为 `versionCode=23`、`versionName=5.9.0`，冷启动返回 `Status: ok`、`LaunchState: COLD`、`TotalTime: 676`。
  - 已用 ADB/UIAutomator 复测相机页 Debug 开关：点击 `展开相机调试信息` 后 UI 变为 `收起相机调试信息`，确认修复 v5.8.0 中的可点击性问题。
- APK：
  - `releases/apk/BlindAssist-v5.9.0-debug-20260519-174352.apk`

## v5.8.0 - 日常使用向导与一键模式

- 状态：已完成，`versionCode=22`，`versionName=5.8.0`。
- 主要变化：
  - Features 页新增日常使用向导，提供 通用日常、室内慢行、走廊通行、密集区域、户外慢行 五个日常预设。
  - 预设只映射到现有本地偏好：使用场景、提醒档位、语音风格、震动强度和 Care Mode；不新增单独模式持久化 key，手动调整后的组合显示为 自定义 / Custom。
  - 相机页显示当前日常模式，并将原来的提醒档位循环按钮改为 调安静 / 调敏感 两个明确快捷操作；两者保留当前场景，只调整提醒强度组合。
  - 新增中英文文案、TalkBack action-oriented content description、48dp+ 触控目标和 Compose 测试覆盖。
  - 本轮不修改 YOLO 模型、CameraX/TFLite 链路、风险阈值、权限、联网、蓝牙、存储、Hilt、多模块、Room 或 DataStore。
- 验证：
  - 已按本仓库已知 Gradle 沙箱限制直接提权运行 `:app:testDebugUnitTest :app:assembleDebug --no-daemon`，命令退出码为 0。
  - JVM 单元测试结果：105 tests，0 failures，0 errors，0 skipped。
  - debug APK 已安装到 `SM-S9280`，包信息核对为 `versionCode=22`、`versionName=5.8.0`，冷启动命令返回 `Status: ok`、`LaunchState: COLD`、`TotalTime: 679`。
  - `connectedDebugAndroidTest` 未执行：设备在线且重复 wireless ADB serial 已断开，但窗口状态仍为 `mDreamingLockscreen=true`、焦点在 `NotificationShade`，不适合运行 Compose 仪器测试。
- APK：
  - `releases/apk/BlindAssist-v5.8.0-debug-20260519-120747.apk`

## v5.3.0 - TalkBack、大字体与中英文切换

- 状态：已完成，`versionCode=21`，`versionName=5.3.0`。
- 主要变化：
  - 新增持久化 `AppLanguage` 偏好，设置页提供 `中文` / `English` App 内语言切换，默认仍为中文。
  - 核心体验文案支持中英文：语音提醒模板、提醒档位、使用场景、语音风格、震动强度、相机状态、风险解释、现场测试摘要和 TalkBack 语义。
  - `FeedbackController` 根据 App 内语言切换 TTS locale，中文使用 `Locale.CHINA`，英文使用 `Locale.US`。
  - 设置页选择器改为更适合大字体的 48dp+ 全宽控件，并补强 state description、action-oriented content description 和 heading 语义。
  - 本轮不修改 YOLO 模型、风险阈值、场景策略、权限、联网、蓝牙、存储、Hilt、多模块、Room 或 DataStore。
- 验证：
  - 已按本仓库已知 Gradle 沙箱限制直接提权运行 `:app:testDebugUnitTest :app:assembleDebug --no-daemon`，构建和 JVM 单元测试通过。
  - 首次 `connectedDebugAndroidTest` 在 `SM-S9280` wireless ADB 设备在线时尝试执行，但由于重复 mDNS serial / UTP 状态卡住，在约 184 秒后超时。
  - 随后断开重复的 `(2)` wireless ADB serial，只保留 `adb-R5CX10M8Y8X-nkVxqz._adb-tls-connect._tcp`，安装 v5.3.0 APK 到 `SM-S9280`，包信息核对为 `versionCode=21`、`versionName=5.3.0`，应用冷启动成功。
  - 设置 `ANDROID_SERIAL` 后重跑 `:app:connectedDebugAndroidTest --no-daemon`，在 `SM-S9280 - 16` 上完成 5 个 Compose 仪器测试，0 skipped，0 failed，`BUILD SUCCESSFUL in 45s`。
- APK：
  - `releases/apk/BlindAssist-v5.3.0-debug-20260519-113731.apk`

## v4.8.0 - 单模块质量升级

- 状态：已完成，`versionCode=20`，`versionName=4.8.0`。
- 主要变化：
  - 保持单模块、原生 Android/Kotlin、CameraX、TFLite、Compose 和 SharedPreferences，不引入 Hilt、多模块、新权限、联网、蓝牙、定位、Room 或 DataStore。
  - 新增 `ObjectDetector`、`DetectorFrameResult`、`FrameSource`、`CameraXFrameSource`、`AssistSessionCoordinator`、`FpsTracker`、`CameraGuidanceMapper` 和 `FieldTestSummaryMapper`，把检测输出、相机取流、会话编排、FPS 统计和 UI 状态映射从 `MainActivity` 中拆出。
  - `MainActivity` 缩减为生命周期、权限、Compose 绑定和用户设置转发入口；v4.3.0 移除 App 内展示中心的主界面形态继续保留。
  - 新增 JVM 测试覆盖会话编排、FPS 统计、风险指导 UI 映射和现场测试摘要映射。
- 验证：
  - `:app:testDebugUnitTest` 和 `:app:assembleDebug` 构建验证通过。
  - `connectedDebugAndroidTest` 已尝试两次：mDNS serial 未被 Gradle 识别，普通 Wi-Fi serial 随后变为 `offline`，因此本轮未获得仪器测试通过结果。
- APK：
  - `releases/apk/BlindAssist-v4.8.0-debug-20260519-005155.apk`

## v4.3.0 - 移除项目展示中心

- 状态：已完成，`versionCode=19`，`versionName=4.3.0`。
- 主要变化：
  - 暂时移除 Features 页里的 App 内“项目展示中心”，减少当前主界面的展示包装。
  - Features 页保留 `使用手机摄像头`、`连接眼镜设备` 占位、安全边界和模型/版本状态。
  - 新手引导回放继续保留在 Settings 页，课堂/答辩材料继续保留在 `README.md`、`CHANGELOG.md`、`DEMO_GUIDE.md` 和 APK 归档中。
  - 本轮不修改 CameraX/TFLite 检测链路、场景化提醒策略、权限、联网、蓝牙、存储或架构形态。
- 验证：
  - `:app:testDebugUnitTest`、`:app:assembleDebug` 和 `:app:assembleDebugAndroidTest` 构建验证通过。
  - debug APK 已安装到 `SM-S9280`，包信息核对为 `versionCode=19`、`versionName=4.3.0`。
  - Compose 仪器测试删除展示中心专项覆盖，保留底部导航、手机摄像头入口、设置反馈控件和相机页场景/风险解释区域覆盖。
- APK：
  - `releases/apk/BlindAssist-v4.3.0-debug-20260519-003109.apk`

## v4.2.0 - 场景化提醒与风险解释

- 状态：已完成，`versionCode=18`，`versionName=4.2.0`。
- 主要变化：
  - 新增手动 `使用场景` 偏好：通用、室内慢行、走廊通行、密集区域、户外慢行。
  - 通用场景保持 v4.1.0 提醒行为；其他场景只调整规则层的中风险确认、提醒保持、近距冷却和震动时长。
  - 相机控制面板显示当前场景和最近风险解释，说明已触发、未稳定、距离较远、冷却中、提醒保持或暂无可反馈风险等原因。
  - 现场测试摘要追加当前场景和最近解释；Care Mode 下保留简短解释，不暴露性能调试细节。
  - 本轮不新增自动场景识别、联网、定位、蓝牙、存储权限、模型变更或大型架构框架。
- 验证：
  - `:app:testDebugUnitTest` 和 `:app:assembleDebug` 构建验证通过。
  - debug APK 已安装到 `SM-S9280`，包信息核对为 `versionCode=18`、`versionName=4.2.0`。
  - Compose 仪器测试增加使用场景选择和相机页解释区域覆盖；本轮 `connectedDebugAndroidTest` 已尝试执行，但设备处于锁屏/Bouncer 状态，报告 `No compose hierarchies found in the app`，未作为通过证据。
- APK：
  - `releases/apk/BlindAssist-v4.2.0-debug-20260519-000200.apk`

## v4.1.0 - 展示交付加强

- 状态：已完成，`versionCode=17`，`versionName=4.1.0`。
- 主要变化：
  - 新增 App 内“项目展示中心”，集中说明本地识别、语音/震动提醒、现场测试摘要和原型安全边界。
  - 展示中心提供“开始演示”和“查看引导”操作，复用现有手机摄像头权限说明与新手引导流程。
  - 新增 `DEMO_GUIDE.md`，整理课堂/答辩演示脚本、环境准备、无设备 fallback、隐私与安全边界。
  - 扩展 Compose 仪器测试，覆盖底部导航、展示中心、相机演示入口和新手引导入口。
- 验证：
  - `:app:testDebugUnitTest`、`:app:assembleDebug` 和 `:app:assembleDebugAndroidTest` 构建验证通过。
  - 重新连接设备后，`connectedDebugAndroidTest` 已在 `SM-S9280 - 16` 上完成 4 个 Compose 仪器测试并通过。
  - debug APK 已安装到 `SM-S9280`，包信息核对为 `versionCode=17`、`versionName=4.1.0`。
- APK：
  - `releases/apk/BlindAssist-v4.1.0-debug-20260518-231542.apk`

## v3.6.0 - 日常使用体验增强

- 主要变化：
  - 新增语音风格：简短、标准、详细。
  - 新增震动强度：轻柔、标准、强。
  - 新增非迫近近距提醒疲劳控制，减少连续提醒打扰。
  - Overlay 检测框增加显示层平滑，风险规则近距阈值略收紧。
  - 新增并修复最小 Compose 仪器测试宿主。
- 验证：
  - JVM 单元测试、debug APK 构建、`connectedDebugAndroidTest` 和手机安装均通过。
- APK：
  - `releases/apk/BlindAssist-v3.6.0-debug-20260518-214947.apk`

## v3.5.0 - ViewModel 与 StateFlow 轻量状态拆分

- 主要变化：
  - Compose 可观察状态集中到 `BlindAssistViewModel`，通过只读 `StateFlow` 暴露。
  - `MainActivity` 保留 CameraX、权限、TFLite、反馈控制和生命周期边界。
- 验证：
  - JVM 单元测试、debug APK 构建和手机安装均通过。
- APK：
  - `releases/apk/BlindAssist-v3.5.0-debug-20260518-193819.apk`

## v3.4.0 - 现场测试摘要与无障碍语义

- 主要变化：
  - 新增内存态现场测试摘要，展示运行时长、风险次数、提醒次数、FPS、推理耗时和当前提醒档位。
  - 设置页、相机控制区和摘要标题补充更自然的 TalkBack 语义。
- 验证：
  - JVM 单元测试、debug APK 构建和手机安装均通过。
- APK：
  - `releases/apk/BlindAssist-v3.4.0-debug-20260518-192333.apk`

## v3.3.0 - 首次使用引导与相机权限说明

- 主要变化：
  - 新增三页新手引导，说明手机摄像头本地识别、语音/震动提醒和原型安全边界。
  - 相机权限请求前增加应用内解释，说明不上传、不联网、不保存视频。
- 验证：
  - JVM 单元测试和 debug APK 构建通过；该轮手机安装因 ADB 无设备未完成。
- APK：
  - `releases/apk/BlindAssist-v3.3.0-debug-20260518-154943.apk`

## v3.2.0 - 相机返回路径与个人主页精简

- 主要变化：
  - 相机沉浸页支持系统返回手势，统一关闭相机并回到主界面。
  - 个人主页移除展示说明卡，保留用户、设备、版本和偏好状态。
- 验证：
  - JVM 单元测试、debug APK 构建和手机安装均通过。
- APK：
  - `releases/apk/BlindAssist-v3.2.0-debug-20260518-152635.apk`

## v3.1.0 - Compose 应用壳层与界面革新

- 主要变化：
  - 引入 Compose + Material 3 主壳、品牌启动页、底部导航、功能页、个人主页、设置页和沉浸式相机子页。
  - 保留原有 CameraX、TFLite、风险分析、提醒档位、语音和震动链路。
- 验证：
  - JVM 单元测试、debug APK 构建和手机安装均通过。
- APK：
  - `releases/apk/BlindAssist-v3.1.0-debug-20260518-151146.apk`

## v2.6.0 - 显示可信度打磨

- 主要变化：
  - 区分当前帧检测和短暂保持提醒。
  - 默认用户文案隐藏数值 urgency，调试信息保留详细指标。
  - 中心区域改为观察参考区，避免误解为真实检测框。
- 验证：
  - debug APK 构建和手机安装均通过。
- APK：
  - `releases/apk/BlindAssist-v2.6.0-debug-71f921d-current.apk`

## v2.5.0 - 现场可测助行体验

- 主要变化：
  - 新增纯 Kotlin `AssistEngine` 和 `SessionTrace`，把检测结果、风险分析、稳定策略和反馈决策串成可测试会话层。
  - 调试区显示最近会话摘要，近处和迫近语音更偏行动提示。
- 验证：
  - JVM 单元测试、debug APK 构建和手机安装均通过。
- APK：
  - `releases/apk/BlindAssist-v2.5.0-debug-e803d1f-rebuilt.apk`

## v2.0.0 - 提醒档位与 CameraX API 更新

- 主要变化：
  - 新增 Quiet、Standard、Sensitive 三档提醒策略。
  - 风险稳定、语音冷却、震动时长随档位调整。
  - CameraX 分析分辨率迁移到 `ResolutionSelector`。
- 验证：
  - JVM 单元测试、debug APK 构建和手机安装均通过。
- APK：
  - `releases/apk/BlindAssist-v2.0.0-debug-52a0c93-rebuilt.apk`

## v1.5.0 - 用户偏好持久化

- 主要变化：
  - 持久化语音提醒、震动提醒和关怀模式。
  - 检测开关保持 session-only，每次启动默认开启。
- 验证：
  - JVM 单元测试、debug APK 构建和手机安装均通过。
- APK：
  - `releases/apk/BlindAssist-v1.5.0-debug-f6b6d5e-rebuilt.apk`

## v1.4.0 - 相机画面填充与覆盖层映射

- 主要变化：
  - 预览画面改为填满屏幕，减少顶部黑边。
  - Overlay 坐标映射对齐填充裁剪后的预览。
- 验证：
  - debug APK 构建通过。
- APK：
  - `releases/apk/BlindAssist-v1.4.0-debug-f96c6f7-rebuilt.apk`

## v1.3.0 - 相机界面重设计

- 主要变化：
  - 新增品牌/状态头部、风险徽章、两行控制区和关怀模式。
  - 关怀模式放大指导语、提高对比度并隐藏调试细节。
- 验证：
  - debug APK 构建通过。
- APK：
  - `releases/apk/BlindAssist-v1.3.0-debug-e29b99a-rebuilt.apk`

## v0.8.0 - 实时界面交互升级

- 主要变化：
  - 分离主风险状态、控制开关和可折叠调试信息。
  - 改善检测、语音、震动开关的可访问描述。
- 验证：
  - debug APK 构建通过。
- APK：
  - `releases/apk/BlindAssist-v0.8.0-debug-4bf9ad2-rebuilt.apk`

## v0.7.0 - 相对距离风险提醒

- 主要变化：
  - 新增 FAR、MID、NEAR、CRITICAL 相对距离分层和 urgency score。
  - FAR/MID 主要用于视觉状态，NEAR/CRITICAL 才进入语音和震动提醒路径。
- 验证：
  - JVM 单元测试、debug APK 构建和手机安装均通过。
- APK：
  - `releases/apk/BlindAssist-v0.7.0-debug-d948f6b-rebuilt.apk`

## v0.2.0 - 风险提醒稳定化

- 主要变化：
  - 新增 `RiskStabilizer`，高风险单帧确认，中风险需要连续帧确认。
  - 短暂漏检时保持已确认提醒，减少语音/震动抖动。
- 验证：
  - JVM 单元测试和 debug APK 构建通过。
- APK：
  - `releases/apk/BlindAssist-v0.2.0-debug-fb937da-rebuilt.apk`

## v0.1.0 - 第一版本地检测原型

- 主要变化：
  - CameraX 获取实时摄像头画面。
  - 本地加载 YOLO11n TFLite 模型，解析检测框并绘制 overlay。
  - 初版规则层生成助盲避障提醒。
- 验证：
  - debug APK 构建通过。
- APK：
  - `releases/apk/BlindAssist-v0.1.0-debug-958d5a9-rebuilt.apk`
