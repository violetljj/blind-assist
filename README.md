# BlindAssist Android 原型

BlindAssist 是 Android Kotlin + Jetpack Compose 助盲避障原型：Compose/Material 3 提供启动页、功能入口、底部导航和设置体验，CameraX 实时取流，TFLite 本地运行 YOLO11n，规则层判断危险区域，并通过语音和震动提醒。

## 版本

- 当前项目版本：`v8.2.0`
- 版本规则：小更新增加 `v0.1`，较大更新增加 `v0.5`，阶段性质变增加 `v1.0`。
- 版本影响由 Codex/Agent 根据每次变更的范围和风险判断。
- 会影响项目状态、使用方式、功能行为、构建流程、模型资产、测试结论或重要技术决策的更新，应同步保持 README 与当前状态一致。
- 普通措辞、错别字、格式整理或轻量协作规则说明不计为版本更新。

## 近期状态

- 2026-05-27：完成 `yolo26n` 专项真机验证，但不替换默认模型。新增 COCO val2017 固定抽样脚本，已在 `.downloads/detector-lab/datasets/coco100/` 准备 100 张图片和 manifest；新增 yolo26n instrumentation benchmark，候选模型只进入 androidTest APK 资产，正式 debug APK 仍只包含 `assets/yolo11n_fp16_320.tflite` 与 `assets/coco_labels.txt`。在 Samsung `SM-S9280` / Android 16 上，`yolo26n_fp16_320.tflite` 纯 TFLite CPU 4 线程 invoke P50/P95 为 `36.996/42.060ms`，BlindAssist 应用链路 inference P50/P95 为 `37/39ms`，total P50/P95 为 `49/51ms`，100 张图片无失败；证据目录为 `test-artifacts.local-yolo26n-device-benchmark-20260527-015039`。随后默认模型路径执行 `scripts/run_device_regression.ps1 -SampleSeconds 90` 通过，证据目录为 `test-artifacts.local-device-regression-20260527-015153`。本轮是实验验证工具与测试证据补充，不改变用户可见行为，不调整 `versionName=8.2.0` / `versionCode=30`。

- 2026-05-27：完成实时检测器横向评测框架。新增项目内 detector lab，本地下载 COCO8 smoke dataset 和 `yolo26n`、`yolo12n`、`yolov10n` 候选权重到 `.downloads/detector-lab/`，导出 320 FP16 TFLite 并用 `ai-edge-litert` 完成多模型 shape/dtype 检查和 CPU smoke benchmark。当前默认 App 模型仍为 `YOLO11n FP16 320 TFLite`，不改变 `ObjectDetector` 运行路径、风险规则、用户界面或 APK 行为；本轮不调整版本号。详细流程见 `docs/DETECTOR_BENCHMARK.md`，本地证据目录为 `test-artifacts.local-detector-benchmark-20260527-010222`。

- 2026-05-26：完成 `v8.2.0` 相机可靠性与无障碍修复。相机关闭后会清空旧 `PreviewView` 就绪状态，重新打开必须等待新的预览 View，避免绑定已移除预览导致黑屏；TFLite 检测与关闭使用同一生命周期锁，并在 analyzer executor 关闭时做有界等待，降低销毁竞态风险。相机页普通动作不再被 TalkBack 读成“已启用/未启用”，debug 展开态和核心开关保留本地化 state description；底部面板在大字体和 debug 展开时限制高度并支持滚动。`scripts/inspect_tflite.py` 已升级为模型 shape/dtype 断言脚本，CI 增加模型检查与 `:app:assembleDebugAndroidTest`。当前版本为 `v8.2.0` / `versionCode=30`；模型检查、多模块单测、多模块 lint、debug/androidTest APK 构建、7 个 Compose `connectedDebugAndroidTest` 真机测试和 `scripts/run_device_regression.ps1 -SampleSeconds 90` 均通过，设备证据目录为 `test-artifacts.local-device-regression-20260526-231417`。本轮属于 `+0.1` 小版本，APK 已归档到完整本地目录 `E:\linnan\blind-assist-apk-archive\apks\BlindAssist-v8.2.0-debug-20260526-215736.apk`，不提交 Git 里程碑 APK。

- 2026-05-25：完成 `v8.1.0` 真实使用体验 UI 升级。功能页调整为任务启动台，优先展示当前行走任务、主摄像头入口和日常模式选择；相机页重排底部面板，把风险状态、行动建议和检测/语音/震动核心开关放在前面，并让 Care Mode 成为更大字号、更低干扰的相机体验；设置页按界面与辅助、提醒方式、行走场景、调试与记录分组。当前版本为 `v8.1.0` / `versionCode=29`，已通过 UI 相关单测、lint、debug/androidTest APK 构建、6 个 Compose 真机测试和 `scripts/run_device_regression.ps1 -SampleSeconds 90` 真机回归；设备证据目录为 `test-artifacts.local-device-regression-20260525-222502`，Git 里程碑 APK 为 `releases/apk/BlindAssist-v8.1.0-debug-20260525-222724.apk`。
- 2026-05-25：已在 Samsung `SM-S9280` / Android 16 上完成当前 `v7.6.0` 真机验证。旧 `v5.9.0` 安装包因 debug 签名不同，已在确认后卸载再安装当前 APK。模型检查、`:app:testDebugUnitTest :app:assembleDebug`、6 个 Compose `connectedDebugAndroidTest` 用例和 `scripts/run_device_regression.ps1 -SampleSeconds 90` 均通过。设备端包信息为 `versionName=7.6.0` / `versionCode=28`，本地证据目录为 `test-artifacts.local-device-regression-20260525-012352`。
- 2026-05-25：完成 `v7.6.0` 运行时管线和发布卫生更新。`AssistRuntimeController` 已收敛为更薄的入口，运行时执行、相机生命周期、帧处理、渲染、配置同步和性能统计拆到更小的协作者中。CameraX 实时路径改用可关闭的 `VisionFrame` / `RgbaVisionFrame`，减少逐帧 `Bitmap` 分配。当前版本为 `v7.6.0` / `versionCode=28`，Git 里程碑 APK 为 `releases/apk/BlindAssist-v7.6.0-debug-20260525-004833.apk`。
- 2026-05-24：完成 `v7.1.0` 可靠性和工程质量更新，补充模型/风险回放、运行时故障注入、反馈不可用状态、CameraX 失败清理和多模块测试/lint 验证。
- 2026-05-22 至 2026-05-24：项目从单模块演进为 `:app`、`:feature:assist` 和 `:core:*` 多模块结构，并完成 Hilt 运行时拆分、本机工具链修复、E 盘工作区迁移、APK 归档规则和仓库卫生脚本。

更完整的版本路线见 `CHANGELOG.md`，逐次执行证据见 `DEVELOPMENT_LOG.md`。

## 项目材料

- [新电脑交接说明](docs/NEW_COMPUTER_HANDOFF.md)：Windows 环境准备、Git 克隆、Android 构建验证、手机安装和 Codex skills 恢复说明。
- [真机回归说明](docs/DEVICE_REGRESSION.md)：安装、冷启动、包状态、UI dump、截图、`gfxinfo`、`meminfo` 和可选 connected Compose 测试的真机证据采集流程。
- [APK 归档策略](docs/APK_ARCHIVE.md)：Git 只保留累计 `versionName` 差值 `>= 0.5` 的里程碑 APK，完整本地归档位于 `E:\linnan\blind-assist-apk-archive\apks`，SHA256 证据写入 `APK_ARCHIVE_MANIFEST.csv`。
- [实时检测器横向评测说明](docs/DETECTOR_BENCHMARK.md)：说明候选检测器下载、导出、多模型检查、COCO8 smoke benchmark 和真实助行图片集边界。
- [Codex skills 快照清单](codex/skills-snapshot/MANIFEST.md)：`codex/skills-snapshot/codex-skills-20260522.zip` 的 SHA256、大小、条目数和恢复提示。
- [真实版本更新记录](CHANGELOG.md)：按真实版本整理功能变化、验证证据和 APK 归档路径，方便课堂展示、答辩材料和版本对比。
- [演示指南](DEMO_GUIDE.md)：面向老师/答辩的演示脚本，包含环境准备、手机安装、现场演示顺序、无设备 fallback、隐私与安全边界说明。
- [回顾式阶段进度说明](PROJECT_PROGRESS_REVIEW.md)：面向课程汇报、阶段检查和毕设展示的整理稿，按 2026 年 5 月 1 日前的“调研、方案、原型、测试、迭代”脉络说明项目工作量。
- [v5.8.0 真机完整测试与 v5.9.0 修复复测报告](TEST_REPORT_2026-05-19.md)：记录 2026-05-19 在 `SM-S9280` 上执行的清数据真机功能、UI、性能、稳定性和 instrumentation 测试结果，以及 v5.9.0 对遗留问题的修复复测。

## 架构

BlindAssist 当前是 Gradle 多模块 Android 项目。`:app` 保持较薄，只负责启动壳层、Manifest、资源、模型资产和 APK 配置；主要业务由 `:feature:assist` 协调，底层能力拆分到多个 `:core:*` 模块。

- `:core:assist`：纯 Kotlin 助盲领域模型、风险分析、提醒策略、会话统计、本地化和偏好映射。
- `:core:vision`：TFLite YOLO 检测器、图像预处理、YOLO 输出解析和视觉帧处理。
- `:core:device`：CameraX 帧源、Android 语音/震动反馈、SharedPreferences 用户偏好和设备侧适配。
- `:core:ui`：Compose/UI 状态模型、检测框覆盖层、相机引导和现场测试摘要映射。
- `:feature:assist`：Hilt ViewModel、运行时状态机、CameraX/TFLite 协调、配置同步、渲染和性能日志边界。

## 界面行为

应用启动后先进入 Compose 应用壳层，不会立即打开相机：

- 可见界面状态由 Hilt 注入的 `BlindAssistViewModel` 和只读 `StateFlow` 驱动。`MainActivity` 保持为启动、权限请求和 Compose 绑定入口，`:feature:assist` 负责协调 CameraX、检测器、反馈、运行时状态和 overlay 更新。
- 冷启动使用 Android SplashScreen API，随后展示短暂的 BlindAssist 品牌页；点击品牌页可跳过。
- 首次使用会展示三页引导，说明本地手机摄像头识别、语音/震动提醒和原型安全边界；完成或跳过后会在本地保存引导状态。
- 主界面使用 Material 3 底部导航，包含“功能”“个人主页”“设置”三个顶层入口。
- “功能”页是日常辅助启动台，先展示当前行走任务、场景、提醒档位和 Care Mode 状态，再提供高优先级 `使用手机摄像头` 主入口。日常使用向导仍提供 通用日常、室内慢行、走廊通行、密集区域、户外慢行 五个一键预设。`连接眼镜设备` 保留为次级未来设备占位入口；该占位不扫描蓝牙、不申请蓝牙权限、不联网，也不表示硬件连接已经完成。
- “个人主页”展示本地原型状态、当前提醒档位、版本信息和辅助偏好，不包含登录、云同步、账号数据或展示型说明卡片。
- “设置”页按“界面与辅助、提醒方式、行走场景、调试与记录”分组，控制界面语言、Care Mode、语音提醒、震动提醒、语音风格、震动强度、提醒档位、手动使用场景和调试详情，并提供 `查看新手引导` 入口。用户手动调整后若不再匹配一键预设，主界面会显示为 自定义 / Custom。
- 点击 `使用手机摄像头` 后，只有在相机权限可用时才进入沉浸式相机页。如果权限未授予，应用会先说明相机帧只在本地实时处理、不上传、不保存视频，然后再引导用户打开 Android 系统权限弹窗。
- 相机权限被拒绝时，应用留在主界面，并说明手机摄像头辅助路径无法在无权限状态下启动。
- 相机子页面隐藏底部导航，显示全屏 `PreviewView`、检测框 overlay、顶部返回按钮和紧凑底部控制面板。点击返回或使用系统返回手势会回到主界面、解绑 CameraX 并清空 overlay。

实时相机页以全屏预览为主，底部面板承载交互：

- 面板层级按真实助行优先级组织：主要风险状态、一句行动建议、当前场景/模式、检测/语音/震动核心开关，再到快捷调节、场景切换和调试信息。
- 风险状态变化使用克制的短过渡，保持响应感但不干扰预览画面。
- 竖屏使用时相机预览填满屏幕；检测框坐标与同一套填充裁剪逻辑对齐。
- 主风险区展示当前风险等级、相对距离、方向、当前帧目标数和已锁定的主要提醒来源。
- 如果稳定后的提醒在下一帧短暂丢失目标后继续保持，目标行会明确说明提醒来自前一帧，避免把旧目标名称和当前 0 个目标误配。
- 检测、语音、震动、手动场景和 Care Mode 使用紧凑高对比按钮，并可独立切换。语音、震动、提醒档位、使用场景、语音风格、震动强度、Care Mode 及其组合会在下次启动恢复；检测开关每次启动默认开启。
- 底部面板显示当前日常模式，并保留两个直接提醒强度快捷操作：调安静会应用 Quiet + Brief speech + Soft vibration，调敏感会应用 Sensitive + Standard speech + Strong vibration。两者都会保留当前使用场景并写入现有偏好存储。
- 提醒档位可在 Quiet、Standard、Sensitive 之间切换。Quiet 降低提醒频率和震动时长，Standard 保持平衡策略，Sensitive 更快确认中风险并缩短提醒冷却。
- 使用场景可在 通用、室内慢行、走廊通行、密集区域、户外慢行 之间切换。通用保持基础策略，其他场景只调整规则层确认、保持、冷却和震动参数，不声称自动识别场景。
- 语音风格可选 简短、标准、详细；震动强度可选 轻柔、标准、强。它们只调整反馈措辞和触感强度，不改变检测或风险分析。
- Care Mode 会放大主要指导语、提高面板对比度、隐藏快捷调节和调试细节，只保留检测、语音、震动和关怀模式等必要控制，并在 overlay 中增加中心参考线，以支持低视力或高压力使用场景。
- 调试信息默认折叠。展开后显示 FPS、总耗时、预处理/推理/后处理耗时、模型状态、最新原始风险、稳定风险、紧急度分数、当前提醒档位、当前场景、反馈原因、风险解释，以及设置页使用的同一份现场测试摘要。
- 检测框会突出当前风险来源，弱化其他检测目标；中心区域绘制为观察参考区，不当作已检测目标框。

## 风险提醒行为

BlindAssist 是辅助原型，不是可以替代人工判断的安全设备。检测结果会先经过平滑处理，再进入语音和震动反馈：

- 应用不估算真实米制距离，只根据检测框位置和大小推导相对距离分层。
- FAR 检测会保留在视觉显示中，但不触发语音或震动。
- MID 检测显示为低风险视觉/状态反馈。
- NEAR 检测在风险等级为中或高时可以触发常规语音和震动。语音提示由所选语音风格生成：简短减少字数，标准保持平衡提示，详细会在可用时补充目标类别。
- CRITICAL 检测使用更短冷却和更强震动；正前方 critical 提示使用短句“前方很近，放慢”。
- HIGH 风险无需等待帧确认即可提醒。
- MEDIUM 风险需要连续两帧方向/文案匹配后才确认。
- 已确认的中/高风险提醒在下一帧短暂丢失目标时最多保持 600ms，减少漏检导致的闪烁。
- Standard 档位下，常规 near 语音和震动使用 1500ms 冷却；短时间重复的非 critical near 提醒会获得更长的有效冷却，以降低提醒疲劳。critical 高风险提醒仍走紧急冷却路径，不受疲劳控制压制。
- Quiet 档位下，中风险需要三帧确认，提醒保持 450ms，near 冷却 2200ms、震动 100ms，critical 冷却 1200ms、震动 260ms。
- Standard 档位下，中风险需要两帧确认，提醒保持 600ms，near 冷却 1500ms、震动 160ms，critical 冷却 850ms、震动 420ms。
- Sensitive 档位下，中风险首帧确认，提醒保持 800ms，near 冷却 1000ms、震动 220ms，critical 冷却 650ms、震动 520ms。
- 手动使用场景只调整提醒策略，不改变检测器，也不声称自动理解场景：Indoor Slow 略微增加保持时间和 near 冷却，Corridor 更快确认中风险并略微增强震动，Crowded 增加中风险确认要求并延长冷却，Outdoor Slow 延长保持时间并强化震动清晰度。
- 相机面板会用普通语言解释最新反馈决策：已触发、不稳定、距离较远、冷却中、提醒保持、反馈关闭或暂无可反馈风险。该解释用于透明化和调试，不代表安全认证。

## 环境

当前仓库是 Android Studio/Gradle 项目。构建前需要安装：

- JDK 17
- Android Studio 或 Android SDK + Platform Tools
- Android SDK Platform 35

验证命令：

```powershell
java -version
adb version
```

### 当前本机工具链

2026-05-22 后续修复后，当前 Windows 机器已具备仓库本地验证工具链：

```powershell
$env:JAVA_HOME=(Resolve-Path '.\.jdk\jdk17.0.19_10').Path
$env:PATH="$env:JAVA_HOME\bin;D:\Git\cmd;$((Resolve-Path '.\.android-sdk\platform-tools').Path);$env:PATH"
$env:GRADLE_USER_HOME=(Resolve-Path '.\.gradle-local').Path
.\gradlew.bat :core:assist:test :core:vision:testDebugUnitTest :core:device:testDebugUnitTest :core:ui:testDebugUnitTest :feature:assist:testDebugUnitTest :app:testDebugUnitTest --no-daemon
.\gradlew.bat :app:lintDebug :core:vision:lintDebug :core:device:lintDebug :core:ui:lintDebug :feature:assist:lintDebug --no-daemon
.\gradlew.bat :app:assembleDebug :app:assembleDebugAndroidTest --no-daemon
```

上述命令已验证可得到 `BUILD SUCCESSFUL`；模型检查使用 `.venv-export312\Scripts\python.exe scripts\inspect_tflite.py`。

## 模型资产

第一版默认从 assets 加载真实 YOLO11n TFLite 模型：

```text
app/src/main/assets/yolo11n_fp16_320.tflite
app/src/main/assets/coco_labels.txt
```

当前仓库已包含 Android 运行和 CI 模型检查所需的受控模型资产。若需要重新生成或校验资产，推荐用本仓库脚本导出，导出参数固定为 `imgsz=320`、`half=True`、`nms=False`，这样 Android 端可以解析 raw YOLO 输出并自行执行 NMS。已验证的本机导出路径是 Python 3.12 + TensorFlow 2.19：

```powershell
.\.venv-export\Scripts\python.exe -m pip install uv
.\.venv-export\Scripts\uv.exe python install 3.12
.\.venv-export\Scripts\uv.exe venv .venv-export312 --python 3.12
.\.venv-export\Scripts\uv.exe pip install --python .\.venv-export312\Scripts\python.exe -r requirements-export.txt
.\.venv-export312\Scripts\python.exe scripts\export_yolo11n_tflite.py
.\.venv-export312\Scripts\python.exe scripts\inspect_tflite.py
```

期望输出：

```text
assertions=passed
input shape=[1, 320, 320, 3] dtype=float32
output shape=[1, 84, 2100] dtype=float32
```

## 构建

在 Android Studio 打开项目并同步依赖；或使用仓库本地 JDK 17 和 `.gradle-local` 运行：

```powershell
$env:JAVA_HOME=(Resolve-Path '.\.jdk\jdk17.0.19_10').Path
$env:GRADLE_USER_HOME=(Resolve-Path '.\.gradle-local').Path
.\gradlew.bat :app:testDebugUnitTest :app:lintDebug :app:assembleDebug --no-daemon
```

当前验证矩阵显式覆盖各模块，避免遗漏 library lint：

```powershell
.\.venv-export312\Scripts\python.exe scripts\inspect_tflite.py
.\gradlew.bat :core:assist:test :core:vision:testDebugUnitTest :core:device:testDebugUnitTest :core:ui:testDebugUnitTest :feature:assist:testDebugUnitTest :app:testDebugUnitTest --no-daemon --console=plain
.\gradlew.bat :app:lintDebug :core:vision:lintDebug :core:device:lintDebug :core:ui:lintDebug :feature:assist:lintDebug --no-daemon --console=plain
.\gradlew.bat :app:assembleDebug :app:assembleDebugAndroidTest --no-daemon --console=plain
```

APK 输出位置：

```text
app/build/outputs/apk/debug/app-debug.apk
```

## 版本 APK 归档

用于演示和版本对比的 Git 里程碑 APK 位于：

```text
releases/apk/
```

Git 只保留累计 `versionName` 差值达到 `>= 0.5` 的 APK，或用户明确指定为 Git 里程碑的 APK。更小更新构建出的 APK 应复制到完整本地归档目录，不默认提交。

完整本地归档目录为：

```text
E:\linnan\blind-assist-apk-archive\apks
```

SHA256 清单为：

```text
E:\linnan\blind-assist-apk-archive\APK_ARCHIVE_MANIFEST.csv
```

当前 Git 里程碑列表和校验命令见 [APK 归档策略](docs/APK_ARCHIVE.md)。

CI 会运行 `scripts/check_repo_hygiene.ps1`，阻止新增本地缓存、普通测试产物和未列入白名单的大型二进制文件进入 Git。

## 安装到手机

打开 Android 手机 USB 调试后：

```powershell
.\.android-sdk\platform-tools\adb.exe devices
.\.android-sdk\platform-tools\adb.exe install -r app\build\outputs\apk\debug\app-debug.apk
```
