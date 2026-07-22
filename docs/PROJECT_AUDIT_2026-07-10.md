# BlindAssist 项目综合评估报告（2026-07-10）

> 本报告是基于 2026-07-10 当前工作树进行的代码、架构、测试、工程化、产品完成度与无障碍只读审查。它是阶段性工程判断，不是医疗器械、安全产品认证或真实道路可用性证明。

> 修复状态（v9.4.0）：本报告列出的前三项优先问题已实施修复，包括非安全承诺措辞与证据状态、token 化 session 提交边界、卫生 smoke/Gradle 资产依赖以及功能测试与 `:device-benchmark` 隔离。本地 JVM、Lint、组合 APK 构建、模型与 APK 元数据检查已通过；Samsung `SM-S9280` / Android 16 上 Compose 11/11、Detector A/B、Depth-fusion 和强化后的 90 秒相机回归也已完成。候选 YOLO26n 与 MiDaS Depth-fusion 均未通过助盲风险晋级门槛。下文保留原评估时基线和问题描述，便于追溯。

## 一、结论摘要

BlindAssist 已经是一个真实可运行、工程完成度较高的 Android 助盲原型，不是只包含界面或演示素材的空壳。CameraX 实时取流、TFLite YOLO11n 推理、风险规则、连续帧稳定、界面反馈、TTS、震动、偏好保存、测试和 APK 构建链路均有真实实现。

综合判断：

- 作为研究生毕设、课程设计或阶段性技术原型：约 `8/10`。
- 作为可持续迭代的移动端工程项目：约 `7/10`。
- 作为可以让视障用户依赖的助盲安全产品：当前约 `3/10`。

项目当前最大的缺口已不是“有没有功能”，而是以下三个方面：

1. 系统需要严格区分“未检测到可提醒目标”和“确认安全”，避免错误安全暗示。
2. 相机帧、风险结果、反馈和 UI 的 session 生命周期需要进一步收拢，防止关闭或快速重开时出现跨代状态污染。
3. 需要真实连续场景、真实障碍物和目标用户参与的安全验证，不能只依赖静态 COCO 图像和自动化采集结果。

## 二、原评估范围与当时基线

- 实际 Git 仓库：`E:\linnan\linnan`。
- 当前分支：`codex/stabilize-project-structure`，相对远端超前 1 个提交。
- 当前应用版本：`versionName=8.9.0`，`versionCode=33`。
- Kotlin 源文件：116 个。
- JVM 测试文件：31 个。
- AndroidTest/benchmark Kotlin 文件：3 个。
- 当前工作树原本已有 `AGENTS.md`、`DEVELOPMENT_LOG.md`、`idea.md` 修改，以及 `.android-home/`、`scripts/__pycache__/`、`work/` 未跟踪目录；本次评估没有回滚或清理这些内容。
- 当前没有连接 Android 真机，因此本次没有执行设备回归和 `connectedDebugAndroidTest`。

评估覆盖：

- 多模块依赖与领域划分。
- CameraX、TFLite、风险分析、反馈和 UI 主运行链路。
- session 生命周期、并发、异常恢复和性能热点。
- JVM 测试、仪器测试、benchmark、Lint、Gradle 和 CI。
- 仓库卫生、模型资产、供应链与本地生成物。
- 产品真实能力、占位入口、隐私权限、无障碍与文档一致性。

## 三、原评估时实际验证结果

| 验证项目 | 本次结果 | 说明 |
| --- | --- | --- |
| TFLite 模型检查 | 通过 | 输入 `[1, 320, 320, 3] float32`，输出 `[1, 84, 2100] float32` |
| 核心模块 JVM 测试 | 通过 | 使用 `--rerun-tasks` 强制重跑 189 项，0 failure、0 error、0 skipped |
| Android Lint | 通过 | `app`、`core:vision`、`core:device`、`core:ui`、`feature:assist` 单独矩阵通过 |
| Debug APK 构建 | 通过 | `app-debug.apk`，47,288,840 bytes |
| AndroidTest APK 构建 | 通过 | `app-debug-androidTest.apk`，76,238,831 bytes |
| 完整合并 Gradle 矩阵 | 失败 | benchmark 资产任务与 `generateDebugAndroidTestLintModel` 存在未声明的隐式依赖 |
| 仓库卫生脚本 | 表面通过但结果不可信 | 脚本输出通过，但扩展名与缓存路径正则实际不能匹配目标路径 |
| ADB 设备检查 | 无设备 | `adb devices` 列表为空，未进行真机验证 |

强制重跑的 JVM 测试分布：

| Module | 测试数 | 失败 | 错误 | 跳过 |
| --- | ---: | ---: | ---: | ---: |
| `core:assist` | 84 | 0 | 0 | 0 |
| `core:vision` | 16 | 0 | 0 | 0 |
| `core:device` | 32 | 0 | 0 | 0 |
| `core:ui` | 11 | 0 | 0 | 0 |
| `feature:assist` | 46 | 0 | 0 | 0 |
| 合计 | 189 | 0 | 0 | 0 |

## 四、主要优势

### 4.1 项目不是空壳，核心链路真实存在

- `FrameSourceFactory` 创建真实 `CameraXFrameSource`，通过 CameraX 获取手机后置相机帧。
- 生产运行时注入真实 `TfliteYoloDetector`，默认模型为 `yolo11n_fp16_320.tflite`。
- 风险层根据检测框位置、面积、底部位置、方向和连续帧趋势生成风险结果。
- TTS、震动、冷却、疲劳控制、稳定器、偏好保存和双语核心界面均有实际代码。
- 深度模型和其他候选模型被隔离在 benchmark/AndroidTest 资产中，没有直接污染默认生产 APK。

### 4.2 多模块方向总体正确

- `core:assist` 为纯 Kotlin 领域 module，不依赖 Android framework，测试速度和可导航性较好。
- `core:vision` 集中 TFLite 检测、图像预处理和输出解析。
- `core:device` 集中 CameraX、TTS、震动和 SharedPreferences 等 Android adapter。
- `feature:assist` 负责运行时协调、ViewModel、配置同步和渲染。
- 当前模块依赖无环，领域 module 没有反向依赖 UI 或设备实现。

### 4.3 测试基础明显强于普通课程项目

- 风险分析、连续帧追踪、稳定器、反馈、偏好、CameraX 安全、预处理、坐标映射、运行时状态机和故障注入均有专项测试。
- 本次 189 项 JVM 测试全部重新执行并通过，而不是只读取历史报告。
- 项目对 benchmark 未通过、真机测试 signal 9、未执行验证等情况总体记录诚实，没有简单粉饰为成功。

### 4.4 安全与隐私定位总体克制

- Manifest 生产权限主要是相机和震动，没有生产 INTERNET、定位、存储或蓝牙权限。
- 文档和界面多处说明原型不能替代盲杖、导盲犬或人工判断。
- “连接眼镜设备”入口明确属于占位能力，当前不会扫描蓝牙或连接真实 ESP32。

## 五、优先问题与风险

### P0-1：把“没有识别到”表达成“安全”

证据位置：

- `core/assist/src/main/java/com/linnan/blindassist/risk/RiskAnalyzer.kt:58-83`
- `core/ui/src/main/java/com/linnan/blindassist/ui/CameraGuidanceMapper.kt:202-210`

`RiskAnalyzer` 只允许 person、bicycle、car、motorcycle、bus、truck、traffic light、stop sign、bench、chair、potted plant 这 11 类进入风险候选。候选为空时返回 `RiskLevel.NONE` 和“未发现风险”，UI 又将 `RiskLevel.NONE` 映射为“安全观察中 / Safe observing”。

这会把以下状态折叠成同一种“安全”表达：

- 模型没有检测到目标。
- 检测到了不在白名单中的目标。
- 真实障碍物不属于 COCO 类别。
- 目标被遮挡、曝光异常或模型漏检。
- 当前画面确实没有风险目标。

建议优先改为“暂未识别到可提醒目标，请继续使用盲杖或人工判断”等非安全承诺措辞，并在领域结果中保留“无证据”“模型不可用”“已识别风险”等不同状态。

### P0-2：旧帧可能越过关闭动作进入新 session

证据位置：

- `feature/assist/src/main/java/com/linnan/blindassist/runtime/AssistRuntimeLifecycleGate.kt:21-67`
- `feature/assist/src/main/java/com/linnan/blindassist/runtime/AssistRuntimeEffectExecutor.kt:34-47`
- `feature/assist/src/main/java/com/linnan/blindassist/runtime/AssistFrameProcessor.kt:51-84`
- `core/device/src/main/java/com/linnan/blindassist/camera/CameraXFrameSource.kt:66-75`

`stopSession()` 和 `shutdown()` 最多等待 1 秒并返回 Boolean，但 effect executor 丢弃该返回值，随后无条件 reset frame processor 和 session coordinator。如果在途检测、反馈或 UI 投递未在超时内结束，旧帧仍可能在关闭后产生反馈，或在快速重开后污染新 session。

建议把“停止接收新帧 → drain 在途帧 → 校验 generation → reset → 允许新 session”收拢到一个运行时 session module，并让 drain 失败成为显式状态，不能静默忽略。

### P1-1：session 摘要存在跨线程重复读取

证据位置：

- `feature/assist/src/main/java/com/linnan/blindassist/runtime/AssistFrameProcessor.kt:61-68`
- `feature/assist/src/main/java/com/linnan/blindassist/runtime/AssistRuntimeRenderer.kt:79-95`
- `core/assist/src/main/java/com/linnan/blindassist/session/AssistEngine.kt:49-63`

`AssistFrameResult` 已经携带同一帧对应的不可变 `sessionSummary`，但 UI 渲染时又通过 provider 重新读取 coordinator 内部的可变 trace。下一帧可能已经在分析线程更新 trace，导致 UI 看到与当前帧不完全对应的摘要。

建议直接使用 `frameResult.sessionSummary`，其他跨线程读取统一通过不可变 snapshot。

### P1-2：反馈冷却与疲劳状态跨 session 保留

证据位置：

- `core/assist/src/main/java/com/linnan/blindassist/session/AssistSessionCoordinator.kt`
- `core/device/src/main/java/com/linnan/blindassist/feedback/FeedbackController.kt`
- `core/device/src/main/java/com/linnan/blindassist/feedback/FeedbackFatigueController.kt`

领域 session reset 时不会清理 feedback adapter 的 `lastAlertAt` 和疲劳窗口。用户关闭检测后快速重开时，新 session 的第一条提醒可能被旧 session 的 cooldown 抑制。

建议明确反馈疲劳是 session 级还是 Activity 级策略，并让命名、生命周期和测试与该决定一致。

### P1-3：仓库卫生脚本存在假通过

证据位置：

- `scripts/check_repo_hygiene.ps1:78-110`
- `.github/workflows/android.yml:19-22`

脚本已经把路径统一为 `/`，但多个正则仍使用 `\\.`。在 PowerShell/.NET 正则中，这会寻找反斜杠，而不是匹配普通文件扩展名中的点。因此 APK、ZIP、keystore 和部分本地缓存规则不能命中。

本次脚本输出 `Repository hygiene check passed for 200 changed path(s).`，但仓库实际存在受跟踪 APK 和大型二进制文件，所以该结果不能作为可靠门禁证据。

### P1-4：完整 Gradle 验证矩阵不可组合执行

证据位置：

- `app/build.gradle.kts:20-57`
- `app/build.gradle.kts:130-140`

`prepareYolo26nBenchmarkAssets` 和 `prepareDepthBenchmarkAssets` 的输出被 AndroidTest Lint model 使用，但当前只让 `mergeDebugAndroidTestAssets` 依赖这两个任务。把测试、Lint、Debug APK 和 AndroidTest APK 放在同一 Gradle 调用中时，Gradle 8.10.2 会报告隐式任务依赖并停止构建。

拆开运行时测试、Lint 和 APK 构建均可通过，但仓库文档声明的完整矩阵不能一次可靠执行。应补充正确的 task input/dependency 关系，或将 benchmark 资产彻底移入独立 benchmark module。

### P1-5：仪器测试、功能测试和 benchmark 没有完成隔离

证据位置：

- `.github/workflows/android.yml:34-52`
- `app/src/androidTest/java/com/linnan/blindassist/ui/compose/BlindAssistComposeTest.kt`
- `app/src/androidTest/java/com/linnan/blindassist/benchmark/DetectorAbDeviceBenchmarkTest.kt`
- `app/src/androidTest/java/com/linnan/blindassist/benchmark/Yolo26nDeviceBenchmarkTest.kt`

CI 只运行 JVM 测试和构建 AndroidTest APK，没有真正执行 Compose instrumentation。Compose 功能测试与耗时 detector benchmark 又位于同一个 AndroidTest APK 中，全量 `connectedDebugAndroidTest` 容易被 benchmark 时长、内存或 signal 9 影响。

建议：

- PR/CI 设备门禁只执行快速、确定性的 Compose 功能测试。
- detector/depth benchmark 迁移到独立 module、source set 或独立 runner。
- benchmark 放到 nightly、手动或固定设备工作流。

### P1-6：现有数据不能支持真实助行安全结论

证据位置：

- `docs/BLINDASSIST_EVALSET.md`
- `TEST_REPORT_2026-05-19.md`
- `scripts/run_device_regression.ps1`

当前专用评测集主要由 150 张 COCO 静态图组成，风险字段来自检测框几何自动标注，仍需要隔离的多模型逐张复核；连续帧逼近证据不足。设备回归脚本主要负责安装、启动和采集日志/截图，只要 ADB 命令成功就可能写入 `status=passed`，没有充分断言相机预览、有效帧、模型 ready、无 crash/ANR 和真实提醒行为。

产品化前至少需要：

- 连续视频与真实行走轨迹。
- 低矮障碍、台阶、坑洞、透明物体、非 COCO 障碍。
- 光照变化、遮挡、运动模糊和相机角度变化。
- 关键漏报门槛和风险级别混淆矩阵。
- 受控安全环境下的目标用户试用。

### P1-7：文档与真实 UI 已经漂移

证据位置：

- `README.md:78-84`
- `DEMO_GUIDE.md:5-47`
- `core/ui/src/main/java/com/linnan/blindassist/ui/compose/CameraExperienceScreen.kt:303-315`
- `app/src/androidTest/java/com/linnan/blindassist/ui/compose/BlindAssistComposeTest.kt:193`

README 和演示指南描述相机页存在“调安静、调敏感、场景切换”等快捷入口，但对应 `AnimatedVisibility` 被固定为 `visible=false`，Compose 测试也明确断言相关节点不存在。演示指南仍要求检查 v7.6.0/versionCode 28，而当前工程为 v8.9.0/versionCode 33。

这不会导致 App 崩溃，但会直接影响现场演示和文档可信度。

### P1/P2：无障碍仍有局部缺口

主要问题：

- 高风险徽标白字与 `#FF8469`、`#FF6377` 背景的对比度约为 2.40:1 和 2.88:1，低于普通文本 4.5:1 标准。
- 设置项整行是可点击 `Role.Switch`，内部 `Switch` 又能单独点击，可能产生重复 TalkBack 焦点和动作。
- 风险标题没有 `liveRegion` 等动态播报语义，关闭内置 TTS 时 TalkBack 不一定及时播报状态变化。
- 相机权限永久拒绝后，没有“打开应用设置”的恢复路径。
- onboarding 存在中文硬编码，页码进度没有完整语义和焦点管理。
- 震动仅表达强弱和时长，不能在没有语音时独立表达左右方向。

证据位置：

- `core/ui/src/main/java/com/linnan/blindassist/ui/compose/SettingsRows.kt:50-83`
- `core/ui/src/main/java/com/linnan/blindassist/ui/compose/CameraExperienceScreen.kt:251-265`
- `core/ui/src/main/java/com/linnan/blindassist/ui/CameraGuidanceMapper.kt:376-386`
- `app/src/main/java/com/linnan/blindassist/MainActivity.kt:79`
- `core/ui/src/main/java/com/linnan/blindassist/ui/compose/OnboardingScreen.kt`

## 六、架构深化候选

### 候选 1：运行时 session module

把相机启停、frame lease、generation、drain、reset、结果提交和反馈提交集中到一个更 deep 的 module。调用者只提交运行时事件，不再分别操纵 9 个协作者。

收益：生命周期知识集中、旧帧不能越界、真实资源顺序可从同一个 interface 测试。

### 候选 2：保留不确定性的风险解释 module

让领域结果明确区分“已识别风险”“未识别到可提醒目标”“模型/相机不可用”，UI adapter 只做本地化，不把未知重新解释为安全。

收益：安全语义集中，多种 UI、TTS 和 Care Mode 共享同一含义。

### 候选 3：可信验证 module

将仓库卫生、模型合同、JVM 测试、Lint、APK 合同和设备断言整理成同一套可执行判据，本地和 CI 使用不同 adapter 但保持相同语义。

收益：减少“每一步都绿、组合却失败”的假闭环。

### 候选 4：分离功能测试与 benchmark

让 Compose 功能测试和 detector/depth benchmark 成为两个真实 seam，分别拥有独立资产、执行入口、超时与设备策略。

收益：PR 反馈更快，benchmark 结果更稳定，AndroidTest APK 更小。

### 候选 5：反馈策略 module

明确冷却、疲劳、硬件能力和 session 生命周期的关系，把反馈状态集中到同一 implementation。

收益：新 session 首条提醒可预测，也方便未来加入眼镜、双马达或其他反馈 adapter。

### 候选 6：助盲专属 UI 回归 feature locality

当前 `core:ui` 实际包含整套 BlindAssist 专属屏幕、mapper 和状态语义，interface 偏宽。后续可只保留真正复用的主题和 UI primitives，把产品专属 UI 收回 `feature:assist`。

该候选优先级低于运行时安全和验证链问题，适合在核心风险修复后再探索。

## 七、建议实施顺序

### 第一阶段：安全语义与运行时正确性

1. 将“安全观察中”改为非安全承诺措辞，并为未知状态补测试。
2. 修复 session drain、generation 和旧帧提交问题。
3. 让 UI 使用当前帧携带的不可变 session summary。
4. 明确并测试 feedback cooldown/fatigue 的生命周期。

### 第二阶段：工程门禁可信化

1. 修复 `check_repo_hygiene.ps1` 正则并新增脚本自测样例。
2. 修复 Gradle benchmark 资产任务依赖。
3. 分离 Compose instrumentation 与 benchmark。
4. 给设备回归增加前台、无 crash/ANR、模型 ready、有效帧和 UI 节点断言。

### 第三阶段：文档与无障碍

1. 同步 README、DEMO_GUIDE 和当前 v8.9.0 UI。
2. 修复高风险徽标对比度和设置开关重复语义。
3. 增加权限永久拒绝恢复路径、动态风险播报和 onboarding 双语支持。

### 第四阶段：真实场景验证

1. 建立由隔离 GPT/Codex 共识复核的连续视频评测集。
2. 采集非 COCO、低矮、透明、光照和运动场景。
3. 在固定真机上记录延迟、P95、温度、功耗和关键漏报。
4. 在受控安全场地进行目标用户验证。

## 八、版本与交付判断

### v9.4.0 修复后更新

- 当前版本已升级为 `versionName=9.4.0` / `versionCode=34`。
- 原评估的安全措辞、session 生命周期、卫生/Gradle 依赖和测试隔离问题已标记为本地修复完成。
- 真机项目已在 Samsung `SM-S9280` / Android 16 上执行：Compose 11/11 通过；Detector A/B 与 Depth-fusion 均完整产生 100 图证据，两个候选都不晋级；强化后的 90 秒回归进入真实 CameraX 推理并通过前台、模型就绪和无 Crash/ANR 断言。
- 真机验证同时暴露两项工程问题并已修复：benchmark 测试 APK 的 Lifecycle 2.3.1/2.8.7 类冲突，以及 Detector/Depth 脚本使用不同 debug keystore。Android 16 的 16KB native library 对齐警告仍保留为后续兼容项。
- v9.4.0 已生成完整本地归档和 Git 里程碑 APK，详见 `docs/APK_ARCHIVE.md`。

### 原评估时判断

本次工作属于分析、验证与报告留存，没有修改 Kotlin、Gradle、Manifest、资源、模型资产或应用行为，因此：

- 不调整 `versionName` 和 `versionCode`。
- 不生成新的功能版本。
- 不归档新的里程碑 APK。
- 项目版本保持 `v8.9.0` / `versionCode=33`。

后续如果只修复卫生脚本、文档漂移和无障碍语义，可按小更新评估；如果实施运行时 session 深化并改变核心提醒链路，应按大更新评估，并重新执行完整 JVM、Lint、AndroidTest 编译、真机回归和 APK 归档。
