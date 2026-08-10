# BlindAssist Android 原型

BlindAssist 是使用 Kotlin、Jetpack Compose、CameraX 和 TFLite 构建的本地助盲避障原型。它通过手机摄像头识别前方目标，结合规则层生成方向、相对距离、语音和震动提醒。

> BlindAssist 仍是辅助原型，不是安全认证设备，不能替代盲杖、导盲犬、人工判断或专业出行训练。

## 当前状态

<!-- research-status-owner: docs/research/README.md -->

- 当前版本：`v10.9.0`，`versionCode=37`。
- 正式 App 默认模型：`app/src/main/assets/yolo11n_fp16_320.tflite`。
- 算法、数据、系统与平台研究的动态状态、唯一 successor、禁止动作和证据权限只由
  [项目研究总入口](docs/research/README.md) 及其分类/路线 current 真源维护；本页不复制研究终态。
- 任何研究、benchmark、导出或设备结果都不自动改变正式 App、默认模型、产品权限或安全结论。
- 可并存安装的实验构建与正式 App 隔离；它们只用于研究或诊断，不获得默认产品权限。
- 正式 App 保持本地推理；眼镜外界硬件入口已能通过局域网连接 AtomS3R-M12 +
  ToF4M，读取设备/距离状态；实时 MJPEG 采用 latest-only 语义进入现有识别与提醒链路。
  ToF 仅作逐帧绑定元数据，标定融合仍暂缓。

发布变化见 [CHANGELOG.md](CHANGELOG.md)，近期工程过程见
[DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md)。研究状态从[项目研究总入口](docs/research/README.md)
进入；日期化审计和实验报告只代表当时快照，不作为当前状态真源。

研发默认端到端无人化：来源发现/获取、采集编排、标注、复核、裁决、隐私与质量检查、数据准入、实验验收和发布证据复核均由 GPT/Codex、多模态模型或自动 Agent 完成，不建立人工待办；统一 receipt、仲裁和失败关闭规则见 [GPT / Codex 端到端自主工作流治理](docs/AI_REVIEW_GOVERNANCE.md)。

## 仓库导航

| 路径 | 职责 |
| --- | --- |
| `app/` | Android 应用入口、依赖装配和正式资产 |
| `feature/assist/` | CameraX、检测、反馈与 UI 状态协调 |
| `core/assist/` | 风险分析、稳定、事件和提醒策略 |
| `core/vision/` | TFLite 检测、图像处理与视觉候选能力 |
| `core/device/` | 语音、震动和设备 adapter |
| `core/ui/` | Compose UI 模型与可视化 |
| `apps/` | 所有非默认 Android benchmark、canary、demo 和候选 App；见 [Apps 索引](apps/README.md) |
| `scripts/` | 构建、验证、数据集和研究脚本；见 [脚本索引](scripts/README.md) |
| `docs/` | 当前协议、操作指南和历史快照；见 [文档索引](docs/README.md) |
| `artifacts.local/` | 本机下载、数据集、benchmark、训练和临时产物；不提交 Git |

编码任务的最短职责入口见 [代码地图](docs/CODE_MAP.md)。

## 环境要求

- JDK 17
- Android SDK Platform 35、Build Tools 和 Platform Tools
- Python 仅用于模型检查、数据集及研究任务

本机通用工具位于 `E:\codex-tools`。`.jdk`、`.android-sdk`、`.android-home` 和 `.kotlin-home` 是指向 canonical toolchain/state 的兼容 junction；`.gradle-local` 是无 tracked 调用方的遗留本地缓存而非 junction，新命令统一使用 `E:\codex-tools\projects\blindassist\state\gradle`。研究 Python 统一通过 `E:\codex-tools\bin\blindassist-python.cmd` 使用 `E:\codex-tools\tools\venvs\blindassist-venv-export312`（Python 3.11.9）。仓库内 `.python311` 仅作遗留兼容，旧 `.venv-export312` 已不存在。新电脑安装见 [新电脑交接说明](docs/NEW_COMPUTER_HANDOFF.md)。

## 构建与验证

Windows/Codex 本地构建只使用 `scripts/run_android_gradle.ps1`。该入口自行锁定仓库根目录和项目声明的 JDK、SDK、Gradle wrapper/state；不要在调用前手工拼接环境变量，也不要直接运行 `gradlew.bat`。

在 `E:\linnan\linnan` 执行：

```powershell
pwsh -NoProfile -File scripts/run_android_gradle.ps1 -PreflightOnly
pwsh -NoProfile -File scripts/run_android_gradle.ps1 :app:testDebugUnitTest :app:lintDebug :app:assembleDebug
pwsh -NoProfile -File scripts/run_android_gradle.ps1 :app:assembleUstrfExperiment
```

完整无设备验证矩阵：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts\inspect_tflite.py
E:\codex-tools\bin\blindassist-python.cmd scripts\run_research_contract_tests.py
pwsh -NoProfile -File scripts/run_android_gradle.ps1 :core:assist:test :core:ustrf:test :core:vision:testDebugUnitTest :core:device:testDebugUnitTest :core:ui:testDebugUnitTest :feature:assist:testDebugUnitTest :app:testDebugUnitTest
pwsh -NoProfile -File scripts/run_android_gradle.ps1 :app:lintDebug :core:vision:lintDebug :core:device:lintDebug :core:ui:lintDebug :feature:assist:lintDebug
pwsh -NoProfile -File scripts/run_android_gradle.ps1 :app:assembleDebug :app:assembleDebugAndroidTest :device-benchmark:assembleDebug
```

设备测试必须显式指定 module：

- 功能测试：`:app:connectedDebugAndroidTest`
- benchmark：`:device-benchmark:connectedDebugAndroidTest`
- 真机回归流程：[DEVICE_REGRESSION.md](docs/DEVICE_REGRESSION.md)

## 模型资产

正式资产：

```text
app/src/main/assets/yolo11n_fp16_320.tflite
app/src/main/assets/coco_labels.txt
```

Android 端消费 raw YOLO 输出并自行执行 NMS。重新导出和静态检查入口见 [scripts/README.md](scripts/README.md)；本地模型源文件和导出结果应放入 `artifacts.local/models/`，不得散落在仓库根目录。

## APK 与安装

构建输出：

```text
app/build/outputs/apk/debug/app-debug.apk
app/build/outputs/apk/ustrfExperiment/app-ustrfExperiment.apk
app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
apps/benchmarks/device-benchmark/build/outputs/apk/debug/device-benchmark-debug.apk
```

连接已开启 USB 调试的 Android 手机后：

```powershell
E:\codex-tools\tools\android-sdk\platform-tools\adb.exe devices
E:\codex-tools\tools\android-sdk\platform-tools\adb.exe install -r app\build\outputs\apk\debug\app-debug.apk
E:\codex-tools\tools\android-sdk\platform-tools\adb.exe install -r app\build\outputs\apk\ustrfExperiment\app-ustrfExperiment.apk
```

实验版包名为 `com.linnan.blindassist.ustrf.experimental`，可与正式包并存；应用名和页顶常驻条幅会明确标出 USTRF 实验边界。

正式归档、校验和与 Git 收据规则见 [APK 归档策略](docs/APK_ARCHIVE.md) 和 [发布与验证](docs/RELEASE_AND_VERIFICATION.md)。原始 APK 保留在经校验的外部归档，不再作为 Git 二进制提交。

## 文档职责

- `README.md`：当前产品、构建方式和导航。
- `CHANGELOG.md`：已发布变化。
- `DEVELOPMENT_LOG.md`：最近 2–4 周的工程变化与验证索引；更早原文按月归档到 `docs/history/development-log/`。
- `idea.md`：尚未决定的产品与研究方向；不是实验流水或当前状态真源。
- `docs/SANPO_CURRENT_STATUS.md`：SANPO 当前研究状态、硬门与下一步。
- `docs/DOCUMENT_GOVERNANCE.md`：文档职责、真源与归档规则。
- `docs/*_YYYY-MM-DD.*`：日期化快照，不覆盖当前协议。
