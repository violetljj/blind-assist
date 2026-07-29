# BlindAssist Android 原型

BlindAssist 是使用 Kotlin、Jetpack Compose、CameraX 和 TFLite 构建的本地助盲避障原型。它通过手机摄像头识别前方目标，结合规则层生成方向、相对距离、语音和震动提醒。

> BlindAssist 仍是辅助原型，不是安全认证设备，不能替代盲杖、导盲犬、人工判断或专业出行训练。

## 当前状态

- 当前版本：`v10.9.0`，`versionCode=37`。
- 正式 App 默认模型：`app/src/main/assets/yolo11n_fp16_320.tflite`。
- 当前论文系统研究主线为
  [神经—几何双环阶段−1准入](docs/research/dual-loop/README.md)，状态仍是
  `PROPOSAL_ONLY / EXECUTION_NOT_AUTHORIZED`；RCLE 已暂停，历史终态与权限由
  [RCLE current 入口](docs/research/rcle/README.md) 保留。研究过程遵循
  [渐进式治理](docs/RESEARCH_GOVERNANCE.md)，任何研究证据都不自动改变正式 App、
  默认模型、Android、人体、安全、产品或生产权限。
- 可并存安装的 `ustrfExperiment` 实验版已接入 USTRF 二维路线代理，并直接替代旧风险分析入口；它只使用画面中心假设路线与检测框生成保守代理风险，不具备米制深度、稳定姿态或真实路线，因此不可用于独立行走，也不改变正式 App。
- SANPO 分割路线仍为研究候选：当前离线质量门未通过，未导出正式 INT8、未执行设备晋级门、未替换 App 默认模型。
- 正式 App 保持本地推理；眼镜设备中心仍是模拟功能，不扫描蓝牙、不连接真实眼镜。

发布变化见 [CHANGELOG.md](CHANGELOG.md)，近期工程过程见
[DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md)。当前论文系统研究入口见
[双环研究主线](docs/research/dual-loop/README.md)，暂停的 RCLE 状态见
[RCLE current 入口](docs/research/rcle/README.md)，SANPO 的当前研究状态见
[SANPO_CURRENT_STATUS.md](docs/SANPO_CURRENT_STATUS.md)；日期化审计和实验报告只代表
当时快照，不作为当前状态真源。

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
| `device-benchmark/` | 与正式 App 隔离的设备 benchmark module |
| `scripts/` | 构建、验证、数据集和研究脚本；见 [脚本索引](scripts/README.md) |
| `docs/` | 当前协议、操作指南和历史快照；见 [文档索引](docs/README.md) |
| `artifacts.local/` | 本机下载、数据集、benchmark、训练和临时产物；不提交 Git |

## 环境要求

- JDK 17
- Android SDK Platform 35、Build Tools 和 Platform Tools
- Python 仅用于模型检查、数据集及研究任务

本机通用工具位于 `E:\codex-tools`。`.jdk`、`.android-sdk`、`.android-home` 和 `.kotlin-home` 是指向 canonical toolchain/state 的兼容 junction；`.gradle-local` 是无 tracked 调用方的遗留本地缓存而非 junction，新命令统一使用 `E:\codex-tools\projects\blindassist\state\gradle`。研究 Python 统一通过 `E:\codex-tools\bin\blindassist-python.cmd` 使用 `E:\codex-tools\tools\venvs\blindassist-venv-export312`（Python 3.11.9）。仓库内 `.python311` 仅作遗留兼容，旧 `.venv-export312` 已不存在。新电脑安装见 [新电脑交接说明](docs/NEW_COMPUTER_HANDOFF.md)。

## 构建与验证

在 `E:\linnan\linnan` 执行：

```powershell
$env:JAVA_HOME='E:\codex-tools\projects\blindassist\toolchain\.jdk\jdk17.0.19_10'
$env:PATH="$env:JAVA_HOME\bin;E:\codex-tools\tools\android-sdk\platform-tools;$env:PATH"
$env:GRADLE_USER_HOME='E:\codex-tools\projects\blindassist\state\gradle'
.\gradlew.bat :app:testDebugUnitTest :app:lintDebug :app:assembleDebug --no-daemon --console=plain
.\gradlew.bat :app:assembleUstrfExperiment --no-daemon --console=plain
```

完整无设备验证矩阵：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts\inspect_tflite.py
E:\codex-tools\bin\blindassist-python.cmd scripts\run_research_contract_tests.py
.\gradlew.bat :core:assist:test :core:ustrf:test :core:vision:testDebugUnitTest :core:device:testDebugUnitTest :core:ui:testDebugUnitTest :feature:assist:testDebugUnitTest :app:testDebugUnitTest --no-daemon --console=plain
.\gradlew.bat :app:lintDebug :core:vision:lintDebug :core:device:lintDebug :core:ui:lintDebug :feature:assist:lintDebug --no-daemon --console=plain
.\gradlew.bat :app:assembleDebug :app:assembleDebugAndroidTest :device-benchmark:assembleDebug --no-daemon --console=plain
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
device-benchmark/build/outputs/apk/debug/device-benchmark-debug.apk
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
