# BlindAssist Android Prototype

原生 Android Kotlin 助盲避障原型：CameraX 实时取流，TFLite 本地运行 YOLO11n，规则层判断危险区域，并通过语音和震动提醒。

## Version

- Current project version: `v0.8.0`
- Version policy: small updates add `v0.1`, major updates add `v0.5`, and milestone-level changes add `v1.0`.
- Version impact is judged by Codex/Agent based on each change's scope and risk.
- Updates that affect project state, usage, behavior, build flow, model assets, tests, or important technical decisions should keep this README aligned with the current state.
- Trivial wording, typo, formatting, or lightweight collaboration-rule clarifications do not count as version updates.

## Recent Updates

- 2026-05-17: Upgraded the real-time front-end interaction layer. The camera screen now separates the main risk status, control switches, and collapsible debug details, improves accessibility text for detection/speech/vibration switches, and makes the overlay risk source easier to distinguish. The app version is now `v0.8.0`.
- 2026-05-17: Added proximity-aware risk reminders. Risk analysis now reports relative proximity bands (`FAR`, `MID`, `NEAR`, `CRITICAL`) and an urgency score, allowing the app to distinguish visual-only mid/far detections from near and critical alerts. The app version is now `v0.7.0`.
- 2026-05-17: Added risk reminder stabilization after the rule-based analyzer. HIGH risks are confirmed immediately, MEDIUM risks require two matching frames, and confirmed alerts are briefly held across short missed detections. The app version is now `v0.2.0`.
- 2026-05-17: Added project collaboration rules for README synchronization and version bump judgment. Later clarified that trivial wording or lightweight process text does not trigger a version bump, so the project remained at `v0.1.0`.

## Interface Behavior

The main camera screen keeps the full-screen preview as the primary surface and uses a compact bottom panel for interaction:

- The main risk area shows the current risk level, relative proximity band, direction, primary target, target count, and urgency score.
- Detection, speech, and vibration can be toggled independently. Disabling detection clears the overlay and stops risk feedback while keeping the camera preview visible.
- Debug information is collapsed by default. Expanding it shows FPS, total/preprocess/inference/postprocess timing, and model status.
- Overlay boxes use stronger highlighting for the current risk source and quieter styling for other detected objects.

## Risk Reminder Behavior

The app is an assistive prototype, not a safety device that can replace human judgment. Detection results are smoothed before speech and vibration feedback:

- The app does not estimate real-world distance in meters. It only derives relative proximity bands from detection box position and size.
- FAR detections are retained visually but do not trigger speech or vibration.
- MID detections are shown as low-risk visual/status feedback.
- NEAR detections can trigger regular speech and vibration when the risk level is medium or high.
- CRITICAL detections trigger a shorter cooldown and stronger vibration pattern.
- HIGH risk reminders are emitted without frame-delay.
- MEDIUM risk reminders require two consecutive matching direction/message frames.
- A confirmed medium/high reminder is held for up to 600ms if the next frame briefly loses the risk, reducing flicker from transient missed detections.
- Regular near speech and vibration use a 1500ms cooldown; critical reminders use an 850ms cooldown.

## Environment

当前仓库是 Android Studio/Gradle 项目。构建前需要安装：

- JDK 17
- Android Studio 或 Android SDK + Platform Tools
- Android SDK Platform 35

验证命令：

```powershell
java -version
adb version
```

## Model Asset

第一版默认从 assets 加载真实 YOLO11n TFLite 模型：

```text
app/src/main/assets/yolo11n_fp16_320.tflite
app/src/main/assets/coco_labels.txt
```

模型文件较大，默认不提交到 Git。推荐用本仓库脚本导出，导出参数固定为 `imgsz=320`、`half=True`、`nms=False`，这样 Android 端可以解析 raw YOLO 输出并自行执行 NMS。已验证的本机导出路径是 Python 3.12 + TensorFlow 2.19：

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
input shape=[1, 320, 320, 3] dtype=float32
output shape=[1, 84, 2100] dtype=float32
```

## Build

在 Android Studio 打开项目并同步依赖；或安装 Gradle 后运行：

```powershell
.\gradlew.bat :app:testDebugUnitTest :app:assembleDebug --no-daemon
```

APK 输出位置：

```text
app/build/outputs/apk/debug/app-debug.apk
```

## Install to Phone

打开 Android 手机 USB 调试后：

```powershell
.\.android-sdk\platform-tools\adb.exe devices
.\.android-sdk\platform-tools\adb.exe install -r app\build\outputs\apk\debug\app-debug.apk
```
