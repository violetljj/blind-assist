# BlindAssist Android Prototype

原生 Android Kotlin 助盲避障原型：CameraX 实时取流，TFLite 本地运行 YOLO11n，规则层判断危险区域，并通过语音和震动提醒。

## Version

- Current project version: `v2.6.0`
- Version policy: small updates add `v0.1`, major updates add `v0.5`, and milestone-level changes add `v1.0`.
- Version impact is judged by Codex/Agent based on each change's scope and risk.
- Updates that affect project state, usage, behavior, build flow, model assets, tests, or important technical decisions should keep this README aligned with the current state.
- Trivial wording, typo, formatting, or lightweight collaboration-rule clarifications do not count as version updates.

## Recent Updates

- 2026-05-18: Implemented the v2.6.0 display-trust polish update. The main panel now separates current-frame detections from short held reminders, hides numeric urgency from the default user-facing target line, moves urgency into debug details, uses more action-oriented risk copy, softens the center overlay into an observation reference area, and improves toggle accessibility descriptions. The debug APK was installed successfully on device `R5CX10M8Y8X` and verified as `versionName=2.6.0`, `versionCode=10`.
- 2026-05-18: Implemented the v2.5.0 field-testable walking-assist upgrade. Risk analysis, stabilization, feedback-display reason calculation, and in-memory session tracing now live in a pure Kotlin assist-session layer. The existing debug panel now shows a recent-session summary, near and critical speech prompts are shorter action-oriented guidance, and the app version is now `v2.5.0`. The debug APK was installed successfully on device `R5CX10M8Y8X` and verified as `versionName=2.5.0`, `versionCode=9`.
- 2026-05-17: Added the v2.0.0 phone-only experience upgrade. The app now has persistent alert profiles (`Quiet`, `Standard`, `Sensitive`) that tune medium-risk confirmation, alert hold time, speech/vibration cooldowns, and vibration duration. The debug panel now explains the latest raw risk, stabilized risk, alert profile, and feedback reason, while the controls include clearer accessibility descriptions. CameraX analysis resolution selection was also updated away from the deprecated target-resolution API. The app version is now `v2.0.0`.
- 2026-05-17: Added lightweight user preference persistence for speech reminders, vibration reminders, and Care Mode. Detection still starts enabled on each launch so the app does not reopen in a silent paused-recognition state. The app version is now `v1.5.0`.
- 2026-05-17: Polished the camera screen based on phone screenshot feedback. The preview now fills the display instead of leaving a large top letterbox, overlay mapping matches the filled preview crop, and the bottom controls use compact high-contrast mode buttons instead of bulky platform switches. The app version is now `v1.4.0`.
- 2026-05-17: Redesigned the real-time camera interface as a calmer assistive workspace. The screen now includes a brand/status header, risk badge, two-row control area, smoother status transitions, and a Care Mode that enlarges key guidance, increases contrast, hides developer debug details, and adds a center guide in the overlay. The app version is now `v1.3.0`.
- 2026-05-17: Upgraded the real-time front-end interaction layer. The camera screen now separates the main risk status, control switches, and collapsible debug details, improves accessibility text for detection/speech/vibration switches, and makes the overlay risk source easier to distinguish. The app version is now `v0.8.0`.
- 2026-05-17: Added proximity-aware risk reminders. Risk analysis now reports relative proximity bands (`FAR`, `MID`, `NEAR`, `CRITICAL`) and an urgency score, allowing the app to distinguish visual-only mid/far detections from near and critical alerts. The app version is now `v0.7.0`.
- 2026-05-17: Added risk reminder stabilization after the rule-based analyzer. HIGH risks are confirmed immediately, MEDIUM risks require two matching frames, and confirmed alerts are briefly held across short missed detections. The app version is now `v0.2.0`.
- 2026-05-17: Added project collaboration rules for README synchronization and version bump judgment. Later clarified that trivial wording or lightweight process text does not trigger a version bump, so the project remained at `v0.1.0`.

## Project Materials

- [回顾式阶段进度说明](PROJECT_PROGRESS_REVIEW.md)：面向课程汇报、阶段检查和毕设展示的整理稿，按 3 月至 5 月 1 日前的“调研、方案、原型、测试、迭代”脉络说明项目工作量。该文档是回顾式材料，不替代真实开发日志。

## Interface Behavior

The main camera screen keeps the full-screen preview as the primary surface and uses a compact bottom panel for interaction:

- The panel follows a camera/navigation-app style hierarchy: product identity, current state badge, large risk instruction, supporting detail, then controls.
- Status changes use a short restrained transition so risk updates feel responsive without distracting from the camera preview.
- The camera preview fills the screen in portrait use. Detection overlay coordinates follow the same filled-preview crop so boxes and guide areas stay aligned.
- The main risk area shows the current risk level, relative proximity band, direction, current-frame target count, and the primary alert source when one is currently locked.
- If a stabilized reminder is briefly held after the current frame loses the target, the target line explicitly says the reminder is being held from the previous frame instead of pairing the old target name with a current count of zero.
- Detection, speech, vibration, alert profile, and Care Mode use compact high-contrast mode buttons and can be toggled independently. Speech, vibration, alert profile, and Care Mode restore the user's last choice on the next launch, while detection starts enabled every time. Disabling detection clears the overlay and stops risk feedback while keeping the camera preview visible.
- Alert profile cycles between Quiet, Standard, and Sensitive. Quiet reduces reminder frequency and vibration length, Standard keeps the original balanced behavior, and Sensitive confirms medium risks sooner with shorter reminder cooldowns.
- Care Mode enlarges the main instruction, simplifies the supporting copy, increases panel contrast, hides debug details, and adds a center guide line to support lower-vision or high-stress use.
- Debug information is collapsed by default. Expanding it shows FPS, total/preprocess/inference/postprocess timing, model status, the latest raw and stabilized risk, raw/stabilized urgency scores, the active alert profile, the feedback reason, and a recent-session summary for the last 30 processed frames.
- Overlay boxes use stronger highlighting for the current risk source and quieter styling for other detected objects. The center region is drawn as an observation reference area, not as a detected object box.

## Risk Reminder Behavior

The app is an assistive prototype, not a safety device that can replace human judgment. Detection results are smoothed before speech and vibration feedback:

- The app does not estimate real-world distance in meters. It only derives relative proximity bands from detection box position and size.
- FAR detections are retained visually but do not trigger speech or vibration.
- MID detections are shown as low-risk visual/status feedback.
- NEAR detections can trigger regular speech and vibration when the risk level is medium or high. Speech prompts are short guidance phrases such as “前方近处，减速” or “左前方近处，注意避让”.
- CRITICAL detections trigger a shorter cooldown and stronger vibration pattern. Center critical prompts use the short guidance phrase “前方很近，放慢”.
- HIGH risk reminders are emitted without frame-delay.
- MEDIUM risk reminders require two consecutive matching direction/message frames.
- A confirmed medium/high reminder is held for up to 600ms if the next frame briefly loses the risk, reducing flicker from transient missed detections.
- Regular near speech and vibration use a 1500ms cooldown; critical reminders use an 850ms cooldown.
- In Quiet profile, medium risks require three matching frames, held alerts last 450ms, near reminders use a 2200ms cooldown with 100ms vibration, and critical reminders use a 1200ms cooldown with 260ms vibration.
- In Standard profile, medium risks require two matching frames, held alerts last 600ms, near reminders use a 1500ms cooldown with 160ms vibration, and critical reminders use an 850ms cooldown with 420ms vibration.
- In Sensitive profile, medium risks confirm on the first frame, held alerts last 800ms, near reminders use a 1000ms cooldown with 220ms vibration, and critical reminders use a 650ms cooldown with 520ms vibration.

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

## Versioned APK Archive

用于展示和对比的版本 APK 保存在：

```text
releases/apk/
```

当前已补存 v0.1.0、v0.2.0、v0.7.0、v0.8.0、v1.3.0、v1.4.0、v1.5.0、v2.0.0、v2.5.0 和 v2.6.0 的 debug APK。带 `rebuilt` 的文件表示从对应 Git 历史提交重新构建得到，适合用于演示版本演进；它不是当时原始构建产物的文件时间复刻。

## Install to Phone

打开 Android 手机 USB 调试后：

```powershell
.\.android-sdk\platform-tools\adb.exe devices
.\.android-sdk\platform-tools\adb.exe install -r app\build\outputs\apk\debug\app-debug.apk
```
