# BlindAssist Android Prototype

Android Kotlin + Jetpack Compose 助盲避障原型：Compose/Material 3 提供启动页、功能入口、底部导航和设置体验，CameraX 实时取流，TFLite 本地运行 YOLO11n，规则层判断危险区域，并通过语音和震动提醒。

## Version

- Current project version: `v5.9.0`
- Version policy: small updates add `v0.1`, major updates add `v0.5`, and milestone-level changes add `v1.0`.
- Version impact is judged by Codex/Agent based on each change's scope and risk.
- Updates that affect project state, usage, behavior, build flow, model assets, tests, or important technical decisions should keep this README aligned with the current state.
- Trivial wording, typo, formatting, or lightweight collaboration-rule clarifications do not count as version updates.

## Recent Updates

- 2026-05-19: Implemented the v5.9.0 test-hardening and camera debug-control fix. The camera page debug toggle now uses the same stable `CompactAction` button pattern as the other camera controls, sits above Care Mode/scenario controls, exposes explicit `展开相机调试信息` / `收起相机调试信息` content descriptions, and keeps the `camera_debug_toggle` Compose test tag. The connected Compose tests now grant camera permission before launching `MainActivity`, detect the camera page through either text or content descriptions, reset the camera-panel test to 通用日常 so Debug is visible, and always close the camera page after camera-path tests. This resolves the two issues found in the v5.8.0 phone test: `connectedDebugAndroidTest` now passes 6 tests with 0 failures on `SM-S9280 - 16`, and ADB UIAutomator can tap the debug control and observe it change to `收起相机调试信息`. Full JVM tests still pass with 105 tests and 0 failures; the debug APK was archived at `releases/apk/BlindAssist-v5.9.0-debug-20260519-174352.apk`, installed on `SM-S9280`, cold-started successfully, and verified as `versionCode=23` / `versionName=5.9.0`. This update does not change the YOLO model, CameraX/TFLite pipeline, risk thresholds, permissions, networking, Bluetooth, storage, Hilt, multi-module structure, Room or DataStore.
- 2026-05-19: Completed a detailed v5.8.0 real-phone validation pass on Samsung `SM-S9280` with app data cleared, fixed wireless ADB serial selection, model shape inspection, full JVM test/build validation, APK reinstall, UIAutomator navigation, camera-permission flow, camera-page checks, and a 90-second CameraX/TFLite performance sample. The model still reports input `[1, 320, 320, 3] float32` and output `[1, 84, 2100] float32`; the full Gradle JVM/debug build command passed with 105 tests, 0 failures, and the APK was archived at `releases/apk/BlindAssist-v5.8.0-debug-20260519-170622.apk`. Manual ADB UI checks passed for onboarding, Features/Profile/Settings navigation, language switching, glasses placeholder, permission explanation, camera entry, quiet/sensitive shortcuts, and camera return. The 90-second camera sample produced 88 `BlindAssistPerf` entries with average total processing `55.40ms`, average inference `37.76ms`, average FPS `14.97`, `gfxinfo` jank `20/1920 frames (1.04%)`, TOTAL PSS `269,790 KB`, and no crash/ANR keyword matches. Two issues remain documented in `TEST_REPORT_2026-05-19.md`: `connectedDebugAndroidTest` failed with 2 failures and an instrumentation `Process crashed` message, and the camera page `展开调试信息` button node was visible but did not respond to repeated ADB taps near the bottom of the screen. This was a testing/documentation pass only, so the project version remains `v5.8.0` / `versionCode=22`.
- 2026-05-19: Implemented the v5.8.0 daily usage guide and one-tap mode update. The Features page now includes a persistent daily usage guide before the phone-camera entry, with 通用日常、室内慢行、走廊通行、密集区域 and 户外慢行 presets. Each preset applies an existing local preference bundle across assist scenario, reminder profile, speech style, vibration strength and Care Mode; no separate mode key is persisted, so manually adjusted combinations appear as 自定义 / Custom. The camera panel now shows the active daily mode and replaces the old profile-cycle button with clear 调安静 / 调敏感 shortcuts that keep the current scenario while adjusting reminder intensity. New JVM tests cover mode mapping, custom fallback, ViewModel persistence and reminder shortcuts; Compose tests cover the guide, settings reflection, camera labels, shortcuts and English accessibility text. This update does not change the YOLO model, CameraX/TFLite pipeline, risk thresholds, permissions, networking, Bluetooth, storage, Hilt, multi-module structure, Room or DataStore. The app version is now `v5.8.0` / `versionCode=22`; the debug APK was archived at `releases/apk/BlindAssist-v5.8.0-debug-20260519-120747.apk`, installed on `SM-S9280`, and package info verified as `versionCode=22` / `versionName=5.8.0`. Full JVM tests and debug APK build passed with 105 tests, 0 failures; `connectedDebugAndroidTest` was not run because the phone was online but still on the lockscreen (`mDreamingLockscreen=true`).
- 2026-05-19: Implemented the v5.3.0 TalkBack, large-text and in-app Chinese/English language update. Settings now include a persistent `界面语言` / `Interface language` selector for Chinese and English; Chinese remains the default. Core assistive text now has a lightweight localization boundary covering speech reminder templates, vibration/profile/scenario labels, camera guidance, risk explanations, field-test summaries and TalkBack-oriented content descriptions/state descriptions. Settings selectors now use 48dp+ full-width rows to better tolerate large font sizes, and the camera panel uses localized compact controls and action-oriented accessibility text. Android string resources now include matching Chinese and English core labels. This update does not change the YOLO model, risk thresholds, scenario policies, permissions, networking, Bluetooth, storage, Hilt, multi-module structure, Room or DataStore. The app version is now `v5.3.0` / `versionCode=21`; the debug APK was archived at `releases/apk/BlindAssist-v5.3.0-debug-20260519-113731.apk`. The first `connectedDebugAndroidTest` attempt timed out while duplicate wireless ADB serials were present; after disconnecting the duplicate serial and setting `ANDROID_SERIAL`, the APK was installed on `SM-S9280`, package info verified as `versionCode=21` / `versionName=5.3.0`, the app cold-started successfully, and `connectedDebugAndroidTest` completed 5 tests on `SM-S9280 - 16` with 0 failures.
- 2026-05-19: Implemented the v4.8.0 single-module quality upgrade. The app remains a native single-module Android/Kotlin prototype with CameraX, TFLite, Compose and SharedPreferences, but the runtime responsibilities are now split more clearly: `CameraXFrameSource` owns CameraX startup and frame delivery, `ObjectDetector` / `DetectorFrameResult` define detector output, `AssistSessionCoordinator` owns per-frame assist evaluation and feedback dispatch, `FpsTracker` owns FPS windows, and `CameraGuidanceMapper` / `FieldTestSummaryMapper` own UI state mapping. `MainActivity` is now a thinner lifecycle, permission and Compose binding entry point, while the v4.3.0 feature-page cleanup remains in place. New JVM tests cover the coordinator, FPS tracker, guidance mapper and field-test mapper. This update does not change the YOLO model, risk thresholds, scenario policies, permissions, networking, Bluetooth, storage, Hilt, multi-module structure, Room or DataStore. The app version is now `v4.8.0` / `versionCode=20`; the debug APK was archived at `releases/apk/BlindAssist-v4.8.0-debug-20260519-005155.apk`.
- 2026-05-19: Implemented the v4.3.0 feature-page cleanup update. The App 内“项目展示中心” has been temporarily removed from the Features page because it is not useful for the current daily prototype flow. The Features page now keeps only the practical phone-camera entry, the future glasses-device placeholder, the safety-boundary strip, and model/version status; onboarding replay remains available from Settings. Showcase documents such as `CHANGELOG.md`, `DEMO_GUIDE.md`, and the APK archive are kept for classroom or thesis materials. This update does not change the detector, reminder model, scenario policies, permissions, networking, Bluetooth, storage, Hilt, multi-module structure, Room or DataStore. The app version is now `v4.3.0` / `versionCode=19`; the debug APK was archived at `releases/apk/BlindAssist-v4.3.0-debug-20260519-003109.apk`, installed on device `SM-S9280`, and verified as `versionName=4.3.0`.
- 2026-05-19: Implemented the v4.2.0 scenario-aware reminder and risk-explanation update. Settings now include a persistent manual `使用场景` selector with 通用、室内慢行、走廊通行、密集区域 and 户外慢行 options; the default 通用 scenario preserves v4.1.0 behavior, while the other scenarios adjust medium-risk confirmation, held-alert timing, near-risk cooldown and vibration duration through the existing rule layer. The camera panel now shows the active scenario and a concise explanation for why feedback was triggered, held, cooled down, considered unstable or skipped; Care Mode keeps this explanation short and avoids debug/performance detail. The field-test summary now records the current scenario and latest explanation. This update does not add automatic scene recognition, networking, location, Bluetooth, storage permissions, model changes, Hilt, multi-module architecture, Room or DataStore. The app version is now `v4.2.0` / `versionCode=18`; the debug APK was archived at `releases/apk/BlindAssist-v4.2.0-debug-20260519-000200.apk`, installed on device `SM-S9280`, and verified as `versionName=4.2.0`. `connectedDebugAndroidTest` was attempted, but the device was on the lockscreen/Bouncer (`mDreamingLockscreen=true`), so it did not provide a passing result in this run.
- 2026-05-18: Implemented the v4.1.0 showcase delivery update. The Features page now includes a project showcase center for classroom and thesis-demo scenarios, covering local recognition, speech/vibration reminders, field-test summaries, and the prototype safety boundary. The showcase center reuses the existing phone-camera permission flow for “开始演示” and the existing onboarding replay for “查看引导”, without adding networking, location, Bluetooth, storage permissions, model changes, Hilt, multi-module architecture, Room, or DataStore. New `CHANGELOG.md` and `DEMO_GUIDE.md` materials document the real version route, APK archive, demo script, no-device fallback, privacy boundary, and safety wording. Compose instrumentation coverage now includes top-level navigation and showcase actions; after reconnecting/unlocking the device, `connectedDebugAndroidTest` completed 4 tests on `SM-S9280 - 16` and passed. The app version is now `v4.1.0` / `versionCode=17`; the debug APK was archived at `releases/apk/BlindAssist-v4.1.0-debug-20260518-231542.apk`, installed on device `SM-S9280`, and verified as `versionName=4.1.0`.
- 2026-05-18: Fixed the connected-device Compose instrumentation test failure that UTP reported as `Process crashed`. Direct `am instrument` showed the real failure was `No compose hierarchies found`, caused by the Compose test class not being run with `AndroidJUnit4` plus unstable merged semantics around selector chips. The test class now uses `@RunWith(AndroidJUnit4::class)`, AndroidX runner/rules are explicit test dependencies, and feedback selector cards no longer merge child chip semantics. `connectedDebugAndroidTest` now passes on `SM-S9280 - 16`; the current v3.6.0 debug APK was re-archived at `releases/apk/BlindAssist-v3.6.0-debug-20260518-214947.apk` and installed successfully.
- 2026-05-18: Implemented the v3.6.0 daily-use feedback polish update. Settings now include persistent speech style (`简短` / `标准` / `详细`) and vibration strength (`轻柔` / `标准` / `强`) controls, while alert profiles remain Quiet/Standard/Sensitive. Feedback now generates speech text through templates, scales vibration duration/amplitude by strength, and lengthens repeated non-critical near-risk cooldowns without suppressing critical high-risk alerts. The overlay now applies lightweight display-only box smoothing, near-distance thresholds are slightly more conservative, and tests cover the new preferences, feedback plans, fatigue control, risk matrix, overlay smoothing, ViewModel state, and Compose settings/camera controls. The app version is now `v3.6.0` / `versionCode=16`; the debug APK was archived, installed on device `SM_S9280`, and verified as `versionName=3.6.0`.
- 2026-05-18: Implemented the v3.5.0 lightweight ViewModel/StateFlow state split. Compose-observed app shell state, settings preference state, dialog flags, camera active state, model status, guidance state, and field-test summary now flow through `BlindAssistViewModel` as read-only `StateFlow`, collected in Compose with `collectAsStateWithLifecycle()`. `MainActivity` still owns CameraX, permissions, TFLite detector, overlay view, feedback controller, and lifecycle cleanup, so the architecture is cleaner without introducing Hilt, DataStore, Room, networking, location, new permissions, or modules. The app version is now `v3.5.0` / `versionCode=15`, the debug APK was archived at `releases/apk/BlindAssist-v3.5.0-debug-20260518-193819.apk`, installed on device `SM_S9280`, and verified as `versionName=3.5.0`.
- 2026-05-18: Implemented the v3.4.0 field-test summary and accessibility polish update. The app now keeps an in-memory field-test summary for the current or last camera session, including runtime, latest 30-frame risk counts, speech/vibration trigger counts, average FPS, average inference time, and the active alert profile. The summary appears in Settings and in the camera debug area without adding storage, network, location, or file permissions. Settings switches, compact camera controls, alert-profile selection, and summary headings now expose clearer TalkBack state semantics while preserving 48dp touch targets. The app version is now `v3.4.0` / `versionCode=14`, and the debug APK was archived at `releases/apk/BlindAssist-v3.4.0-debug-20260518-192333.apk`, installed on device `SM_S9280`, and verified as `versionName=3.4.0`.
- 2026-05-18: Implemented the v3.3.0 onboarding and camera-permission explanation update. First-time users now see a three-page Compose onboarding flow covering phone-camera local recognition, speech/vibration assistive reminders, and the prototype safety boundary. The Settings page includes a `查看新手引导` entry for replaying the guide. Tapping `使用手机摄像头` without camera permission now shows an in-app explanation before the Android system permission sheet, and denied permission leaves the user in the main shell with a short explanation instead of entering the camera subpage. The app version is now `v3.3.0` / `versionCode=13`, and the debug APK was archived at `releases/apk/BlindAssist-v3.3.0-debug-20260518-154943.apk`. Phone installation was not completed in this run because ADB reported `no devices/emulators found`.
- 2026-05-18: Implemented the v3.2.0 camera-back and profile cleanup update. The immersive camera subpage now handles Android system back gestures with the same close-camera path as the top return button, so swiping back returns to the main app shell instead of exiting to the launcher. The Profile page no longer shows the project showcase/explanation card and keeps only user, device, version, and assist-preference status. The app version is now `v3.2.0` / `versionCode=12`, and the debug APK was archived at `releases/apk/BlindAssist-v3.2.0-debug-20260518-152635.apk`, installed on device `R5CX10M8Y8X`, and verified as `versionName=3.2.0`.
- 2026-05-18: Implemented the v3.1.0 app-shell UI renewal. The app now starts with the Android SplashScreen API and a short Compose brand launch screen, then opens a Material 3 main shell with bottom navigation for Features, Profile, and Settings. The Features page now offers a phone-camera entry and a placeholder for future glasses connection, while the real-time CameraX/TFLite assist flow moved into an immersive camera subpage that starts only after the user taps the phone-camera action. Compose is enabled with a compileSdk 35-compatible BOM, the original CameraX `PreviewView`, detection overlay, assist engine, alert profiles, speech, and vibration logic are retained, and the app version is now `v3.1.0` / `versionCode=11`. The debug APK was archived at `releases/apk/BlindAssist-v3.1.0-debug-20260518-151146.apk`, installed on device `R5CX10M8Y8X`, and verified as `versionName=3.1.0`.
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

- [真实版本更新记录](CHANGELOG.md)：按真实版本整理功能变化、验证证据和 APK 归档路径，方便课堂展示、答辩材料和版本对比。
- [演示指南](DEMO_GUIDE.md)：面向老师/答辩的演示脚本，包含环境准备、手机安装、现场演示顺序、无设备 fallback、隐私与安全边界说明。
- [回顾式阶段进度说明](PROJECT_PROGRESS_REVIEW.md)：面向课程汇报、阶段检查和毕设展示的整理稿，按 3 月至 5 月 1 日前的“调研、方案、原型、测试、迭代”脉络说明项目工作量。该文档是回顾式材料，不替代真实开发日志。
- [v5.8.0 真机完整测试与 v5.9.0 修复复测报告](TEST_REPORT_2026-05-19.md)：记录 2026-05-19 在 `SM-S9280` 上执行的清数据真机功能、UI、性能、稳定性和 instrumentation 测试结果，以及 v5.9.0 对遗留问题的修复复测。

## Interface Behavior

The app now opens into a polished Compose app shell before starting any camera work:

- Compose-visible app state is now driven by a lightweight `BlindAssistViewModel` and read-only `StateFlow`. The Activity remains the boundary for Android lifecycle, permissions, CameraX, detector, feedback, and overlay objects.
- Cold launch uses the Android SplashScreen API, followed by a short BlindAssist brand screen with restrained scan/pulse motion. The launch screen can be skipped by tapping it.
- First-time users then see a three-page onboarding flow for local phone-camera recognition, speech/vibration reminders, and the prototype safety boundary. Completing or skipping the guide saves the onboarding state locally.
- The main shell uses Material 3 bottom navigation with three top-level destinations: Features, Profile, and Settings.
- The Features page is the default entry. It first presents a daily usage guide with five persistent one-tap presets: 通用日常、室内慢行、走廊通行、密集区域 and 户外慢行. Each preset applies an existing preference bundle for scenario, reminder profile, speech style, vibration strength and Care Mode. It then presents `使用手机摄像头` as the active local detection path and `连接眼镜设备` as a future-device placeholder, followed by the prototype safety boundary and model/version status. The previous project showcase center is temporarily removed from the in-app flow; demo and version materials remain in repository documents. The glasses card only shows an explanatory dialog; it does not scan Bluetooth, request Bluetooth permissions, connect to a network, or imply that hardware support is already finished.
- The Profile page is a compact local status page for user/device state, current alert profile, version information, and assist preferences. It does not include login, cloud sync, showcase explanation cards, or account data.
- The Settings page controls interface language, speech reminders, vibration reminders, speech style, vibration strength, Care Mode, debug details, alert profile, manual assist scenario, and includes a `查看新手引导` entry. If the user manually adjusts a preference combination so it no longer matches a one-tap daily preset, the main UI reports the daily mode as 自定义 / Custom. Settings also shows the current or last in-memory field-test summary so a session can review runtime, risk counts, reminders, FPS, inference time, active alert profile, active scenario, and latest risk explanation without writing files.
- Tapping `使用手机摄像头` opens an immersive camera subpage only after camera permission is available. If permission is missing, the app first shows an in-app explanation that camera frames stay local, are not uploaded, and are not saved as video; only then can the user continue to the Android system permission sheet.
- If camera permission is denied, the app stays in the main shell and explains that the phone-camera assist path cannot start without permission.
- The camera subpage hides the bottom navigation, shows a full-screen `PreviewView` with the existing detection overlay, and provides a top return button plus a compact bottom control panel. Tapping the return button or using the Android system back gesture returns to the main app shell, unbinds CameraX, and clears the overlay.

The real-time camera page keeps the full-screen preview as the primary surface and uses a compact bottom panel for interaction:

- The panel follows a camera/navigation-app style hierarchy: product identity, current state badge, large risk instruction, supporting detail, then controls.
- Status changes use a short restrained transition so risk updates feel responsive without distracting from the camera preview.
- The camera preview fills the screen in portrait use. Detection overlay coordinates follow the same filled-preview crop so boxes and guide areas stay aligned.
- The main risk area shows the current risk level, relative proximity band, direction, current-frame target count, and the primary alert source when one is currently locked.
- If a stabilized reminder is briefly held after the current frame loses the target, the target line explicitly says the reminder is being held from the previous frame instead of pairing the old target name with a current count of zero.
- Detection, speech, vibration, manual scenario, and Care Mode use compact high-contrast mode buttons and can be toggled independently. Speech, vibration, alert profile, assist scenario, speech style, vibration strength, Care Mode, and the resulting daily preset combination restore the user's last choice on the next launch, while detection starts enabled every time. Disabling detection clears the overlay and stops risk feedback while keeping the camera preview visible.
- The camera panel shows the active daily mode and keeps two direct reminder-intensity shortcuts: 调安静 applies Quiet + Brief speech + Soft vibration, while 调敏感 applies Sensitive + Standard speech + Strong vibration. Both shortcuts keep the current scenario and persist through the existing preference store.
- Alert profile cycles between Quiet, Standard, and Sensitive. Quiet reduces reminder frequency and vibration length, Standard keeps the original balanced behavior, and Sensitive confirms medium risks sooner with shorter reminder cooldowns.
- Assist scenario cycles between 通用、室内慢行、走廊通行、密集区域 and 户外慢行. 通用 preserves the existing profile behavior; the other scenarios only tune rule-layer confirmation, hold, cooldown and vibration values. They are manual presets, not automatic scene recognition.
- Speech style can be 简短, 标准, or 详细. Vibration strength can be 轻柔, 标准, or 强; these settings adjust feedback wording and tactile strength without changing detection or risk analysis.
- Care Mode enlarges the main instruction, simplifies the supporting copy, increases panel contrast, hides debug details, and adds a center guide line to support lower-vision or high-stress use.
- Debug information is collapsed by default. Expanding it shows FPS, total/preprocess/inference/postprocess timing, model status, the latest raw and stabilized risk, raw/stabilized urgency scores, the active alert profile, active scenario, feedback reason, risk explanation, and the same field-test summary used in Settings.
- Overlay boxes use stronger highlighting for the current risk source and quieter styling for other detected objects. The center region is drawn as an observation reference area, not as a detected object box.

## Risk Reminder Behavior

The app is an assistive prototype, not a safety device that can replace human judgment. Detection results are smoothed before speech and vibration feedback:

- The app does not estimate real-world distance in meters. It only derives relative proximity bands from detection box position and size.
- FAR detections are retained visually but do not trigger speech or vibration.
- MID detections are shown as low-risk visual/status feedback.
- NEAR detections can trigger regular speech and vibration when the risk level is medium or high. Speech prompts are generated from the selected speech style: brief prompts reduce wording, standard prompts keep the balanced guidance phrase, and detailed prompts add the target class when available.
- CRITICAL detections trigger a shorter cooldown and stronger vibration pattern. Center critical prompts use the short guidance phrase “前方很近，放慢”.
- HIGH risk reminders are emitted without frame-delay.
- MEDIUM risk reminders require two consecutive matching direction/message frames.
- A confirmed medium/high reminder is held for up to 600ms if the next frame briefly loses the risk, reducing flicker from transient missed detections.
- Regular near speech and vibration use a 1500ms cooldown in Standard profile; repeated non-critical near reminders within a short window receive a longer effective cooldown to reduce reminder fatigue. Critical high-risk reminders keep the urgent cooldown path and are not suppressed by fatigue control.
- In Quiet profile, medium risks require three matching frames, held alerts last 450ms, near reminders use a 2200ms cooldown with 100ms vibration, and critical reminders use a 1200ms cooldown with 260ms vibration.
- In Standard profile, medium risks require two matching frames, held alerts last 600ms, near reminders use a 1500ms cooldown with 160ms vibration, and critical reminders use an 850ms cooldown with 420ms vibration.
- In Sensitive profile, medium risks confirm on the first frame, held alerts last 800ms, near reminders use a 1000ms cooldown with 220ms vibration, and critical reminders use a 650ms cooldown with 520ms vibration.
- Manual assist scenarios adjust the profile policy without changing the detector or claiming automatic scene understanding: Indoor Slow adds a little hold time and near cooldown, Corridor confirms medium risks sooner and strengthens vibration slightly, Crowded requires more medium-risk confirmation with a longer cooldown, and Outdoor Slow holds alerts longer with clearer vibration.
- The camera panel explains the latest feedback decision in plain language: triggered, unstable, distance too far, cooldown, held alert, disabled feedback, or no feedback risk. This explanation is for transparency and debugging, not a safety certification.

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

当前已补存 v0.1.0、v0.2.0、v0.7.0、v0.8.0、v1.3.0、v1.4.0、v1.5.0、v2.0.0、v2.5.0、v2.6.0、v3.1.0、v3.2.0、v3.3.0、v3.4.0、v3.5.0、v3.6.0、v4.1.0、v4.2.0、v4.3.0、v4.8.0、v5.3.0、v5.8.0 和 v5.9.0 的 debug APK。带 `rebuilt` 的文件表示从对应 Git 历史提交重新构建得到，适合用于演示版本演进；当前 v5.9.0 最近一次测试归档为 `releases/apk/BlindAssist-v5.9.0-debug-20260519-174352.apk`。

## Install to Phone

打开 Android 手机 USB 调试后：

```powershell
.\.android-sdk\platform-tools\adb.exe devices
.\.android-sdk\platform-tools\adb.exe install -r app\build\outputs\apk\debug\app-debug.apk
```
