# BlindAssist v5.8.0 真机完整测试与 v5.9.0 修复复测报告

> 状态说明：这是 2026-05-19 的历史真机测试报告，用于保留当时的失败、修复和复测证据。报告中记录的 v5.8.0 遗留问题已在 v5.9.0 修复复测；当前工程版本和最新真机验证状态请以 `README.md`、`CHANGELOG.md` 和 `DEVELOPMENT_LOG.md` 为准。

## v5.9.0 修复复测补充

- 修复时间：2026-05-19 17:45 +08:00
- 修复版本：`versionName=5.9.0`，`versionCode=23`
- 修复 APK：`releases/apk/BlindAssist-v5.9.0-debug-20260519-174352.apk`

本轮已修复 v5.8.0 真机完整测试中暴露的两个问题：

- `connectedDebugAndroidTest` 已从 5 executed / 2 failures / `Process crashed` 修复为 6 tests / 0 failures / 0 errors / 0 skipped。
- 相机页 Debug 控件已从底部 `TextButton` 调整为稳定的 `CompactAction` 按钮，并暴露 `展开相机调试信息` / `收起相机调试信息` content description；ADB/UIAutomator 复测可以点击展开并观察到收起状态。

复测证据：

- JVM 测试：105 tests，0 failures，0 errors，0 skipped。
- Connected Compose 测试报告：`app/build/outputs/androidTest-results/connected/debug/TEST-SM-S9280 - 16-_app-.xml`，6 tests，0 failures。
- 安装包版本核对：`versionCode=23 minSdk=26 targetSdk=35`，`versionName=5.9.0`。
- 冷启动：`Status: ok`，`LaunchState: COLD`，`TotalTime: 676`，`WaitTime: 679`。
- Debug 控件 ADB 复测：`test-artifacts/2026-05-19-v5.8.0-device-test/v5.9.0_final_debug_button_retest.txt`。

- 测试时间：2026-05-19 17:04-17:24 +08:00
- 测试设备：Samsung SM-S9280，Android 16，wireless ADB
- 固定 serial：`adb-R5CX10M8Y8X-nkVxqz (2)._adb-tls-connect._tcp`
- 应用版本：`versionName=5.8.0`，`versionCode=22`
- 测试 APK：`app/build/outputs/apk/debug/app-debug.apk`
- 本次归档 APK：`releases/apk/BlindAssist-v5.8.0-debug-20260519-170622.apk`
- 证据目录：`test-artifacts/2026-05-19-v5.8.0-device-test/`

## 结论

本轮完成了清数据后的真机构建、安装、首启、引导、导航、设置、语言切换、眼镜占位、相机权限、相机页、90 秒性能采样和稳定性检查。核心 App 可启动、可进入相机页，CameraX/TFLite 链路在 90 秒采样内持续输出性能日志，未发现 crash 或 ANR。

需要后续处理的风险有两个：

- `connectedDebugAndroidTest` 未通过：设备端 Compose 测试执行 5 个用例，其中 2 个失败，随后 instrumentation 报 `Process crashed`。
- 相机页底部 `展开调试信息` 节点存在，但在 SM-S9280 当前 1440x3120 竖屏布局中，ADB 多次点击其 clickable parent 未触发展开；疑似底部交互区域过低或自动化命中不稳定，需要 UI 布局/触控区域复查。

本轮是测试与文档更新，不改变业务代码、模型、权限、构建配置或版本号；项目版本保持 `v5.8.0` / `versionCode=22`。

## 构建与资产检查

| 项目 | 结果 |
| --- | --- |
| `git status --short` 初始检查 | 仅发现既有未跟踪 PPTX，本轮未处理该文件 |
| 模型检查 | 通过 |
| TFLite input | `images` `[1, 320, 320, 3]` `float32` |
| TFLite output | `Identity` `[1, 84, 2100]` `float32` |
| Gradle 命令 | `.\gradlew.bat :app:testDebugUnitTest :app:assembleDebug --no-daemon` |
| Gradle 结果 | `BUILD SUCCESSFUL in 31s` |
| JVM 测试 | 105 tests, 0 failures, 0 errors, 0 skipped |
| Debug APK 大小 | 47,068,480 bytes |
| APK 归档 | `releases/apk/BlindAssist-v5.8.0-debug-20260519-170622.apk` |

Gradle 验证按仓库已知沙箱限制直接提权执行，未重复制造普通沙箱失败。

## 真机功能与 UI 检查

| 场景 | 结果 | 证据 |
| --- | --- | --- |
| 清空应用数据 | 通过，`pm clear com.linnan.blindassist` 输出 `Success` | 命令输出 |
| 安装 APK | 通过，`install -r` 输出 `Success` | 命令输出 |
| 包版本验证 | 通过，`versionCode=22` / `versionName=5.8.0` | `dumpsys package` |
| 冷启动 | 通过，`Status: ok`，`LaunchState: COLD`，`TotalTime=692ms` | `am start -W` |
| 首次引导 | 通过，覆盖本地摄像头识别、语音/震动、不能替代人工判断 | `ui_01` 到 `ui_06` XML/截图 |
| Features 页 | 通过，日常使用向导、手机摄像头、眼镜占位入口可见 | `ui_07_main_features.xml` |
| Profile 页 | 通过，用户和版本状态可见 | `ui_09_profile_page.xml` |
| Settings 页 | 通过，语音、震动、关怀、调试、场景、引导入口可见 | `ui_11` 到 `ui_18` |
| 中英文切换 | 通过，`English` 后显示 `Speech reminders`，切回 `Chinese` 后显示中文 | `ui_16` / `ui_18` |
| 日常模式 | 通过，点击 `走廊通行` 后可进入后续流程；快捷提醒调整后显示自定义组合符合设计 | `ui_21` / `ui_after_camera_back.xml` |
| 眼镜占位 | 通过，打开并关闭占位弹窗，未观察到蓝牙权限申请 | `ui_24_glasses_dialog.xml` |
| 相机权限说明 | 通过，先显示 App 内说明，再进入 Android 权限弹窗 | `ui_27` / `ui_29` |
| 相机页 | 通过，返回、检测/语音/震动、调安静/调敏感、场景、日常模式、风险解释可见 | `ui_30_camera_page.xml` |
| 调安静/调敏感 | 通过，两个快捷操作可点击 | `ui_32` / `ui_34` |
| 相机返回 | 通过，点击返回后回到 Features 页，CameraX 页面退出 | `ui_after_camera_back.xml` |
| Debug 展开 | 未通过，节点存在但 ADB 点击未触发展开 | `ui_debug_after_parent_tap.xml` |

## 设备端 Compose 测试

命令：

```powershell
$env:JAVA_HOME='C:\Program Files\Android\Android Studio\jbr'
$env:PATH="$env:JAVA_HOME\bin;$env:PATH"
$env:ANDROID_SERIAL='adb-R5CX10M8Y8X-nkVxqz (2)._adb-tls-connect._tcp'
.\gradlew.bat :app:connectedDebugAndroidTest --no-daemon
```

结果：失败。

- 报告 XML：`app/build/outputs/androidTest-results/connected/debug/TEST-SM-S9280 - 16-_app-.xml`
- 已执行：5 tests
- 失败：2 failures
- 通过：`settingsScreenChangesFeedbackDetailControls`、`featureDailyGuideAppliesCorridorModeToSettings`、`mainShellBottomNavigationSwitchesTopLevelPages`
- 失败 1：`cameraPanelShowsScenarioAndRiskExplanationWhenCameraPathOpens`，`ComposeTimeoutException: Condition still not satisfied after 5000 ms`，位置 `BlindAssistComposeTest.kt:120`
- 失败 2：`phoneCameraEntryUsesExistingCameraPath`，XML failure 节点为空，随后 system-err 记录 `Instrumentation run failed due to Process crashed`
- 设备在测试失败后回到 Launcher，且测试流程卸载了 `com.linnan.blindassist`；随后已重新安装同一个 debug APK 继续性能采样。

## 性能与稳定性

采样方式：进入相机页后清空 logcat，连续运行约 90 秒，采集 `BlindAssistPerf`、`gfxinfo framestats`、`meminfo`、窗口焦点和 logcat。

| 指标 | 样本 | 平均 | P95 | 最大 | 最小 |
| --- | ---: | ---: | ---: | ---: | ---: |
| totalMs | 88 | 55.40ms | 72ms | 91ms | 40ms |
| preMs | 88 | 16.38ms | 22ms | 25ms | 11ms |
| inferMs | 88 | 37.76ms | 54ms | 64ms | 27ms |
| postMs | 88 | 0.01ms | 0ms | 1ms | 0ms |
| fps | 88 | 14.97 | 15.5 | 15.8 | 14.3 |
| detections | 88 | 0 | 0 | 0 | 0 |

采样期间画面未检测到目标，所有 `BlindAssistPerf` 样本为：

- profile：`standard`
- scenario：`general`
- rawRisk：`NONE/NONE/FAR`
- stableRisk：`NONE/NONE/FAR`
- feedbackReason：`距离较远`
- model status：`模型已加载`

渲染和内存：

- `gfxinfo`：总渲染帧数 `1920`
- 卡顿帧：`20 (1.04%)`
- 50/90/95/99 分位耗时：`9ms / 11ms / 12ms / 16ms`
- 丢失 Vsync 次数：`10`
- `meminfo`: TOTAL PSS `269,790 KB`，TOTAL RSS `397,904 KB`
- Java Heap `24,508 KB`，Native Heap `61,372 KB`，Graphics `81,948 KB`
- 窗口焦点保持在 `com.linnan.blindassist/.MainActivity`
- crash/ANR 关键词匹配数：`0`

## 遗留风险与建议

- 优先修复或复查 `connectedDebugAndroidTest` 失败。最可疑路径是相机权限/相机页等待条件与真实设备状态不一致，第二个 failure 为空且伴随 `Process crashed`，需要单独跑 `am instrument` 或加更细 logcat 过滤定位。
- 检查相机页底部控制面板。`展开调试信息` 的 clickable parent 为 `[98,2854][551,3022]`，位置贴近底部系统区域，ADB 点击未生效；建议提高按钮位置、减少面板高度、增加滚动容器或把 Debug 控制移到更稳定的可触达区域。
- 本轮 ADB 自动化没有真实摆放障碍物，因此不评价 YOLO 识别精度、避障有效性或安全可靠性，只验证当前摄像头画面下的相机、推理、提醒状态和性能链路。
- 后续做人工实景测试时，建议用人、椅子、桌角、走廊正前方、侧向经过等安全场景分别采集 `BlindAssistPerf` 与截图，记录 speech/vibration 是否符合预期。
