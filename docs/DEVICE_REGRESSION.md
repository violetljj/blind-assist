# 真机回归说明

当一次变更需要在 APK 交付或发布前留下边界清晰的真机证据时，使用 `scripts/run_device_regression.ps1`。该脚本的范围刻意保持克制：验证安装、启动、包状态，并采集可重复对比的设备快照；这些原始证据默认不进入 Git。

```powershell
.\scripts\run_device_regression.ps1 -ApkPath .\app\build\outputs\apk\debug\app-debug.apk
```

脚本要求当前只有一台在线 ADB 设备。它会安装 APK、清空 `com.linnan.blindassist` 数据、冷启动 `.MainActivity`，处理 Android 兼容提示与 onboarding，并确认已进入相机页、检测已开启、UI 已给出稳定相机语义且两次 `BlindAssistPerf` 的 model-loaded 帧持续增长。随后在默认 90 秒内采样 `BlindAssistPerf`、`gfxinfo`、`meminfo`、UI XML 和截图；模型不可用、未进入相机页或性能帧停止增长都会失败。需要在同一台设备上运行 connected Compose 测试时，可追加 `-RunConnectedAndroidTest`。

常用参数：

- `-ApkPath`：要安装的 APK，默认是 `app\build\outputs\apk\debug\app-debug.apk`。
- `-SampleSeconds`：采样秒数，默认是 `90`。
- `-RunConnectedAndroidTest`：额外运行 connected Compose 测试。
- `-AdbPath`：在仓库本地 Android SDK 不可用时指定某个 `adb.exe`。

输出会写入带时间戳的 `test-artifacts.local/device-regression/<timestamp>/` 目录。这些目录只作为本机后续回归对比证据：保留在工作电脑上，不提交到 Git。

## 无障碍回归矩阵

自动化只能证明声明的语义和交互条件，不证明 TalkBack 用户能够安全、有效地完成任务。
涉及 UI、反馈、导航或输入方式的改动，按变更风险选择下面最小覆盖；未执行的模式必须明确写成
`NOT_TESTED`，不能被一次 connected test 隐含覆盖。

| 模式 | 最小关键流程 | 通过条件 | 保留证据 | 结论上限 |
|---|---|---|---|---|
| Compose/connected 自动化 | 启动、onboarding、进入相机页、主要控件 | semantics、role/state、可点击性和焦点断言通过；适用时启用 Android accessibility checks | 测试任务、设备/API、结果与失败节点 | 只证明被断言的自动条件 |
| TalkBack 手工 | 从冷启动到检测开关、状态理解、退出/恢复 | 读屏顺序可理解；动态状态能被感知；没有只能靠颜色或视觉位置理解的关键动作 | 设备、系统与 TalkBack 版本、步骤、结果、截图/录屏或文字记录 | 只覆盖所测语言、设备和流程 |
| Switch Access / 键盘 / D-pad | 完成同一关键流程 | 所有动作可达；焦点可见且顺序稳定；无触摸专属阻塞 | 输入方式、焦点路径、阻塞点 | 不代表所有辅助开关配置 |
| 字体与显示缩放 | 系统大字体及放大显示下重复关键页 | 文本不裁切；主要操作不重叠、不消失；滚动和状态仍可理解 | 缩放档位与页面截图 | 只证明所测窗口/设备组合 |
| 触控与视觉可辨识性 | 检查关键操作和状态 | 触控目标通常至少 `48dp`；普通文本对比度至少 `4.5:1`、大文本至少 `3:1`；状态不只依赖颜色 | 测量或审计记录 | 设计/静态符合性，不是用户研究 |

建议的专项顺序是：先跑受影响的 Compose/connected 测试，再用 TalkBack 完成一条最关键流程；
只有改动影响焦点、布局、导航或输入时，才追加 Switch Access、键盘/D-pad 与缩放矩阵。
自动检查、人工可达性检查和目标用户可用性研究必须在报告中分开。

报告至少包含：APK/commit、设备与 Android 版本、辅助技术版本、所测语言、窗口/缩放设置、
实际步骤、PASS/FAIL/NOT_TESTED、原始证据目录，以及“未证明什么”。TalkBack 或其他辅助技术
失败时，不得用普通触摸流程成功替代。

如果 `adb install -r` 失败并出现 `INSTALL_FAILED_UPDATE_INCOMPATIBLE`，通常表示手机里已有同包名但不同 debug 签名的旧安装包。先用下面命令确认手机端版本：

```powershell
.\.android-sdk\platform-tools\adb.exe shell dumpsys package com.linnan.blindassist | Select-String -Pattern 'versionCode|versionName'
```

只有在用户确认后，才卸载旧包：

```powershell
.\.android-sdk\platform-tools\adb.exe uninstall com.linnan.blindassist
```

任何 APK 交付前，先归档生成的 APK：

```powershell
.\scripts\archive_apk.ps1
```

该脚本会把 debug APK 复制到 `E:\linnan\blind-assist-apk-archive\apks`，在本地归档清单中记录 SHA256，并可在用户批准后通过 `-Milestone` 同步一份 Git 里程碑 APK 到 `releases/apk`。
