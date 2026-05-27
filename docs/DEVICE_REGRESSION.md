# 真机回归说明

当一次变更需要在 APK 交付或发布前留下边界清晰的真机证据时，使用 `scripts/run_device_regression.ps1`。该脚本的范围刻意保持克制：验证安装、启动、包状态，并采集可重复对比的设备快照；这些原始证据默认不进入 Git。

```powershell
.\scripts\run_device_regression.ps1 -ApkPath .\app\build\outputs\apk\debug\app-debug.apk
```

脚本要求当前只有一台在线 ADB 设备。它会安装 APK、清空 `com.linnan.blindassist` 数据、冷启动 `.MainActivity`，采集包版本信息，并在默认 90 秒内采样 `BlindAssistPerf`、`gfxinfo`、`meminfo`、UI XML 和截图。需要在同一台设备上运行 connected Compose 测试时，可追加 `-RunConnectedAndroidTest`。

常用参数：

- `-ApkPath`：要安装的 APK，默认是 `app\build\outputs\apk\debug\app-debug.apk`。
- `-SampleSeconds`：采样秒数，默认是 `90`。
- `-RunConnectedAndroidTest`：额外运行 connected Compose 测试。
- `-AdbPath`：在仓库本地 Android SDK 不可用时指定某个 `adb.exe`。

输出会写入带时间戳的 `test-artifacts.local/device-regression/<timestamp>/` 目录。这些目录只作为本机后续回归对比证据：保留在工作电脑上，不提交到 Git。

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
