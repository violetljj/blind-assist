# Device Regression

Use `scripts/run_device_regression.ps1` for bounded real-device smoke and performance evidence before publishing an APK.

```powershell
.\scripts\run_device_regression.ps1 -ApkPath .\app\build\outputs\apk\debug\app-debug.apk
```

The script requires exactly one online ADB device. It installs the APK, clears `com.linnan.blindassist`, cold-starts `.MainActivity`, captures package/version dumps, and samples `BlindAssistPerf`, `gfxinfo`, `meminfo`, UI XML, and screenshots for 90 seconds. Add `-RunConnectedAndroidTest` when the attached device should also run `connectedDebugAndroidTest`.

Outputs are written to a timestamped `test-artifacts.local-device-regression-*` directory. These directories are forward local evidence only: keep them on the workstation, use them when comparing future regressions, and do not commit them to Git.

Archive generated APKs before any release handoff:

```powershell
.\scripts\archive_apk.ps1
```

This copies the debug APK into `E:\linnan\blind-assist-apk-archive\apks`, records SHA256 in the local archive manifest, and can also sync a user-approved milestone copy into `releases/apk` with `-Milestone`.
