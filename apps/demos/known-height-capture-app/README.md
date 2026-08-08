# 高度标定采集 App

这是独立于 BlindAssist 正式提示链、也独立于旧 ARCore/TFLite benchmark APK 的轻量现场
采集工具。安装 `debug` APK 后，桌面入口名称为“高度标定采集”。

## 默认现场流程（开发数据）

1. 把手机后摄镜头中心实际架高到离地 `80–220 cm`。`15 cm` 低支架不在当前协议范围，
   不能用于这条路线。首次填写支架名称和实际镜头高度；App 会显示米制换算并保存。
2. 打开三星“快速测量”，读取当前目标的厘米距离。
3. 返回后填入当前距离，点击“采集当前目标”；每个目标采 25 帧。
4. 完成后点“采下一个目标”，只需更换距离，支架信息不再重复填写。
5. 导出 ZIP 采集包。快速测量是同机 AR 参考，因此数据固定标记为
   `DEVELOPMENT_ONLY`，不冒充 P0/R2 独立金标。

主动停止、CameraX/文件错误或时间戳不递增都会写 `capture_hold.json`，部分帧不得进入评价。
有效完成目录包含 `receipt.json`、`frames.json`、`intrinsics.json`、哈希 PNG，以及由现场
表单自动生成的 `reference/reference.json`。该 App 不产生导航提示、不修改正式 App 状态，
也不授权 production。

## 三星“快速测量”自动导入

手机固定后，可以让电脑自动等待“快速测量”的稳定读数、保存截图、切回采集 App，
并把厘米换算成米填入近/中/远字段。运行后只需在手机上完成快速测量的校正：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  .\apps\demos\known-height-capture-app\quick-measure-import.ps1 `
  -LaunchQuickMeasure -Slot Near
```

后两次把 `Near` 改成 `Middle`、`Far`。脚本要求连续两帧读数一致才接受，并在
`artifacts.local/quick-measure-captures/` 保存原始截图和文字回执；识别失败时不会切换
或填写采集 App。

## 验证

```powershell
.\gradlew.bat :known-height-capture-app:testDebugUnitTest `
  :known-height-capture-app:assembleDebug `
  :known-height-capture-app:assembleDebugAndroidTest
```

真机 UI 合同测试类：
`com.linnan.blindassist.ustrfbenchmark.KnownHeightCaptureActivityTest`。
