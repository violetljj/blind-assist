# 高度标定采集 App

这是独立于 BlindAssist 正式提示链、也独立于旧 ARCore/TFLite benchmark APK 的轻量现场
采集工具。安装 `debug` APK 后，桌面入口名称为“高度标定采集”。

## 现场流程

1. 选择 P0（120 帧）或 R2（25 帧），确认自动生成的 Session ID 唯一，并填写固定支架编号。
2. 用卷尺现场测量相机镜头光心高度；输入米制高度和不超过 0.02 m 的不确定度。
3. 选择独立卷尺/激光参考 JSON 或文本清单。App 自动复制并计算 SHA-256。
4. 检查启动条件卡片全部通过，点击“开始采集”。采集期间保持支架不动。
5. 完成后点击“导出 ZIP 采集包”，再交给 host preflight。App 显示“完成”不代表 P0/R2 通过。

主动停止、CameraX/文件错误或时间戳不递增都会写 `capture_hold.json`，部分帧不得进入评价。
有效完成目录包含 `receipt.json`、`frames.json`、`intrinsics.json`、哈希 PNG 和复制后的独立
参考清单。该 App 不产生导航提示、不修改正式 App 状态，也不授权 production。

## 验证

```powershell
.\gradlew.bat :known-height-capture-app:testDebugUnitTest `
  :known-height-capture-app:assembleDebug `
  :known-height-capture-app:assembleDebugAndroidTest
```

真机 UI 合同测试类：
`com.linnan.blindassist.ustrfbenchmark.KnownHeightCaptureActivityTest`。
