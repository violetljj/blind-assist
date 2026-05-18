# BlindAssist Demo Guide

本文档用于课程展示、阶段检查和毕业设计答辩时快速演示 BlindAssist。它描述的是当前仓库真实功能，不把原型能力描述为安全认证产品。

## 1. 演示目标

- 说明 BlindAssist 是 Android Kotlin + Compose 的本地助盲避障原型。
- 展示手机摄像头实时识别、检测框覆盖、语音/震动提醒、现场测试摘要和用户偏好设置。
- 强调隐私与安全边界：摄像头画面本地处理，不上传、不联网、不保存视频；提醒只作为辅助参考。

## 2. 演示前准备

1. 确认 Android 手机已打开 USB 调试或 ADB over Wi-Fi。
2. 确认当前 APK 已安装：

```powershell
.\.android-sdk\platform-tools\adb.exe devices
.\.android-sdk\platform-tools\adb.exe install -r app\build\outputs\apk\debug\app-debug.apk
```

3. 核对版本：

```powershell
.\.android-sdk\platform-tools\adb.exe shell dumpsys package com.linnan.blindassist | Select-String -Pattern 'versionCode|versionName'
```

预期当前演示版本为：

```text
versionCode=19
versionName=4.3.0
```

## 3. 推荐演示顺序

1. 打开 App，展示启动页和首次使用引导。
2. 进入“功能”页，说明两个入口：
   - “使用手机摄像头”：当前可用的本地检测路径。
   - “连接眼镜设备”：未来扩展占位，当前不申请蓝牙权限。
3. 说明当前已暂时移除 App 内“项目展示中心”，主界面只保留实际可用入口和安全边界；课堂材料改由 `CHANGELOG.md`、`README.md`、`PROJECT_PROGRESS_REVIEW.md` 和 APK 归档展示。
4. 点击“使用手机摄像头”，进入相机权限说明。
5. 授权后进入沉浸式相机页，展示：
   - 实时预览和检测框。
   - 风险指导语、目标行、提醒档位。
   - 检测、语音、震动、关怀模式开关。
6. 打开调试信息，展示 FPS、推理耗时、风险判定和现场测试摘要。
7. 回到主界面，进入“设置”页，展示：
   - 语音提醒、震动提醒、关怀模式、调试信息。
   - 提醒档位、语音风格、震动强度。
   - 现场测试摘要和“查看新手引导”。
8. 进入“个人主页”，展示本地原型状态、当前版本和辅助偏好。

## 4. 无设备 fallback

如果现场没有连接手机或 ADB 不稳定：

- 使用 `releases/apk/` 展示不同版本 APK 的归档列表。
- 打开 `CHANGELOG.md` 说明版本演进。
- 打开 `PROJECT_PROGRESS_REVIEW.md` 说明项目阶段进度。
- 打开 `README.md` 说明当前功能、模型资产、构建方式和风险提醒策略。
- 可展示 Compose 仪器测试和 Gradle 构建记录，证明 UI 路径和核心逻辑经过验证。

## 5. 隐私与安全边界表述

推荐说明：

> BlindAssist 当前是本地运行的助盲避障原型。摄像头画面只在手机端用于实时识别，不上传、不联网、不保存视频。语音和震动提醒受光照、遮挡、模型识别结果和设备性能影响，只能作为辅助参考，不能替代盲杖、导盲犬、人工判断或专业安全设备。

避免说明：

- “可以保证避障安全”
- “可以替代盲杖或导盲犬”
- “能识别所有障碍物”
- “已经完成真实眼镜设备连接”

## 6. 验收材料清单

- `README.md`：当前状态、使用方式、模型和构建说明。
- `CHANGELOG.md`：真实版本路线和 APK 归档路径。
- `PROJECT_PROGRESS_REVIEW.md`：阶段进度回顾材料。
- `DEVELOPMENT_LOG.md`：逐次开发、验证和版本判断记录。
- `releases/apk/`：按版本留存的 debug APK。
