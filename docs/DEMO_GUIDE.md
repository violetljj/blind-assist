# BlindAssist 演示指南

本文档用于课程展示、阶段检查和毕业设计答辩时快速演示 BlindAssist。它描述的是当前仓库真实功能，不把原型能力描述为安全认证产品。

当前仓库版本：`v10.9.0` / `versionCode=37`。演示前必须用下方命令核对实际安装包；如果现场使用历史 APK，应说明对应版本和功能差异。构建或测试是否通过以本次演示前的新鲜输出为准，不沿用旧报告中的通过结论。

## 1. 演示目标

- 说明 BlindAssist 是 Android Kotlin + Compose 的本地助盲避障原型。
- 展示手机摄像头实时识别、检测框覆盖、语音/震动提醒、现场测试摘要和用户偏好设置。
- 强调隐私与安全边界：摄像头画面本地处理，不上传、不联网、不保存视频；提醒只作为辅助参考。

## 2. 演示前准备

以下命令均从仓库根 `E:\linnan\linnan` 执行。

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
versionCode=37
versionName=10.9.0
```

## 3. 推荐演示顺序

1. 打开 App，展示启动页和首次使用引导。
2. 进入“功能”页，先说明日常使用向导和一键预设：
   - 通用日常、室内慢行、走廊通行、密集区域、户外慢行会组合调整场景、提醒档位、语音风格、震动强度和 Care Mode。
   - 手动调整后若不再匹配预设，界面会显示为自定义组合。
3. 说明两个功能入口：
   - “使用手机摄像头”：当前可用的本地检测路径。
   - “眼镜设备模拟中心”：正式版可见的交互演示，所有状态均明确为模拟，不扫描蓝牙、不联网、不连接真实眼镜。
4. 进入模拟中心，依次演示“模拟连接”（800ms 后 82%）、“模拟低电量”（15%）、“模拟断连”和“重置模拟”。debug 版本连接后可选择 HIGH/MEDIUM/LOW/NONE 本地素材并开始离线回放；release 不显示回放入口。
5. 返回功能页，点击“使用手机摄像头”，进入相机权限说明。
6. 授权后进入沉浸式相机页，展示：
   - 实时预览和检测框。
   - 风险指导语、目标行、提醒档位。
   - 检测、语音、震动、场景、Care Mode、调安静和调敏感快捷控制。
   - `持续检测中 / Monitoring` 只表示未达到提醒等级；必须说明这不代表环境安全。
7. 打开调试信息，展示 FPS、推理耗时、风险判定和现场测试摘要。
8. 回到主界面，进入“设置”页，展示：
   - 语音提醒、震动提醒、关怀模式、调试信息。
   - 提醒档位、语音风格、震动强度。
   - 现场测试摘要和“查看新手引导”。
9. 进入“个人主页”，展示本地原型状态、当前版本和辅助偏好。

## 4. 无设备 fallback

如果现场没有连接手机或 ADB 不稳定：

- 使用 [APK_ARCHIVE.md](APK_ARCHIVE.md) 的收据清单和外部归档说明历史 APK；`releases/apk/` 不保存原始 APK。
- 打开 [CHANGELOG.md](../CHANGELOG.md) 说明版本演进。
- 打开 [历史阶段进度说明](history/project-materials/PROJECT_PROGRESS_REVIEW.md) 讲述早期阶段，并明确它不是当前状态。
- 打开 [README.md](../README.md) 说明当前功能、模型资产、构建方式和风险提醒策略。
- 打开 [DEVICE_REGRESSION.md](DEVICE_REGRESSION.md) 说明当前真机回归脚本和本地证据目录规则。
- 可展示 Compose 仪器测试和 Gradle 构建记录，证明 UI 路径和核心逻辑经过验证。
- debug APK 可用离线回放展示背景图、真实检测框、风险解释和 session 摘要；应说明素材回放不等同于实际场景验证。

## 5. 隐私与安全边界表述

推荐说明：

> BlindAssist 当前是本地运行的助盲避障原型。摄像头画面只在手机端用于实时识别，不上传、不联网、不保存视频。语音和震动提醒受光照、遮挡、模型识别结果和设备性能影响，只能作为辅助参考，不能替代盲杖、导盲犬、人工判断或专业安全设备。

避免说明：

- “可以保证避障安全”
- “可以替代盲杖或导盲犬”
- “能识别所有障碍物”
- “已经完成真实眼镜设备连接”

## 6. 验收材料清单

- [README.md](../README.md)：当前状态、使用方式、模型和构建说明。
- [CHANGELOG.md](../CHANGELOG.md)：真实版本路线和 APK 归档路径。
- [历史阶段进度说明](history/project-materials/PROJECT_PROGRESS_REVIEW.md)：早期阶段回顾，不作为当前状态。
- [DEVELOPMENT_LOG.md](../DEVELOPMENT_LOG.md)：近期开发、验证和版本判断记录。
- [DEVICE_REGRESSION.md](DEVICE_REGRESSION.md)：真机安装、冷启动、截图、UI dump 和性能采样脚本说明。
- [APK_ARCHIVE.md](APK_ARCHIVE.md)：Git 收据与外部 APK 归档位置。
