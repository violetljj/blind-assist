# BlindAssist 演示指南

本页用于课程展示、阶段检查和毕业设计答辩。当前演示以真实 Atom RGB、H3 ToF 和
BMI270 IMU 接入为主；具体操作以[硬件辅助与融合演示](HARDWARE_OBSTACLE_DEMO.md)、
[三路数据采集](PHONE_DATA_CAPTURE.md)为准。

## 选择演示内容

| 首页入口 | 展示内容 | 操作说明 |
| --- | --- | --- |
| 融合演示 | 显示 HEAD/BODY 区域、ToF/RGB 来源及确认后的提醒 | [融合演示](HARDWARE_OBSTACLE_DEMO.md) |
| 开始辅助 | 当前 H3 固件上的基础 ToF 提醒；显式开始与结束 | [默认辅助入口](HARDWARE_OBSTACLE_DEMO.md) |
| 数据采集 | Atom RGB、8×8 ToF、IMU 预览，原始记录保存与 ZIP 导出 | [采集步骤与文件内容](PHONE_DATA_CAPTURE.md) |

手机相机与旧设备模拟页面属于其他已有路径。演示时按实际打开的入口说明输入来源，
不要把旧模拟页面或素材回放当成真实硬件连接。

## 演示前准备

1. 按[快速开始](QUICKSTART_ZH.md)准备构建工具；使用已有 APK 时，核对其版本与
   [CHANGELOG.md](../CHANGELOG.md)中的功能记录。源码版本在
   [app/build.gradle.kts](../app/build.gradle.kts)中，不在本页另存一份版本号。
2. 现场融合展示使用 `demo` 包。按硬件指南部署匹配的实验模型与本地运行库；仅安装
   APK 不会自动下载模型。构建入口：

   ```powershell
   pwsh -NoProfile -File scripts/run_android_gradle.ps1 :app:assembleDemo
   ```

   普通 clone 的包位于 `app/build/outputs/apk/demo/app-demo.apk`；配置外置 artifact
   junction 的工作区位于 `artifacts.local/work/gradle-build/app/outputs/apk/demo/app-demo.apk`。
   安装、预编译与所需资产见[硬件指南](HARDWARE_OBSTACLE_DEMO.md)，需要留存的演示包按
   [APK 归档策略](APK_ARCHIVE.md)保存。
3. 手机与硬件接入同一局域网或已配置热点，在“数据采集”中寻找设备并确认启用输入有
   有效画面和读数。H3 只允许一个 TCP 消费者，切换到辅助或融合前先结束当前采集，
   并关闭电脑上的 ToF 预览或采集客户端。
4. 若通过 ADB 核对安装版本，使用本机 Android SDK 的 `adb`：

   ```powershell
   adb shell dumpsys package com.linnan.blindassist | Select-String -Pattern 'versionCode|versionName'
   ```

## 推荐演示顺序

1. 在首页介绍各入口，说明本次使用的硬件、APK 和演示模式。
2. 打开“融合演示”，点“开始演示”后观察 HEAD/BODY、输入来源、未知状态与提醒；
   所用演示规则和冻结评价版本的区别按硬件指南说明。结束后停止演示。
3. 如需展示基础路径，回首页打开“开始辅助”，展示 ToF 状态及语音／震动设置，再结束辅助。
4. 打开“数据采集”，连接后显式开始一段录制，停止保存并导出 ZIP；展示原始文件种类，
   说明采集本身不提供障碍真值。采集时保持页面在前台，已启动的独立 ZIP 导出可以在离页后继续。
5. 展示设置页中的语言、语音、震动偏好。结束时返回首页，并确认辅助或采集已停止。

这些步骤用于展示已有操作；研究数字引用对应报告中的设备、数据范围和分母。
模型推理耗时、手机接收时刻和完整提醒延迟是不同测量，不相互替代。

## 没有硬件时

可以构建并查看首页和设置，结合 [README](../README.md)、[版本记录](../CHANGELOG.md)、
[项目状态](PROJECT_STATE.md)说明功能。展示既有截图、录像、回放或设备报告时，标明其
版本和来源；UI 构建、素材回放或历史测试不证明现场硬件已连接。历史材料从
[历史索引](history-index.md)查找。

## 数据与能力边界

硬件通过局域网向手机传输数据，感知在手机端处理；不能笼统表述为“不联网”。数据采集
会保存原始 JPEG、ToF、IMU 和事件记录，并按用户选择的位置导出 ZIP。基础辅助也提供
显式留证和可选画面抽帧，保存内容与导出行为见各入口指南；不能笼统表述为“不保存画面”。

BlindAssist 是盲杖互补的前视感知研究原型。演示响应、离线评价和真实使用效果分别报告；
`UNKNOWN` 不等于无障碍，设备接收时间匹配不等于传感器曝光同步。提醒受安装、光照、
遮挡、网络和模型影响，当前材料不构成可通行或导航安全证明。
