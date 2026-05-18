# BlindAssist Changelog

本文件按真实版本记录 BlindAssist 的功能演进、验证证据和可展示 APK 归档。它用于课程汇报、答辩材料整理和版本对比，不替代 `DEVELOPMENT_LOG.md` 的逐次工作记录。

## v4.2.0 - 场景化提醒与风险解释

- 状态：已完成，`versionCode=18`，`versionName=4.2.0`。
- 主要变化：
  - 新增手动 `使用场景` 偏好：通用、室内慢行、走廊通行、密集区域、户外慢行。
  - 通用场景保持 v4.1.0 提醒行为；其他场景只调整规则层的中风险确认、提醒保持、近距冷却和震动时长。
  - 相机控制面板显示当前场景和最近风险解释，说明已触发、未稳定、距离较远、冷却中、提醒保持或暂无可反馈风险等原因。
  - 现场测试摘要追加当前场景和最近解释；Care Mode 下保留简短解释，不暴露性能调试细节。
  - 本轮不新增自动场景识别、联网、定位、蓝牙、存储权限、模型变更或大型架构框架。
- 验证：
  - `:app:testDebugUnitTest` 和 `:app:assembleDebug` 构建验证通过。
  - debug APK 已安装到 `SM-S9280`，包信息核对为 `versionCode=18`、`versionName=4.2.0`。
  - Compose 仪器测试增加使用场景选择和相机页解释区域覆盖；本轮 `connectedDebugAndroidTest` 已尝试执行，但设备处于锁屏/Bouncer 状态，报告 `No compose hierarchies found in the app`，未作为通过证据。
- APK：
  - `releases/apk/BlindAssist-v4.2.0-debug-20260519-000200.apk`

## v4.1.0 - 展示交付加强

- 状态：已完成，`versionCode=17`，`versionName=4.1.0`。
- 主要变化：
  - 新增 App 内“项目展示中心”，集中说明本地识别、语音/震动提醒、现场测试摘要和原型安全边界。
  - 展示中心提供“开始演示”和“查看引导”操作，复用现有手机摄像头权限说明与新手引导流程。
  - 新增 `DEMO_GUIDE.md`，整理课堂/答辩演示脚本、环境准备、无设备 fallback、隐私与安全边界。
  - 扩展 Compose 仪器测试，覆盖底部导航、展示中心、相机演示入口和新手引导入口。
- 验证：
  - `:app:testDebugUnitTest`、`:app:assembleDebug` 和 `:app:assembleDebugAndroidTest` 构建验证通过。
  - 重新连接设备后，`connectedDebugAndroidTest` 已在 `SM-S9280 - 16` 上完成 4 个 Compose 仪器测试并通过。
  - debug APK 已安装到 `SM-S9280`，包信息核对为 `versionCode=17`、`versionName=4.1.0`。
- APK：
  - `releases/apk/BlindAssist-v4.1.0-debug-20260518-231542.apk`

## v3.6.0 - 日常使用体验增强

- 主要变化：
  - 新增语音风格：简短、标准、详细。
  - 新增震动强度：轻柔、标准、强。
  - 新增非迫近近距提醒疲劳控制，减少连续提醒打扰。
  - Overlay 检测框增加显示层平滑，风险规则近距阈值略收紧。
  - 新增并修复最小 Compose 仪器测试宿主。
- 验证：
  - JVM 单元测试、debug APK 构建、`connectedDebugAndroidTest` 和手机安装均通过。
- APK：
  - `releases/apk/BlindAssist-v3.6.0-debug-20260518-214947.apk`

## v3.5.0 - ViewModel 与 StateFlow 轻量状态拆分

- 主要变化：
  - Compose 可观察状态集中到 `BlindAssistViewModel`，通过只读 `StateFlow` 暴露。
  - `MainActivity` 保留 CameraX、权限、TFLite、反馈控制和生命周期边界。
- 验证：
  - JVM 单元测试、debug APK 构建和手机安装均通过。
- APK：
  - `releases/apk/BlindAssist-v3.5.0-debug-20260518-193819.apk`

## v3.4.0 - 现场测试摘要与无障碍语义

- 主要变化：
  - 新增内存态现场测试摘要，展示运行时长、风险次数、提醒次数、FPS、推理耗时和当前提醒档位。
  - 设置页、相机控制区和摘要标题补充更自然的 TalkBack 语义。
- 验证：
  - JVM 单元测试、debug APK 构建和手机安装均通过。
- APK：
  - `releases/apk/BlindAssist-v3.4.0-debug-20260518-192333.apk`

## v3.3.0 - 首次使用引导与相机权限说明

- 主要变化：
  - 新增三页新手引导，说明手机摄像头本地识别、语音/震动提醒和原型安全边界。
  - 相机权限请求前增加应用内解释，说明不上传、不联网、不保存视频。
- 验证：
  - JVM 单元测试和 debug APK 构建通过；该轮手机安装因 ADB 无设备未完成。
- APK：
  - `releases/apk/BlindAssist-v3.3.0-debug-20260518-154943.apk`

## v3.2.0 - 相机返回路径与个人主页精简

- 主要变化：
  - 相机沉浸页支持系统返回手势，统一关闭相机并回到主界面。
  - 个人主页移除展示说明卡，保留用户、设备、版本和偏好状态。
- 验证：
  - JVM 单元测试、debug APK 构建和手机安装均通过。
- APK：
  - `releases/apk/BlindAssist-v3.2.0-debug-20260518-152635.apk`

## v3.1.0 - Compose 应用壳层与界面革新

- 主要变化：
  - 引入 Compose + Material 3 主壳、品牌启动页、底部导航、功能页、个人主页、设置页和沉浸式相机子页。
  - 保留原有 CameraX、TFLite、风险分析、提醒档位、语音和震动链路。
- 验证：
  - JVM 单元测试、debug APK 构建和手机安装均通过。
- APK：
  - `releases/apk/BlindAssist-v3.1.0-debug-20260518-151146.apk`

## v2.6.0 - 显示可信度打磨

- 主要变化：
  - 区分当前帧检测和短暂保持提醒。
  - 默认用户文案隐藏数值 urgency，调试信息保留详细指标。
  - 中心区域改为观察参考区，避免误解为真实检测框。
- 验证：
  - debug APK 构建和手机安装均通过。
- APK：
  - `releases/apk/BlindAssist-v2.6.0-debug-71f921d-current.apk`

## v2.5.0 - 现场可测助行体验

- 主要变化：
  - 新增纯 Kotlin `AssistEngine` 和 `SessionTrace`，把检测结果、风险分析、稳定策略和反馈决策串成可测试会话层。
  - 调试区显示最近会话摘要，近处和迫近语音更偏行动提示。
- 验证：
  - JVM 单元测试、debug APK 构建和手机安装均通过。
- APK：
  - `releases/apk/BlindAssist-v2.5.0-debug-e803d1f-rebuilt.apk`

## v2.0.0 - 提醒档位与 CameraX API 更新

- 主要变化：
  - 新增 Quiet、Standard、Sensitive 三档提醒策略。
  - 风险稳定、语音冷却、震动时长随档位调整。
  - CameraX 分析分辨率迁移到 `ResolutionSelector`。
- 验证：
  - JVM 单元测试、debug APK 构建和手机安装均通过。
- APK：
  - `releases/apk/BlindAssist-v2.0.0-debug-52a0c93-rebuilt.apk`

## v1.5.0 - 用户偏好持久化

- 主要变化：
  - 持久化语音提醒、震动提醒和关怀模式。
  - 检测开关保持 session-only，每次启动默认开启。
- 验证：
  - JVM 单元测试、debug APK 构建和手机安装均通过。
- APK：
  - `releases/apk/BlindAssist-v1.5.0-debug-f6b6d5e-rebuilt.apk`

## v1.4.0 - 相机画面填充与覆盖层映射

- 主要变化：
  - 预览画面改为填满屏幕，减少顶部黑边。
  - Overlay 坐标映射对齐填充裁剪后的预览。
- 验证：
  - debug APK 构建通过。
- APK：
  - `releases/apk/BlindAssist-v1.4.0-debug-f96c6f7-rebuilt.apk`

## v1.3.0 - 相机界面重设计

- 主要变化：
  - 新增品牌/状态头部、风险徽章、两行控制区和关怀模式。
  - 关怀模式放大指导语、提高对比度并隐藏调试细节。
- 验证：
  - debug APK 构建通过。
- APK：
  - `releases/apk/BlindAssist-v1.3.0-debug-e29b99a-rebuilt.apk`

## v0.8.0 - 实时界面交互升级

- 主要变化：
  - 分离主风险状态、控制开关和可折叠调试信息。
  - 改善检测、语音、震动开关的可访问描述。
- 验证：
  - debug APK 构建通过。
- APK：
  - `releases/apk/BlindAssist-v0.8.0-debug-4bf9ad2-rebuilt.apk`

## v0.7.0 - 相对距离风险提醒

- 主要变化：
  - 新增 FAR、MID、NEAR、CRITICAL 相对距离分层和 urgency score。
  - FAR/MID 主要用于视觉状态，NEAR/CRITICAL 才进入语音和震动提醒路径。
- 验证：
  - JVM 单元测试、debug APK 构建和手机安装均通过。
- APK：
  - `releases/apk/BlindAssist-v0.7.0-debug-d948f6b-rebuilt.apk`

## v0.2.0 - 风险提醒稳定化

- 主要变化：
  - 新增 `RiskStabilizer`，高风险单帧确认，中风险需要连续帧确认。
  - 短暂漏检时保持已确认提醒，减少语音/震动抖动。
- 验证：
  - JVM 单元测试和 debug APK 构建通过。
- APK：
  - `releases/apk/BlindAssist-v0.2.0-debug-fb937da-rebuilt.apk`

## v0.1.0 - 第一版本地检测原型

- 主要变化：
  - CameraX 获取实时摄像头画面。
  - 本地加载 YOLO11n TFLite 模型，解析检测框并绘制 overlay。
  - 初版规则层生成助盲避障提醒。
- 验证：
  - debug APK 构建通过。
- APK：
  - `releases/apk/BlindAssist-v0.1.0-debug-958d5a9-rebuilt.apk`
