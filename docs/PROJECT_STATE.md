# 项目现在做到哪里

更新：2026-10-13（本轮任务始于 10 月 12 日）。研究决定以[总决定](CURRENT_DECISION.md)为准；实机研究见[硬件当前状态](../research/active/hardware-bringup/CURRENT.md)。

## 当前可用与本轮接入

当前 Android v10.19.0 支持 Atom RGB、H3 完整 8×8×16 CNH 与 BMI270 IMU 三路采集、原始数据保存、连接异常记录和后台 ZIP 导出。首页“开始辅助”在当前 H3 固件上运行基础 ToF 提醒；旧 UDP 时序协议保留 A+LOCAL 实验模式。[采集说明](PHONE_DATA_CAPTURE.md) · [辅助入口](HARDWARE_OBSTACLE_DEMO.md)

本轮 **58 / 10.19.0 已安装，工程实验待佩戴验证**：从“数据采集 → 打开融合实验”进入独立页面，再点“开始实验”才连接和推理；显示 HEAD/BODY 的 ToF 强档或 RGB 融合，并作最近 3 帧至少 2 帧确认。提供低头空地高度标定，离开页面停止；实验页无语音震动，不替换默认辅助。

构建、lint（0 errors / 88 warnings）、APK 签名与 16 KB 检查通过，4 项模型/数值设备检查和默认入口检查通过。DAV2 三次模型推理 121–156 ms，不是整机延迟；首次真实流 8 帧均超出姿态范围，未执行 RGB 分支，因此端到端融合仍待验证。[本轮记录](../research/active/hardware-bringup/MOBILE_FUSION_INTEGRATION_20261012.md)

10月13日后续[工程修复](../research/active/hardware-bringup/REAL_FUSION_ENGINEERING_DEV_20261013.md)完成源码交付：手机因果RGB配对、输入代次/失败重试，及标签前输入预检；7项JVM与11项预检检查通过。尚未覆盖安装，以上58 / 10.19.0设备证据不包含本次修复。

## 研究结论与关键缺口

- **ToF**：保留原 M3/5 格基线。模拟默认候选与实机俯角/地面/噪声联合栈分别记录；已消费真实回放中墙、柱、家具响应较明显，细竖杆和持续轻提醒仍是瓶颈。[实机联合栈](../research/active/hardware-bringup/REAL_POSE_FLOOR_NOISE_DR_DEV_20261012.md)
- **RGB**：已标定 Atom 相机并估计相机/ToF 旋转；DAV2-Small 相对深度配合 ToF/地面尺度锚定，在已消费片段中补到部分 ToF 漏掉的细杆。[小模型报告](../research/active/hardware-bringup/REAL_RGB_SMALL_MODEL_DEV_20261012.md)。既有手机网络推理计时不等于整机端到端延迟。
- **融合**：真实头戴回放已显示互补，但尚无新头戴冻结检验结论。本轮 Android 接入是工程迁移，不自动证明与离线脚本等价或新场景有效。半合成 ToF 的 v4 中带排序结果保留其原证据身份，不替代实机验证。

## 接下来

数据研究可先复用现有素材：[已整理5407张未缓存RGB，并实际读通VisTa3D细结构参考小切片](../research/active/hardware-bringup/DATA_POTENTIAL_DEV_20261013.md)。优先完整RGB时序与细结构距离检查；不以新采集为这些Development工作的前提。

1. 用户佩戴设备，低头空地标定安装高度，再在有效姿态下检查实时融合；软件工程检查不能替代这一步。
2. 用户佩戴设备录制新片段并导出；按已冻结规则只看 RGB 写标签，提交后运行 v1/v2。[执行说明](../research/active/hardware-bringup/FROZEN_FUSION_EVAL_RUNBOOK.md) · [冻结协议](../research/active/hardware-bringup/FROZEN_FUSION_EVAL_V1_PROTOCOL.md)。模型、阈值、标签与判据不据结果调整；所有片段和分母完整报告。
3. 根据独立数据决定下一步融合取舍，再验证完整响应时间、持续使用与提醒体验。

定位仍为盲杖互补（cane-complementary）的前视感知研究原型。当前采用头朝向、身体朝向与行进方向共向假设；设备时钟独立，手机接收匹配不是曝光同步。UNKNOWN 不等于无障碍，开发回放和工程检查不构成真实导航安全证明。

## 按任务进入

[研究决定](CURRENT_DECISION.md) · [避障研究](../research/active/dtr-r0/CURRENT.md) · [代码地图](CODE_MAP.md) · [硬件路线](GLASSES_HARDWARE_ROUTE.md) · [设备回归](DEVICE_REGRESSION.md)

本页整理前全文保留在 Git `83fbc28a` 同路径；旧实验结果、失败与各自停止规则仍以原报告为准。
L10 暂停、DTR R2 动态路线保留历史证据（`L10_R0_PAUSED / DTR_R2_DYNAMIC_RETAINED`），不作为当前推进线。
