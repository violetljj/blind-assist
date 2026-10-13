# 项目现在做到哪里

更新：2026-10-13（本轮任务始于 10 月 12 日）。研究决定以[总决定](CURRENT_DECISION.md)为准；实机研究见[硬件当前状态](../research/active/hardware-bringup/CURRENT.md)。

## 当前可用与本轮接入

当前 Android v10.19.1 支持 Atom RGB、H3 完整 8×8×16 CNH 与 BMI270 IMU 三路采集、原始数据保存、连接异常记录和后台 ZIP 导出。首页“开始辅助”在当前 H3 固件上运行基础 ToF 提醒；旧 UDP 时序协议保留 A+LOCAL 实验模式。[采集说明](PHONE_DATA_CAPTURE.md) · [辅助入口](HARDWARE_OBSTACLE_DEMO.md)

本轮 **59 / 10.19.1 已保留数据覆盖安装，融合实验待佩戴验证**。数据采集页相机有界预读和预览不再等待持久写盘，同样12秒录制窗口预览12.58→23.74fps，RGB保存13.33→25.36Hz，IMU84.17→99.96Hz、缺序号201→0。队列高水位1/8，无满队列或异常丢弃；原始文件与索引一致。[完整性能与数值记录](../research/active/hardware-bringup/FUSION_INPUT_AND_MEASURED_GEOMETRY_DEV_20261013.md)

融合入口仍显式开始、HEAD/BODY独立2/3确认，可低头空地标定高度；静音、离页停止。原始模型接入检查见[10.19.0记录](../research/active/hardware-bringup/MOBILE_FUSION_INTEGRATION_20261012.md)。用户本轮不便佩戴，故未执行有效姿态下的实时RGB/完整融合，也未打开新冻结eval。

此前[因果输入/代次修复](../research/active/hardware-bringup/REAL_FUSION_ENGINEERING_DEV_20261013.md)本版已纳入安装；新增float32边界与退化拟合修复，oracle6/6、常规回归7/7通过。三视图已量横移几何与离线标线工具可用，真实位姿/距离仍需测量；合成自洽不算实测精度。

## 研究结论与关键缺口

- **ToF**：保留原 M3/5 格基线。模拟默认候选与实机俯角/地面/噪声联合栈分别记录；已消费真实回放中墙、柱、家具响应较明显，细竖杆和持续轻提醒仍是瓶颈。[实机联合栈](../research/active/hardware-bringup/REAL_POSE_FLOOR_NOISE_DR_DEV_20261012.md)
- **RGB**：已标定 Atom 相机并估计相机/ToF 旋转；DAV2-Small 相对深度配合 ToF/地面尺度锚定，在已消费片段中补到部分 ToF 漏掉的细杆。[小模型报告](../research/active/hardware-bringup/REAL_RGB_SMALL_MODEL_DEV_20261012.md)。既有手机网络推理计时不等于整机端到端延迟。
- **融合**：真实头戴回放已显示互补，但尚无新头戴冻结检验结论。本轮 Android 接入是工程迁移，不自动证明与离线脚本等价或新场景有效。半合成 ToF 的 v4 中带排序结果保留其原证据身份，不替代实机验证。

## 接下来

数据研究可先复用现有素材：[首轮RGB与VisTa小样](../research/active/hardware-bringup/DATA_POTENTIAL_DEV_20261013.md)、[扩展及第三轮实读](../research/active/hardware-bringup/DATA_OPPORTUNITY_CATALOG_20261013.md)已落地。第三轮整理54个额外RGB复核入口，确认Hypersim原生几何采样损失/ZJU区内混合，并读通OctoNet的232帧ST直方图与LLRGBD照明配对。优先原生几何支持、ST bin合同及完整RGB时序；未训练或改变模型。

1. 用户佩戴设备，低头空地标定安装高度，再在有效姿态下检查实时融合；软件工程检查不能替代这一步。
2. 用户佩戴设备录制新片段并导出；按已冻结规则只看 RGB 写标签，提交后运行 v1/v2。[执行说明](../research/active/hardware-bringup/FROZEN_FUSION_EVAL_RUNBOOK.md) · [冻结协议](../research/active/hardware-bringup/FROZEN_FUSION_EVAL_V1_PROTOCOL.md)。模型、阈值、标签与判据不据结果调整；所有片段和分母完整报告。
3. 根据独立数据决定下一步融合取舍，再验证完整响应时间、持续使用与提醒体验。

定位仍为盲杖互补（cane-complementary）的前视感知研究原型。当前采用头朝向、身体朝向与行进方向共向假设；设备时钟独立，手机接收匹配不是曝光同步。UNKNOWN 不等于无障碍，开发回放和工程检查不构成真实导航安全证明。

## 按任务进入

[研究决定](CURRENT_DECISION.md) · [避障研究](../research/active/dtr-r0/CURRENT.md) · [代码地图](CODE_MAP.md) · [硬件路线](GLASSES_HARDWARE_ROUTE.md) · [设备回归](DEVICE_REGRESSION.md)

本页整理前全文保留在 Git `83fbc28a` 同路径；旧实验结果、失败与各自停止规则仍以原报告为准。
L10 暂停、DTR R2 动态路线保留历史证据（`L10_R0_PAUSED / DTR_R2_DYNAMIC_RETAINED`），不作为当前推进线。
