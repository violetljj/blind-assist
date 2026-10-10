# 项目现在做到哪里

更新：2026-10-10（研究路由；设备记录沿用）。研究决定以[总决定](CURRENT_DECISION.md)为准。

## 运行与使用边界

Android v10.15.1保留原首页，手动“开始辅助”运行相机+8×8 ToF的A+LOCAL实验模式，可切回基础ToF，结束即停止会话。模型离线运行，几何配准仍是名义参数，未完成物理标定。[设备证据](HARDWARE_OBSTACLE_DEMO.md)

定位为盲杖互补的类别无关前视障碍感知（cane-complementary forward perception），关注墙体、身体/头部突出物、悬空障碍和杆状物。使用者决定如何移动；研究展示原型未证明真实导航安全，UNKNOWN不等于无障碍。

受控仿真A→A+LOCAL：24→29/32正事件检出，误报帧7→22、漏报帧106→65（576帧/48段）；UNKNOWN538/576，可与提醒并存。[原结果](../research/active/dtr-r0/nearfield/LOCAL_RESCUE_RESULTS_20260923.md)。手机10秒窗口101/101 ToF样本完成A+LOCAL，处理时间中位53ms/P95 61ms，未含提醒全链路时延或真实障碍真值；不能作为M3成绩。

## 当前工作

RGB独立子线：[查询外推与负参考](../research/active/dtr-r0/nearfield/RGB_BODY_QUERY_QUERY_LEVEL_DEV_20261009.md)。固定27子盒、冻结旧head/cal且不重选阈值；136帧3672格，575格sampledFREE（572来自三ARKit capture），不是整盒/身体空闲。新3RScan正见证head352/866、affine-margin487/866，救/损109/244；旧ARKit采样free支持7/105对0/105；新增两个官方visit组head31/220、22/247，对照12/220、10/247。未证明冻结head对任意query的稳定增量，保留旧RGB距离信息Development收益；下一改距离/不确定性表征和query覆盖，以强几何及全量救回损失检验。第三硬件TUM仍因吞吐停止NOT_RUN，同iPad家族补参考不替代硬件迁移；完整身体/细障碍/步行仍缺。旧query/32特征不续训，真实硬目标/同源锚点限制/CNH预算不变。

[步行走廊参考](../research/active/dtr-r0/nearfield/CNH_WALKING_CORRIDOR_DEV_20261010.md)已接通全部1025个BlindWays缓存片段：10位参与者、615000帧、10250native anchor×T0.5/1/1.5秒=30750窗口。显式60Hz名义相对时钟，无实测时间戳/头部朝向；原XY/Z上合同沿用，米尺度未找到官方明文，公制结果以旧单位假设为条件。

past≥0.30m预定子集的past1s/future-chord差异RMS18.38/18.82/19.62°，不是设备yaw或head-to-travel误差；全部时域/参与者/缺失与低motion都保留，不按结果选T/δ。完整未来折线+0.30m圆盘仅中心轨迹proxy，中心距端点弦>0.30m为1/10250、2/9225、4/9225，非身体面积遗漏/碰撞。片段关联未知、不是iid。

6fixture、30750行账本/66摘要/28700完整路径与100anchor的past公式/prefix核验通过。仅CPU离线整理（本run300s上限、GPU0）；无新模型/投影/训练/采集/数据下载/480。因果位置仍NOT_IMPLEMENTED；下一接合格因果位置与同参考/同时间窗评价，再验证物理T、身体几何、外参和同步。

旧[Charades方向合同](../research/active/dtr-r0/nearfield/CNH_DIRECTION_CONTRACT_DEV_20261009.md)、全部聚合/锚定/高度混合结果和旧2°mixed RMS结论按原run保留，不能与本轮轨迹差异直接换算。M3/5格/L2/bodytruth/bin/480及停止配方保留；更新前CNH正文见Git 4cfcca53同路径。

方向估计仍是既有瓶颈。480新模拟unit确认保留39/51及时损失（/1002）的原身份；后续回放/筛查属于已消费Development。[扰动续训](../research/active/dtr-r0/nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)三门槛失败、配方停止，不加seed/epoch或换分布；方位微调及稀疏空间/关联L3暂停，M3/L2及负结果保留。读出、UNKNOWN与三态没有升级为安全证据。

[Nymeria](../research/active/dtr-r0/nearfield/CNH_NYMERIA_SAMPLE_AUDIT_DEV_20261008.md)样本异常未修补、合法段注册仅离线参考。原始IMU→CPF姿态前缀检查通过，准确度未评；位置/PDR未实现，E1整链不可评价。该audit run的位移接口与参与者抽样当时暂缓；新步行参考run见上述报告。

恢复设备时，M3/CNH链路单独核对真实输入处理与逐query标定真值，再评价迁移效果。City、保护test、新UE及硬件第二阶段暂停。`L10_R0_PAUSED / DTR_R2_DYNAMIC_RETAINED`仅历史状态；TARO、PanoLab、语义锚点不作独立推进线。SANPO由原“不作独立推进线”调整为RGB真实评价候选；旧SANPO筛查停止与负结果按原run保留。

## 按任务进入

[研究决定](CURRENT_DECISION.md) · [避障证据](../research/active/dtr-r0/CURRENT.md) · [代码地图](CODE_MAP.md) · [硬件路线](GLASSES_HARDWARE_ROUTE.md) · [设备回归](DEVICE_REGRESSION.md) · [关键备份](operations/CRITICAL_EVIDENCE_BACKUP.md)

本次更新前全文：Git `41d4baf3` 同路径；更早设备记录见`4f174009da62a8fcd9f219bb6758375f3f1ce2aa`。旧待决不覆盖当前决定。
