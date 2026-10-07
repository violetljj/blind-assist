# 项目现在做到哪里

更新：2026-10-07（研究路由；设备记录沿用）。本页负责运行能力和工作路由，研究决定以[总决定](CURRENT_DECISION.md)为准。

## 运行与使用边界

Android v10.15.1 保留原首页，手动“开始辅助”后运行相机 + 8×8 ToF 的 A+LOCAL 实验模式，可切回基础 ToF；结束辅助即停止会话。模型在手机离线运行，几何配准仍为名义参数，未完成物理标定。[运行及设备证据](HARDWARE_OBSTACLE_DEMO.md)

定位为盲杖互补的类别无关前视障碍感知（cane-complementary forward perception），关注墙体、身体/头部突出物、悬空障碍和杆状物。使用者决定如何移动；这是研究展示原型，未证明真实导航安全，UNKNOWN 不等于无障碍。

受控仿真固定对照为576帧、48段、32个正事件：A→A+LOCAL检出24→29/32，误报帧7→22、漏报帧106→65；UNKNOWN为538/576，可与提醒并存。不能换算实机准确率。[结果与分母](../research/active/dtr-r0/nearfield/LOCAL_RESCUE_RESULTS_20260923.md)

手机已有10秒窗口中，101/101个ToF样本完成A+LOCAL，处理时间中位数53ms、P95 61ms。仅证明该窗口运行，不含感测到听到提醒的全链路时延，也无真实障碍真值。新模拟模型未据此替换手机方案。

## 当前工作

唯一研究主线是前视障碍感知，用户已设定“持续进行算法探索”目标。冻结M3保留；独立单调标定消除否决但无性能改进，事后两固定阈值oracle余量有限。下一步比较原始径向摘要及过去变化，区分分数压缩损失与接近时序信息。详细结论、分母与输入范围在[避障当前页](../research/active/dtr-r0/CURRENT.md)维护。既有章节及真实头动确认保留，行进意图仍是开放问题。

当前单调标定与精确两阈值oracle已完成，前轮GPU已释放。持续目标内继续执行、修复和验证；各子实验在运行前明确范围，不继承已结束预算或改写停止结果。研究方法见[研究工作方式](../research/WORKFLOW.md)。

L10_R0_PAUSED；DTR_R2_DYNAMIC_RETAINED仅为历史算法状态，不表示恢复动态研究。TARO、SANPO、PanoLab、语义锚点不作独立推进线。City、保护test、新UE采集及硬件第二阶段暂停；真实回放待设备会话。

## 按任务进入

[研究决定](CURRENT_DECISION.md) · [避障证据与下一问题](../research/active/dtr-r0/CURRENT.md) · [代码地图](CODE_MAP.md) · [硬件路线](GLASSES_HARDWARE_ROUTE.md) · [设备回归](DEVICE_REGRESSION.md) · [关键证据备份](operations/CRITICAL_EVIDENCE_BACKUP.md)

整理前全文保存在Git revision `4f174009da62a8fcd9f219bb6758375f3f1ce2aa` 的 `docs/PROJECT_STATE.md`；旧待决状态不覆盖当前决定。
