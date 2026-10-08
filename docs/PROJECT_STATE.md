# 项目现在做到哪里

更新：2026-10-08（研究路由；设备记录沿用）。本页负责运行能力和工作路由，研究决定以[总决定](CURRENT_DECISION.md)为准。

## 运行与使用边界

Android v10.15.1 保留原首页，手动“开始辅助”后运行相机 + 8×8 ToF 的 A+LOCAL 实验模式，可切回基础 ToF；结束辅助即停止会话。模型在手机离线运行，几何配准仍为名义参数，未完成物理标定。[运行及设备证据](HARDWARE_OBSTACLE_DEMO.md)

定位为盲杖互补的类别无关前视障碍感知（cane-complementary forward perception），关注墙体、身体/头部突出物、悬空障碍和杆状物。使用者决定如何移动；这是研究展示原型，未证明真实导航安全，UNKNOWN 不等于无障碍。

受控仿真固定对照为576帧、48段、32个正事件：A→A+LOCAL检出24→29/32，误报帧7→22、漏报帧106→65；UNKNOWN为538/576，可与提醒并存。不能换算实机准确率。[结果与分母](../research/active/dtr-r0/nearfield/LOCAL_RESCUE_RESULTS_20260923.md)

手机已有10秒窗口中，101/101个ToF样本完成A+LOCAL，处理时间中位数53ms、P95 61ms。仅证明该窗口运行，不含感测到听到提醒的全链路时延，也无真实障碍真值。新模拟模型未据此替换手机方案。

## 当前工作

粗方位首检已完成：96物理yaw缓存unit、229事件/384clear，同全量提示条目预算下，top1首条及时且受接触物方位范围支持相对EMA四挑战单路净−38/−39/−52/−62，双路−28/−39/−41/−52；检测也落后。降低当前固定三查询读出优先级，本次不进入去重或NAT重渲染。支持范围不等于唯一定位；11项契约及独立重算通过，推理611.531秒，无新渲染/采集，计算结束。下一可考虑观测空间支持直接读方位，未实施。[报告与口径](../research/active/dtr-r0/nearfield/CNH_SECTOR_NOTICE_DEV_20261008.md)。

唯一研究主线是前视障碍感知，方向估计优先、读出保留不上调。步态与冻结EMA推理级对照完成：复用96unit物理yaw观测，229事件、全13实际FA124/4992各臂残差0；四挑战gait−EMA单路净−2/−2/−8/−1，降步态优先级；双路+5/+1/−5/0，取舍未决保留，两配置均不自动追加冷启动。输入为理想60Hz fullclip代理与带噪5Hz窗口EMA的整套方案比较。冻结M3不替换，候选query完整覆盖/三态待验证；不择参、不开同类确认批。详细边界、成本和报告在[避障当前页](../research/active/dtr-r0/CURRENT.md)维护。

已有480新模拟unit确认与后续已消费Development分开：方向损失39/51 of1002，EMA收回约三分之二；并集降低静默并增加UNKNOWN，双路及时优势未建立，非实机/新人群。章节与答辩问答已整合。读出HB策略净及时区间跨零，不确认改善/非劣；启动误报保留。新EMA推理215.578/1200秒、分析检查15.284/180秒通过，计算已结束，载荷保留；旧机制、预算及停止结果不改。[研究工作方式](../research/WORKFLOW.md)。

L10_R0_PAUSED；DTR_R2_DYNAMIC_RETAINED仅为历史算法状态，不表示恢复动态研究。TARO、SANPO、PanoLab、语义锚点不作独立推进线。City、保护test、新UE采集及硬件第二阶段暂停；真实回放待设备会话。

## 按任务进入

[研究决定](CURRENT_DECISION.md) · [避障证据与下一问题](../research/active/dtr-r0/CURRENT.md) · [代码地图](CODE_MAP.md) · [硬件路线](GLASSES_HARDWARE_ROUTE.md) · [设备回归](DEVICE_REGRESSION.md) · [关键证据备份](operations/CRITICAL_EVIDENCE_BACKUP.md)

整理前全文保存在Git revision `4f174009da62a8fcd9f219bb6758375f3f1ce2aa` 的 `docs/PROJECT_STATE.md`；旧待决状态不覆盖当前决定。
