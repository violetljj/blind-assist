# 项目现在做到哪里

更新：2026-10-10（研究路由；设备记录沿用）。研究决定以[总决定](CURRENT_DECISION.md)为准。

## 运行与使用边界

Android v10.15.1保留原首页，手动“开始辅助”运行相机+8×8 ToF的A+LOCAL实验模式，可切回基础ToF，结束即停止会话。模型离线运行，几何配准仍是名义参数，未完成物理标定。[设备证据](HARDWARE_OBSTACLE_DEMO.md)

定位为盲杖互补的类别无关前视障碍感知（cane-complementary forward perception），关注墙体、身体/头部突出物、悬空障碍和杆状物。使用者决定如何移动；研究展示原型未证明真实导航安全，UNKNOWN不等于无障碍。

受控仿真A→A+LOCAL：24→29/32正事件检出，误报帧7→22、漏报帧106→65（576帧/48段）；UNKNOWN538/576，可与提醒并存。[原结果](../research/active/dtr-r0/nearfield/LOCAL_RESCUE_RESULTS_20260923.md)。手机10秒窗口101/101 ToF样本完成A+LOCAL，处理时间中位53ms/P95 61ms，未含提醒全链路时延或真实障碍真值；不能作为M3成绩。

## 当前工作

RGB 当前冻结全局 affine，0.8m以上为候选public query职责；当前近带固定recipe停止调参。两step600残差及全部失败/局部排序结果按[原报告](../research/active/dtr-r0/nearfield/RGB_BODY_QUERY_RESIDUAL_READOUT_DEV_20261010.md)保留，不把配方失败写成单目近带原理否定。后续提前量/语义尚待评价。[本次输入核查](../research/active/dtr-r0/nearfield/CNH_RGB_FROZEN_E2E_INPUT_INVENTORY_20261010.md)确认没有未消费且满足真值合同的同步RGB/CNH输入，完整融合NOT_EVALUABLE，未运行新RGB推理。

用户于2026-10-10明确：本阶段暂按“头朝向＝身体朝向＝行进方向”处理，ToF身体/头部query共用这一方向。以ideal/共向条件为主比较；方向估计、PDR及方向不确定性合同暂不作为本阶段前置工作。这是当前任务假设，旧偏差实验的结果和身份保留。

目标回到ToF任务收益与分级提醒：明显风险给强提醒，接触/擦边难分时允许轻提醒；不把精确分界作为继续推进的前提。分级只依据运行时可见的分数与证据质量，真值仅评价；弱证据保留不确定含义。重点是弱/细目标证据在投影、累积和完整bin读出中的保留，以及分级能否增加有用及时提醒。强/轻提醒分别报告接触检出、完整救/损、pass与clear clip/slot成本，强转轻也单列；轻提醒可接受但打扰成本不记为零，不设零损失门槛。共向假设不替代外参、时钟或空间投影核验。

[冻结配置新场景端到端评价](../research/active/dtr-r0/nearfield/CNH_RGB_FROZEN_E2E_RESULTS_20261010.md)完成ToF→通知比较：cal/hold各768新物理scene、K2，新背景family分开，主seed955预声明。主HEAD细横杆厚1.7cm，16scene/32相关事件；原5格与同通知成本匹配臂均2/32，救0损0。HEAD158/256不变，BODY116→119/256救3损0；hold空闲/纯擦边联合通知均28/37。原both为HEAD202、BODY154各/256、细横杆3/32，但通知49/286；956/957匹配HEAD细救回也0，956擦边38比原37多1。不支持主目标同成本收益，不升级App，固定配方结束不在该hold调参。原M3/5格/旧both及head50、L2和所有失败保留；旧剩余检出诊断见Git1311e202同路径。仅解析AABB共向ideal/有限背景、短窗口离线通知，完整RGB＋ToF不可评价。

[HEAD细横杆期望回波诊断](../research/active/dtr-r0/nearfield/CNH_HEAD_THIN_SIGNAL_ORACLE_DEV_20261010.md)完成：厚度/反射率受控扫中，1.7cm在f13前距0.97m最佳bin SNR为1.37/3.59（rho .19/.57）；0.65m原薄杆整盒出横向视场。固定bin移动8帧SNR降至0.78/2.27，真值matched为2.38/6.54，静态√K不可搬用。原M3细杆3/32及时、29静默，但27/32及时窗曾有相容top8，支持中位2帧；query线性投影描述比例约16.4%，非网络因果。sub32敏感性显著，不能宣布硬件物理上限。下一优先可见窗口内随距离移动的稀疏读出，新读出NOT_RUN；不同统计量阈值不可直接比较，不在已消费E2E hold调参，完整融合仍NOT_EVALUABLE。

[弱支持读出对照](../research/active/dtr-r0/nearfield/CNH_GRADED_WEAK_MASS_DEV_20261010.md)完成：在旧consumed ideal输入上，正负分离allbin质量28项与12格前向距离profile96项均未改善主要弱群，不采用。validation相对head50：mass HEAD救/损2:0、0:4、1:2，BODY1:1、1:6、0:5；profile HEAD2:1、0:9、0:5，BODY0:3、0:6、0:8，各/384。两方案BODY弱横杆救回均0/0/0、损0/2/0，各/48。cal9/11匹配clear union slots与purepass整clip上限，validation成本漂移单列，不能称等成本收益。控制重fit与旧分数完全一致，旧强提醒和ordinary轻提醒保留；18浅层fit/predict及独立18cells/540cohorts通过，失败留证。两聚合配方结束，下一保留native zone/range局部身份、可见短窗内公共运动对齐（NOT_RUN），不在已消费E2E hold调参。[原生支持诊断](../research/active/dtr-r0/nearfield/CNH_GRADED_TARGET_SUPPORT_DEV_20261010.md)与raw成本负结果保留；目标truth只评价，旧M3/5格/L2/480及stop不变，非实机证据。更新前当前页见Git4ab513c4。

已完成的[步行参考](../research/active/dtr-r0/nearfield/CNH_WALKING_CORRIDOR_DEV_20261010.md)、[方向合同](../research/active/dtr-r0/nearfield/CNH_DIRECTION_CONTRACT_DEV_20261009.md)和全部方向聚合/锚定/高度混合结果作为历史诊断保留，不再驱动本阶段主线。步行参考的18–20°是过去/未来轨迹差异，非ToF姿态精度；单位/名义时钟限制仍按原报告。更新前CNH当前正文见Git 53bb1069同路径。

[Nymeria](../research/active/dtr-r0/nearfield/CNH_NYMERIA_SAMPLE_AUDIT_DEV_20261008.md)样本异常未修补、合法段注册仅离线参考。原始IMU→CPF姿态前缀检查通过，准确度未评；位置/PDR未实现，E1整链不可评价。该audit run的位移接口与参与者抽样当时暂缓；新步行参考run见上述报告。

恢复设备时，M3/CNH链路单独核对真实输入处理与逐query标定真值，再评价迁移效果。City、保护test、新UE及硬件第二阶段暂停。`L10_R0_PAUSED / DTR_R2_DYNAMIC_RETAINED`仅历史状态；TARO、PanoLab、语义锚点不作独立推进线。SANPO由原“不作独立推进线”调整为RGB真实评价候选；旧SANPO筛查停止与负结果按原run保留。

## 按任务进入

[研究决定](CURRENT_DECISION.md) · [避障证据](../research/active/dtr-r0/CURRENT.md) · [代码地图](CODE_MAP.md) · [硬件路线](GLASSES_HARDWARE_ROUTE.md) · [设备回归](DEVICE_REGRESSION.md) · [关键备份](operations/CRITICAL_EVIDENCE_BACKUP.md)

本次更新前全文：Git `41d4baf3` 同路径；更早设备记录见`4f174009da62a8fcd9f219bb6758375f3f1ce2aa`。旧待决不覆盖当前决定。
