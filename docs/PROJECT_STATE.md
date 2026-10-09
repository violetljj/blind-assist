# 项目现在做到哪里

更新：2026-10-09（研究路由；设备记录沿用）。研究决定以[总决定](CURRENT_DECISION.md)为准。

## 运行与使用边界

Android v10.15.1保留原首页，手动“开始辅助”运行相机+8×8 ToF的A+LOCAL实验模式，可切回基础ToF，结束即停止会话。模型离线运行，几何配准仍是名义参数，未完成物理标定。[设备证据](HARDWARE_OBSTACLE_DEMO.md)

定位为盲杖互补的类别无关前视障碍感知（cane-complementary forward perception），关注墙体、身体/头部突出物、悬空障碍和杆状物。使用者决定如何移动；研究展示原型未证明真实导航安全，UNKNOWN不等于无障碍。

受控仿真A→A+LOCAL：24→29/32正事件检出，误报帧7→22、漏报帧106→65（576帧/48段）；UNKNOWN538/576，可与提醒并存。[原结果](../research/active/dtr-r0/nearfield/LOCAL_RESCUE_RESULTS_20260923.md)。手机10秒窗口101/101 ToF样本完成A+LOCAL，处理时间中位53ms/P95 61ms，未含提醒全链路时延或真实障碍真值；不能作为M3成绩。

## 当前工作

RGB独立子线：[冻结迁移与锚点旁路](../research/active/dtr-r0/nearfield/RGB_BODY_QUERY_TRANSFER_DEV_20261009.md)。原模型/归一化/cal实际cutoff冻结，无训练；8个未用3RScan环境64帧，正/free query-ray3400061/4344666，depth-only召回/误支持.59236/.08489 vs geometry .50464/.09513，6/8环境TP净增、2/8净减；相对raw DP .64005/.15911是取舍。ARKitScenes iPadPro单捕获16帧，depth-only .29567/.04188 vs geometry .42204/.05606，两项同时改善与充分近场增量未复现；场景、原图分辨率/K、预测距离及读出适配原因未定。28全采样free格各臂支持0/28，非整盒/身体误报；步行提前量仍缺事件/身体参考。同源实测锚点旁路改善直接几何但不证明ToF增量。GPU154.136/900s，官方数据357569481B/2GiB，独立保存预测复算及旧帧CUDA逐值核验PASS。下一高分辨率不同相机多环境与预测质量诊断、完整query/步行参考；旧两query/32特征不续训，原负结果/真实硬目标/CNH预算保留。

[三方向聚合比较](../research/active/dtr-r0/nearfield/CNH_DIRECTION_AGGREGATION_DEV_20261009.md)完成全部三seed/两模型/三聚合/两分支/三个点，共108 validation组合、54 cal阈值。δ=3°固定网格，raw先聚合后原平滑，权重.25/.50/.25，阈值只在ideal cal按原clear/pass预算选择。24中心一致性检查958464值差0，原身体truth/完整bin保持。

weak加权相对中心single：+3的9个seed×点1cm HEAD净增全正、BODY8正1负；ideal 1cm BODY9格全负。40% +3 H救/损3/1、5/1、4/0，B3/2、5/1、6/0各/128；clear clip139/135/142 vs148/120/140各/512，pass130/127/123 vs136/122/122各/256。max在浅HEAD有收益但BODY损失明显；完整表/M3配对保留，两类clip成本不折算，无一致支配或正式升级。M3/原5格/L2和weak独立读出候选继续保留。

82944事件账本和2592方向切片留证，argmax只描述不校正。声明方向误差未获实机依据，本轮是已消费模拟Development，未训练、读取480或采集硬件，未评价以−3°为中心的反号偏差分支。[机制链](../research/active/dtr-r0/nearfield/CNH_PASS_MECHANISM_CHAIN_DEV_20261009.md)、[CF/摘要负结果](../research/active/dtr-r0/nearfield/CNH_COUNTERFACTUAL_DEV_20261009.md)和旧stop保留。加权仅作为方向偏差下的取舍方案；核验、实际预算与资源记录见新报告。更新前CNH段见Git cc1a9737.

方向估计仍是既有瓶颈。480新模拟unit确认保留39/51及时损失（/1002）的原身份；后续回放/筛查属于已消费Development。[扰动续训](../research/active/dtr-r0/nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)三门槛失败、配方停止，不加seed/epoch或换分布；方位微调及稀疏空间/关联L3暂停，M3/L2及负结果保留。读出、UNKNOWN与三态没有升级为安全证据。

[Nymeria](../research/active/dtr-r0/nearfield/CNH_NYMERIA_SAMPLE_AUDIT_DEV_20261008.md)样本异常未修补、合法段注册仅离线参考。原始IMU→CPF姿态前缀检查通过，准确度未评；位置/PDR未实现，E1整链不可评价。相关位移接口与参与者抽样本轮暂缓。

恢复设备时，M3/CNH链路单独核对真实输入处理与逐query标定真值，再评价迁移效果。City、保护test、新UE及硬件第二阶段暂停。`L10_R0_PAUSED / DTR_R2_DYNAMIC_RETAINED`仅历史状态；TARO、PanoLab、语义锚点不作独立推进线。SANPO由原“不作独立推进线”调整为RGB真实评价候选；旧SANPO筛查停止与负结果按原run保留。

## 按任务进入

[研究决定](CURRENT_DECISION.md) · [避障证据](../research/active/dtr-r0/CURRENT.md) · [代码地图](CODE_MAP.md) · [硬件路线](GLASSES_HARDWARE_ROUTE.md) · [设备回归](DEVICE_REGRESSION.md) · [关键备份](operations/CRITICAL_EVIDENCE_BACKUP.md)

本次更新前全文：Git `41d4baf3` 同路径；更早设备记录见`4f174009da62a8fcd9f219bb6758375f3f1ce2aa`。旧待决不覆盖当前决定。
