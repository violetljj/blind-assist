# 项目现在做到哪里

更新：2026-10-09（研究路由；设备记录沿用）。研究决定以[总决定](CURRENT_DECISION.md)为准。

## 运行与使用边界

Android v10.15.1保留原首页，手动“开始辅助”运行相机+8×8 ToF的A+LOCAL实验模式，可切回基础ToF，结束即停止会话。模型离线运行，几何配准仍是名义参数，未完成物理标定。[设备证据](HARDWARE_OBSTACLE_DEMO.md)

定位为盲杖互补的类别无关前视障碍感知（cane-complementary forward perception），关注墙体、身体/头部突出物、悬空障碍和杆状物。使用者决定如何移动；研究展示原型未证明真实导航安全，UNKNOWN不等于无障碍。

受控仿真A→A+LOCAL：24→29/32正事件检出，误报帧7→22、漏报帧106→65（576帧/48段）；UNKNOWN538/576，可与提醒并存。[原结果](../research/active/dtr-r0/nearfield/LOCAL_RESCUE_RESULTS_20260923.md)。手机10秒窗口101/101 ToF样本完成A+LOCAL，处理时间中位53ms/P95 61ms，未含提醒全链路时延或真实障碍真值；不能作为M3成绩。

## 当前工作

RGB独立子线：[尺度、场景信息与度量表征诊断](../research/active/dtr-r0/nearfield/RGB_BODY_QUERY_METRIC_DIAGNOSTIC_DEV_20261009.md)。沿用3RScan96帧/12环境7/2/3隔离，1440格825正/615UNKNOWN，无满足全部可观测采样射线free判据的负query。冻结Depth Pro的RGB预测深度+learned query读出，validation正/free query-ray1071771/1795858，召回/误支持.70048/.10663，匹配geometry .61966/.11535；新增32特征较弱(.65990/.10880)。depth-only打乱退化至.64691/.19574，相同head ray/query的417179近邻反标签pair排序.84421，支持场景信息。三环境和全距离带取舍并列，第三环境召回下降，前两环境误支持增加；cal阈值固定，不以validation改模型。GPU166.096/600s，独立SHA/预测复算通过。保留简洁预测深度读出候选，下一充分参考负query与步行/细结构覆盖；真实射线Development非完整身体/整盒/事件证据。旧两query/SANPO负结果及停止规则、真实硬目标、CNH预算保留。

[逐帧机制链](../research/active/dtr-r0/nearfield/CNH_PASS_MECHANISM_CHAIN_DEV_20261009.md)完成固定seed956/30%点19个有目的选例：209历史/418输入，1672冻结模型同输入回放分数与保存值逐值一致。全64个1cm场景×11及时帧×两branch=1408几何行；+3正侧HEAD/BODY各16场景在全窗口均无完整目标与contact query交集，负侧均仍相交。f10正侧最小x=.36549m，8/11帧仍进入±.4m扩展区。身体contact真值不改，不能据错位query外目标判监督无价值。

17/19选例历史正支持反而增加（含正HEAD6/6）；负BODY loss在f8历史signed支持+8.130，weak阈值余量+.350→−.047，另有支持下降却救回例。全域汇总含背景噪声，不能单调解释报警，完整bin/[双条件候选](../research/active/dtr-r0/nearfield/CNH_PASS_POSE_DIAGNOSTIC_DEV_20261009.md)保留。固定height/shape/bg/rho后侧别可比0/64；静态两高度共享横向合同不认证实物pose/time。无训练/新阈值/−3/新光子/设备/480；下一方向不确定性与身体走廊合同的冻结读出成本小试，未执行。M3/原5格/L2/旧stop与[CF/摘要负结果](../research/active/dtr-r0/nearfield/CNH_COUNTERFACTUAL_DEV_20261009.md)保留，更新前CNH段见Git 183358af。

方向估计仍是既有瓶颈。480新模拟unit确认保留39/51及时损失（/1002）的原身份；后续回放/筛查属于已消费Development。[扰动续训](../research/active/dtr-r0/nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)三门槛失败、配方停止，不加seed/epoch或换分布；方位微调及稀疏空间/关联L3暂停，M3/L2及负结果保留。读出、UNKNOWN与三态没有升级为安全证据。

[Nymeria](../research/active/dtr-r0/nearfield/CNH_NYMERIA_SAMPLE_AUDIT_DEV_20261008.md)样本异常未修补、合法段注册仅离线参考。原始IMU→CPF姿态前缀检查通过，准确度未评；位置/PDR未实现，E1整链不可评价。相关位移接口与参与者抽样本轮暂缓。

恢复设备时，M3/CNH链路单独核对真实输入处理与逐query标定真值，再评价迁移效果。City、保护test、新UE及硬件第二阶段暂停。`L10_R0_PAUSED / DTR_R2_DYNAMIC_RETAINED`仅历史状态；TARO、PanoLab、语义锚点不作独立推进线。SANPO由原“不作独立推进线”调整为RGB真实评价候选；旧SANPO筛查停止与负结果按原run保留。

## 按任务进入

[研究决定](CURRENT_DECISION.md) · [避障证据](../research/active/dtr-r0/CURRENT.md) · [代码地图](CODE_MAP.md) · [硬件路线](GLASSES_HARDWARE_ROUTE.md) · [设备回归](DEVICE_REGRESSION.md) · [关键备份](operations/CRITICAL_EVIDENCE_BACKUP.md)

本次更新前全文：Git `41d4baf3` 同路径；更早设备记录见`4f174009da62a8fcd9f219bb6758375f3f1ce2aa`。旧待决不覆盖当前决定。
