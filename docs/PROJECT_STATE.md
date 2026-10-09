# 项目现在做到哪里

更新：2026-10-09（研究路由；设备记录沿用）。研究决定以[总决定](CURRENT_DECISION.md)为准。

## 运行与使用边界

Android v10.15.1保留原首页，手动“开始辅助”运行相机+8×8 ToF的A+LOCAL实验模式，可切回基础ToF，结束即停止会话。模型离线运行，几何配准仍是名义参数，未完成物理标定。[设备证据](HARDWARE_OBSTACLE_DEMO.md)

定位为盲杖互补的类别无关前视障碍感知（cane-complementary forward perception），关注墙体、身体/头部突出物、悬空障碍和杆状物。使用者决定如何移动；研究展示原型未证明真实导航安全，UNKNOWN不等于无障碍。

受控仿真A→A+LOCAL：24→29/32正事件检出，误报帧7→22、漏报帧106→65（576帧/48段）；UNKNOWN538/576，可与提醒并存。[原结果](../research/active/dtr-r0/nearfield/LOCAL_RESCUE_RESULTS_20260923.md)。手机10秒窗口101/101 ToF样本完成A+LOCAL，处理时间中位53ms/P95 61ms，未含提醒全链路时延或真实障碍真值；不能作为M3成绩。

## 当前工作

RGB独立子线已完成[真实连续段参考与基线评价](../research/active/dtr-r0/nearfield/RGB_BODY_QUERY_EVAL_DEV_20261009.md)：45帧/270格，59正/25空域负/186UNKNOWN；空域负排除评分，GT定位oracle不能证明独立检出或误报。共同18帧Depth Pro原图/128像素IoU .3284/.3261，高分辨率增益未建立；VDA全45帧完成但近盒正格0/10。金属细杆漏标，下一补真实细障碍标注与非空角域负例，再训练query机制；提前量NOT_EVALUABLE。继承旧BodyQuery、对照COPILOT；真实定量仍为硬目标，预算独立于CNH，身体外参未闭合。

[擦边工作点与匹配续训](../research/active/dtr-r0/nearfield/CNH_PASS_BOUNDARY_DEV_20261009.md)完成零训练cal选点、32历史坐标等价检查及三seed各12轮原mask/弱pass对照。30%cal预算下独立读出弱pass−control HEAD净+5/+18/+13、BODY+4/+30/+25各/384；validation pass80/73/72→76/77/75各/256，clear0/0/0→3/1/0各/6656，不是严格同成本。预设40%点1cm HEAD净+4/+1/+3、BODY+1/+6/+8各/128，pass减少3/9/5但clear增加7/4/2；保留低擦边独立读出候选，不由validation选上线点。

OR30% HEAD净−9/−5/−3、BODY−12/+1/+5各/384；+3°独立读出1cm HEAD净−6/−5/−7各/128，成本仍不稳定。原M3/5格/L2不升级。旧OR cal pass下限55使20%预算51不可行；坐标gauge输入/logits相同只排除覆盖的坐标依赖，query/pose时间与符号合同未闭合。弱pass新增2288槽，同时重归一化新增数据原有效权重×.985034，旧39936行不变；不能归因为mask单一原因。[此前反事实和摘要负结果](../research/active/dtr-r0/nearfield/CNH_COUNTERFACTUAL_DEV_20261009.md)保留。科学阶段266.219/1200s、有记录CPU85.249/600s（初始映射时长未知），聚焦检查、6模型日程及258048账本行独立PASS；6衍生数组约5.153GB经SHA清理，证据/模型保留。已消费模拟Development，无新光子/硬件/保护480访问；更新前CNH段见Git d2afbaf1。

方向估计仍是既有瓶颈。480新模拟unit确认保留39/51及时损失（/1002）的原身份；后续回放/筛查属于已消费Development。[扰动续训](../research/active/dtr-r0/nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)三门槛失败、配方停止，不加seed/epoch或换分布；方位微调及稀疏空间/关联L3暂停，M3/L2及负结果保留。读出、UNKNOWN与三态没有升级为安全证据。

[Nymeria](../research/active/dtr-r0/nearfield/CNH_NYMERIA_SAMPLE_AUDIT_DEV_20261008.md)样本异常未修补、合法段注册仅离线参考。原始IMU→CPF姿态前缀检查通过，准确度未评；位置/PDR未实现，E1整链不可评价。相关位移接口与参与者抽样本轮暂缓。

恢复设备时，M3/CNH链路单独核对真实输入处理与逐query标定真值，再评价迁移效果。City、保护test、新UE及硬件第二阶段暂停。`L10_R0_PAUSED / DTR_R2_DYNAMIC_RETAINED`仅历史状态；TARO、PanoLab、语义锚点不作独立推进线。SANPO由原“不作独立推进线”调整为RGB真实评价候选；旧SANPO筛查停止与负结果按原run保留。

## 按任务进入

[研究决定](CURRENT_DECISION.md) · [避障证据](../research/active/dtr-r0/CURRENT.md) · [代码地图](CODE_MAP.md) · [硬件路线](GLASSES_HARDWARE_ROUTE.md) · [设备回归](DEVICE_REGRESSION.md) · [关键备份](operations/CRITICAL_EVIDENCE_BACKUP.md)

本次更新前全文：Git `41d4baf3` 同路径；更早设备记录见`4f174009da62a8fcd9f219bb6758375f3f1ce2aa`。旧待决不覆盖当前决定。
