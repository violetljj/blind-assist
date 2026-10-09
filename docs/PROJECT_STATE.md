# 项目现在做到哪里

更新：2026-10-09（研究路由；设备记录沿用）。研究决定以[总决定](CURRENT_DECISION.md)为准。

## 运行与使用边界

Android v10.15.1保留原首页，手动“开始辅助”运行相机+8×8 ToF的A+LOCAL实验模式，可切回基础ToF，结束即停止会话。模型离线运行，几何配准仍是名义参数，未完成物理标定。[设备证据](HARDWARE_OBSTACLE_DEMO.md)

定位为盲杖互补的类别无关前视障碍感知（cane-complementary forward perception），关注墙体、身体/头部突出物、悬空障碍和杆状物。使用者决定如何移动；研究展示原型未证明真实导航安全，UNKNOWN不等于无障碍。

受控仿真A→A+LOCAL：24→29/32正事件检出，误报帧7→22、漏报帧106→65（576帧/48段）；UNKNOWN538/576，可与提醒并存。[原结果](../research/active/dtr-r0/nearfield/LOCAL_RESCUE_RESULTS_20260923.md)。手机10秒窗口101/101 ToF样本完成A+LOCAL，处理时间中位53ms/P95 61ms，未含提醒全链路时延或真实障碍真值；不能作为M3成绩。

## 当前工作

RGB另立独立研究子线：[真实身体空间查询](../research/active/dtr-r0/nearfield/RGB_BODY_QUERY_DEV_20261009.md)。已完成真实数据/文献核查、50真实帧输入诊断、10帧Depth Pro配对烟测及相机局部query读出，另取得45连续真实帧/180配套文件。128输入细节中位保留指标12.7%，不是检出率；真实身体/事件标签未闭合，训练与提前量NOT_RUN。下一核验连续段query标签及因果视频深度基线。继承旧BodyQuery、对照COPILOT；真实定量仍为硬目标，预算独立于CNH。

[简化bin-token试点](../research/active/dtr-r0/nearfield/CNH_DELAYED_QUERY_DEV_20261009.md)复用训练39936行与492场景×K4；8833名义参数去signed-face臂H578/B510各/688、clear46/4576，相对M3救77/117、损16/13。本轮不支持signed-face或extent增量。优先保留旧5格融合＋简化臂互补候选H562/B481，相对旧融合救19/20、损0，暗4cm横杆H12/B7各/56保留；clear46格/25段/23clip，但pass74→101/352。+3°固定ideal阈值成本旧175→191格（M3为176），H514/B422；48训练外同生成器单位AUC .839265低于M3 .868327。已消费Development选择，不升级M3或声称实机/泛化效果。

当前配方本轮不再加训，新机制开放；下一优先train-only反事实配对及不同背景、擦边和迁移。v1 pair是同观测HEAD/BODY query contrast，完整背景反事实尚未做；旧192输入响应NOT_RUN仅辅诊。11项聚焦检查及独立账本核验通过；GPU成功397.609s加两次启动失败5.7036556s，预算403.313/1200s；CPU准备/分析预算600s。20.457GB/16个衍生features已保存SHA manifest后清理，raw/模型/输入/ledger和失败保留；无新光子、硬件或保护480访问。

方向估计仍是既有瓶颈。480新模拟unit确认保留39/51及时损失（/1002）的原身份；后续回放/筛查属于已消费Development。[扰动续训](../research/active/dtr-r0/nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)三门槛失败、配方停止，不加seed/epoch或换分布；方位微调及稀疏空间/关联L3暂停，M3/L2及负结果保留。读出、UNKNOWN与三态没有升级为安全证据。

[Nymeria](../research/active/dtr-r0/nearfield/CNH_NYMERIA_SAMPLE_AUDIT_DEV_20261008.md)样本异常未修补、合法段注册仅离线参考。原始IMU→CPF姿态前缀检查通过，准确度未评；位置/PDR未实现，E1整链不可评价。相关位移接口与参与者抽样本轮暂缓。

恢复设备时，M3/CNH链路单独核对真实输入处理与逐query标定真值，再评价迁移效果。City、保护test、新UE及硬件第二阶段暂停。`L10_R0_PAUSED / DTR_R2_DYNAMIC_RETAINED`仅历史状态；TARO、PanoLab、语义锚点不作独立推进线。SANPO由原“不作独立推进线”调整为RGB真实评价候选；旧SANPO筛查停止与负结果按原run保留。

## 按任务进入

[研究决定](CURRENT_DECISION.md) · [避障证据](../research/active/dtr-r0/CURRENT.md) · [代码地图](CODE_MAP.md) · [硬件路线](GLASSES_HARDWARE_ROUTE.md) · [设备回归](DEVICE_REGRESSION.md) · [关键备份](operations/CRITICAL_EVIDENCE_BACKUP.md)

本次更新前全文：Git `41d4baf3` 同路径；更早设备记录见`4f174009da62a8fcd9f219bb6758375f3f1ce2aa`。旧待决不覆盖当前决定。
