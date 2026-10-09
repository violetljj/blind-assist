# 项目现在做到哪里

更新：2026-10-09（研究路由；设备记录沿用）。研究决定以[总决定](CURRENT_DECISION.md)为准。

## 运行与使用边界

Android v10.15.1保留原首页，手动“开始辅助”运行相机+8×8 ToF的A+LOCAL实验模式，可切回基础ToF，结束即停止会话。模型离线运行，几何配准仍是名义参数，未完成物理标定。[设备证据](HARDWARE_OBSTACLE_DEMO.md)

定位为盲杖互补的类别无关前视障碍感知（cane-complementary forward perception），关注墙体、身体/头部突出物、悬空障碍和杆状物。使用者决定如何移动；研究展示原型未证明真实导航安全，UNKNOWN不等于无障碍。

受控仿真A→A+LOCAL：24→29/32正事件检出，误报帧7→22、漏报帧106→65（576帧/48段）；UNKNOWN538/576，可与提醒并存。[原结果](../research/active/dtr-r0/nearfield/LOCAL_RESCUE_RESULTS_20260923.md)。手机10秒窗口101/101 ToF样本完成A+LOCAL，处理时间中位53ms/P95 61ms，未含提醒全链路时延或真实障碍真值；不能作为M3成绩。

## 当前工作

RGB另立独立研究子线：[真实身体空间查询](../research/active/dtr-r0/nearfield/RGB_BODY_QUERY_DEV_20261009.md)。已完成真实数据/文献核查、50真实帧输入诊断、10帧Depth Pro配对烟测及相机局部query读出，另取得45连续真实帧/180配套文件。128输入细节中位保留指标12.7%，不是检出率；真实身体/事件标签未闭合，训练与提前量NOT_RUN。下一核验连续段query标签及因果视频深度基线。继承旧BodyQuery、对照COPILOT；真实定量仍为硬目标，预算独立于CNH。

[三臂三seed扩充](../research/active/dtr-r0/nearfield/CNH_COUNTERFACTUAL_DEV_20261009.md)固定简化bin-token/BCE及7488步，新增普通与完整交叉反事实样本等量。普通OR在不同几何背景H315/313/323、B267/257/282各/384（旧融合298/244），clear均58/6656；pass60→89/103/115各/256。+3°固定校准阈值clear1297/1261/1352（旧1245），不是扰动同成本。CF对普通的独立读出H净−2/+13/0、B0/−4/−7，OR H−7/0/+2、B−10/−2/−10，无稳定增量；保留普通扩充Development候选，M3/原5格/L2不升级。

下一优先公共query/pose输入合同与擦边成本（+3°基线M3 clear64→1255、旧融合58→1245），不重启旧扰动训练。当前完整交叉配方本轮收敛，新机制开放。新增背景族train/cal/validation隔离；继承39936行/M3背景相似性未审计。BCE未读pair身份，不称反事实关系学习；AABB/有限背景/相关K仍为模拟探索。3标量摘要对完整bin base独立读出H净+11/−46/−49、B−29/−13/−30，clear102/104/140（base87/46/49）/6656；只支持保留完整bin相对该适配器，不作普适机制证明。实验累计1190.955/3600s含失败10.187s，13聚焦检查、96epoch日程及73728行独立账本复核通过；12衍生数组10.305GB经SHA留证清理，raw/模型/账本/失败保留。旧bin-token试点和预算保留在[原报告](../research/active/dtr-r0/nearfield/CNH_DELAYED_QUERY_DEV_20261009.md)，更新前CNH段见Git 3564255a；设备与保护480未访问。

方向估计仍是既有瓶颈。480新模拟unit确认保留39/51及时损失（/1002）的原身份；后续回放/筛查属于已消费Development。[扰动续训](../research/active/dtr-r0/nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)三门槛失败、配方停止，不加seed/epoch或换分布；方位微调及稀疏空间/关联L3暂停，M3/L2及负结果保留。读出、UNKNOWN与三态没有升级为安全证据。

[Nymeria](../research/active/dtr-r0/nearfield/CNH_NYMERIA_SAMPLE_AUDIT_DEV_20261008.md)样本异常未修补、合法段注册仅离线参考。原始IMU→CPF姿态前缀检查通过，准确度未评；位置/PDR未实现，E1整链不可评价。相关位移接口与参与者抽样本轮暂缓。

恢复设备时，M3/CNH链路单独核对真实输入处理与逐query标定真值，再评价迁移效果。City、保护test、新UE及硬件第二阶段暂停。`L10_R0_PAUSED / DTR_R2_DYNAMIC_RETAINED`仅历史状态；TARO、PanoLab、语义锚点不作独立推进线。SANPO由原“不作独立推进线”调整为RGB真实评价候选；旧SANPO筛查停止与负结果按原run保留。

## 按任务进入

[研究决定](CURRENT_DECISION.md) · [避障证据](../research/active/dtr-r0/CURRENT.md) · [代码地图](CODE_MAP.md) · [硬件路线](GLASSES_HARDWARE_ROUTE.md) · [设备回归](DEVICE_REGRESSION.md) · [关键备份](operations/CRITICAL_EVIDENCE_BACKUP.md)

本次更新前全文：Git `41d4baf3` 同路径；更早设备记录见`4f174009da62a8fcd9f219bb6758375f3f1ce2aa`。旧待决不覆盖当前决定。
