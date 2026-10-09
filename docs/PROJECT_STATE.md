# 项目现在做到哪里

更新：2026-10-09（研究路由；设备记录沿用）。研究决定以[总决定](CURRENT_DECISION.md)为准。

## 运行与使用边界

Android v10.15.1保留原首页，手动“开始辅助”运行相机+8×8 ToF的A+LOCAL实验模式，可切回基础ToF，结束即停止会话。模型离线运行，几何配准仍是名义参数，未完成物理标定。[设备证据](HARDWARE_OBSTACLE_DEMO.md)

定位为盲杖互补的类别无关前视障碍感知（cane-complementary forward perception），关注墙体、身体/头部突出物、悬空障碍和杆状物。使用者决定如何移动；研究展示原型未证明真实导航安全，UNKNOWN不等于无障碍。

受控仿真A→A+LOCAL：24→29/32正事件检出，误报帧7→22、漏报帧106→65（576帧/48段）；UNKNOWN538/576，可与提醒并存。[原结果](../research/active/dtr-r0/nearfield/LOCAL_RESCUE_RESULTS_20260923.md)。手机10秒窗口101/101 ToF样本完成A+LOCAL，处理时间中位53ms/P95 61ms，未含提醒全链路时延或真实障碍真值；不能作为M3成绩。

## 当前工作

RGB独立子线：[低参数校正与迁移](../research/active/dtr-r0/nearfield/RGB_BODY_QUERY_CALIBRATED_TRANSFER_DEV_20261009.md)。只原train56帧/7环境1591791米制点拟合shift/affine；只原cal16帧选δ+.12/−.02m，实际FPR .143116/.139822非精确目标；head/checkpoint/归一化/实际cutoff冻结。原val affine-margin召回/误支持.704999/.132615 vs head .700483/.106626，TP≥16查询见证149/198 vs163/198；新3RScan .653427/.106888 vs .592361/.084894，384/570 vs411/570，射线取舍且query覆盖减少。ARKit单capture .297995/.014912 vs .295673/.041885，77/140 vs76/140；不能由一次比较裁定贡献。包括预定shift-direct的全部8臂及全部5cohort/环境/距离带/救回损失并列；训练距离监督不同，不作孤立架构归因。独立14400臂query/18000配对及冻结身份PASS，GPU0/600s、CPU保守200/1500s。TUM官方新镜像Python/curl均失败，源0B、第三相机NOT_RUN，官方HTML1110177B，网络保守175.857/600s，资源释放；fr3硬件勘误Asus Xtion。下一补可取得的真实参考/任务可观测覆盖与事件定义，以强基线检验度量表征/时序；旧640→256小变化/28采样free各0/28、完整身体/细结构/步行缺口保留。旧两query/32特征不续训，真实硬目标/同源锚点限制/CNH预算不变。

[校准尾部与中心锚定读出](../research/active/dtr-r0/nearfield/CNH_DIRECTION_ANCHOR_DEV_20261009.md)完成三seed×control/weak_pass×20/30/40%点×ideal/+3°，36center参照与72新政策共108 validation格、54本轮cal记录。仅复用冻结方向raw分数，无训练或模型回放。中心锚定OR：中心θ保持single，偏离两侧逐帧max后原last5平滑；ideal-cal中心未报警联合clear slot及整段未报警pass clip的最大限制分数取nextafter，覆盖f3–15，校准零增量。高度混合为HEAD .25/.50/.25、BODY center，raw后原平滑，在原joint预算下只选一个共享θ。

尾部诊断：max的18个cal阈值全部上移，前驱阻断均为pass clip预算；同θ分数单调且报警无损失，损失来自阈值变化。weighted阈值12升6降，分数和阈值两步分别留证。具体尾部样本/方向仅定位约束，不证明侧墙回波来源。

OR全格对center零损失是超集结构，不能推广为validation零成本。40% weak +3°的1cm H救/损3/0、0/0、5/0，B2/0、4/0、4/0各/128；clear clip148/120/140→155/133/150各/512，pass136/122/122→136/124/128各/256。ideal该点H最多救1、B无新增，pass仍增4/1/2。hybrid +3°浅H3/1、4/1、4/0、B0/0、0/2、0/0各/128；ideal浅H0/6、1/3、1/4、B0/6、0/3、0/0。BODY raw用中心仍会受共享θ影响；不设hybrid零损失门槛，不由validation选赢家或升级。

8项policy/5项tail聚焦检查与独立账本审计PASS；完整82944政策事件、55296分解事件及4548尾部单位保留。CPU实测21.749s+根端保守扣额40s，总61.749/600s，GPU0。原身体truth、完整bin、M3/原5格/L2/weak中心与加权候选保留；人工δ3°与已消费相关模拟Development不构成实机确认。下一围绕有依据的方向误差与独立背景复核附加报警成本；零增量OR严于用满剩余预算，后者未运行，不能判为无空间。完整表、两类clip/slot单位、失败/来源及未验证项见报告。

更新前CNH段见Git a91cb6c7同路径；RGB独立子线继续沿用。

方向估计仍是既有瓶颈。480新模拟unit确认保留39/51及时损失（/1002）的原身份；后续回放/筛查属于已消费Development。[扰动续训](../research/active/dtr-r0/nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)三门槛失败、配方停止，不加seed/epoch或换分布；方位微调及稀疏空间/关联L3暂停，M3/L2及负结果保留。读出、UNKNOWN与三态没有升级为安全证据。

[Nymeria](../research/active/dtr-r0/nearfield/CNH_NYMERIA_SAMPLE_AUDIT_DEV_20261008.md)样本异常未修补、合法段注册仅离线参考。原始IMU→CPF姿态前缀检查通过，准确度未评；位置/PDR未实现，E1整链不可评价。相关位移接口与参与者抽样本轮暂缓。

恢复设备时，M3/CNH链路单独核对真实输入处理与逐query标定真值，再评价迁移效果。City、保护test、新UE及硬件第二阶段暂停。`L10_R0_PAUSED / DTR_R2_DYNAMIC_RETAINED`仅历史状态；TARO、PanoLab、语义锚点不作独立推进线。SANPO由原“不作独立推进线”调整为RGB真实评价候选；旧SANPO筛查停止与负结果按原run保留。

## 按任务进入

[研究决定](CURRENT_DECISION.md) · [避障证据](../research/active/dtr-r0/CURRENT.md) · [代码地图](CODE_MAP.md) · [硬件路线](GLASSES_HARDWARE_ROUTE.md) · [设备回归](DEVICE_REGRESSION.md) · [关键备份](operations/CRITICAL_EVIDENCE_BACKUP.md)

本次更新前全文：Git `41d4baf3` 同路径；更早设备记录见`4f174009da62a8fcd9f219bb6758375f3f1ce2aa`。旧待决不覆盖当前决定。
