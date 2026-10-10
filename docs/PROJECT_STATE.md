# 项目现在做到哪里

更新：2026-10-10（研究路由；设备记录沿用）。研究决定以[总决定](CURRENT_DECISION.md)为准。

## 运行与使用边界

Android v10.15.1保留原首页，手动“开始辅助”运行相机+8×8 ToF的A+LOCAL实验模式，可切回基础ToF，结束即停止会话。模型离线运行，几何配准仍是名义参数，未完成物理标定。[设备证据](HARDWARE_OBSTACLE_DEMO.md)

定位为盲杖互补的类别无关前视障碍感知（cane-complementary forward perception），关注墙体、身体/头部突出物、悬空障碍和杆状物。使用者决定如何移动；研究展示原型未证明真实导航安全，UNKNOWN不等于无障碍。

受控仿真A→A+LOCAL：24→29/32正事件检出，误报帧7→22、漏报帧106→65（576帧/48段）；UNKNOWN538/576，可与提醒并存。[原结果](../research/active/dtr-r0/nearfield/LOCAL_RESCUE_RESULTS_20260923.md)。手机10秒窗口101/101 ToF样本完成A+LOCAL，处理时间中位53ms/P95 61ms，未含提醒全链路时延或真实障碍真值；不能作为M3成绩。

## 当前工作

RGB独立子线：[点迁移与查询校准](../research/active/dtr-r0/nearfield/RGB_BODY_QUERY_MIGRATION_DIAGNOSTIC_DEV_20261010.md)。冻结缓存点迁移与三折query级cal探针完成，无训练/推理/下载。152帧5301769匹配点，新3RScan点MALE .282250 vsaffine .301547；追加ARKit整体ratio .812/.775但近/远带偏移不同，不定位单一尺度或硬件原因。另两capture严格sampledFREE按5%经验比例cal后，depth点FREE 73/220、79/247→11/220、27/247，正见证171/202、165/179→67/202、74/179，同折affine120/202、131/179；迁移支持率并未稳定5%，当前配方不升级。下一检验几何主干受约束残差（实施提议）并补跨环境query负cal/新相机/完整身体细障碍步行参考。相关mask/分母/分数与冻结核验通过，CPU保守490/1200s，GPU/下载0，故障和计时缺口留证、任务释放。已消费同相机家族/有限相关FREE不证明整盒空闲；旧阈值/head32停止配方、真实硬目标与CNH预算保留；前文见Git53bb1069。

用户于2026-10-10明确：本阶段暂按“头朝向＝身体朝向＝行进方向”处理，ToF身体/头部query共用这一方向。以ideal/共向条件为主比较；方向估计、PDR及方向不确定性合同暂不作为本阶段前置工作。这是当前任务假设，旧偏差实验的结果和身份保留。

目标回到ToF任务收益与分级提醒：明显风险给强提醒，接触/擦边难分时允许轻提醒；不把精确分界作为继续推进的前提。分级只依据运行时可见的分数与证据质量，真值仅评价；弱证据保留不确定含义。重点是弱/细目标证据在投影、累积和完整bin读出中的保留，以及分级能否增加有用及时提醒。强/轻提醒分别报告接触检出、完整救/损、pass与clear clip/slot成本，强转轻也单列；轻提醒可接受但打扰成本不记为零，不设零损失门槛。共向假设不替代外参、时钟或空间投影核验。

[新增clear与BODY匹配诊断](../research/active/dtr-r0/nearfield/CNH_GRADED_PEAK_MATCH_DEV_20261010.md)完成固定current中档全cal/val、575事件/1041尝试/133pairs；BODY救回4/6/8、提前53/31/44仍保留，strict同背景实例/形状＋公共z/extent/share/count匹配clear仅0/1/0、pass1/2/3，未配为NOT_EVALUABLE，不放宽边界。新增clear23/40/27slots中HEADonly11/28/9、BODYonly12/12/18、both0；新增clear clips21/31/23。内走廊峰更远也在BODY救回1/3/4件出现，仅峰深度对照、不确立背景可见/free或静默收益。current联合候选及原strong/fullsingle等级/首次保留，temporal不默认追加。下一优先固定评分/阈值的BODY定向追加对照，完整计HEAD新增救回4/14/6及提前的放弃代价，再决定是否分高度校准；尚未执行/采用，不直接静默或继续本匹配放宽。匹配102297与深度18146项独立复算PASS；depth配对列名错误初source/PLAN/partialevents留证，repair-1同条件通过。CPU保守190/600s、GPU/fit/预测/新采样0、无常驻资源。仅已消费模拟Development、未接入App，原M3/5格/L2/body truth/fullbin/480、weak_pass及旧stop保留；[联合评分原结果](../research/active/dtr-r0/nearfield/CNH_GRADED_PEAK_JOINT_DEV_20261010.md)保留，更新前CNH正文见Git eb7572e7。

已完成的[步行参考](../research/active/dtr-r0/nearfield/CNH_WALKING_CORRIDOR_DEV_20261010.md)、[方向合同](../research/active/dtr-r0/nearfield/CNH_DIRECTION_CONTRACT_DEV_20261009.md)和全部方向聚合/锚定/高度混合结果作为历史诊断保留，不再驱动本阶段主线。步行参考的18–20°是过去/未来轨迹差异，非ToF姿态精度；单位/名义时钟限制仍按原报告。更新前CNH当前正文见Git 53bb1069同路径。

[Nymeria](../research/active/dtr-r0/nearfield/CNH_NYMERIA_SAMPLE_AUDIT_DEV_20261008.md)样本异常未修补、合法段注册仅离线参考。原始IMU→CPF姿态前缀检查通过，准确度未评；位置/PDR未实现，E1整链不可评价。该audit run的位移接口与参与者抽样当时暂缓；新步行参考run见上述报告。

恢复设备时，M3/CNH链路单独核对真实输入处理与逐query标定真值，再评价迁移效果。City、保护test、新UE及硬件第二阶段暂停。`L10_R0_PAUSED / DTR_R2_DYNAMIC_RETAINED`仅历史状态；TARO、PanoLab、语义锚点不作独立推进线。SANPO由原“不作独立推进线”调整为RGB真实评价候选；旧SANPO筛查停止与负结果按原run保留。

## 按任务进入

[研究决定](CURRENT_DECISION.md) · [避障证据](../research/active/dtr-r0/CURRENT.md) · [代码地图](CODE_MAP.md) · [硬件路线](GLASSES_HARDWARE_ROUTE.md) · [设备回归](DEVICE_REGRESSION.md) · [关键备份](operations/CRITICAL_EVIDENCE_BACKUP.md)

本次更新前全文：Git `41d4baf3` 同路径；更早设备记录见`4f174009da62a8fcd9f219bb6758375f3f1ce2aa`。旧待决不覆盖当前决定。
