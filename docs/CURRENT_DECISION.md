# 当前研究决定

更新：2026-10-10。主线为盲杖互补前视感知，M3和A+LOCAL保留；原5格局部融合和L2保留。

RGB独立子线：[残差目标与输入兼容性](../research/active/dtr-r0/nearfield/RGB_BODY_QUERY_RESIDUAL_TARGET_DEV_20261010.md)完成56train/304cal目标与固定public512诊断，未读136eval。四bound .2/.742222/1.148101/3.184721；envelope约24.16倍且ARKitcal近83404点仅74.958%可达。16邻居代理近带ARKit MAE1.9614→.8289，但3RScan .6008→.6104、trainLOEO整体.3171→.4193；ARKit3/3访问整体改善，train7/7及cal3RScan17/17环境整体变差。像素改善不等于query救损/FREE，局部冲突不证明不可学习。全局affine保留，下一实际训练有界log残差＋可观测局部深度上下文，对照三项depth/ray；不默认范围回退、不用eval设bound。360采样/4bound/全邻距与汇总、2原帧/4exact16及角色核验PASS，审计失败与metadata语义修正留证。CPU保守329/1200s（整合全额200），GPU/训练/下载/新推理0，资源已释放。仅已消费Development/first-return参考，旧A0全δ/三折/成本取舍/stop/CNH保留；前RGB正文Gitabda0a31。

## 当前：共向假设下推进ToF检出与分级提醒

用户于2026-10-10明确：本阶段暂按“头朝向＝身体朝向＝行进方向”处理，ToF身体/头部query共用这一方向。以ideal/共向条件为主比较；方向估计、PDR及方向不确定性合同暂不作为本阶段前置工作。这是当前任务假设，旧偏差实验的结果和身份保留。

目标回到ToF任务收益与分级提醒：明显风险给强提醒，接触/擦边难分时允许轻提醒；不把精确分界作为继续推进的前提。分级只依据运行时可见的分数与证据质量，真值仅评价；弱证据保留不确定含义。重点是弱/细目标证据在投影、累积和完整bin读出中的保留，以及分级能否增加有用及时提醒。强/轻提醒分别报告接触检出、完整救/损、pass与clear clip/slot成本，强转轻也单列；轻提醒可接受但打扰成本不记为零，不设零损失门槛。共向假设不替代外参、时钟或空间投影核验。

[先轻后强与断续诊断](../research/active/dtr-r0/nearfield/CNH_GRADED_ALERT_CHAIN_DEV_20261010.md)完成固定baseline/both/head50、三seed全cal/val query/joint82944流。head50 BODY新增救回4/6/8中3/6/7窗口内无强，不能要求后续升强才保留轻档；HEAD救1/13/5中升强1/9/5均仅晚f14–15。BODY原已及时提前53/31/44中升强46/28/41、间断升级27/14/22，其中最大静默1帧17/9/13件。完整joint纯pass也有先轻后强55/71/67，不以升级顺序/连续性当接触确认。下一每query固定允许1静默帧事件合并：首轻即报、首强即升、同级不重复，2静默帧解除；与0帧合并对照首提醒/升级时刻及contact救损、pass/clear重复成本，尚未执行。head50成本候选与both高检出对照、全部原strong/fullsingle保留，深度静默证据不足、temporal不默认。本轮等级/阈值不变，82944streams/162summary/144effects及4021229项、4直接/30继承hash独立PASS，CPU保守155/300s、GPU/fit/预测/新raw0、无常驻资源。仅已消费模拟Development、query流非物体轨迹、窗口不升强非低风险，未接入App；原M3/5格/L2/body truth/fullbin/480/weak_pass及旧stop保留。[分高度工作点](../research/active/dtr-r0/nearfield/CNH_GRADED_PEAK_HEAD_CAL_DEV_20261010.md)保留，前CNH正文Git86c45f54。

已完成的[步行参考](../research/active/dtr-r0/nearfield/CNH_WALKING_CORRIDOR_DEV_20261010.md)、[方向合同](../research/active/dtr-r0/nearfield/CNH_DIRECTION_CONTRACT_DEV_20261009.md)和全部方向聚合/锚定/高度混合结果作为历史诊断保留，不再驱动本阶段主线。步行参考的18–20°是过去/未来轨迹差异，非ToF姿态精度；单位/名义时钟限制仍按原报告。更新前CNH当前正文见Git 53bb1069同路径。

## 停止与保留

[旧扰动训练](../research/active/dtr-r0/nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)、T/T2、旧参照和门控的停止规则仍按各自run适用；不同机制开放。[480确认](../research/active/dtr-r0/nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)原身份保留，UNKNOWN/三态/query覆盖待验证。原Nymeria位移实现、−19°、设备/City/保护test/新UE及硬件第二阶段按原run暂缓；101/101与53ms属A+LOCAL，M3/CNH实机效果未建立。

模拟AABB、有限背景、人工误差、相关K/帧及探索选择不构成实机、安全或独立确认。更新前CNH当前正文保留于Git 5844eacd同路径；RGB子线继续沿用。

[路线当前页](../research/active/dtr-r0/CURRENT.md) · [RUNS](../research/active/dtr-r0/RUNS.md)。
