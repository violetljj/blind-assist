ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED

# 前视障碍感知：当前状态

更新：2026-10-10。盲杖互补前视感知主线；原M3、5格局部融合、L2和A+LOCAL保留。

RGB独立子线：[冻结残差读出与条件校准](nearfield/RGB_BODY_QUERY_RESIDUAL_READOUT_DEV_20261010.md)冻结两step600预测，完成absolute/normalized×global/band。主60/65原affine见证/FREE120/12、131/10；CTX归一化global1/13、2/18，绝对band60/20、50/8，归一化band4/7、19/19，不升级。held固定带成本辅助DR130/12、131/10，对强curveaffine122/12、132/10救9损1／救0损1，局部排序空间保留但非校准迁移成功。Ark16band辅助144query N/E，commonPOS158/FREE105；nearPOS44/11/0无稳定恢复。全部4组合/3臂/3capture与5补充cohort、逐query救损、3450ties曲线独立PASS。CPU科学51.607s、独立审计12.806s、整体保守270/1200s；GPU/train/forward/download0，任务资源结束。affine保留，本run停止固定recipe调参，下一优先RGB可观测信息/合规train域覆盖，cal不转train，其它机制开放。仅已消费Development/nativefirstreturn，旧A0全δ/CNH/stop保留；前RGB正文Git24688f0d。

## 当前：共向假设下推进ToF检出与分级提醒

用户于2026-10-10明确：本阶段暂按“头朝向＝身体朝向＝行进方向”处理，ToF身体/头部query共用这一方向。以ideal/共向条件为主比较；方向估计、PDR及方向不确定性合同暂不作为本阶段前置工作。这是当前任务假设，旧偏差实验的结果和身份保留。

目标回到ToF任务收益与分级提醒：明显风险给强提醒，接触/擦边难分时允许轻提醒；不把精确分界作为继续推进的前提。分级只依据运行时可见的分数与证据质量，真值仅评价；弱证据保留不确定含义。重点是弱/细目标证据在投影、累积和完整bin读出中的保留，以及分级能否增加有用及时提醒。强/轻提醒分别报告接触检出、完整救/损、pass与clear clip/slot成本，强转轻也单列；轻提醒可接受但打扰成本不记为零，不设零损失门槛。共向假设不替代外参、时钟或空间投影核验。

[0/1静默帧通知合并](nearfield/CNH_GRADED_EPISODE_MERGE_DEV_20261010.md)完成固定baseline/both/head50三seed全cal/val、每query0/1静默帧通知合并。首正等级即报、轻到强即升、同事件同级及强后回轻不重报；1帧方案连续2静默帧重置，静默不发通知。head50 val联合contact通知1186/1249/1191→1137/1196/1144，纯pass246/306/289→221/271/257（减10–11%）、clear58/93/71→56/89/69。全部原grade、首any/首strong与及时指标不变；及时HEAD352/357/351、BODY311/316/320各/384。合并只省重复，受提醒clips不减，head50仍较同合并baseline新增clear15/19/18、pass1/8/6clips。保留episode1离线成本候选、episode0对照和head50成本候选/both高检出对照；下一连续流重置及持续强信号再提示需求，弱/细目标漏检和首报及时性仍主线，不续扫gap。54cells/165888streams/324groups，702prefix/8fixtures/72273487checks及4直接30继承hash独立PASS；CPU保守170/300s，GPU/fit/预测/新raw0，无常驻资源。仅已消费模拟Development，query事件非物体轨迹、13帧clip重置及跨片段/变帧率/新障碍合并未验证，未接App。原M3/5格/L2/body truth/fullbin/480/weak_pass及旧stop保留，深度静默证据不足、temporal不默认。前CNH正文Git9b504791。

已完成的[步行参考](nearfield/CNH_WALKING_CORRIDOR_DEV_20261010.md)、[方向合同](nearfield/CNH_DIRECTION_CONTRACT_DEV_20261009.md)和全部方向聚合/锚定/高度混合结果作为历史诊断保留，不再驱动本阶段主线。步行参考的18–20°是过去/未来轨迹差异，非ToF姿态精度；单位/名义时钟限制仍按原报告。更新前CNH当前正文见Git 53bb1069同路径。

## 停止与保留

[旧扰动训练](nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)、T/T2、旧参照和门控的停止规则仍按各自run适用；不同机制开放。[480确认](nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)原身份保留，UNKNOWN/三态/query覆盖待验证。原Nymeria位移实现、−19°、设备/City/保护test/新UE及硬件第二阶段按原run暂缓；101/101与53ms属A+LOCAL，M3/CNH实机效果未建立。

模拟AABB、有限背景、人工误差、相关K/帧及探索选择不构成实机、安全或独立确认。更新前CNH当前正文保留于Git 5844eacd同路径；RGB子线继续沿用。

[RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)。
