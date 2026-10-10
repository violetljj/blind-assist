# 当前研究决定

更新：2026-10-10。主线为盲杖互补前视感知，M3和A+LOCAL保留；原5格局部融合和L2保留。

RGB独立子线：[有界残差实际训练](../research/active/dtr-r0/nearfield/RGB_BODY_QUERY_BOUNDED_RESIDUAL_DEV_20261010.md)完成两4993参数/同初始化同批次600step实际残差训练。主Ark16/60/65 affine见证/FREE103/5、120/12、131/10；depth/ray47/0、103/10、104/26；context52/0、94/22、102/9。两追加capture无优势，不升级，本run停止固定配方追加调参，affine保留。事后同5FREE context Ark16见证138，对固定affine103救54损19，对完整curve affine113救50损25，近2/44但FREE1/96；60/65同成本context91/102低于curve affine122/132。主/304cal补充trained cuts全>.25结构排除near，不证明无近信息；pooled cal各29/588。两臂直接paired与全5cohort/1725curves保留，65相对depthray省成本的混合收益不丢弃。下一若继续优先新增可观测RGB上下文/训练域覆盖，不把cal转train；不同机制开放。360特征/同600batches、880模型输出/23760原affine分数、全18792pairs/阈值与81原生query独立PASS，input审核源码SHA误断言留证。GPU49.128/900s、CPU保守280/1200s，无新DepthPro/下载，资源释放。仅已消费Development/first-return，旧A0全δ/LOCO/held成本/stop/CNH保留；前RGB正文Gitb97299b4。

## 当前：共向假设下推进ToF检出与分级提醒

用户于2026-10-10明确：本阶段暂按“头朝向＝身体朝向＝行进方向”处理，ToF身体/头部query共用这一方向。以ideal/共向条件为主比较；方向估计、PDR及方向不确定性合同暂不作为本阶段前置工作。这是当前任务假设，旧偏差实验的结果和身份保留。

目标回到ToF任务收益与分级提醒：明显风险给强提醒，接触/擦边难分时允许轻提醒；不把精确分界作为继续推进的前提。分级只依据运行时可见的分数与证据质量，真值仅评价；弱证据保留不确定含义。重点是弱/细目标证据在投影、累积和完整bin读出中的保留，以及分级能否增加有用及时提醒。强/轻提醒分别报告接触检出、完整救/损、pass与clear clip/slot成本，强转轻也单列；轻提醒可接受但打扰成本不记为零，不设零损失门槛。共向假设不替代外参、时钟或空间投影核验。

[先轻后强与断续诊断](../research/active/dtr-r0/nearfield/CNH_GRADED_ALERT_CHAIN_DEV_20261010.md)完成固定baseline/both/head50、三seed全cal/val query/joint82944流。head50 BODY新增救回4/6/8中3/6/7窗口内无强，不能要求后续升强才保留轻档；HEAD救1/13/5中升强1/9/5均仅晚f14–15。BODY原已及时提前53/31/44中升强46/28/41、间断升级27/14/22，其中最大静默1帧17/9/13件。完整joint纯pass也有先轻后强55/71/67，不以升级顺序/连续性当接触确认。下一每query固定允许1静默帧事件合并：首轻即报、首强即升、同级不重复，2静默帧解除；与0帧合并对照首提醒/升级时刻及contact救损、pass/clear重复成本，尚未执行。head50成本候选与both高检出对照、全部原strong/fullsingle保留，深度静默证据不足、temporal不默认。本轮等级/阈值不变，82944streams/162summary/144effects及4021229项、4直接/30继承hash独立PASS，CPU保守155/300s、GPU/fit/预测/新raw0、无常驻资源。仅已消费模拟Development、query流非物体轨迹、窗口不升强非低风险，未接入App；原M3/5格/L2/body truth/fullbin/480/weak_pass及旧stop保留。[分高度工作点](../research/active/dtr-r0/nearfield/CNH_GRADED_PEAK_HEAD_CAL_DEV_20261010.md)保留，前CNH正文Git86c45f54。

已完成的[步行参考](../research/active/dtr-r0/nearfield/CNH_WALKING_CORRIDOR_DEV_20261010.md)、[方向合同](../research/active/dtr-r0/nearfield/CNH_DIRECTION_CONTRACT_DEV_20261009.md)和全部方向聚合/锚定/高度混合结果作为历史诊断保留，不再驱动本阶段主线。步行参考的18–20°是过去/未来轨迹差异，非ToF姿态精度；单位/名义时钟限制仍按原报告。更新前CNH当前正文见Git 53bb1069同路径。

## 停止与保留

[旧扰动训练](../research/active/dtr-r0/nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)、T/T2、旧参照和门控的停止规则仍按各自run适用；不同机制开放。[480确认](../research/active/dtr-r0/nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)原身份保留，UNKNOWN/三态/query覆盖待验证。原Nymeria位移实现、−19°、设备/City/保护test/新UE及硬件第二阶段按原run暂缓；101/101与53ms属A+LOCAL，M3/CNH实机效果未建立。

模拟AABB、有限背景、人工误差、相关K/帧及探索选择不构成实机、安全或独立确认。更新前CNH当前正文保留于Git 5844eacd同路径；RGB子线继续沿用。

[路线当前页](../research/active/dtr-r0/CURRENT.md) · [RUNS](../research/active/dtr-r0/RUNS.md)。
