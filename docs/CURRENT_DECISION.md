# 当前研究决定

更新：2026-10-10。主线为盲杖互补前视感知，M3和A+LOCAL保留；原5格局部融合和L2保留。

RGB独立子线：[新负参考冻结评分](../research/active/dtr-r0/nearfield/RGB_BODY_QUERY_NEGATIVE_SCORE_DEV_20261010.md)。原A0三折完整结果保留。新增240帧冻结DepthPro/旧头推理完成，原cal2＋新cal30的32严格FREE（12环境、near6仅1环境）选全局cutoff，各臂实际1/32；绝对余量affine在Ark16/60/65见证1/114/102、FREE0/10/7，五cutoff>.433m结构性排除near/mid。一次事后区间宽度归一化保留混合信号：A0δ=.1相对归一化affine在60/65净救21/25，FREE各+1；但同臂见证62→34、71→33，已有affine等成本辅助为47/41>34/33，near仍全0。不升级工作点、不提高残差训练优先级，不否定重训。15环境扩到960帧FREE129/13环境，near6→23仍1环境，按stop停止同源扩帧，新720帧未推理评分。下一更换环境/机位补跨环境近负覆盖；当前冻结affine为RGB代表。独立分数/配对/LOEO及25920参考核验PASS，GPU261.108/1000s、CPU保守821/1800s、训练/下载0，无保留进程。仅Development非整盒/身体/实机证明；旧停止/CNH保留，前文Git20e014b5。

## 当前：共向假设下推进ToF检出与分级提醒

用户于2026-10-10明确：本阶段暂按“头朝向＝身体朝向＝行进方向”处理，ToF身体/头部query共用这一方向。以ideal/共向条件为主比较；方向估计、PDR及方向不确定性合同暂不作为本阶段前置工作。这是当前任务假设，旧偏差实验的结果和身份保留。

目标回到ToF任务收益与分级提醒：明显风险给强提醒，接触/擦边难分时允许轻提醒；不把精确分界作为继续推进的前提。分级只依据运行时可见的分数与证据质量，真值仅评价；弱证据保留不确定含义。重点是弱/细目标证据在投影、累积和完整bin读出中的保留，以及分级能否增加有用及时提醒。强/轻提醒分别报告接触检出、完整救/损、pass与clear clip/slot成本，强转轻也单列；轻提醒可接受但打扰成本不记为零，不设零损失门槛。共向假设不替代外参、时钟或空间投影核验。

[按高度评分与校准](../research/active/dtr-r0/nearfield/CNH_GRADED_HEIGHT_CAL_DEV_20261010.md)完成共享/分别评分×共享/分高度预算四组合、两特征路、288单元。强档ordinary OR保留，完整single轻档HEAD351/344/346、BODY307/310/312各/384仍作参照。空间评分filter50分别拟合＋共享预算的HEAD345/335/342、BODY295/299/300，对共享评分BODY救/损8/3、22/1、0/8；pass106/125/130各/256，clear70/74/72各/6656，seed956收益含边缘/横杆，seed957损失和推迟亦集中弱族。分别评分保留候选，硬分预算未稳定追加收益；下一距离峰跟踪后的断续支持，不追加本批固定参数调参。有峰不证明目标存在/覆盖，过滤不证明free；cal预算为cutoff半实例的candidate light-query并集，评价light按最高等级计费，同cal预算非val等成本。仅已消费模拟Development、小评分cal拟合、未接入App，无主干重训/前向/新采样；原M3/5格/L2/body truth/fullbin/480及旧stop保留，前轮见[走廊峰比较](../research/active/dtr-r0/nearfield/CNH_GRADED_CORRIDOR_DEV_20261010.md)。

已完成的[步行参考](../research/active/dtr-r0/nearfield/CNH_WALKING_CORRIDOR_DEV_20261010.md)、[方向合同](../research/active/dtr-r0/nearfield/CNH_DIRECTION_CONTRACT_DEV_20261009.md)和全部方向聚合/锚定/高度混合结果作为历史诊断保留，不再驱动本阶段主线。步行参考的18–20°是过去/未来轨迹差异，非ToF姿态精度；单位/名义时钟限制仍按原报告。更新前CNH当前正文见Git 53bb1069同路径。

## 停止与保留

[旧扰动训练](../research/active/dtr-r0/nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)、T/T2、旧参照和门控的停止规则仍按各自run适用；不同机制开放。[480确认](../research/active/dtr-r0/nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)原身份保留，UNKNOWN/三态/query覆盖待验证。原Nymeria位移实现、−19°、设备/City/保护test/新UE及硬件第二阶段按原run暂缓；101/101与53ms属A+LOCAL，M3/CNH实机效果未建立。

模拟AABB、有限背景、人工误差、相关K/帧及探索选择不构成实机、安全或独立确认。更新前CNH当前正文保留于Git 5844eacd同路径；RGB子线继续沿用。

[路线当前页](../research/active/dtr-r0/CURRENT.md) · [RUNS](../research/active/dtr-r0/RUNS.md)。
