ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED

# 前视障碍感知：当前状态

更新：2026-10-10。盲杖互补前视感知主线；原M3、5格局部融合、L2和A+LOCAL保留。

RGB独立子线：[A0与负参考](nearfield/RGB_BODY_QUERY_A0_DEV_20261010.md)。冻结log-affine主干＋旧depth/ray的log-Z有界代理，δ=.05/.1/.2×3capture全比较、另两capture严格sampledFREE按5%三折cal完成。追加60见证120/121/122各/202、FREE各12/220（affine120）；65见证130/130/129各/179、FREE10/11/11各/247（affine131、10），arkit16退化88/71/62各/228 vs103；所有δ曲线均有低于affine区段，不提高残差训练优先级但不否定重训。新增15官方train缓存环境240帧6480格，FREE30/10环境，原cal2→候选32/12环境；近带6仅1环境。补参考后冻结带条件cal探针取舍更差，不采用；新参考预测评分未运行。当前冻结affine为RGB几何代表，下一补近场跨环境负覆盖并评价新cal冻结预测。独立query/curve和8424参考核验通过；CPU保守674/1800s，训练/推理/GPU/下载0、无常驻资源。仅Development、相关sampledFREE非整盒/身体安全证据；旧停止规则/CNH保留，前文见Git167181b5。

## 当前：共向假设下推进ToF检出与分级提醒

用户于2026-10-10明确：本阶段暂按“头朝向＝身体朝向＝行进方向”处理，ToF身体/头部query共用这一方向。以ideal/共向条件为主比较；方向估计、PDR及方向不确定性合同暂不作为本阶段前置工作。这是当前任务假设，旧偏差实验的结果和身份保留。

目标回到ToF任务收益与分级提醒：明显风险给强提醒，接触/擦边难分时允许轻提醒；不把精确分界作为继续推进的前提。分级只依据运行时可见的分数与证据质量，真值仅评价；弱证据保留不确定含义。重点是弱/细目标证据在投影、累积和完整bin读出中的保留，以及分级能否增加有用及时提醒。强/轻提醒分别报告接触检出、完整救/损、pass与clear clip/slot成本，强转轻也单列；轻提醒可接受但打扰成本不记为零，不设零损失门槛。共向假设不替代外参、时钟或空间投影核验。

[分级提醒与回波诊断](nearfield/CNH_GRADED_EVIDENCE_DEV_20261010.md)已实现并评价ideal缓存级候选。保留ordinary OR强档、ordinary single新增证据轻档：三seed总及时HEAD351/344/346、BODY307/310/312各/384，额外轻档救36/31/23和40/53/30、总损0；clear总73/107/80slots各/6656，pass149/168/158clips各/256。query内双轴降级仅省强clear2slots/pass1clip却损强提醒时序，优先保留强档；max3作次选。相近分数下趋势/波动重叠，背景可见静默未成立；下一补公共扩张走廊/局部峰及匹配背景证据，降低轻档pass成本。原M3/5格/L2/body truth/fullbin/480、weak_pass候选和旧stop保留；仅模拟Development、未接入App，无训练/前向/新采样。

已完成的[步行参考](nearfield/CNH_WALKING_CORRIDOR_DEV_20261010.md)、[方向合同](nearfield/CNH_DIRECTION_CONTRACT_DEV_20261009.md)和全部方向聚合/锚定/高度混合结果作为历史诊断保留，不再驱动本阶段主线。步行参考的18–20°是过去/未来轨迹差异，非ToF姿态精度；单位/名义时钟限制仍按原报告。更新前CNH当前正文见Git 53bb1069同路径。

## 停止与保留

[旧扰动训练](nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)、T/T2、旧参照和门控的停止规则仍按各自run适用；不同机制开放。[480确认](nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)原身份保留，UNKNOWN/三态/query覆盖待验证。原Nymeria位移实现、−19°、设备/City/保护test/新UE及硬件第二阶段按原run暂缓；101/101与53ms属A+LOCAL，M3/CNH实机效果未建立。

模拟AABB、有限背景、人工误差、相关K/帧及探索选择不构成实机、安全或独立确认。更新前CNH当前正文保留于Git 5844eacd同路径；RGB子线继续沿用。

[RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)。
