ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED

# 前视障碍感知：当前状态

更新：2026-10-09。盲杖互补前视感知主线；原M3、5格局部融合、L2和A+LOCAL保留。

RGB独立子线：[真实跨环境参考与query训练](nearfield/RGB_BODY_QUERY_CROSS_SESSION_DEV_20261009.md)。3RScan96帧/12官方train环境按7/2/3隔离，1440格825正/615UNKNOWN，完整负query0；RGB+公开K/query独立推理，传感器距离仅监督/评分。DP validation检出146/198；两query候选均弱于无RGB控制，多尺度RGB射线召回/FPR .50386/.14587，对照.60742/.11727，shuffle退化尚无收益。GPU阶段184.566/600s，6fixture及几何/预测复算通过。下一几何预训练表征与步行/细结构负例覆盖；室内扫描射线非整盒/身体/事件真值。旧SANPO补标与DP/VDA负结果、真实硬目标、旧BodyQuery/COPILOT定位与CNH预算/停止规则各自保留。

## 当前：双条件独立读出候选保留，方向敏感性已定位到侧别

[已有工作点与姿态诊断](nearfield/CNH_PASS_POSE_DIAGNOSTIC_DEV_20261009.md)仅整理既有20/30/40% cal点，无新阈值搜索、训练或推理。40% weak独立读出在ideal H316/309/307、B282/264/279各/384；clear clip8/5/2（M3 29、旧融合28）各/512，pass93/97/93（旧/M3 60）各/256。+3° H306/289/299、B278/258/259各/384；clear clip148/120/140（两基线172）各/512，pass136/122/122（M3 138、旧139）各/256。两条件总体高于M3，但ideal pass代价更大，且+3 BODY后两seed比旧融合少3/2，不能称全面替换。

40% weak同模型ideal→+3的1cm HEAD救9/10/10、损16/28/18，各/128；相对+3 M3救/损14/4、3/3、10/3。30%点weak HEAD跨姿态损失13/25/17全部正x侧，每侧64。两臂均有负x分数上升、正x下降，支持方向敏感性线索，不能归因为弱监督边界变锐；跨模型logit尺度不可直接比较因果。

严格同反射率HEAD/BODY匹配0，双方各128事件未匹配；32对几何对应反射率随高度组交换，不能消除混杂。固定f10前向z距全1.45m，距离分层无新对照。保留weak低擦边独立读出候选，M3/原5格/L2不升级；不要求零损失，不据validation选上线点。下一优先逐事件输入支持与跨阈值对应；反号回放及同rho高度配对保留，均未执行。

[弱pass匹配续训](nearfield/CNH_PASS_BOUNDARY_DEV_20261009.md)与[反事实/摘要负结果](nearfield/CNH_COUNTERFACTUAL_DEV_20261009.md)保留。OR20%仍因旧pass下限55>预算51不可行，不优先继续放宽weak OR预算。沿用已消费模拟Development、有限背景/AABB及相关K/帧，无新光子、设备或保护480访问。完整救/损、分母、CPU收据与核验见报告。

## 停止与保留

[旧扰动训练](nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)、T/T2、旧参照和门控的停止规则仍按各自run适用；不同机制开放。[480确认](nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)原身份保留，UNKNOWN/三态/query覆盖待验证。方向/Nymeria位移、−19°、设备/City/保护test/新UE及硬件第二阶段暂缓；101/101与53ms属A+LOCAL，M3/CNH实机效果未建立。

模拟AABB、有限背景、人工误差、相关K/帧及探索选择不构成实机、安全或独立确认。更新前CNH当前正文保留于Git 44dd8ed6同路径；RGB子线继续沿用。

[RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)。
