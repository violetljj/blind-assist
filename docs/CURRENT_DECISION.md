# 当前研究决定

更新：2026-10-09。主线为盲杖互补前视感知，M3和A+LOCAL保留；原5格局部融合和L2保留。

RGB独立子线：[尺度、场景信息与度量表征诊断](../research/active/dtr-r0/nearfield/RGB_BODY_QUERY_METRIC_DIAGNOSTIC_DEV_20261009.md)。沿用3RScan96帧/12环境7/2/3隔离，1440格825正/615UNKNOWN，无满足全部可观测采样射线free判据的负query。冻结Depth Pro的RGB预测深度+learned query读出，validation正/free query-ray1071771/1795858，召回/误支持.70048/.10663，匹配geometry .61966/.11535；新增32特征较弱(.65990/.10880)。depth-only打乱退化至.64691/.19574，相同head ray/query的417179近邻反标签pair排序.84421，支持场景信息。三环境和全距离带取舍并列，第三环境召回下降，前两环境误支持增加；cal阈值固定，不以validation改模型。GPU166.096/600s，独立SHA/预测复算通过。保留简洁预测深度读出候选，下一充分参考负query与步行/细结构覆盖；真实射线Development非完整身体/整盒/事件证据。旧两query/SANPO负结果及停止规则、真实硬目标、CNH预算保留。

## 当前：加权方向读出改善部分偏差边界，理想BODY损失与成本取舍保留

[三方向聚合比较](../research/active/dtr-r0/nearfield/CNH_DIRECTION_AGGREGATION_DEV_20261009.md)完成三seed×control/weak_pass×single/max/weighted×ideal/+3×20/30/40%点，共108 validation组合及54 ideal-cal阈值。声明δ=3°，方向ideal为−3/0/+3、偏差+3分支为0/+3/+6；逐帧raw聚合后沿用原last5平滑，加权固定.25/.50/.25，single取中心。每聚合器仅在ideal cal按原clear65 slot、pass51/76/102 clip预算选最低可行完整ties阈值，固定用于两validation分支。

weak加权相对同臂同seed同点中心single：+3的9个seed×点1cm HEAD净增全部为正，BODY8正1负；ideal的1cm BODY9格全负。40%点+3 H救/损3/1、5/1、4/0，B3/2、5/1、6/0，各/128；ideal H0/7、1/3、1/4，B0/10、2/8、1/8。+3 clear clip139/135/142（single148/120/140）各/512，pass130/127/123（single136/122/122）各/256，分别报告且不折算，未形成一致同成本支配。

max也有浅HEAD收益与明显BODY损失：40% weak +3 1cm H净+2/+13/+6、B−7/−1/−4，ideal B−29/−17/−36，各/128。完整双模型/seed/点及M3配对均保留，不由validation选赢家、不设零损失门槛。M3/原5格/L2及原weak独立读出候选保留；加权仅保留为方向偏差下的Development取舍方案，未升级。

24组中心0/+3回放共958464个分数差精确0；42方向分数文件、82944接触事件账本及2592方向切片留证。argmax绝对角/相对偏移/端点只作描述，未作为校正器。与[逐帧机制链](../research/active/dtr-r0/nearfield/CNH_PASS_MECHANISM_CHAIN_DEV_20261009.md)一致：原身体truth保持，完整bin继续使用。已消费模拟Development、相关K/帧/seed；声明误差范围未获实机依据，无训练/新观察/−3中心偏差分支/硬件/保护480访问。预算、核验与全表见报告。

## 停止与保留

[旧扰动训练](../research/active/dtr-r0/nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)、T/T2、旧参照和门控的停止规则仍按各自run适用；不同机制开放。[480确认](../research/active/dtr-r0/nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)原身份保留，UNKNOWN/三态/query覆盖待验证。方向/Nymeria位移、−19°、设备/City/保护test/新UE及硬件第二阶段暂缓；101/101与53ms属A+LOCAL，M3/CNH实机效果未建立。

模拟AABB、有限背景、人工误差、相关K/帧及探索选择不构成实机、安全或独立确认。更新前CNH当前正文保留于Git cc1a9737同路径；RGB子线继续沿用。

[路线当前页](../research/active/dtr-r0/CURRENT.md) · [RUNS](../research/active/dtr-r0/RUNS.md)。
