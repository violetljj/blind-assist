# 当前研究决定

更新：2026-10-09。主线为盲杖互补前视感知，M3和A+LOCAL保留；原5格局部融合和L2保留。

RGB独立子线：[冻结迁移与锚点旁路](../research/active/dtr-r0/nearfield/RGB_BODY_QUERY_TRANSFER_DEV_20261009.md)。原模型/归一化/cal实际cutoff冻结，无训练；8个未用3RScan环境64帧，正/free query-ray3400061/4344666，depth-only召回/误支持.59236/.08489 vs geometry .50464/.09513，6/8环境TP净增、2/8净减；相对raw DP .64005/.15911是取舍。ARKitScenes iPadPro单捕获16帧，depth-only .29567/.04188 vs geometry .42204/.05606，两项同时改善与充分近场增量未复现；场景、原图分辨率/K、预测距离及读出适配原因未定。28全采样free格各臂支持0/28，非整盒/身体误报；步行提前量仍缺事件/身体参考。同源实测锚点旁路改善直接几何但不证明ToF增量。GPU154.136/900s，官方数据357569481B/2GiB，独立保存预测复算及旧帧CUDA逐值核验PASS。下一高分辨率不同相机多环境与预测质量诊断、完整query/步行参考；旧两query/32特征不续训，原负结果/真实硬目标/CNH预算保留。

## 当前：加权方向读出改善部分偏差边界，理想BODY损失与成本取舍保留

[三方向聚合比较](../research/active/dtr-r0/nearfield/CNH_DIRECTION_AGGREGATION_DEV_20261009.md)完成三seed×control/weak_pass×single/max/weighted×ideal/+3×20/30/40%点，共108 validation组合及54 ideal-cal阈值。声明δ=3°，方向ideal为−3/0/+3、偏差+3分支为0/+3/+6；逐帧raw聚合后沿用原last5平滑，加权固定.25/.50/.25，single取中心。每聚合器仅在ideal cal按原clear65 slot、pass51/76/102 clip预算选最低可行完整ties阈值，固定用于两validation分支。

weak加权相对同臂同seed同点中心single：+3的9个seed×点1cm HEAD净增全部为正，BODY8正1负；ideal的1cm BODY9格全负。40%点+3 H救/损3/1、5/1、4/0，B3/2、5/1、6/0，各/128；ideal H0/7、1/3、1/4，B0/10、2/8、1/8。+3 clear clip139/135/142（single148/120/140）各/512，pass130/127/123（single136/122/122）各/256，分别报告且不折算，未形成一致同成本支配。

max也有浅HEAD收益与明显BODY损失：40% weak +3 1cm H净+2/+13/+6、B−7/−1/−4，ideal B−29/−17/−36，各/128。完整双模型/seed/点及M3配对均保留，不由validation选赢家、不设零损失门槛。M3/原5格/L2及原weak独立读出候选保留；加权仅保留为方向偏差下的Development取舍方案，未升级。

24组中心0/+3回放共958464个分数差精确0；42方向分数文件、82944接触事件账本及2592方向切片留证。argmax绝对角/相对偏移/端点只作描述，未作为校正器。与[逐帧机制链](../research/active/dtr-r0/nearfield/CNH_PASS_MECHANISM_CHAIN_DEV_20261009.md)一致：原身体truth保持，完整bin继续使用。已消费模拟Development、相关K/帧/seed；声明误差范围未获实机依据，无训练/新观察/−3中心偏差分支/硬件/保护480访问。预算、核验与全表见报告。

## 停止与保留

[旧扰动训练](../research/active/dtr-r0/nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)、T/T2、旧参照和门控的停止规则仍按各自run适用；不同机制开放。[480确认](../research/active/dtr-r0/nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)原身份保留，UNKNOWN/三态/query覆盖待验证。方向/Nymeria位移、−19°、设备/City/保护test/新UE及硬件第二阶段暂缓；101/101与53ms属A+LOCAL，M3/CNH实机效果未建立。

模拟AABB、有限背景、人工误差、相关K/帧及探索选择不构成实机、安全或独立确认。更新前CNH当前正文保留于Git cc1a9737同路径；RGB子线继续沿用。

[路线当前页](../research/active/dtr-r0/CURRENT.md) · [RUNS](../research/active/dtr-r0/RUNS.md)。
