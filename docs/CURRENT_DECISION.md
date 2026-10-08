# 当前研究决定

更新：2026-10-09。主线为盲杖互补的前视障碍感知，冻结M3和手机A+LOCAL保留；DTR_R2_DYNAMIC_RETAINED仅历史，不恢复动态研究。

## 当前：保留固定5格融合，优先query误差成本

[形状能力图](../research/active/dtr-r0/nearfield/CNH_ALIGNED_SHAPES_DEV_20261008.md)492 AABB×K4，原−10°/理想位姿；316物理接触、344高度接触、88pass/88clear。原M3 HEAD517/BODY406及时，各/688；clear46/4576格、25段、24/352clip。非实机提醒频率。

[5格融合](../research/active/dtr-r0/nearfield/CNH_BAR_FUSION_VERTICAL_DEV_20261009.md)θM=.940418、θL=4.625390的OR：H543/B461，救28/56损2/1；暗4cm BODY1→7/56损0；同原46格，clear25段/23clip。四Development点中事后工程选择，不升级原M3政策，真值形状只评价不进门控。

[本轮迁移检查](../research/active/dtr-r0/nearfield/CNH_BAR_TRANSFER_DEV_20261009.md)已完成：5种子互换10次整场景留出（K4/双高度/完整窗），校准k按分母缩2.5，cal全匹配；留出H/B全净增，9次有损，clear重复出现230→232/22880，4次成本增，差−5..+7，不是独立确认。

位姿query/恒定外参/逐帧抖动1/2/3°正负18档，5076已知点注入解析通过；仅改算法位姿，两支共享投影并重跑冻结M3。各档相对同档M3两高度全净增，但4档融合clear更高。query +3°融合BODY461→409/688、clear46→175/4576；末帧外参相消符合公式，历史非零。一个人工Y轴抖动轨迹非真实分布。下一优先定位query误差的报警成本，未追加运行。

两新模拟背景固定阈值：低H515→547/B413→453，救33/43损1/3；强H512→545/B405→457，救36/53损3/1，各/688。暗BODY4→9、6→10/56损0；clear42→40、32→32/4576，强段/clip各+2。另列cal-only适配留出，不当固定迁移；低BODY损5（固定损2），不自动替换。GPU-wall451.860/1800s、117.484/1200s；独立复核通过，无训练/实测，仅背景新光子。

[投影/FP16](../research/active/dtr-r0/nearfield/CNH_BAR_REPRESENTATION_DEV_20261008.md)条件d²保留下界≥96.1847%，远负差未覆盖；[读出](../research/active/dtr-r0/nearfield/CNH_BAR_READOUT_DEV_20261008.md)方差失配68/76，d_J仅敏感性。[局部替换](../research/active/dtr-r0/nearfield/CNH_BAR_LOCAL_READOUT_DEV_20261009.md)全批损40/55仍保留；尺度/形状库/跨窗/远箱支路开放，不自动训练。

## 停止与保留

[扰动训练](../research/active/dtr-r0/nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)原配方exact各损7超2护栏，停止、不加seed/epoch/换分布；实质不同机制仍开放。方位微调、稀疏空间/关联L3及方向/Nymeria位移接口暂缓；−19°不推进。

[480确认](../research/active/dtr-r0/nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)原身份保留，E1较exact少39/51、EMA收回25/34（/1002）；后续已消费Development不称新确认，UNKNOWN/三态/query覆盖待验证。

[Nymeria](../research/active/dtr-r0/nearfield/CNH_NYMERIA_SAMPLE_AUDIT_DEV_20261008.md)姿态准确度未评、位置/PDR未实现，E1整链NOT_EVALUABLE，future pelvis/闭环不进估计器。设备、City、保护test、新UE/硬件第二阶段暂停；101/101、53ms属A+LOCAL，M3/CNH实机效果未建立。

[路线当前页](../research/active/dtr-r0/CURRENT.md) · [RUNS](../research/active/dtr-r0/RUNS.md)。更新前全文：Git 839177bd同路径。
