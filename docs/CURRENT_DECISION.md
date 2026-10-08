# 当前研究决定

更新：2026-10-09。主线是盲杖互补的前视障碍感知，冻结M3和手机A+LOCAL保留；历史动态研究不恢复。

## 当前：保留5格预算交换融合候选，不升级M3政策

用户授权对齐直行、理想位姿、不实测，优先横杆/竖杆/柜体突出物，标牌补充。[形状能力图](../research/active/dtr-r0/nearfield/CNH_ALIGNED_SHAPES_DEV_20261008.md)492 AABB场景×K4：316物理接触、344高度接触、88pass/88clear。原−10°/M3/θ下HEAD517/688、BODY406/688及时，clear46/4576格、25段、24/352clip，非真实提醒频率。

[同成本融合及竖杆诊断](../research/active/dtr-r0/nearfield/CNH_BAR_FUSION_VERTICAL_DEV_20261009.md)：5/10格×中段/无下界四点全匹配原46/4576格。推荐5格无下界OR（θM=.940418、θL=4.625390）继续研究：HEAD517→543/688救28损2，BODY406→461/688救56损1；暗4cm BODY1→7/56救6损0，30/90cm分别0→3、1→4/28。clear仍25段、clip24→23/352，pass仍74/352；竖杆救5/9损0/0。形状真值只分组，不进门控。

37×64保存条件draw复核，5格候选HEAD119→217、BODY70→157/896，各损3，fixedθ clear仍9/832。10格无下界原K4收益更高但BODY损5，MC clear11/832（原9），未再校准。推荐为Development事后工程判断，非独立确认。融合总.828/180s；竖杆投影9.046/600s、分析/审计7.266/180s，无新采样/训练/M3推理，核验通过。

竖杆448事件的T/G/N全保留：原局部替换损15/22件，支持内峰值中位3.250/3.402（原localθ4.069）。几何支持不等于回波归因，三分数不能确立因果。下一固定5格候选检验不同背景下预算与原检出保持，NOT_RUN；形状库/经验尺度/跨窗一致性/远箱支路开放，不自动训练/实测。

[表示/FP16](../research/active/dtr-r0/nearfield/CNH_BAR_REPRESENTATION_DEV_20261008.md)保留条件d²≥96.1847%，远负差几乎未覆盖；[M3读出](../research/active/dtr-r0/nearfield/CNH_BAR_READOUT_DEV_20261008.md)方差失配68/76，d_J仅敏感性。[局部替换](../research/active/dtr-r0/nearfield/CNH_BAR_LOCAL_READOUT_DEV_20261009.md)暗BODY救8损0/56，但全批HEAD/BODY救53/72损40/55，原负结果不覆盖；[−19°](../research/active/dtr-r0/nearfield/CNH_ALIGNED_BOUNDARY_DEV_20261008.md)仍不推进。

## 停止与保留

[扰动续训](../research/active/dtr-r0/nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)原配方停止：exact各损7超2件护栏，不加seed/epoch、不换分布；实质不同机制仍可审查。方位微调、稀疏空间/关联L3、方向/Nymeria位移接口暂缓。

[480确认](../research/active/dtr-r0/nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)原身份保留，E1较exact少39/51及时、EMA收回25/34（/1002）。本轮和BlindWays已消费回放不重称确认；UNKNOWN/三态/query覆盖待验证，盲杖互补不保证安全。

[Nymeria](../research/active/dtr-r0/nearfield/CNH_NYMERIA_SAMPLE_AUDIT_DEV_20261008.md)姿态准确度未评，位置/PDR未实现，E1整链NOT_EVALUABLE；future pelvis/闭环参考不进估计器。设备、City、保护test、新UE/硬件第二阶段暂停；101/101和53ms属A+LOCAL，M3/CNH实机效果未建立。

[路线当前页](../research/active/dtr-r0/CURRENT.md) · [RUNS](../research/active/dtr-r0/RUNS.md)。更新前全文：Git daed724c同路径。
