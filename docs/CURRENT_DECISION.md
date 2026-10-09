# 当前研究决定

更新：2026-10-09。主线为盲杖互补前视感知，M3和A+LOCAL保留；不恢复历史动态研究。

## 当前：竖杆空间混合参与query响应，直接裁剪不采用

[形状能力图](../research/active/dtr-r0/nearfield/CNH_ALIGNED_SHAPES_DEV_20261008.md)492 AABB×K4，原−10°/理想物理位姿；316物理接触、344高度接触、88pass/88clear。原M3 H517/B406及时各/688，clear46/4576格、25段、24/352clip。非实机提醒频率。

[5格融合](../research/active/dtr-r0/nearfield/CNH_BAR_FUSION_VERTICAL_DEV_20261009.md)固定θM=.940418/θL=4.625390的OR：H543/B461，救28/56损2/1；暗4cm BODY1→7/56损0；clear46格、25段/23clip。四Development点中事后工程选择，不升级M3，真值形状不进门控。

[竖杆逐级配对](../research/active/dtr-r0/nearfield/CNH_BAR_VERTICAL_EVIDENCE_DEV_20261009.md)新完成：132竖杆×K4、ideal/±3°；观测不变且24clear竖杆均未入query，f13对应高度投影正均值差在ideal已非零，±3°约增3–4倍；远负差在该query基本未覆盖。支持空间混合参与，不是物体进入/信息损失率。

只改tot/lst、cnt及入网处理保留：裁query外回波ideal clear12→36/1248，H217→192/B177→159各/224；+3°clear117→45，但H损21救1、B损12救12。外部回波单独+3°clear17，zero_echo M3各档0。分布干预不是物理移除，裁剪不采用。无新光子/训练；新增冻结M3 82368 ensemble输入，94.297/900s；观测7.365/180s含序列化失败，独立58.234/120s、5376事件核验。下一保留背景上下文的边界/侧别对比仅建议NOT_RUN。

[上一轮query成本](../research/active/dtr-r0/nearfield/CNH_BAR_QUERY_COST_DEV_20261009.md)：+3°全批fusion175/4576，raised170/local独有5；竖杆净+106。三query min门控ideal损H27/B32，仍不采用。[迁移](../research/active/dtr-r0/nearfield/CNH_BAR_TRANSFER_DEV_20261009.md)：10场景留出全净增但4次成本增；18人工位姿档全相对净增但绝对成本敏感；两新背景暗B4→9、6→10/56损0，clear42→40/32→32，强段/clip各+2，不称普遍迁移。

[投影/FP16](../research/active/dtr-r0/nearfield/CNH_BAR_REPRESENTATION_DEV_20261008.md)暗横杆条件d²保留下界≥96.1847%，与空间归属混合是不同问题；[M3读出](../research/active/dtr-r0/nearfield/CNH_BAR_READOUT_DEV_20261008.md)方差失配68/76，d_J仅敏感性；[局部替换](../research/active/dtr-r0/nearfield/CNH_BAR_LOCAL_READOUT_DEV_20261009.md)全批损40/55，不替换M3。形状库/跨窗/远箱支路仍开放。

## 停止与保留

[扰动训练](../research/active/dtr-r0/nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)原配方exact各损7超2护栏，停止、不加seed/epoch/换分布；实质不同机制仍开放。方位微调、稀疏空间/关联L3及方向/Nymeria位移接口暂缓；−19°不推进。

[480确认](../research/active/dtr-r0/nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)原身份保留，E1较exact少39/51、EMA收回25/34（/1002）；后续已消费Development不称新确认，UNKNOWN/三态/query覆盖待验证。

[Nymeria](../research/active/dtr-r0/nearfield/CNH_NYMERIA_SAMPLE_AUDIT_DEV_20261008.md)姿态准确度未评、位置/PDR未实现，E1整链NOT_EVALUABLE，future pelvis/闭环不进估计器。设备、City、保护test、新UE/硬件第二阶段暂停；101/101、53ms属A+LOCAL，M3/CNH实机效果未建立。

[路线当前页](../research/active/dtr-r0/CURRENT.md) · [RUNS](../research/active/dtr-r0/RUNS.md)。更新前全文：Git 3590176b同路径。
