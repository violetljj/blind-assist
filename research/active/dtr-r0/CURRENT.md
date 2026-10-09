ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED

# 前视障碍感知：当前状态

更新：2026-10-09。主线为盲杖互补前视感知，M3和A+LOCAL保留；不恢复历史动态研究。

## 当前：query成本主要在M3，最小值门控不采用

[形状能力图](nearfield/CNH_ALIGNED_SHAPES_DEV_20261008.md)492 AABB×K4，原−10°/理想位姿；316物理接触、344高度接触、88pass/88clear。原M3 HEAD517/BODY406及时，各/688；clear46/4576格、25段、24/352clip。非实机提醒频率。

[5格融合](nearfield/CNH_BAR_FUSION_VERTICAL_DEV_20261009.md)θM=.940418、θL=4.625390的OR：H543/B461，救28/56损2/1；暗4cm BODY1→7/56损0；同原46格，clear25段/23clip。四Development点中事后工程选择，不升级原M3政策，真值形状只评价不进门控。

[query成本诊断](nearfield/CNH_BAR_QUERY_COST_DEV_20261009.md)新完成：+3°融合175/4576格，其中raised-M3已触发170、local独有5（ideal也5）；成本主要来自M3。+3°相对ideal新增158、移除29、净+129格，竖杆净+106。几何SAT/独立裁剪：±1/2°原clear物体未入query，±3°仅f3少量edge_plate进入；+3°175报警中174当帧无几何接触，158新增全部无，不能解释为大多数物体进入。

三query M3取min、local保持中心：−2..2° clear下降，但ideal H543→516/B461→429 /688，损27/32、救0，不采用。±3缺±4缓存NOT_EVALUABLE；部署需3query/15单网forward。无新推理/训练/采样，主.796+出图1.000/240s、几何60.640/120s、审计3.188/120s通过。下一竖杆clear的hist角距→投影query→M3逐级配对NOT_RUN，不先归因。

[上一轮迁移](nearfield/CNH_BAR_TRANSFER_DEV_20261009.md)保留：10整场景留出H/B全净增但4次成本增；18位姿档相对同档M3全净增，query+3°B461→409/688、clear46→175/4576。两新背景暗B4→9、6→10/56损0，clear42→40、32→32，强段/clip各+2；适配另列，不升级政策。几何真值只评价、人工Y轨迹和有限背景不等于真实鲁棒性。

[投影/FP16](nearfield/CNH_BAR_REPRESENTATION_DEV_20261008.md)条件d²保留下界≥96.1847%，远负差未覆盖；[读出](nearfield/CNH_BAR_READOUT_DEV_20261008.md)方差失配68/76，d_J仅敏感性。[局部替换](nearfield/CNH_BAR_LOCAL_READOUT_DEV_20261009.md)全批损40/55仍保留；尺度/形状库/跨窗/远箱支路开放，不自动训练。

## 停止与保留

[扰动训练](nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)原配方exact各损7超2护栏，停止、不加seed/epoch/换分布；实质不同机制仍开放。方位微调、稀疏空间/关联L3及方向/Nymeria位移接口暂缓；−19°不推进。

[480确认](nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)原身份保留，E1较exact少39/51、EMA收回25/34（/1002）；后续已消费Development不称新确认，UNKNOWN/三态/query覆盖待验证。

[Nymeria](nearfield/CNH_NYMERIA_SAMPLE_AUDIT_DEV_20261008.md)姿态准确度未评、位置/PDR未实现，E1整链NOT_EVALUABLE，future pelvis/闭环不进估计器。设备、City、保护test、新UE/硬件第二阶段暂停；101/101、53ms属A+LOCAL，M3/CNH实机效果未建立。

[RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)。更新前全文：Git 2848e983同路径。
