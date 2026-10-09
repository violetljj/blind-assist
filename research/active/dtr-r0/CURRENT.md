ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED

# 前视障碍感知：当前状态

更新：2026-10-09。主线为盲杖互补前视感知，M3和A+LOCAL保留；不恢复历史动态研究。

## 当前：单帧背景参照预检未通过，保留原M3

[形状能力图](nearfield/CNH_ALIGNED_SHAPES_DEV_20261008.md)492 AABB×K4，原−10°/理想物理位姿；316物理接触、344高度接触、88pass/88clear。原M3 H517/B406及时各/688，clear46/4576格、25段、24/352clip。非实机提醒频率。

[5格融合](nearfield/CNH_BAR_FUSION_VERTICAL_DEV_20261009.md)固定θM=.940418/θL=4.625390的OR：H543/B461，救28/56损2/1；暗4cm BODY1→7/56损0；clear46格、25段/23clip。四Development点事后工程选择，不升级M3，真值形状不进门控。

[背景参照预检](nearfield/CNH_BAR_BACKGROUND_REFERENCE_DEV_20261009.md)新完成：44暗4cm横杆×K4原hist，固定单帧双邻区找峰，上下邻行及时参照H0/B1各/56、同行左右均0/56。clear按侧别×长度整组分校准/验证；验证上下仅5/16384个zone×帧×K可用，目标远峰距离/宽度无可评价样本。几何污染上下37.36%（/1424目标zone×帧）少于同行100%（/1132），仍不足可靠覆盖。预检不通过，不启动联合读出，不证明原hist无远段信号；不处理竖杆query误报。

独立逐值重算5046272字段、224事件及几何一致；新几何计算而无新光子/期望渲染/投影/M3推理/训练。下一可另列同past8、保留角度的原hist远段统计预检，处理位姿/距离变化，NOT_RUN。原M3与5格候选不升级。

[边界内外对比](nearfield/CNH_BAR_BOUNDARY_CONTRAST_DEV_20261009.md)：Dmax全不动，Dwin ideal只省1格；±3°省格少却损浅伸入目标，不采用；降低已失败一致性/裁剪/固定内外差方案优先级，不排除不同机制。
[上一轮逐级诊断](nearfield/CNH_BAR_VERTICAL_EVIDENCE_DEV_20261009.md)：clear竖杆未入query但正均值差进入投影，±3°约3–4倍；直接裁剪ideal clear12→36且损H25/B18，不采用。[query成本](nearfield/CNH_BAR_QUERY_COST_DEV_20261009.md)：+3°全批175格主要raised170；三query min损27/32，不采用。[迁移](nearfield/CNH_BAR_TRANSFER_DEV_20261009.md)：留出/人工位姿/两新背景有相对净增，成本非普遍稳定。

[投影/FP16](nearfield/CNH_BAR_REPRESENTATION_DEV_20261008.md)暗横杆条件d²保留下界≥96.1847%，远负差未覆盖；[读出](nearfield/CNH_BAR_READOUT_DEV_20261008.md)方差失配68/76，d_J仅敏感性；[局部替换](nearfield/CNH_BAR_LOCAL_READOUT_DEV_20261009.md)全批损40/55，不替换M3。

## 停止与保留

[扰动训练](nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)原配方exact各损7超2护栏，停止、不加seed/epoch/换分布；实质不同机制仍开放。方位微调、稀疏空间/关联L3及方向/Nymeria位移接口暂缓；−19°不推进。

[480确认](nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)原身份保留，E1较exact少39/51、EMA收回25/34（/1002）；后续已消费Development不称新确认，UNKNOWN/三态/query覆盖待验证。

[Nymeria](nearfield/CNH_NYMERIA_SAMPLE_AUDIT_DEV_20261008.md)姿态准确度未评、位置/PDR未实现，E1整链NOT_EVALUABLE，future pelvis/闭环不进估计器。设备、City、保护test、新UE/硬件第二阶段暂停；101/101、53ms属A+LOCAL，M3/CNH实机效果未建立。

[RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)。更新前全文：Git d937fcda同路径。
