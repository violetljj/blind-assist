# 当前研究决定

更新：2026-10-09。主线为盲杖互补前视感知，M3和A+LOCAL保留；不恢复历史动态研究。

## 当前：竖杆受控配对诊断完成，原M3不动

[受控配对](../research/active/dtr-r0/nearfield/CNH_VERTICAL_CONTROLLED_PAIR_DEV_20261009.md)复用旧48竖杆期望、固定背景几何与sensor轨迹，生成K4共192个新模拟Poisson序列；共享count/减项与独立残余保持原边际，是人工统计耦合。±3°同光子投影、4224冻结五模型输入，无新期望渲染/训练/实机采集。本轮CPU180秒/GPU240秒上限，GPU28.093秒，独立核验通过。原M3及固定5格融合不动。

近f10–13共1024个同K配对高度窗，非上轮4096个K4×K4出现；八组contact−clear当前缩放支持差均值/中位数全正，内部正量份额差全负。条件线性支持差也全正，但在输入/特征FP16舍入前，不是实际FP16或非线性份额期望。共有随机分量相关，目标遮挡造成的可见背景差保留，未隔离纯目标回波或证明M3增量。

+3°镜像1cm的raw M3差均值+.681，中位数+.636；last5后−.562/−.237。当前贡献+.352、前四窗−.914，19/128窗raw差正而平滑差负。定位的是分数序列贡献，raw仍含past8，不能归为回波拖尾或直接关闭last5。下一仅建议复用已存响应定位前四窗的宽度/ρ/高度/距离条件，再定是否值得同成本读出对照，细分定位NOT_RUN。

[上轮缓存](../research/active/dtr-r0/nearfield/CNH_VERTICAL_VISIBLE_SUPPORT_DEV_20261009.md)远段−3/+3融合59/86各/672，靠近侧历史占比差仅+.417/+.164百分点，主要远段成本仍未解决；原clear名义gap仅15cm，无多间隔规律/定位精度结论。真实方向误差分布未知，竖杆排序仅本批压力条件；得到实际分布后结合尾部、可补回量和验证成本重评。

[5格候选](../research/active/dtr-r0/nearfield/CNH_BAR_FUSION_VERTICAL_DEV_20261009.md)ideal H543/B461各/688、clear46/4576格25段23clip，不升级M3。[分类表](../research/active/dtr-r0/nearfield/CNH_BAR_CATEGORY_PRIORITY_DEV_20261009.md)可补回量未知；暗4cm仍第二缺口BODY7/56，检出也可能受方向误差影响。任何新读出须同配对同报警成本报救回/损失，均值差不代替性能，不做全组d′或借N64阈值。

[单帧参照](../research/active/dtr-r0/nearfield/CNH_BAR_BACKGROUND_REFERENCE_DEV_20261009.md)停止，目标远峰容差null非零信号，联合/扰动/强背景复查不启动。时间累积暂缓；新依据支持可能改善参照或提供增量时可考虑诊断，无需启动前已证明。[旧门控](../research/active/dtr-r0/nearfield/CNH_BAR_BOUNDARY_CONTRAST_DEV_20261009.md)Dwin/Dmax不细化，裁剪/三query min不采用，不关闭不同机制。480确认身份及旧停止结果保留。本轮属于模拟AABB、固定有限背景、人工误差、loss条件选择后的已消费Development，非实机/安全/确认。

## 停止与保留

[扰动训练](../research/active/dtr-r0/nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)原配方exact各损7超2，停止，不加seed/epoch/换分布；不同机制开放。L2、[480确认](../research/active/dtr-r0/nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)原身份保留；UNKNOWN/三态/query覆盖待验证。方位微调、稀疏空间/关联L3、方向/Nymeria位移接口、−19°及设备/City/保护test/新UE/硬件第二阶段暂缓；101/101、53ms属A+LOCAL，M3/CNH实机效果未建立。

[路线当前页](../research/active/dtr-r0/CURRENT.md) · [RUNS](../research/active/dtr-r0/RUNS.md)。更新前全文：Git 1a8b3e8a同路径。
