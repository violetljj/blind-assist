# 当前研究决定

更新：2026-10-09。主线为盲杖互补前视感知，M3和A+LOCAL保留；不恢复历史动态研究。

## 当前：竖杆可见支持缓存诊断完成，原M3不动

[可见支持诊断](../research/active/dtr-r0/nearfield/CNH_VERTICAL_VISIBLE_SUPPORT_DEV_20261009.md)只读已有±3°缓存：远f3–9两侧clear1344格、近f10–13同侧/镜像浅接触4096配对出现。新60秒累计预算内完成独立核验，无新几何/投影/推理/训练/采样。原M3/固定5格候选不动；优先级仅本批压力条件，真实误差分布未知。

远段−3/+3°融合clear59/86各/672格，其中仅last5后报警2/6格；raw输入仍含past8，不是单帧或历史因果对照。靠近侧同场景/帧两态K4的历史支持占比差均值仅+0.417/+0.164百分点，当前占比方向不一致，远段成本机制未解决。[旧几何](../research/active/dtr-r0/nearfield/CNH_VERTICAL_GEOMETRY_JOIN_DEV_20261009.md)真实clear间隔仅15cm，SAT代理不是定位精度。

近段clear联合−3°同侧0/128、镜像6/128，+3°1/112、22/112；每depth接触高度机会128，in1/in4报警窗−3°32/88、+3°11/69。八组contact−clear当前内部正量占比均值/中位数皆负，预期失败；当前缩放支持量皆正但逐配对不全正。不是稳定检出增益，也未证明超出M3的增量。

下一提案：固定背景/配对seed验证浅接触“内部当前缩放支持量较高而内部份额较低”的线索，NOT_RUN，需另定预算和判据；它只回答近段保护，不能解释主要远段成本。统一可见信息/处理规则，不要求输入数值相同；不同seed混杂未排除，镜像一致非充分/必要条件。不做全组d′或旧门控细化；若另立读出，与M3须同配对同报警成本，不借N64阈值。

[5格融合](../research/active/dtr-r0/nearfield/CNH_BAR_FUSION_VERTICAL_DEV_20261009.md)ideal H543/B461各/688，救28/56损2/1；clear46/4576格、25段/23clip。四Development点事后候选，不升级原M3。[类别表](../research/active/dtr-r0/nearfield/CNH_BAR_CATEGORY_PRIORITY_DEV_20261009.md)后续可补回量全未知；暗4cm横杆仍第二位缺口，ideal BODY7/56，但也可能受方向误差影响。

[单帧参照](../research/active/dtr-r0/nearfield/CNH_BAR_BACKGROUND_REFERENCE_DEV_20261009.md)停止：上下及时H0/B1各/56、同行0/0、目标远峰容差null；联合及±3°/强背景复查不启动，不证明hist无远段信号。时间累积暂缓；新依据支持改善参照或提供增量时可考虑诊断，可靠性/增量无需启动前证明，累积判据重冻。[旧门控](../research/active/dtr-r0/nearfield/CNH_BAR_BOUNDARY_CONTRAST_DEV_20261009.md)Dmax不动、Dwin损浅接触；裁剪/三query min也不采用，降低这些配方优先级，不关闭不同机制。

旧索引维护3429243a独立提交并检查通过；旧ca65c9b2在远端历史已核对。整体仍模拟/AABB/有限背景/理想或人工误差的已消费Development，非实机、安全或确认。

## 停止与保留

[扰动训练](../research/active/dtr-r0/nearfield/CNH_QUERY_PERTURB_TRAIN_DEV_20261008.md)原配方exact各损7超2，停止，不加seed/epoch/换分布；不同机制开放。L2、[480确认](../research/active/dtr-r0/nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)原身份保留；UNKNOWN/三态/query覆盖待验证。方位微调、稀疏空间/关联L3、方向/Nymeria位移接口、−19°及设备/City/保护test/新UE/硬件第二阶段暂缓；101/101、53ms属A+LOCAL，M3/CNH实机效果未建立。

[路线当前页](../research/active/dtr-r0/CURRENT.md) · [RUNS](../research/active/dtr-r0/RUNS.md)。更新前全文：Git 2537c921同路径。
