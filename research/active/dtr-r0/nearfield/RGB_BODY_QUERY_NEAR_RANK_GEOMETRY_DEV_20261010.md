# 近带分数排序与原生点级几何诊断（Development）

2026-10-10 · `RGB_BODY_QUERY_NEAR_RANK_GEOMETRY_DEV_20261010` · COMPLETE · 基线 `f053479c` · EXPLORE。

**近带问题已经超出单纯阈值位置：ARKit参考近正点存在很大的米制深度高估，同时正见证与FREE的分数排序不利。** absolute affine在arkit16、40777060恢复首个近带正见证的最小事后成本分别是41/96、56/133个FREE；旧head可把arkit16首个成本降至7/96，但40777060仍需53/133。A0三个固定δ有所改善，未得到跨capture有利取舍，不采用新工作点，不提高训练优先级。

在两capture的参考近正点上，affine预测/参考深度中位3.735、4.330倍；旧head为2.677、2.160倍，A0=.2仍为3.086、3.545倍。所检查六臂（含rawDP）的这些已知正点均位于原生query射线区间之后。旧±.2代理的修正尺度明显不足，但不能从已消费eval误差倒推训练残差界限。3RScan的raw误差方向不同，不采用统一倍率修正，不将原因定位为单一相机或硬件问题。

建议下一优先检查**ARKit模型输入与米制尺度链、近正点对应的空间/深度误差结构**，再决定是否能用train/cal确定可转移的有界残差。当前全局affine作为RGB代表保持，明确其近带能力未建立；本轮到此停止，不生成新模型预测、重新拟合或训练。

## 固定范围、角色与统计口径

跑前固定：排序用原304帧cal与136帧eval全部近query缓存，两套原读出、五臂；几何只读全部136eval＋固定新ARKitcal48，共184帧。没有因为结果选frame/capture，无新增source。GT depth/labels仅在evaluator侧作诊断，不参与输入生成、模型refit或校准选主阈值。原A0三折、全局588FREE、分带5%负结果和停止条件全部保留。

近带以公开query几何0.3–0.8m定义。排序比较known-positive-score（已知正像素中第16高margin）与FREE query-score（有效ray中第16高margin）。这是“见证对FREE的成对排序概率”，不是普通部署分类AUC；另报告POS query-trigger-score对FREE排序，不将任意POSquery触发冒称已知正见证。概率为P(POS score>FREE score)+.5P(tie)，相关query/ray/frame不当独立样本。

首个见证所需FREE成本使用该组最高known-positive-score作为事后阈值，保留ties；曲线只使用预定held-FREE成本0/5/10/20/50/100%，报告actual≤budget。它们使用eval标签，都是事后可达性诊断，不是新的cal-only主工作点，也不用于每capture选δ。缺POS或FREE的排序/首成本为NOT_EVALUABLE；POS=0的curve witness=0只是空计数。

| 评价cohort | near POS | near FREE | 可检验内容 |
| --- | --- | --- | --- |
| 原val24 | 24 | 0 | 见证恢复描述，缺FREE无法评价取舍 |
| 追加3RScan64 | 168 | 0 | 见证恢复描述，缺FREE无法评价取舍 |
| arkit16/47333462 | 44 | 96 | 正见证与负支持取舍 |
| 40777060 | 11 | 133 | 正见证与负支持取舍 |
| 40777065 | 0 | 144 | 负支持描述，缺POS不检正能力 |

## absolute排序主诊断

| capture | 臂 | 见证对FREE排序概率 | POS触发对FREE排序概率 | 首见证最少FREE成本 |
| --- | --- | --- | --- | --- |
| arkit16/47333462 | affine | 0.1747 | 0.2603 | 41/96 |
| arkit16/47333462 | 旧depth/ray | 0.5249 | 0.5784 | 7/96 |
| arkit16/47333462 | A0 δ=.05 | 0.1824 | 0.2562 | 28/96 |
| arkit16/47333462 | A0 δ=.1 | 0.1914 | 0.2713 | 26/96 |
| arkit16/47333462 | A0 δ=.2 | 0.2008 | 0.3136 | 12/96 |
| 40777060 | affine | 0.1141 | 0.2303 | 56/133 |
| 40777060 | 旧depth/ray | 0.2761 | 0.4074 | 53/133 |
| 40777060 | A0 δ=.05 | 0.1162 | 0.2283 | 54/133 |
| 40777060 | A0 δ=.1 | 0.1196 | 0.2283 | 51/133 |
| 40777060 | A0 δ=.2 | 0.1278 | 0.2303 | 44/133 |

预定held-FREE成本曲线；单元格 **已知正见证 / 实际FREE支持**，列为请求百分比，分母见前表。100%端点仅显示极宽松阈值的可达性，不视为有用工作点。

| capture | 臂 | 0% | 5% | 10% | 20% | 50% | 100% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| arkit16/47333462 | affine | 0 / 0 | 0 / 4 | 0 / 9 | 0 / 19 | 5 / 48 | 44 / 96 |
| arkit16/47333462 | 旧depth/ray | 0 / 0 | 0 / 4 | 1 / 9 | 3 / 19 | 28 / 48 | 44 / 96 |
| arkit16/47333462 | A0 δ=.05 | 0 / 0 | 0 / 4 | 0 / 9 | 0 / 19 | 4 / 48 | 44 / 96 |
| arkit16/47333462 | A0 δ=.1 | 0 / 0 | 0 / 4 | 0 / 9 | 0 / 19 | 6 / 48 | 44 / 96 |
| arkit16/47333462 | A0 δ=.2 | 0 / 0 | 0 / 4 | 0 / 9 | 2 / 19 | 7 / 48 | 44 / 96 |
| 40777060 | affine | 0 / 0 | 0 / 6 | 0 / 13 | 0 / 26 | 2 / 66 | 11 / 133 |
| 40777060 | 旧depth/ray | 0 / 0 | 0 / 6 | 0 / 13 | 0 / 26 | 3 / 66 | 11 / 133 |
| 40777060 | A0 δ=.05 | 0 / 0 | 0 / 6 | 0 / 13 | 0 / 26 | 2 / 66 | 11 / 133 |
| 40777060 | A0 δ=.1 | 0 / 0 | 0 / 6 | 0 / 13 | 0 / 26 | 2 / 66 | 11 / 133 |
| 40777060 | A0 δ=.2 | 0 / 0 | 0 / 6 | 0 / 13 | 0 / 26 | 2 / 66 | 11 / 133 |

## normalized排序辅助诊断

| capture | 臂 | 见证对FREE排序概率 | POS触发对FREE排序概率 | 首见证最少FREE成本 |
| --- | --- | --- | --- | --- |
| arkit16/47333462 | affine | 0.2509 | 0.3513 | 9/96 |
| arkit16/47333462 | 旧depth/ray | 0.5180 | 0.5781 | 8/96 |
| arkit16/47333462 | A0 δ=.05 | 0.2633 | 0.3506 | 9/96 |
| arkit16/47333462 | A0 δ=.1 | 0.2730 | 0.3565 | 8/96 |
| arkit16/47333462 | A0 δ=.2 | 0.3018 | 0.3892 | 2/96 |
| 40777060 | affine | 0.1955 | 0.2994 | 54/133 |
| 40777060 | 旧depth/ray | 0.2365 | 0.3486 | 63/133 |
| 40777060 | A0 δ=.05 | 0.1900 | 0.2973 | 61/133 |
| 40777060 | A0 δ=.1 | 0.1880 | 0.3001 | 64/133 |
| 40777060 | A0 δ=.2 | 0.1900 | 0.3008 | 59/133 |

预定held-FREE成本曲线；单元格 **已知正见证 / 实际FREE支持**，列为请求百分比，分母见前表。100%端点仅显示极宽松阈值的可达性，不视为有用工作点。

| capture | 臂 | 0% | 5% | 10% | 20% | 50% | 100% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| arkit16/47333462 | affine | 0 / 0 | 0 / 4 | 1 / 9 | 3 / 19 | 9 / 48 | 44 / 96 |
| arkit16/47333462 | 旧depth/ray | 0 / 0 | 0 / 4 | 3 / 9 | 11 / 19 | 25 / 48 | 44 / 96 |
| arkit16/47333462 | A0 δ=.05 | 0 / 0 | 0 / 4 | 1 / 9 | 3 / 19 | 11 / 48 | 44 / 96 |
| arkit16/47333462 | A0 δ=.1 | 0 / 0 | 0 / 4 | 2 / 9 | 3 / 19 | 12 / 48 | 44 / 96 |
| arkit16/47333462 | A0 δ=.2 | 0 / 0 | 1 / 4 | 1 / 9 | 5 / 19 | 11 / 48 | 44 / 96 |
| 40777060 | affine | 0 / 0 | 0 / 6 | 0 / 13 | 0 / 26 | 1 / 66 | 11 / 133 |
| 40777060 | 旧depth/ray | 0 / 0 | 0 / 6 | 0 / 13 | 0 / 26 | 1 / 66 | 11 / 133 |
| 40777060 | A0 δ=.05 | 0 / 0 | 0 / 6 | 0 / 13 | 0 / 26 | 1 / 66 | 11 / 133 |
| 40777060 | A0 δ=.1 | 0 / 0 | 0 / 6 | 0 / 13 | 0 / 26 | 1 / 66 | 11 / 133 |
| 40777060 | A0 δ=.2 | 0 / 0 | 0 / 6 | 0 / 13 | 0 / 26 | 3 / 66 | 11 / 133 |

normalized A0=.2在arkit16的局部首见证成本2/96；这一局部混合信息保留，不能外推到40777060（首见证59/133）或无近POS的40777065。全部cal源/环境的分布和曲线是fitted-in诊断，完整记录在`ranking/`，不当独立转移收益。

## 原生近正点的米制偏差

统计参考POS>=16的近query内label=1原生像素，每frame先union去重；跨frame仍重复观察同一表面，不当独立物理点。GT是现有native sensor/reference深度，不保证物理绝对精度。预测来自既有raw DepthPro、train log-affine、旧head与A0；同一native ray/K、valid mask，全部SHA校验。GT像素选择不是运行时可用信号。

表中Z error=预测−参考（m），log ratio=log(预测/参考)，ratio=预测/参考；各为点加权中位数，不能平均ratio或把重复像素当置信样本。

| cohort | 臂 | union像素数 | Z error中位(m) | log ratio中位 | ratio中位 | Z error q95(m) |
| --- | --- | --- | --- | --- | --- | --- |
| 原val24 | raw DepthPro | 44686 | -0.2013 | -0.3508 | 0.7042 | 0.0448 |
| 原val24 | affine | 44686 | 0.3122 | 0.3951 | 1.4845 | 0.4941 |
| 原val24 | 旧depth/ray | 44686 | 0.1978 | 0.2626 | 1.3003 | 0.3104 |
| 原val24 | A0 δ=.05 | 44686 | 0.2699 | 0.3502 | 1.4194 | 0.4396 |
| 原val24 | A0 δ=.1 | 44686 | 0.2414 | 0.3178 | 1.3741 | 0.3913 |
| 原val24 | A0 δ=.2 | 44686 | 0.2209 | 0.2912 | 1.3380 | 0.3104 |
| 追加3RScan64 | raw DepthPro | 347142 | -0.0185 | -0.0313 | 0.9692 | 0.8816 |
| 追加3RScan64 | affine | 347142 | 0.4484 | 0.5457 | 1.7258 | 0.8510 |
| 追加3RScan64 | 旧depth/ray | 347142 | 0.3212 | 0.4218 | 1.5248 | 0.9668 |
| 追加3RScan64 | A0 δ=.05 | 347142 | 0.4112 | 0.5165 | 1.6761 | 0.9221 |
| 追加3RScan64 | A0 δ=.1 | 347142 | 0.3769 | 0.4913 | 1.6344 | 0.9795 |
| 追加3RScan64 | A0 δ=.2 | 347142 | 0.3372 | 0.4503 | 1.5687 | 1.0145 |
| arkit16/47333462 | raw DepthPro | 150703 | 3.6126 | 1.9673 | 7.1514 | 11.8653 |
| arkit16/47333462 | affine | 150703 | 1.6309 | 1.3177 | 3.7347 | 2.6963 |
| arkit16/47333462 | 旧depth/ray | 150703 | 1.0119 | 0.9847 | 2.6771 | 1.4917 |
| arkit16/47333462 | A0 δ=.05 | 150703 | 1.5234 | 1.2687 | 3.5561 | 2.5344 |
| arkit16/47333462 | A0 δ=.1 | 150703 | 1.4253 | 1.2207 | 3.3897 | 2.3808 |
| arkit16/47333462 | A0 δ=.2 | 150703 | 1.2703 | 1.1270 | 3.0864 | 2.0938 |
| 40777060 | raw DepthPro | 10243 | 9.5701 | 2.6511 | 14.1693 | 9999.2350 |
| 40777060 | affine | 10243 | 2.3577 | 1.4656 | 4.3301 | 39.0274 |
| 40777060 | 旧depth/ray | 10243 | 0.8672 | 0.7701 | 2.1599 | 1.3269 |
| 40777060 | A0 δ=.05 | 10243 | 2.2072 | 1.4156 | 4.1190 | 37.0867 |
| 40777060 | A0 δ=.1 | 10243 | 2.0645 | 1.3656 | 3.9181 | 35.2407 |
| 40777060 | A0 δ=.2 | 10243 | 1.7994 | 1.2656 | 3.5452 | 31.8143 |
| 40777065 | raw DepthPro | 0 | NOT_EVALUABLE | NOT_EVALUABLE | NOT_EVALUABLE | NOT_EVALUABLE |
| 40777065 | affine | 0 | NOT_EVALUABLE | NOT_EVALUABLE | NOT_EVALUABLE | NOT_EVALUABLE |
| 40777065 | 旧depth/ray | 0 | NOT_EVALUABLE | NOT_EVALUABLE | NOT_EVALUABLE | NOT_EVALUABLE |
| 40777065 | A0 δ=.05 | 0 | NOT_EVALUABLE | NOT_EVALUABLE | NOT_EVALUABLE | NOT_EVALUABLE |
| 40777065 | A0 δ=.1 | 0 | NOT_EVALUABLE | NOT_EVALUABLE | NOT_EVALUABLE | NOT_EVALUABLE |
| 40777065 | A0 δ=.2 | 0 | NOT_EVALUABLE | NOT_EVALUABLE | NOT_EVALUABLE | NOT_EVALUABLE |
| 新ARKit cal48 | raw DepthPro | 83370 | 9.7786 | 2.6660 | 14.3827 | 9999.3200 |
| 新ARKit cal48 | affine | 83370 | 2.3894 | 1.4495 | 4.2611 | 39.1124 |
| 新ARKit cal48 | 旧depth/ray | 83370 | 0.8987 | 0.8150 | 2.2592 | 1.3818 |
| 新ARKit cal48 | A0 δ=.05 | 83370 | 2.2370 | 1.3995 | 4.0533 | 37.1717 |
| 新ARKit cal48 | A0 δ=.1 | 83370 | 2.0919 | 1.3495 | 3.8556 | 35.3257 |
| 新ARKit cal48 | A0 δ=.2 | 83370 | 1.8226 | 1.2495 | 3.4887 | 31.8993 |

arkit16的150703个点、40777060的10243个点、新ARKitcal的83370个点，在六臂下预测均高于GT；arkit16/60全部已知near正query-ray pair在query exit之后，原生inside=0。这是零容差3D射线区间包含诊断，与分带负margin容差下的输出支持分开。40777065没有近POS，相关point统计为NOT_EVALUABLE。

固定affine参数a=.3713312368564329、b=.2635894826641125，`log z_affine=a log z_DP+b`。它减少ARKit raw DP的巨大高估，但3RScan raw近点ratio中位原val=.704、追加=.969，经affine变为1.485、1.726；全局强基线并不保证各距离带点级最优。此方向差异不证明rawDP在整体query成本上胜过affine，也不支持所有来源套同一尺度。

## 原生区间、参考与重尾边界

以下为相关query-ray pair数量，不是独立像素/场景。POS pair只取POSITIVE>=16近query的label=1；FREE pair只取整个query严格FREE的label=0可用预测ray。before/inside/after使用原生零容差entry/exit，不套任何cal cutoff；depth落入0.3–0.8轴向带与完整XYZ query inside另行保存，不混用。

| cohort | 臂 | POS pairs before/inside/after | 严格FREE pairs | FREE原生false-entry pairs |
| --- | --- | --- | --- | --- |
| 原val24 | affine | 0/0/44686 | 0 | 0 |
| 原val24 | 旧depth/ray | 180/4169/40337 | 0 | 0 |
| 原val24 | A0 δ=.2 | 0/1655/43031 | 0 | 0 |
| 追加3RScan64 | affine | 0/1278/345864 | 0 | 0 |
| 追加3RScan64 | 旧depth/ray | 1031/46660/299451 | 0 | 0 |
| 追加3RScan64 | A0 δ=.2 | 0/18459/328683 | 0 | 0 |
| arkit16/47333462 | affine | 0/0/150703 | 905379 | 0 |
| arkit16/47333462 | 旧depth/ray | 0/0/150703 | 905379 | 0 |
| arkit16/47333462 | A0 δ=.2 | 0/0/150703 | 905379 | 0 |
| 40777060 | affine | 0/0/10243 | 1374974 | 0 |
| 40777060 | 旧depth/ray | 0/0/10243 | 1374974 | 0 |
| 40777060 | A0 δ=.2 | 0/0/10243 | 1374974 | 0 |
| 40777065 | affine | 0/0/0 | 1472192 | 0 |
| 40777065 | 旧depth/ray | 0/0/0 | 1472192 | 0 |
| 40777065 | A0 δ=.2 | 0/0/0 | 1472192 | 0 |
| 新ARKit cal48 | affine | 0/0/83370 | 3979880 | 0 |
| 新ARKit cal48 | 旧depth/ray | 0/0/83370 | 3979880 | 0 |
| 新ARKit cal48 | A0 δ=.2 | 0/0/83370 | 3979880 | 0 |

ARKit eval/cal六臂在严格nearFREE query内的原生false-entry pairs为0。它与分带cal的arkit16 affine FREE35/96不矛盾：后者cutoff−.775622m允许预测位于box外，margin仍超过负阈值。“支持了FREE query”不等于预测深度进入原生near盒；arkit16分带affine另有1个POS query触发但仍无known-positive见证。

raw DP在40777060、新cal近正点保留了10000m有限正值重尾，不删除或转UNKNOWN。核验现有DepthPro执行源码SHA与上轮新cal `model_identity.json`一致；本地`depth_pro.py:293`为`depth=1/torch.clamp(inverse_depth,min=1e-4,max=1e4)`，10000m与倒数下限端点吻合。未保存raw pre-clamp inverse_depth，故不能据端点定位为何到达下限，更不能归为相机硬件单因。source trace见`upstream_readout_trace.json`；没有改模型有效性或重跑FP32。

## 决策

已有证据支持两个判断：一是当前ARKit近正点存在非常大的度量偏差，旧head方向能减小但仍未恢复原生近区间；二是affine见证分数对FREE排序不利，在这两个有近POS的ARKit capture低成本区域不能恢复近见证，旧head/A0的arkit16局部收益未转移到40777060。两者可能关联，但未通过受控输入/结构对照定位原因；真实RGB输入、focal/resize、reference对齐、深度表示及模型转移等解释仍需区分。

下一优先沿已有ARKit缓存审计模型输入与米制尺度链，包括倒数饱和的数值来源，再看能否用train/cal学习有界、按输入条件修正的残差。旧A0±.2不足以抵消这里的偏移，不能因此用eval中位logratio来设新训练bound；也不能以旧head在arkit16排序改善就自动提高训练优先级。若输入链正常，才把这项诊断转换成明确的重新训练残差目标和train/cal界限检验。

所有结果仅限已消费Development、现有参考及模型输出；相关query/ray/frame、cal标签事后分布和held-FREE曲线不是独立确认，不证明身体/整盒clearance、实机安全或一般化能力。原全局affine工作点仍保留，本轮没有发布替代模型。

## 核验、失败、预算与复现

排序一次完成，CPU .798904/180s；几何184帧分析15.336910s末端CSV发布遇空POS组字段差异而失败，原receipt/source与全部184份native-point-values、1104frame-arm、9936querypair保留；只复用这些保存值发布修复1.389072s，累计16.725982/420s，不重新读取184原始像素。最终geometry输出在`geometry/repair-1/`，原失败在`geometry/`，不覆盖。

生产核验184身份/mask、1656近domain/state/GT与270代表cached score/witness。独立排序核290汇总、870分布、1740固定成本点、580冻结阈值描述；独立几何从184保存点值重聚合504标量、792pair计数及1620分位数，另复算原val与arkit16两个原始frame，score/witness最大误差0。审计累计1.858064/180s，不重跑全部184原始像素或模型，记录见`focused-audit.json`。

保守CPU计费622/1200s：ranking2＋geometry20（含失败/修复与进程启动）＋audit180＋integration420（含plot/source trace/报告）；GPU、下载、训练、新模型推理均0。matplotlib图形来自固定6个成本点，连线仅辅助阅读，不表示插值工作点实际可达到；PNG已视觉检查，无裁切，图形生成.373s计入集成。无task-owned常驻进程/worker/GPU/下载session，原始payload保留。

[固定成本诊断图PNG](../../../../artifacts.local/work/rgb-body-query-near-rank-geometry-dev-20261010/near_rank_curves.png) · [SVG](../../../../artifacts.local/work/rgb-body-query-near-rank-geometry-dev-20261010/near_rank_curves.svg)。40777065因POS=0不绘制见证曲线，其FREE描述完整留在CSV。

所有payload位于canonical `artifacts.local/work/rgb-body-query-near-rank-geometry-dev-20261010/`，junction到F盘；Git只入两个helper、报告和RGB current/RUNS。执行snapshot/hash、父计划、inputs和failure链完整保留。AST、scoped diff whitespace及hot-doc index检查见`integration_checks.json`。

```powershell
E:/codex-tools/bin/blindassist-research-gpu.cmd research/active/dtr-r0/nearfield/rgb_body_query_near_ranking.py --repo . --output artifacts.local/work/<fresh>/ranking --budget-s 180
E:/codex-tools/bin/blindassist-research-gpu.cmd research/active/dtr-r0/nearfield/rgb_body_query_near_geometry.py --repo . --output artifacts.local/work/<fresh>/geometry --budget-s 420
```

先在父输出目录保存本轮同等plan；新输出目录不覆盖既有证据。交付geometry helper已修空组CSV schema，新运行可直接发布；本轮仍按实际失败/修复路径留证。ranking SHA `08e1634c4cb7a2a52432ec95a109c7dcd42e9da04c11e1493c97b8151ab30155`；geometry交付SHA `9cd02c7c777f4785005da10174a32b85bc4287f35b7f25cab5a26b5ff7cb4ced`。

前文：[固定分带5%负结果](RGB_BODY_QUERY_DISTANCE_CAL_DEV_20261010.md)、[新ARKit全局cal](RGB_BODY_QUERY_ARKIT_CAL_DEV_20261010.md)、[A0原三折](RGB_BODY_QUERY_A0_DEV_20261010.md)。
