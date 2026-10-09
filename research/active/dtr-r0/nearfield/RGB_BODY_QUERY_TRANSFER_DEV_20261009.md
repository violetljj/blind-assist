# RGB冻结候选：未用环境、不同相机与锚点旁路

2026-10-09，EXPLORE / COMPLETE / SAME_SOURCE_GAIN_RETAINED_CROSS_CAMERA_TRADEOFF。8个未用3RScan环境总体保留对geometry增量；一个ARKitScenes捕获中召回与误支持都更低，未复现总体两项同时改善，近场收益也未充分迁移。用户“推进”授权执行[上一诊断](RGB_BODY_QUERY_METRIC_DIAGNOSTIC_DEV_20261009.md)之后的迁移验证与评价路径核查。结果只回答指定新环境的迁移表现，不称广泛泛化；不能仅据跨相机退化归因硬件、标定或预测偏差。

本轮范围与预算：CPU辅助累计1800s、GPU执行壁时900s、新数据下载最多2GiB、无新模型下载/训练。网页说明文本另记bytes。主线最多8个未用3RScan环境×8帧；不同相机最多一个公开RGB-D序列×16帧，先核查数据许可/K/深度语义，适用则在预算内下载并执行。旁路旧96帧粗分区深度锚点；核查充分负query、细障碍与步行事件路径。payload `artifacts.local/work/rgb-body-query-transfer-dev-20261009/`；到限保存partial，不挪旧run/CNH预算。主线GPU600s、不同相机GPU240s、冻结head最多GPU60s；CPU准备/锚点各240s、资料180s，余量核验和收尾。此预算是本执行阶段实施选择，不冒充此前圆桌共识。

冻结合同：Depth Pro同权重，depth-only与geometry最终200步checkpoint、train归一化、原15个相机局部query。实际cutoff depth-only=0.5618626475334167、geometry=0.6106688380241394；它们原在cal匹配目标射线FPR 0.14190356447903094，不把该FPR当cutoff。新数据不用来改阈值/模型/归一化；原始DP直接几何及原cal固定尺度1.012320716490867并列。只用RGB生成预测距离、公用K/query推理；传感器深度只评分。未用环境排除原全部12组及其rescans；按环境reference SHA排序选前8组、每组reference优先一scan、8均匀配对帧，先保存选择，不看深度或模型输出选择。这个选择法是本run实施方法。

决策检查：旧validation1071771正/1795858 free相关query-ray，depth-only .70048/.10663优于geometry .61966/.11535，但3环境各有取舍；近场召回仍低，宽near及far存在饱和/退化。新环境正/free及充分负query数量未知，缺分母记NOT_EVALUABLE，UNKNOWN不当负。首轮验收是冻结工作点下指定新环境的全量/逐环境/距离带表现与救回损失，结果改变下一步选择；不是完整查询/步行/细结构全部任务的验收门槛。

锚点旁路：8×8固定网格，偶checkerboard有效点取区实测及预测中位数，区有效点≥16；区中位比的log中位得到单帧标量。奇checkerboard重生成评分标签，锚点评分不重叠；比较raw、原cal-scale、frame-anchor的直接几何读出，不训练head。无锚点帧独列覆盖。即使分开采样仍同源深度、可能共享系统误差，只证明实测生成锚点在本诊断作用，非混合回波或真实ToF增量。

评价边界：相关query-ray非独立样本；全部可观测采样free不证明整盒清空。缺充分负查询参考使查询层面误报不可评；缺真实步行序列/身体空间定义/事件参考使提醒提前量不可评。已有真实射线评价保留。SANPO只说已检查稀疏core不能补当前近场正例，不推广整个数据集。当前不续训旧两query或32特征臂；尺度影响局部收益仍可能，直接几何尺度响应不用于学习头因果归因。

## 冻结迁移结果

35个缓存官方train环境，排除旧12组剩23组，预选前8组×8帧；64帧960格570正/390UNKNOWN，无满足全部可观测采样射线free判据的query。15query×64独立XYZ/slab标签核验一致，环境/重扫隔离、publicK、RGB/ref SHA及观察无评分字段检查通过，prepare6.882s、audit1.032s。

不同相机选[ARKitScenes raw Training/47333462](https://github.com/apple/ARKitScenes/blob/main/raw/README.md)，来源是iPad Pro Apple LiDAR/RGB。官方示例capture，三包公开直链，无账号/token，357569481B；[数据许可](https://github.com/apple/ARKitScenes/blob/main/LICENSE)允许非商业研究，本run不再分发数据。[DATA.md](https://github.com/apple/ARKitScenes/blob/main/DATA.md)与[官方loader](https://github.com/apple/ARKitScenes/blob/main/threedod/benchmark_scripts/utils/tenFpsDataLoader.py)支持同像素RGB/depth、逐帧pincam和depth/1000的光学Z。三类timestamp确切交集均匀选16帧，选择先于depth读取；保留native256×192、逐帧K，未沿用Tango外参假定。240/240独立XYZ标签核验一致，140正/72UNKNOWN/28全采样free格。这个捕获是室内扫描，非身体帧或已标步行事件，native RGB分辨率也显著低于3RScan的960×540。

| 指定数据 | 臂 | TP/FN/FP/TN | 正射线召回/free误支持 | 正query支持/已知TP≥16 |
| --- | --- | --- | --- | --- |
| 新3RScan 8环境/64帧 | 冻结depth-only head | 2014062/1385999/368837/3975829 | .59236/.08489 | 429/570 · 411/570 |
| 同上 | 冻结geometry head | 1715819/1684242/413299/3931367 | .50464/.09513 | 323/570 · 321/570 |
| 同上 | raw Depth Pro直接几何 | 2176198/1223863/691265/3653401 | .64005/.15911 | 447/570 · 425/570 |
| 同上 | 原cal-scale直接几何 | 2177672/1222389/663755/3680911 | .64048/.15277 | 446/570 · 426/570 |
| ARKit 1捕获/16帧 | 冻结depth-only head | 415724/990304/73850/1689322 | .29567/.04188 | 84/140 · 76/140 |
| 同上 | 冻结geometry head | 593401/812627/98845/1664327 | .42204/.05606 | 81/140 · 79/140 |
| 同上 | raw Depth Pro直接几何 | 289401/1116627/18650/1744522 | .20583/.01058 | 76/140 · 68/140 |
| 同上 | 原cal-scale直接几何 | 279974/1126054/17263/1745909 | .19912/.00979 | 76/140 · 66/140 |

新3RScan正/free分母3400061/4344666相关query-ray；ARKit1406028/1763172，不是独立射线样本。3RScan head相对geometry召回+8.772个百分点、误支持−1.023个百分点；相对raw/cal DP却是召回更低、误支持更低，不能称强深度基线也被全面胜过。ARKit head相对geometry召回−12.637个百分点、误支持−1.418个百分点；相对raw DP召回更高但误支持更高，两比较均是工作点取舍。这些阈值源自原cal，不保证新数据相同FPR，不在新数据重选工作点。

## 全环境、距离带及负查询支持

depth-only对geometry全量：新3RScan正救437566/损139323，free消除117981/新增73519；ARKit正救8105/损185782，free消除26180/新增1185。其余对raw/cal DP配对全表同步保存，不只展示获益比较。

| 新3RScan环境前缀 | 正/free分母 | 召回：head/geometry | 误支持：head/geometry | 正救/损 | free消除/新增 |
| --- | --- | --- | --- | --- | --- |
| 1d233ff6 | 453313/534453 | .72881/.49830 | .11199/.07949 | 105838/1345 | 937/18306 |
| 422885e5 | 489798/271753 | .44443/.30938 | .01157/0 | 72629/6483 | 0/3145 |
| 1c211546 | 416874/629551 | .66838/.57627 | .10870/.09142 | 50201/11801 | 7926/18804 |
| 4acaebc0 | 453976/474357 | .62803/.43939 | .05267/.05862 | 91665/6027 | 6737/3917 |
| 48699c02 | 399610/642432 | .56071/.62602 | .07029/.09872 | 13224/39320 | 22771/4511 |
| 422885e9 | 350024/589018 | .44961/.52700 | .11823/.17943 | 7966/35053 | 36467/421 |
| 43b8cae9 | 421043/601949 | .64686/.55772 | .07838/.09307 | 61061/23531 | 19024/10179 |
| 4a9a43e4 | 415423/601153 | .59811/.55184 | .08390/.10034 | 34982/15763 | 24119/14236 |

6/8环境正TP净增，2/8净减；3/8环境误支持净增。总体结果不能替代这些取舍。

| 距离带 | 3RScan正/free分母 | 3RScan召回head/geo · 误支持head/geo | ARKit正/free分母 | ARKit召回head/geo · 误支持head/geo |
| --- | --- | --- | --- | --- |
| .3–3m | 1688014/1103884 | .81447/.78061 · .14865/.19758 | 702332/492478 | .47321/.71251 · .08685/.13342 |
| 3–6m | 23045/4704 | 1/1 · 1/1 | 510/852 | 1/1 · 1/1 |
| .3–.8m | 347185/1995831 | .13344/0 · .00217/0 | 150717/898450 | 0/0 · 0/0 |
| .8–1.5m | 933228/995242 | .22675/0 · .03185/0 | 448017/327884 | .00286/0 · 0/0 |
| 1.5–3m | 408589/245005 | .87680/.91803 · .66945/.77751 | 104452/43508 | .78104/.88530 · .69468/.74212 |

旧小距离带优势在新3RScan仍出现，但召回仅.133/.227；ARKit .3–.8m完全未检出，.8–1.5m仅救1282/448017。两路far两臂全部支持，同时100%free误支持，不能把满召回作为好结果。query距离带互相重叠，分母不能相加当独立物理射线。`readout/`与`camera-readout/`的frame/环境/距离带/环境×距离带CSV含全部比较。

ARKit预先固定15网格中的28个FREE_ON_SAMPLED_RAYS格，各臂支持0/28；state定义要求采样free≥16、无采样occupied或unknown。这只评价全采样free格上的支持率，28格来自同一个capture、相关query/帧，不证明整盒清空、普遍零误报或真实身体查询FPR。支持数量与已知TP见证数量仍分开报告。完整步行序列、身体空间/相机到身体关系和事件参考没有补齐，提醒提前量保持NOT_EVALUABLE。

## 预测距离诊断与读出复现

独立评分域诊断，不修改预测/归一化/阈值：新3RScan1880552有效native传感器点，mean abs-log .34222、median AbsRel .24725、median实测/预测比1.05126；ARKit786432点相应.99260/.99893/.50027。ARKit按实测Z的.3–.8/.8–1.5/1.5–3/3–6m各带median比.13983/.48538/.81235/2.71351，不能用单一全局尺度解释局部查询。

预测≥1000m尾部3RScan3782/1880552、ARKit4737/786432，最大10000m；实测无此尾部。mean AbsRel16.0956/85.0675受尾部影响，仍在完整JSON保存，没有删除尾部或裁预测来改善工作点。诊断提示该ARKit捕获的预测距离质量较差，不能定位是场景、原图分辨率、成像/K、传感器参考或读出适配哪项所致，不能归因Tango硬件。直接深度误差也不独自解释learned head收益。

独立复算两路4臂全部records/summary/all/environment/band/environment_band，以及128+32保存head数组、参考/模型/归一化/实际cutoff SHA通过。新主adapter最初每query分块；CPU旧首validation帧geometry有1个cutoff边界差异：CPU=.6106688380241394等于cutoff、旧CUDA低5.96e−8，最大概率差≤1.788e−7。它不是已证明的逻辑错误。为运行复现，将新adapter改回原跨query-point65536批布局，保留v1目录/源码/审计；仅重算head不重跑DP。旧首帧在CUDA逐概率完全一致、两臂二值差0；新两路v1/v2四臂主计数都未改变。路径映射及精确源码SHA在`preservation.json`，CPU边界限制如实保留，不声称跨后端普遍逐值相同。

## 同源锚点旁路

旧96帧全部可用，28–64区域/帧，单帧尺度.188993–5.404701，未按效果筛区/裁剪尺度。validation奇checkerboard正/free分母535822/897896：

| 直接几何读出 | 召回/free误支持 |
| --- | --- |
| 原始RGB预测深度 | .581271/.187899 |
| 原cal固定尺度 | .583403/.181417 |
| 同源逐帧锚点尺度 | .859535/.061594 |

raw→anchor正救172230/损23130，free消除140829/新增27420。12环境全量均TP净增/FP净减，但3–6m召回.40072→.35420、误支持.00277→.09246；.3–.8m召回.87650→.86240、误支持.11698→.00992。完整split/env/band保留。1440标签重生成、4320直接XYZ、棋盘交集0、异质ratio-of-medians fixture/保存计数与SHA检查通过。诊断10.599s、独立.102s，CPU保守15s。额外同源实测信息不同，不能和RGB-only head公平直接排名；结果只支持锚点在此旁路的作用，不是尺度因果、上界、伪ToF或真实ToF收益。

## 完整评价路径与下一决定

本轮补到不同相机实测射线和28全采样free格，未补整盒/身体/步行事件。数据核查详见payload `data-audit/README.md`，13份已读官方源码/网页URL、bytes、SHA在`source_receipt.json`。

- [HEADS-UP](https://www.epfl.ch/labs/vita/research/prediction/heads-up/)有头戴ZED步行；[官方文件库](https://huggingface.co/datasets/Yassaman/HEADS-UP/tree/main)有登录及分享联系信息的真实门槛，本轮未提交或下载。轨迹来源与身体体积/事件参考仍要核查，不能拿行人中心轨迹当身体碰撞真值。
- [SANPO](https://github.com/google-research-datasets/sanpo_dataset)仍可按metadata选真实导航序列；已检查稀疏core的近场缺口不推广整个集。CREStereo不能单独当独立真值，官方更正后的fixed_camera_poses.csv须与旧pose区别。
- [ARKitScenes](https://github.com/apple/ARKitScenes/blob/main/DATA.md)另有FARO高分辨率投影参考/upsampling子集，可作为下一静态高分辨率与多视角覆盖路径；本轮未下载。ScanNet/ScanNet++需要实际协议或账号审批，未虚构访问权。

下一优先检查原图细节/预测距离质量与跨场景训练分布，保留该冻结候选及跨相机负结果；不把当前低分辨率单捕获结果上升为RGB整体失败，也不据此直接给Tango偏差下结论。后续应补不同相机的高分辨率RGB及多个环境，再检验近场收益是否迁移，查询完整覆盖与步行事件路线继续。锚点只作后续真实米制输入的机制提示，不能代替独立ToF。旧两query与32特征臂不续训，原真实硬目标、M3/CNH身份与预算保留。

## 成本与代码

GPU执行总154.136s/900s：新3RScan DP80.993s、ARKit DP49.987s，head v1/v2及旧帧CUDA窄核验23.157s；含启动/准备/保存，不是手机部署时延。CPU辅助保守记账300s/1800s（资料160、两准备12+5、锚点15、独立审计40、主评分/收尾余量），不是性能测量。下载357569481B/2GiB，新模型0B，官方网页源码另842481B。全部子进程完成，无服务/worker留存；官方包/参考/预测/模型/失败及v1/v2证据保留在canonical artifacts树供复算。

代码：[新环境准备](rgb_body_query_transfer_prepare.py)、[ARKit适配](rgb_body_query_arkit_prepare.py)、[冻结读出](rgb_body_query_frozen_transfer.py)、[独立审计](rgb_body_query_transfer_audit.py)、[锚点旁路](rgb_body_query_anchor_probe.py)。总输入/输出/源码SHA、分阶段成本和释放说明在payload `completion_receipt.json`；docs index与scoped diff检查用于交付。

后续已完成[同帧分辨率与可调基线诊断](RGB_BODY_QUERY_INPUT_BASELINE_DEV_20261009.md)：8对640→256仅很小查询变化，δ0校准合同退化，第三相机源下载未完成；旧16帧评分与本轮共同域不同，原结果保留，不作跨域因果比较。
