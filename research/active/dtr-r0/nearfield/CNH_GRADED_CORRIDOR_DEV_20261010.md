# 公共走廊峰值证据与轻提醒成本

2026-10-10；EXPLORE，起始`20e014b5`。接续[分级与回波诊断](CNH_GRADED_EVIDENCE_DEV_20261010.md)，复用`CNH_COUNTERFACTUAL_DEV_20261009`的ideal已消费模拟Development。

**扩张走廊、外环和局部峰已实际提取并进入分级比较。空间交互在较紧预算下出现HEAD及时检出和减少pass的收益线索，但BODY与clear代价仍需取舍。当前保留完整ordinary single轻提醒参照，空间交互保留为候选；下一优先检验HEAD/BODY分别评分、校准和定预算，再考虑跟踪距离峰后的断续支持。**

## 输入与证据边界

每split384scene×4replica×13决策帧(f3–15)×HEAD/BODY；及时为f3–13，晚提醒f14–15。HEAD/BODY各384接触事件，joint-clear6656slots/512clips，纯pass3328slots/256clips。三个ordinary seed为2026100955/956/957，分别拟合和评价；运行时每个候选只用单模型分数，无多seed投票。沿用共向/ideal公共位姿，不访问保护480、设备、新采集或新光子采样。

本次新增特征路径只读取hist、ambient、bias、公共sensor/query和缓存身份，不读取category、目标框或authoring参数。每次观测读取和标准化后，共用于内走廊、扩张走廊、外环：x分别±.30m、±.40m及两者差集；沿用HEAD/BODY的y范围，z=.30–3m。每native angular zone/bin使用固定9节点的硬几何membership，保留current和过去≤8帧投影到当前query的signed-log均值。

各区域/时间模式输出14项：top1/top3正局部峰、signed/positive总量、参与度、正bin/峰数、公共membership量，以及top1的公共前向深度、径向距离、支持范围和原生z-score；另有扩张当前最大峰的径向bin±1内inner/ring竞争，共88项和逐项有效mask。缺支持/缺峰为NaN+无效；不会转成free。校准与validation全量公共范围都有节点支持、也都有正观测峰，包含噪声；**有峰不证明障碍存在，节点支持不证明真实物理覆盖**。past8并未跟踪同一物体，历史峰的位置与当前峰也不自动对应。此处16个径向bin宽约.3002784m，不把bin内公共深度当作精确目标距离。

## 评分与校准

强档逐slot固定为上一轮ordinary_OR：旧5格融合 OR ordinary smoothed score≥原追加阈值。强档及时HEAD315/313/323、BODY267/257/282各/384；强clear58/6656slots、28/512clips；强pass355/362/461各/3328slots，89/103/115各/256clips。所有本轮候选都保留这条强档的等级和时序，强转轻为0；它并非原M3的逐slot超集。

轻档比较两种策略：filter只过滤原single新增轻档；replace在所有非强slot中重新评分，允许补回single以下局部证据，也允许丢掉原轻提醒。filter减少提醒不能解释为“明确不侵入”。

线性三路为score-only、spatial-only、score+spatial。分数路使用single阈值余量、过去≤5帧raw slope、去趋势RMS；空间路固定12个峰/参与度/深度/同bin竞争描述，另有current/past8正支持密度、top3集中度、current−past8峰差和inner−ring同bin峰差，附缺测指示共34维。共享HEAD/BODY的L2=.1 logistic排序分数，不称物理风险概率。

cal每背景族的较小实例ID(8/10)拟合，另实例(9/11)仅定阈值；每scene×replica×query按可用非强帧等总权重。各模型/filter与replace分别固定三个成本档：原轻档cal-cutoff实例的pass clips和joint-clear slots各乘25%/50%/75%，取floor；以完整score ties选择满足两预算的最低cutoff。fit与cutoff实例隔离，validation为12–15；旧ordinary阈值仍曾使用全部旧cal，整个管线不能称独立校准。**同cal预算不是validation同成本**，不在validation重新匹配或调阈值。

观察线性增益混合后，另先声明`interaction/PLAN.json`，仅跑一次固定浅层HGB交互对照，score-only与score+spatial使用同特征、实例和成本档。参数为100轮、learning_rate=.1、最多7叶/深3、min_samples_leaf=50、L2=1、early_stopping=False、seed20261010；等episode权重归一到均值1。它是消费线性结果后的探索，没有参数/seed选择或独立确认身份；只拟合小型轻档评分，未重训/前向运行CNH主干。

## 检出、成本和及时性

以下75%档仅为便于比较的工作点，完整25%/50%/75%×filter/replace表均保留。及时HEAD/BODY各/384，clear slots/6656、pass clips/256；loss为相对原完整轻提醒候选的及时事件损失，filter rescue均0。

| 模型/策略 | seed尾号 | 及时HEAD/BODY | loss HEAD/BODY | clear slots | pass clips |
|---|---|---|---|---|---|
| 原完整轻提醒 | 955 | 351/307 | 0/0 | 73 | 149 |
| 原完整轻提醒 | 956 | 344/310 | 0/0 | 107 | 168 |
| 原完整轻提醒 | 957 | 346/312 | 0/0 | 80 | 158 |
| 线性score-only/filter75 | 955 | 347/303 | 4/4 | 68 | 141 |
| 线性score-only/filter75 | 956 | 332/295 | 12/15 | 70 | 139 |
| 线性score-only/filter75 | 957 | 342/308 | 4/4 | 69 | 144 |
| 线性score+spatial/filter75 | 955 | 350/305 | 1/2 | 68 | 133 |
| 线性score+spatial/filter75 | 956 | 334/291 | 10/19 | 84 | 137 |
| 线性score+spatial/filter75 | 957 | 343/308 | 3/4 | 72 | 150 |
| HGB score-only/filter75 | 955 | 350/303 | 1/4 | 72 | 140 |
| HGB score-only/filter75 | 956 | 342/304 | 2/6 | 80 | 146 |
| HGB score-only/filter75 | 957 | 343/309 | 3/3 | 77 | 143 |
| HGB score+spatial/filter75 | 955 | 348/302 | 3/5 | 72 | 131 |
| HGB score+spatial/filter75 | 956 | 337/298 | 7/12 | 84 | 135 |
| HGB score+spatial/filter75 | 957 | 343/308 | 3/4 | 79 | 141 |

线性空间增量对同cal预算score-only，三个seed总及时净差+5/−2/+1，pass差−8/−2/+6，clear差0/+14/+3；没有稳定优势。交互空间档对相应score-only总及时净差−3/−11/−1，pass减少9/11/2，但clear差0/+4/+2；它进一步压低pass，也真实丢掉及时接触，不能说“分清了接触/擦边”。这些是取舍而非同成本优劣定论，不否定全部峰值机制。

HGB score-only/filter75相对原完整轻档，pass减少9/22/15clips，clear减少1/27/3slots，及时损失HEAD/BODY共5/8/6件。仍比强档多及时HEAD35/29/20、BODY36/47/27，损0来自保留强档的构造；相对强档的共同及时事件提前HEAD116/102/116、BODY92/99/89件，共同事件提前中位数仍0帧。相对完整轻档，已有及时事件推迟HEAD18/32/16、BODY10/31/12件；因此不能只报损失少而省略提醒延迟。

该工作点的clear轻slots14/22/19、pass轻slots177/182/127，pass轻clips112/113/91；强/轻joint-slot互斥，clip可以同时含先轻后强，不能相加。pass轻提醒最长连续3帧，仅覆盖本批13帧窗口。其它工作点的完整cost、segments、最长提醒、paired时序/晚提醒/分组见metrics。

弱目标仍是主要损失位置。HGB score-only/filter75的横杆及时HEAD69/65/64、BODY61/56/60，标牌边缘89/85/87、71/73/73，竖杆96/96/96、75/79/80，各族各高度/96；突出物96/96每seed。空间/filter75的对应横杆68/63/64、60/56/59，边缘88/82/87、71/70/73；它没有恢复BODY整体损失。

replace确实可补single以下事件，也会损原轻提醒；例如HGB空间replace75三seed救HEAD3/10/7、BODY4/6/5，同时损HEAD6/8/6、BODY15/24/4；clear94/113/86，pass116/128/135。seed957总及时347/313高于原346/312，但不能只报这一seed或净计数掩盖救损，不据此升级候选。

不能只据75%档关闭空间机制。HGB空间交互在全部18组同fraction validation对照中减少pass 2–19clips；更紧的25% replace相对对应score-only，HEAD及时净增24/21/19，BODY净差+4/−9/+15，pass减少4/7/2，clear增加4/9/0slots。空间候选本身的HEAD347/343/344、BODY287/268/299、pass97/109/121、clear65/72/67；相对完整轻档救HEAD3/9/6、BODY3/1/3，同时损HEAD7/10/8、BODY23/43/16。这说明空间交互有部分任务收益，不是全面失败；BODY损失尤其seed956仍大。没有设“全seed必胜”或零损失门槛，保留候选但不作为默认的理由是弱BODY及时提醒与HEAD/pass收益尚未形成可接受的统一取舍。

## 核验、载荷与下一步

- 特征：`cnh_graded_corridor_features_dev.py`；`features/{cal,validation}_features.npz`各`[384,4,13,2,88]`float32＋bool有效mask，公共几何/行身份/输入哈希和schema配套保存。原FP16历史代表行精确相同，cache-row身份全量一致，未来扰动前缀通过。
- `audit_cnh_graded_corridor_features_dev.py`用独立节点/峰值计算复算6观测、156 query帧，最大差7.15e−7、bin/峰计数精确。CPU32观测.562s，CUDA3.609s且159个计数与CPU不一致；保留失败，正式采用CPU，未把CUDA当等价部署实现。末次补齐缺公共支持competition的mask，本批所有公共mass>0，结果数组不变；旧/新source哈希及合成缺测检查见`features/source_revision_validation.json`。
- 线性：`cnh_graded_corridor_eval_dev.py`、`audit_cnh_graded_corridor_eval_dev.py`；108 cells/9拟合/54阈值、165888逐事件账本。独立复算normalization、目标、完整ties、grade、first、救损/分组和强轻cost通过，1,239,364标量项。
- 交互：`cnh_graded_corridor_interaction_dev.py`、`audit_cnh_graded_corridor_interaction_dev.py`；72 cells/6拟合/36阈值，独立核验45,946标量项。6个任务自产pickle先核hash，再复算12预测数组逐值相同；cal-only调用/权重、参数、ties、等级、完整metrics通过，没有重新训练审计模型。pickle仅是本地证据，不作为App资源。
- 载荷统一在`artifacts.local/work/cnh-graded-corridor-dev-20261010/`。特征CPU保守90/600s、GPU基准阶段17.344/180s；线性主命令3.703s、交互10.390s，分析含审计/运行时探测/整合保守90/900s，无远端/付费算力或常驻资源。默认Python缺NumPy失败、CPU/CUDA计数失败、source补mask的证据均保留。

当前不采用本批静默/过滤档作为默认，不关闭分级或峰值方向。下一优先用同批cal检验HEAD/BODY分别评分与预算，保留score-only对照、强档和全部救损；“共享评分与阈值混合了两高度证据尺度”只是待验假设，不以validation调参。若不能改善BODY取舍，再转向公共空间中跟踪距离峰及支持范围后的断续积累，比较首次提醒和pass持续时间。跟踪需处理峰跳转、约.30m径向bin及背景峰；先检验身份连续性，不能把score上升直接解释为目标接近。原背景可见规则失败与严格匹配NOT_EVALUABLE保留，尚无新的free/静默依据。

本次有限背景、AABB、exact位姿、相关replica/帧及消费Development只支持上述探索取舍。未接入App，未建立实机风险级别、真实用户打扰或独立泛化。原M3、5格、L2、body truth/fullbin、480和旧stop身份保留。
