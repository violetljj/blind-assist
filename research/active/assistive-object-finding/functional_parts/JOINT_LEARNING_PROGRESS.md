# UMD固定读点与局部证据联合学习

2026-10-16，Development，用户授权继续执行，暂不定题。继承[完整基线](FULL_BASELINE_RESULTS.md)，先完成UMD固定读点强对照，再开展同数据、计算成本匹配的结构比较；大型任务均在注册副机，主机仅轻量调度／汇总。

## 本轮问题与范围

先量化稳定峰值、预测区域限制与内部点能解决多少UMD错误，保留原始点和混淆矩阵作为复现约束；再区分真实局部证据融合、新候选与点质量学习各自是否提供额外收益。不把数值解码称结构创新，不用UMD选头／阈值，不修改前轮冻结训练／评分源码。

复用IIT train4,417行／4,400 RGB、cleanval1,757行／1,754 RGB；UMD9,939作者手标帧／108工具或场景组、19,370正类查询。旧采集相关性、语义并集及正类查询的能力边界不变。所有这些结果已用于Development，不再称独立未见确认。无新数据、PACO预训练、PLSP再训练或付费资源。

本轮payload：`artifacts.local/work/functional-parts-joint-20261016/`。旧数据与预测保留在上一轮payload；新结果不得覆盖旧输出。worker按其自身artifact volume定位，不复制整套数据或Python环境。

## 先补齐UMD强对照

[固定读点脚本](full_umd_point_controls.py)复用[既有IIT读点规则](full_point_gap_diagnostics.py)，每图先严格复现两模型原生100格混淆矩阵、原点坐标、GT类别、命中、面积和ignore分账，再做：原softmax峰、稳定log-odds64、固定T2、预测类别区域内稳定峰、最大预测连通域最深内部点。三候选GT oracle只作评价上界，不作为可部署输出。

保留全部10通道；UMD原生−1及scoop3分开记UNKNOWN／ignore。全量、逐类、小区域、工具／clutter分层，并以相同108来源组、1,000次bootstrap、种子20261015比较；同时报告总正类保守分母与公共可评分母。已有正确分割像素却错点、无正确分割像素、旧候选均未命中分别统计。

CPU Torch只用于与旧结果完全一致的原生双线性插值，隐藏CUDA、不加载模型。单线程BelowNormal、每10图原子恢复块、累计10,800秒上限；低可用RAM停止，不无限重试。合成及两图工程验收通过后才进入全量，两图不能代替完整结论。

## 结构对照与消融

全部六组使用同一原始DINO-S/16预训练、最后4个block／norm可训练，以及相同真实RGB stride4／8细节金字塔。普通高分辨率基线拥有真实细节，不能只给主模型增加RGB信息。共同模块按命名seed逐tensor同初始化，数据顺序、每图颜色与翻转保持一致。

| arm | 真实RGB高分辨率融合 | 点质量 | 新offset候选 |
| --- | --- | --- | --- |
| plain | 普通加性／FPN | 无 | 无 |
| plain_quality | 相同普通融合 | 有 | 无 |
| detail | 条件局部门控 | 无 | 无 |
| detail_quality | 条件局部门控 | 有 | 无 |
| candidate_quality | 相同普通融合 | 有 | 有 |
| joint | 条件局部门控 | 有 | 有 |

普通与门控融合使用相同RGB、语义／熵引导、卷积参数及MAC，分别加性和乘性组合；零初始化时函数一致。门控只是待验证机制，不预称新颖。所有候选从密集网格及预测offset生成，质量评分对应候选最终落点，推理不读GT；不是只重排原三个固定点。

区域、内部质量、最近可达区域offset及困难背景／错类负点监督从同一人工mask自动生成；不新增人工点标注。全部六组统一coarse辅助CE权重0.2，确保引导分布具有原生类别监督；质量权重0.5、offset权重2。训练448 mask缓存一次，翻转和坐标对应须验收，薄区域不能被额外腐蚀清空。ordinary-quality与gate-quality，以及candidate-quality与joint，分别获得相同监督及权重。

主要结构判断来自joint对candidate_quality、detail_quality对plain_quality等同监督比较；plain用于区分增加点目标本身的作用。每个输出都报告固定稳定峰／内部点及学习点规则，防止只给主模型更好的解码。按预定验证本地化指标选模，同时报告mIoU；UMD不参与训练或选模。

## 计算与效率安排

本轮先做六组单训练seed20261016的机制消融，不把组bootstrap当训练随机性。训练进程壁钟总上限24小时，CPU读点上限另为3小时；这是本轮执行边界，旧轮次已结束的24小时不复用。先副机实际profile前向／反向、累积、显存、CPU和数据开销，再封存共同epoch数（目标80）、微批／有效batch及最终命令。若六组80轮预计超出训练上限，在正式训练前统一缩减轮数，不让不同模型悄悄采用不同充分程度。当前封存micro8／effective16、每4轮全量验证；成本抽样使用固定等间距16张val、单独测loader启动，不能把启动时间逐图放大。工程profile每组至多900秒、训练后完整评估另有6小时边界；这些不冒充纯GPU kernel成本。

相同epoch不等于严格相同算力。共同输入448、有效batch16和数据曝光，记录活跃参数、算子FLOPs覆盖范围、实测update时间、CPU、显存与完整验证／I/O开销。普通／门控同卷积成本；质量／offset新增成本实测披露，不能用无用计算填预算。训练同时保留预定共同update时间预算内的验证最佳checkpoint，提供相同曝光和相同计算预算两种比较，明确预算验证边界粒度及实际成本差。

不缓存随训练变化的DINO视觉特征。缓存仅为训练GT几何；完整推理每图一次forward同时解出各规则，保留原生uint8语义预测、全部类别点和可复核SHA，不保存六组全部37通道浮点图。只取回thin evidence及所需checkpoint，完整可重建payload留副机。

单GPU训练写者，恢复前核实际PID／creationtime、子树、锁与完整model／optimizer／scaler／RNG／样本位置。活跃源码和输入SHA冻结；失败保留后修复具体故障，不无限重试，不恢复主机GPU。训练与评估有独占输出、进度、断点和终态。达到预算时诚实交付未完成项，不把启动或中期成绩当完整收益。

## 当前状态

- UMD副机合成检查PASS；两图原坐标／混淆矩阵／ignore严格复现，壁钟4.68秒，峰值约772MiB，只是工程验收。
- 全量CPU任务`functional-parts-umd-controls-20261016-cpu-a1`已派发，实际进度／终态以payload `controls/full_a1/`及runner为准；源SHA `1d4f2d0018e7e928a44ae74f5372ac8ea126ab96459f344e4ddc30bcb1bbdf13`，既有规则SHA `2d98441c642a94182732f55c00067ae904458a3b19603752db4de11c92158f53`。
- [模型](full_joint_models.py)、[训练](full_joint_train.py)、[统一评估](full_joint_evaluate.py)、[完整汇总](full_joint_report.py)的副机CPU合成自检均已实际PASS；实际本地预训练DINO＋joint模型在CPU合成64px上的forward／backward亦PASS、未初始化CUDA。这些只证明工程兼容，尚无新训练效果。
- [持久campaign](full_joint_campaign.py)当前权威runner为`functional-parts-joint-20261016-campaign-a2`，request `dispatch/campaign_request_a2.json` SHA `81ebad39f057ec8766b08637c64b5cd489476d53691fcc5926ace81096074612`；controller源SHA `b2a2bc545f55b44ef6c067404c93e8e107b63d0348b2233c4e19fb9e87c9e9d9`。先实际逐SHA核对12,348个IIT RGB／mask文件、准备训练几何缓存，UMD强对照成功前不启动GPU训练。
- 原campaign-a1在prepare阶段检查了不应访问的val_duplicate路径，副机按原范围没有迁移该排除项，因而停止，未运行profile或训练。原manifest同时保留2,651个test元数据；已修正为先按train／val过滤再stat／hash，排除项与test仅保留元数据，不新增下载、不读test像素。核原进程／锁释放后，旧campaign目录、终态、request及source保存在`attempts/campaign_prepare_a1/`与dispatch，a2仅修路径准备逻辑，模型／训练／评估／报告源及科学输入未变；累计训练消耗仍为0。
- a2已实际完成12,348份训练／干净验证RGB和mask的逐SHA复核，原预训练权重SHA也一致。`inputs/receipt.json`记录原IIT清单SHA `2d6304cbb8ed91d769ad7bdb37599afbbfb1e6dae3d7acd91792f3648006e68d`、路径派生清单SHA `8bf25fb2cea887b6c5d468af969b52902f31319fb1e7ed9cbd3902d38192634e`，只有image／mask字符串重定位，其他字段一致；test及排除项未访问。当前实际campaign1948／launcher28704，正在准备仅训练几何缓存；状态仅为此时快照，后续以实时owner／terminal为准。
- source五文件冻结，原始UMD规则源单独冻结；预部署报告更新的a1因缺目标dispatch目录失败，源未改写，a2创建目录后更新并实际自测PASS，原日志保留。活跃源码不再修改；故障按owner和完整断点修复，不能重启旧baseline队列。
- 每小时由既有线程heartbeat检查，正常进度静默，完成／失败或改变决定的发现才通知。自动跟进记录归本run；只有两项持久任务，严禁因同步超时或旧终态重复派发。

本轮判断看真实命中增量、救回与损害、小区域／逐类／跨源分组区间和成本，不只看mIoU。若固定解码已解释差距，按其剩余问题解释结构收益；若普通高分辨率或相同点监督达到相当效果，保留简单方案。最终交付完整矩阵、消融、曲线、错误案例及成本，停止已完成的监控，暂不定题。
