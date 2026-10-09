# RGB真实连续段：查询标签与基线评价（2026-10-09）

状态：EXPLORE / SCOPED_EVALUATION_COMPLETE；承接用户“继续”，前阶段见[来源与输入诊断](RGB_BODY_QUERY_DEV_20261009.md)。

本阶段目标：把已取得45帧真实连续片段变为可评分的相机局部可见结构查询，并比较高低分辨率Depth Pro及因果视频深度。先保留原6query结果；标签不完整的查询保留UNKNOWN，真实身体碰撞/安全效果不在本阶段声称。

预算独立于前阶段/CNH：CPU准备/参考标签/评价/核验累计1200秒；GPU模型启动及推理累计600秒，其中Depth Pro子项240秒、VDA子项240秒，失败计入。公开模型/源码新增下载≤2GiB；不购买资源、不采集、不访问保护test。调整范围为标签可评价规则、观察端公开K变换及基线执行方式，所有改变保留前版结果。交付标签覆盖/双目差异表、逐query配对基线结果、因果性/输入隔离检查、实际成本与下一决定。

评价单位是1个已消费train session中的45个相关帧、6query共270格；不能按270个独立真实事件推断泛化。主任务先定义“可见已标注结构的米制区域侵入”，区别于完整身体空间占用。正标签必须有明确可见掩码与两种估计深度的几何支持；缺失/遮挡/参考冲突UNKNOWN；负标签只对明确的可见目标任务成立，不把全空间无深度支持标成安全。双目参考是估计值，人工/传播和tracking质量分列。

最小可辨识效应是1个可评分query-frame或1个接近事件；分母未知时先报告覆盖，不能先定百分比胜出线。当前探索不设推广门槛。既有输入细节结果不自动证明检出收益。所有模型仅看同一左RGB历史和允许的公开K；depth/mask/未来pose只在评价端，禁止GT尺度拟合。正式效果解释取决于标签可评价性，而非执行是否成功。

## 结论与下一决定

后继已完成[首批细结构补标](RGB_BODY_QUERY_LABEL_COMPLETION_DEV_20261009.md)，以下保留本阶段交付时的原域结果与计划；补标为独立Development增补，不覆盖原类池/预算身份。

真实视频的参考域与两类深度基线已实际跑通，但当前片段只支持**已标注结构角域内的几何诊断**。不能用它声称独立RGB细杆检测、低误报或提醒提前量。固定共同18帧中，Depth Pro原图/128输入的像素侵入IoU为.3284/.3261，未建立高分辨率增益；公开K居中重投影的原图臂更差(.0351)，不采纳该适配器。VDA因果模型全45帧完成，却漏掉后段全部10个近距离参考正query，不能因全段IoU较高就称其优于只完成前段的Depth Pro。

下一优先补**真实细障碍标注与非空角域负例**：沿用原10个train session，以固定人工anchor核验杆/横杆掩码覆盖、两种深度的边界和遮挡，不按本次模型成绩挑帧；建立按session隔离的可训练/可评价清单。金属细杆漏标的区域需单独核验，不能直接扩semantic类池或把双目一致当物理真值。基线继续保留raw Depth Pro与VDA因果模型，先修数据可评价性，再做query驱动高分辨率局部读取与时序训练。真实定量仍为硬目标，身体外参不足时先评价相机局部可见米制侵入。该后续工作未计为本轮已训练成果。

## 参考标签：59正 / 25空域负 / 186 UNKNOWN

固定session `-PqSDmiEe2pXjmYHgxh4YEBsj0T5LU10`、胸左相机392–436帧，7人工anchor为396/402/408/414/420/426/432，其余38传播；READY42/45不代表pose准确。相机右/下/前，x左右中三个区，y±.55m，z近[.3,3]/远[3,6]m，闭边界与原6query一致，不使用世界pose。

参考网格为CRES原生1280×720；ZED与semantic最近邻对齐，K按half-pixel缩放。目标域为官方类4/9/22/24/28（guardrail/handrail/traffic sign/pole/tree），保留instance0空间掩码但身份UNKNOWN。每条目标射线求与query盒相交的光学Z区间。两种估计depth有限、正、≤80m且`abs(CRES−ZED)≤max(.25m,.25×min)`时，整个双目端点区间在盒内为正、整个区间在同侧盒外为负；缺失、冲突或跨界保持UNKNOWN。≥16正像素才构成正query；无正且无未解目标射线为目标集合负。该规则是双目估计一致性诊断，不是独立物理精度保证。

| query | 正 | 目标集合负 | UNKNOWN |
| --- | ---: | ---: | ---: |
| center_near | 0 | 0 | 45 |
| left_near | 0 | 0 | 45 |
| right_near | 10 | 0 | 35 |
| center_far | 18 | 4 | 23 |
| left_far | 9 | 20 | 16 |
| right_far | 22 | 1 | 22 |
| 合计 | 59 | 25 | 186 |

人工子集42格为8正/2集合负/32UNKNOWN。**25集合负全部是目标角域为空的空域负**（人工2格也一样），oracle预测必为阴性；修正版评价将其全部排除query混淆，单列计数。不存在可用于query误报率的非空域负样本。像素指标仍可评价每个query内已知的正/盒外负像素，UNKNOWN和非目标射线不进TP/FN/FP/TN；UNKNOWN query内部已有的已知像素可以进入像素指标。

实际查看五张固定核验图：392/414/436全景、414栏杆点探针及432人工近结构。414明显金属栏杆探针semantic0/ID0，guardrail/handrail在45帧全部无组；车架金属探针为semantic3/ID0，不在该目标池。432近结构正域主要是围栏旁绑着的干叶/玉米秆状装饰，对应class28/ID3；不能称精确树干、树枝或细杆人工真值。Pole24/ID0还合并多根杆，不能当单一跟踪对象。先前audit建议“84格可评分”已被本次空域检查修正为59个非空正格/0个非空负格，原audit文本与v1结果保留。

right_near首次双目支持正帧427为76像素，随后428–436为5194/9420/15663/19764/11580/4481/2562/1663/1194；此前420–426全UNKNOWN，没有已核验NEG→POS起点。frame/fps只给名义时间，因此事件首提醒与真实碰撞提前量均NOT_EVALUABLE。

## 观察端与配对比较

模型仅消费固定左RGB及允许的公开相机信息；stereo/mask/pose不作模型输入，无GT尺度拟合。Depth Pro消费公开fx，raw输入保留原图主点偏心，不能称模型内部完整使用K。K-centered臂根据公开K将原图重投影到居中等焦2271×1313画布（低分辨率128×74），不读取GT；含边缘填充与图像重采样。VDA API不接收K，公开K只供共同几何评价。所有depth投回同一原相机1280×720射线格。

Depth Pro预算到点停在74/180调用：392–409每帧四臂完成，410只完成raw两臂；故共同比较固定392–409的18帧/108格。双目已知参考正像素35947、负像素2280614，都是相关的query-pixel计数，闭盒共边界可重复，不能当独立样本。人工子集396/402/408三帧，正5876、负380929。

| 共同18帧臂 | TP / FN / FP / TN | 像素IoU | 正像素召回 | 负像素FPR | 正query支持 /24 |
| --- | --- | ---: | ---: | ---: | ---: |
| Depth Pro raw原图 | 30497 / 5450 / 56924 / 2223690 | .3284 | .8484 | .02496 | 22 |
| Depth Pro raw128×72 | 32724 / 3223 / 64402 / 2216212 | .3261 | .9103 | .02824 | 23 |
| Depth Pro K-centered原图 | 12319 / 23628 / 315297 / 1965317 | .0351 | .3427 | .13825 | 22 |
| Depth Pro K-centered128 | 26920 / 9027 / 103716 / 2176898 | .1927 | .7489 | .04548 | 22 |
| VDA metric因果 | 2577 / 33370 / 1298 / 2279316 | .0692 | .0717 | .000569 | 8 |

这些是**GT定位角域内的geometry诊断**：pixel只在已标注且双目已知的射线上评分；query支持使用完整GT目标掩码作为定位oracle，≥16预测盒内像素算支持。query支持不保证支持点就是参考正点；25空域负不评分。全图包含地面/其他类的预测可见支持也已保存，但不拿不完整目标池当全场景真值评分。这不是独立RGB任务检出/FPR，也没有同误报预算比较。

三人工帧的raw原图IoU .5787、raw128 .3440，VDA .0177；与全18帧表现不同，提示标签来源和表面组成影响很大。独立XYZ落盒复算五臂全部计数精确一致。共同正域的CRES/ZED深度中位5.359/5.316m，raw原图/128/centered原图/centered128/VDA相对双目中点的逐像素误差中位−1.266/−1.482/−2.570/−1.781/+1.204m；centered原图相对raw正域预测降低1.341m，20436个原TP转FN，其23628个FN中23564低于该射线进入盒所需深度。劣化与低估有关，但重投影、黑边、FOV及上下文同时改变，未隔离原因。高/低输入两臂都经过Depth Pro自己的输入变换，不能把结果简化为任意网络原生分辨率定律。当前既不支持“高分辨率必胜”，也不能用低分辨率近似成绩排除细障碍信息价值。

VDA全45帧可见已知正像素286085、负4885469，TP143835/FN142250/FP4409/TN4881060；IoU .4951、recall .5028、pixel FPR .0009025，人工7帧IoU .4903。它在59个正query支持31个，但right_near427–436 **0/10**，连全场景该近盒预测支持也为0。不能把全段.4951与Depth Pro前18帧.3284直接比较，或把因果视频架构与不同单帧模型的差异称为隔离时序收益。

固定人工432帧的11580近盒参考正像素，CRES/ZED中位2.752/2.764m，VDA为3.700m且最小3.430m；全部预测z>3m，也沿同一射线放大为x>.9m（最小x .924m）。这解释了当前几何读出的漏检：相对双目参考高估距离后，点越出近盒纵深与右边界；不称独立定位模型偏移。`vda/diagnostic_000432.png`为已查看的三幅对照图，仅评价端可视化，无GT选crop模型输入。

## 执行成本、核验与保存

Depth Pro既有权重复用，启动/推理/重投影/I/O累计239.985s/240s，到点PARTIAL_BUDGET；模型调用108.536s，首调用19.056s，稳态中位.8406s，峰值分配3.993GB。压缩NPZ写出和CPU映射消耗明显，下阶段可复用VDA的无压缩保存方式；本轮不扩预算、不补未完成帧。

VDA官方Small metric模型28.4M参数，官方没有单独outdoor版，发布权重训练源为Virtual KITTI＋IRS；采用官方当前帧`infer_video_depth_one`，每次特征输入T=1，缓存frame ID≤当前帧，首帧复制历史隐藏态，45/45完成。未用未来帧或GT拟合；缺xformers使用官方Torch后备实现。启动/推理/保存/释放42.828s/240s，模型调用26.441s，冷启动16.953s，中位.2144s，峰值分配1.332GB；Torch2.11.0+cu130、RTX5060 Laptop、FP16。两臂计282.813s/600s；不同架构、输入、初始化与保存成本不可当手机部署收益。

官方源码`4f5ae23172ba60fd7bc11ef671cca678842c7072`，Small metric权重116444063B，HF revision `273d090f2ce17df50c2872d82c8322c45da5b4dd`，实际模型卡Apache-2.0，权重SHA256 `3c28432b4e1f0d7bb31cad5151b6313b49457db5aa58d82e85bfb0f8b1311b33`。源码本地15993721B，easydict1.13仅安装artifact-local依赖；requests TLS EOF改Windows curl、首次引号失败均记录，无credential。新增公开模型/源码远低于2GiB；本地源码大小不冒充精确网络传输量。

参考准备28.394s；初次重复frame参数错误1.77s修复；评价v1 28.480s、空域负剔除修正版28.228s均计CPU。audit两脚本87.730s及默认Python缺numpy失败.472s保留；独立五臂XYZ复算50.949s，432诊断脚本1.474s（含shell启动3.295s）。6项reference fixture与4项K映射fixture通过，74个Depth Pro输出与90个VDA原生/统一格输出SHA/K/有效掩码核验通过；45参考计数由独立脚本复算一致。原v1错误评分与未完成预测保留，不覆盖旧失败。

CPU准备总计不能宣称精确实测：VDA异步获取缺完整单调计时，已知等待下界46.569s、文件时间估算约63s，保守扣额80s（含其CPU核验/诊断，不重复加）。本轮合并已测参考/评价/audit/独立复算及未逐条计时终端准备的预留，保守预算扣额400s/1200s；扣额是预算记账，不作性能数字。账本见`completion_receipt.json`与`vda/cpu_preparation_ledger.json`。文档索引和项目结构检查通过；无Android改动，不跑无关全仓测试。

源码：[参考与配对评价](rgb_body_query_reference_eval.py)、[参考fixture](test_rgb_body_query_reference_eval.py)、[Depth Pro连续臂](rgb_body_query_sequence_depthpro.py)、[K映射fixture](test_rgb_body_query_sequence_depthpro.py)、[因果VDA](rgb_body_query_vda_stream.py)。复现评价需本轮两个prediction manifest与reference manifest，不需重跑模型。

payload：`artifacts.local/work/rgb-body-query-eval-dev-20261009/`；`reference/`保存逐帧已知/未知标签和来源hash，`evaluation.json`含119×6=714行预测query记录及共同18帧表，`evaluation_v1_vacuous_negatives.json`保存空域修正前结果，`audit/`保存核验图/参考独立统计，`depthpro/`和`vda/`保存执行源码/模型输出/收据。两task-owned模型进程已退出，CUDA资源释放；模型和证据留在canonical artifact junction以便后续复用。此段为已消费train Development，1session/45相关帧，不是跨场景泛化或独立确认，CNH/旧SANPO停止run/旧RGB负结果均按原身份保留。

来源：[SANPO官方](https://github.com/google-research-datasets/sanpo_dataset)、[ZED光学Z API](https://www.stereolabs.com/docs/development/zed-sdk/modules/depth-sensing/using-the-api)、[VDA官方因果源码](https://github.com/DepthAnything/Video-Depth-Anything/blob/4f5ae23172ba60fd7bc11ef671cca678842c7072/video_depth_anything/video_depth_stream.py)、[实际Small metric模型卡](https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Small/blob/273d090f2ce17df50c2872d82c8322c45da5b4dd/README.md)。记录见[RUNS](../RUNS.md)。
