# RGB：冻结残差的归一化余量与距离带校准

日期：2026-10-10。`RGB_BODY_QUERY_RESIDUAL_READOUT_DEV_20261010`，EXPLORE，已消费 Development。起点提交 `24688f0d`。用户授权继续；沿用[实际训练有界残差](RGB_BODY_QUERY_BOUNDED_RESIDUAL_DEV_20261010.md)的最后 step600 预测，测试不同读出／校准机制。

## 结果与决定

**本轮没有恢复可迁移的有利取舍，全局绝对余量 affine 继续作为 RGB 代表。** 三折留一中，两个追加 capture 的 affine 见证/FREE 为120/12、131/10；CTX 的归一化全局结果1/13、2/18，绝对距离带结果60/20、50/8，归一化距离带结果4/7、19/19。新增机制均未形成相对原强对照的主结果优势。距离带 affine 自己在60/65退化至0/0、6/1，不能把残差胜过这个弱化后的同组合 affine 写成胜过原 affine。

**局部事后信号仍保留。** 固定原 affine 每带 FREE 成本后，绝对距离带 DR 在60/65为130/12、131/10；原绝对全局 affine 同成本曲线平台为122/12、132/10，净见证+8/−1。CTX为122/12、131/10。它说明这两 frozen 臂在部分带成本分配下仍有排序余地，但该阈值只用 held 自身 FREE，主三折 DR 仅41/10、26/11，不能升级为可迁移校准收益；也不能归结为“所有排序都无信息”。未满足双判据不自动是负信号，本次主结果下降与局部辅助混合信号分开记录。

近带恢复仍带有成本：主绝对带 CTX 在arkit16见证2/44且FREE3/96；60见证0/11且FREE7/133；65没有近带POS，见证增益N/E，FREE6/144。归一化全局近带CTX仍无见证。归一化去掉了绝对余量≤半区间宽的结构上限，却未恢复本次固定排序与FREE取舍，不能确定单一原因。

本run完成并停止，不追加归一化尺度、每带预算、阈值或子集调参；原训练run停止约束保持。下一步优先改变模型可观测信息或合规train域覆盖，再用冻结读出对照复核；距离条件校准保留为备选，但不以继续挪阈值替代模型改进，不把cal转成train。旧A0三个δ／三capture、完整失败与旧stop均保留。

## 冻结方法与证据范围

三臂为affine、DR（训练depth/ray）、CTX（训练depth/ray＋局部DP上下文）；两残差均原56train帧、同4993参数/初始化/600batch，B=3.1847211408780134，不重新训练。四组合事先固定并全部报告。绝对评分沿用第16大 `min(z−entry,exit−z)`；归一化评分为第16大 `2*min(z−entry,exit−z)/(exit−entry)`，只用冻结public valid和query domain，width>1e−12，其他射线−Infinity。归一化有效分数≤1，非负当且仅当绝对余量非负；query分数与evaluator已知正射线见证分数分别排序。

三距离带为public query Z上下界0.3–0.8、0.8–1.5、1.5–3m，绝不用GT距离选择带。POS≥16已知正射线；严格FREE要求正0、UNKNOWN0、free≥16；UNKNOWN保持UNKNOWN。主阈值只用另两个capture的FREE，每个全局或每带预算≤floor(.05*N)，ties整组保留。补充阈值先由304cal冻结，再打开新归一化eval：588严格FREE，近/中/远395/180/13，每带预算19/9/0，全局29。cal仍是cal。

如果cal源某带没有FREE，cutoff=None、N/E，不回退全局或held阈值，不记为零错误。主/补本次均全部432或各cohort查询可校准；辅助band在arkit16远带没有held FREE，144query不可评估，必须用共同子集比较。所有数值仅已消费Development、native first-return、固定采样query-ray；相关帧不构成独立重复，FREE不等于整身体体积清空、硬件或安全证明。

## 主三折：四组合完整报告

单元为正见证W／严格FREE支持F。Ark16/60/65的POS分母228/202/179，FREE分母105/220/247；各组合均432query，校准缺失0。DR/CTX不可逐capture挑最佳。

| Capture/臂 | absolute-global | absolute-band | normalized-global | normalized-band |
| --- | --- | --- | --- | --- |
| 16/affine | 103/5 | 94/78 | 97/5 | 86/39 |
| 16/DR | 47/0 | 24/5 | 31/0 | 10/4 |
| 16/CTX | 52/0 | 29/3 | 4/0 | 3/2 |
| 60/affine | 120/12 | 0/0 | 120/12 | 0/0 |
| 60/DR | 103/10 | 41/10 | 66/14 | 6/12 |
| 60/CTX | 94/22 | 60/20 | 1/13 | 4/7 |
| 65/affine | 131/10 | 6/1 | 131/10 | 3/2 |
| 65/DR | 104/26 | 26/11 | 51/18 | 7/10 |
| 65/CTX | 102/9 | 50/8 | 2/18 | 19/19 |

### 0.3–0.8m 主结果单列

近POS分母44/11/0，FREE分母96/133/144；65的0见证计数不表示可检验近带收益。

| Capture/臂 | absolute-global | absolute-band | normalized-global | normalized-band |
| --- | --- | --- | --- | --- |
| 16/affine | 0/0 | 8/69 | 0/0 | 5/31 |
| 16/DR | 0/0 | 3/5 | 0/0 | 2/4 |
| 16/CTX | 0/0 | 2/3 | 0/0 | 0/2 |
| 60/affine | 0/0 | 0/0 | 0/0 | 0/0 |
| 60/DR | 0/0 | 1/9 | 0/1 | 1/10 |
| 60/CTX | 0/0 | 0/7 | 0/0 | 0/5 |
| 65/affine | 0/0 | 0/0 | 0/0 | 0/1 |
| 65/DR | 0/0 | 0/0 | 0/0 | 0/1 |
| 65/CTX | 0/0 | 0/6 | 0/0 | 0/12 |

### 与原 absolute-global affine 配对

各单元“正见证救/损；FREE增/减”；三条均共同432query，无cal缺失。原强对照固定，不以同组合 affine 退化掩盖损失。

| Capture/臂 | absolute-global | absolute-band | normalized-global | normalized-band |
| --- | --- | --- | --- | --- |
| 16/affine | 0/0; 0/0 | 59/68; 73/0 | 0/6; 0/0 | 51/68; 34/0 |
| 16/DR | 8/64; 0/5 | 10/89; 5/5 | 7/79; 0/5 | 9/102; 4/5 |
| 16/CTX | 4/55; 0/5 | 6/80; 3/5 | 3/102; 0/5 | 1/101; 2/5 |
| 60/affine | 0/0; 0/0 | 0/120; 0/12 | 0/0; 0/0 | 0/120; 0/12 |
| 60/DR | 9/26; 2/4 | 5/84; 9/11 | 9/63; 5/3 | 1/115; 10/10 |
| 60/CTX | 3/29; 12/2 | 0/60; 13/5 | 0/119; 13/12 | 0/116; 7/12 |
| 65/affine | 0/0; 0/0 | 0/125; 0/9 | 0/0; 0/0 | 0/128; 1/9 |
| 65/DR | 3/30; 18/2 | 0/105; 11/10 | 2/82; 11/3 | 0/124; 10/10 |
| 65/CTX | 2/31; 4/5 | 1/82; 8/10 | 2/131; 18/10 | 0/112; 18/9 |

### CTX直接对DR

| Capture | absolute-global | absolute-band | normalized-global | normalized-band |
| --- | --- | --- | --- | --- |
| 16 | 18/13; 0/0 | 24/19; 1/3 | 4/31; 0/0 | 3/10; 1/3 |
| 60 | 9/18; 14/2 | 26/7; 17/7 | 0/65; 11/12 | 4/6; 5/10 |
| 65 | 14/16; 4/21 | 29/5; 8/11 | 1/50; 14/14 | 15/3; 17/8 |

## 补充：304cal冻结阈值的全部五cohort

单元W/F；全部query可校准。原validation24 POS295/FREE2，新3RScan64 POS866/FREE1，三Ark分母同主表。这些极少eval FREE使原validation与新3RScan零错误结果很弱，不能写成跨环境安全证据。

| Cohort/臂 | absolute-global | absolute-band | normalized-global | normalized-band |
| --- | --- | --- | --- | --- |
| 原validation24/affine | 219/1 | 134/0 | 225/1 | 129/0 |
| 原validation24/DR | 104/0 | 24/0 | 47/0 | 21/0 |
| 原validation24/CTX | 88/0 | 10/0 | 11/0 | 11/0 |
| 新3RScan64/affine | 570/1 | 490/0 | 584/1 | 441/0 |
| 新3RScan64/DR | 256/1 | 88/0 | 119/1 | 73/0 |
| 新3RScan64/CTX | 211/1 | 34/0 | 30/0 | 30/0 |
| 16/affine | 95/5 | 13/35 | 92/5 | 17/17 |
| 16/DR | 50/0 | 3/1 | 23/0 | 2/0 |
| 16/CTX | 53/0 | 9/2 | 6/0 | 4/2 |
| 60/affine | 120/12 | 57/5 | 119/12 | 13/2 |
| 60/DR | 106/11 | 3/2 | 48/9 | 3/4 |
| 60/CTX | 91/19 | 1/4 | 4/17 | 2/8 |
| 65/affine | 131/10 | 48/4 | 131/10 | 8/3 |
| 65/DR | 96/19 | 0/4 | 37/9 | 2/5 |
| 65/CTX | 102/13 | 4/3 | 3/24 | 1/9 |

### 补充近带

| Cohort/臂 | absolute-global | absolute-band | normalized-global | normalized-band |
| --- | --- | --- | --- | --- |
| 原validation24/affine | 3/0 | 24/0 | 1/0 | 21/0 |
| 原validation24/DR | 0/0 | 8/0 | 0/0 | 6/0 |
| 原validation24/CTX | 0/0 | 5/0 | 0/0 | 5/0 |
| 新3RScan64/affine | 3/0 | 160/0 | 1/0 | 116/0 |
| 新3RScan64/DR | 0/0 | 44/0 | 6/0 | 42/0 |
| 新3RScan64/CTX | 0/0 | 14/0 | 4/0 | 15/0 |
| 16/affine | 0/0 | 0/35 | 0/0 | 2/17 |
| 16/DR | 0/0 | 0/1 | 0/0 | 0/0 |
| 16/CTX | 0/0 | 2/2 | 0/0 | 0/2 |
| 60/affine | 0/0 | 0/0 | 0/0 | 0/0 |
| 60/DR | 0/0 | 0/2 | 0/0 | 0/3 |
| 60/CTX | 0/0 | 0/2 | 0/0 | 0/2 |
| 65/affine | 0/0 | 0/0 | 0/0 | 0/2 |
| 65/DR | 0/0 | 0/0 | 0/0 | 0/0 |
| 65/CTX | 0/0 | 0/1 | 0/1 | 0/2 |

原validation近POS24/FREE0，新3RScan近POS168/FREE0；近带零FREE支持没有负例检验力。完整每带／环境／环境×带分母、UNKNOWN支持、same-combo affine／direct DR／old same-arm与strong affine配对均在各`summary.csv`、`query_pairs.csv`和`results.json`，没有只报告总体。

## 事后同成本辅助

Global用held自身FREE选原affine预算5/12/10，各臂包括affine都填满允许平台，并报告完整0..N曲线。Band预算固定为16=(0,5,0)、60=(0,0,12)、65=(0,0,10)，不重新分配。只辅助诊断，不用于选模型或训练；单元W/F。

| Capture/臂 | absolute-global | absolute-band | normalized-global | normalized-band |
| --- | --- | --- | --- | --- |
| 16/affine | 113/5 | 45/5 | 108/5 | 44/5 |
| 16/DR | 103/5 | 61/5 | 98/5 | 61/5 |
| 16/CTX | 138/5 | 104/5 | 81/5 | 108/5 |
| 60/affine | 122/12 | 123/12 | 122/12 | 123/12 |
| 60/DR | 110/12 | 130/12 | 62/12 | 123/12 |
| 60/CTX | 91/12 | 122/12 | 1/12 | 122/12 |
| 65/affine | 132/10 | 132/10 | 132/10 | 132/10 |
| 65/DR | 80/10 | 131/10 | 42/10 | 131/10 |
| 65/CTX | 102/10 | 131/10 | 0/10 | 131/10 |

Ark16 band只评估近＋中288/432query，POS158/228、FREE105/105；远带144query中POS70、FREE0、UNKNOWN74，全部N/E。band的104或108见证不能与full432的138直接相减；原冻结主affine在该共同子集W35，原absolute-global曲线affine在该共同子集W45。

### 对原 affine 完整曲线点：共同子集配对

以下直接从冻结point evaluation派生，每query原支持/见证与共同cal标记保留；不选新cut。单元“救/损；FREE增/减”。

| Capture/臂 | absolute-global | absolute-band | normalized-global | normalized-band |
| --- | --- | --- | --- | --- |
| 16/affine | 0/0; 0/0 | 0/0; 0/0 | 0/5; 0/0 | 0/1; 0/0 |
| 16/DR | 28/38; 2/2 | 37/21; 2/2 | 26/41; 2/2 | 37/21; 2/2 |
| 16/CTX | 50/25; 3/3 | 61/2; 3/3 | 31/63; 3/3 | 65/2; 3/3 |
| 60/affine | 0/0; 0/0 | 1/0; 0/0 | 0/0; 0/0 | 1/0; 0/0 |
| 60/DR | 12/24; 4/4 | 9/1; 0/0 | 8/68; 3/3 | 2/1; 0/0 |
| 60/CTX | 0/31; 2/2 | 1/1; 0/0 | 0/121; 12/12 | 1/1; 0/0 |
| 65/affine | 0/0; 0/0 | 0/0; 0/0 | 0/0; 0/0 | 0/0; 0/0 |
| 65/DR | 0/52; 5/5 | 0/1; 0/0 | 0/90; 6/6 | 0/1; 0/0 |
| 65/CTX | 2/32; 5/5 | 0/1; 0/0 | 0/132; 10/10 | 0/1; 0/0 |

## 校准阈值与实际成本

| 组合/臂 | global或near cutoff; FREE | mid cutoff; FREE | far cutoff; FREE |
| --- | --- | --- | --- |
| absolute-global/affine | -0.0810713576; 29/588 | - | - |
| absolute-global/DR | 0.265199845; 29/588 | - | - |
| absolute-global/CTX | 0.348014469; 29/588 | - | - |
| absolute-band/affine | -0.775622496; 19/395 | 0.00315595399; 9/180 | 0.659210865; 0/13 |
| absolute-band/DR | 0.0349242026; 19/395 | 0.346403977; 9/180 | 0.748525398; 0/13 |
| absolute-band/CTX | 0.169942509; 19/395 | 0.348801767; 9/180 | 0.744213211; 0/13 |
| normalized-global/affine | -0.21899025; 29/588 | - | - |
| normalized-global/DR | 0.989104637; 29/588 | - | - |
| normalized-global/CTX | 0.996531168; 29/588 | - | - |
| normalized-band/affine | -3.96675653; 19/395 | 0.00901701141; 9/180 | 0.997473937; 0/13 |
| normalized-band/DR | 0.355538576; 19/395 | 0.995017844; 9/180 | 0.998239417; 0/13 |
| normalized-band/CTX | 0.980756837; 19/395 | 0.99789995; 9/180 | 0.996528544; 0/13 |

绝对阈值单位m，归一化阈值无单位，不跨评分直接比较数值。Global实际29/588；band实际19＋9＋0=28/588；分带后每带floor预算不是把原29任意分配，报告实际28。far cal仅13个FREE，零预算不等于零风险。主三折全部global／band切点与实际cal支持、tie可达性在`final/*_thresholds.json`。

## 核验、成本与复现

独立focused核验PASS：96个严格FREE-only切点，173,664query决策、505,440完整配对与7,200分组（包含same-arm跨读出、同组合affine与原强affine、直接CTX对DR）、3,450保ties global曲线、54固定带成本切点及6个empty-FREE N/E描述符。独立原生Ark16一帧81臂query／162评分分量最大误差0，649,788像素sign检查；没有重跑模型或训练。补核完整864query确认DR60对原affine曲线救9损1且FREE增减0；65救0损1且FREE增减0；六个旧绝对评分cache的SHA与生产input receipt一致，两step600身份保留。

生产阶段重算35,640归一化arm-query、242,886,612像素上界与零点符号，11,880reference状态与880trained valid mask保持；小/零宽域和query availability改变均0。原absolute-global cal三阈值／主九阈值精确复现，旧分数缓存与冻结model SHA通过；分带改变阈值，不改变query reference。独立核验覆盖数值与源码合同，没有额外全文件访问tracer，不把缓存复核描述为新模型推理。

CPU评分／校准进程墙钟：cal **31.602s**、final **20.005s**，科学累计 **51.607/700s**；postcheck .076612s，focused加headroom **12.806/180s**。整合全额计200s，合计保守上界264.490s，向上十秒取整 **270/1200s**，是墙钟预算保守记账，不是实测CPU算时。科学／focused均无失败；报告派生配对计入整合额度，不修改冻结科学输出。

TASK_NOT_GPU_SUITABLE：native缓存NumPy几何评分、scalar FREE校准及JSON配对；GPU0、训练0、新模型forward0、下载0。cal/final同步session完成，helper没有writer/lock或后台子进程；无保留任务计算资源。旧checkpoint/public预测及新score/ledger/失败历史为持久证据保留。

父planSHA `7d3fa779ac8be9e7c02c17862e4733f28decce5404aeeb93b2d2ba269a6c289a`；cal执行源码 `b893fcde21d229706be25c501d1d88490883a7c1f812df974b921cf2d5dda6ec`，final/交付 `292ef05e3bbd0ac790cf5009732b2228fdaeba1017d4d2ceb8716f30f550ae63`。cal后只增加同臂跨读出与曲线基准配对、父plan与模型身份断言，normalized/select/evaluate/summary/paired五核心函数AST保持；cal原snapshot与final新版分别留证，cal同臂派生配对单独存final，不覆盖cal。report另外15,552逐query强curve点配对仅从冻结point evaluations派生，无新评分／cutoff；来源SHA和共同子集分母独立保存。

两沿用step600 SHA：DR `a3866aac9738d7c3594bd36212a21ccef93c2116e43035f0d54f977b1c569562`；CTX `d629a677641c27de4a7070a4b5a0b035a1ec1e9fe4243f3be62ca30068e99205`。本轮不证明其他归一化、其他校准法或不同训练机制无价值。

复现脚本：[冻结读出与校准](rgb_body_query_residual_readout.py)，参数需显式`--repo`、`--runroot`、`--output`、`--stage calibrate|final`、`--budget-s`，final另传`--calibration`；先提供本run冻结plan。现有stage禁止覆盖或用重启重置预算。项目科学runtime需要NumPy；没有Torch/CUDA模型前向。

证据根：`artifacts.local/work/rgb-body-query-residual-readout-dev-20261010/`。`plan.json`、`evaluation/calibration/thresholds.json`先冻结、两executed snapshots/input SHA/width diagnostics/terminal、final四组main-loco/pooled-supplemental/posthoc完整query ledger、same-arm跨读出pairs、全3,450 global curves、`report_strong_curve_pairs.csv`与summary及后续focused/publication/resource/delivery receipts。旧两step600 checkpoint和304＋136公共预测在原run保留，未删除或修改。
