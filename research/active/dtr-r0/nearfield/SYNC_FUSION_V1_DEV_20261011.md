# 同步源 query 融合器 v1（EXPLORE）

执行日期：2026-10-11；输入版本 `5902640c`；已完成一次eval及独立复算。

**结论：仅半循环来源支持，独立几何未复现。** 主HGB在扰动native近/中带分别W278/371、497/542，F7/690（1.01%）、9/286（3.15%），比最佳单传感器净增20/104个POS命中；远带虽净增26，F7/30（23.33%）超10%目标。FARO近带HGB W145/163、F8/151（5.30%），未超过RGB W146/163、F2/151；中远因cal FREE=0不可校准。logistic在native近/中带同向通过，在FARO近带F14/151仍超标，不能以它替换主HGB来改变判读。

这支持继续把半循环结果作为探索线索，**不支持声明双来源query融合收益或升级默认候选**。后续应先按预声明输入/标签覆盖规则补独立几何cal及新visit评价源；本eval已经消费，不能再改模型/阈值后称一次性评价。近/中HGB相对最佳单传感器净收益的visit bootstrap区间均跨0，4visit证据较弱；本判读按预声明点估计规则，未冒充显著性结论。

## 已冻结的实验口径

用户决定进入小模型融合阶段。主来源为 v1.1 扰动 native，始终标为**半循环**；融合 FARO 是独立几何交叉检验，仍有官方筛选偏差和静态表面长尾误差。原连续源 80/80 失败保留，不升级为合格实机 ToF 来源。

[PLAN](SYNC_FUSION_V1_PLAN_DEV_20261011.json) 在训练及 eval 前固定。原 visit 分区保持 train6/cal2/eval4；native 保留全部192/64/128帧，FARO 使用固定 K0 `joint_pass_zones>=52` 门槛，分别57/7/39帧。FARO 的 train/cal 实际仅有5/1个 visit 贡献过门帧，不把筛空 visit 迁入其他分区。K0/K1先汇成一条 query 特征，不翻倍分母。

固定20个输入仅来自两传感器预测：ToF第16余量、支持像素射线数、UNKNOWN比例、峰相对区间的前/内/后比例及偏移；DAV2、UniDepth各自第16余量和有效比例；缺测标志、有效K数、距离带与固定query位置。无参考标签、类别、几何真值或visit身份输入。模型余量缺测编码−10并带缺测标志；基线保留原始−∞，不把缺测当有限支持。射线数是投影像素射线数，并非独立物理sub-ray数。

每来源固定 HGB（100轮、lr .1、7叶、深3、min_leaf50、L2=1、无早停）及标准化 logistic（C1/L2/lbfgs/1000轮）；seed955/956/957概率均值。仅train的POS与严格FREE拟合，UNKNOWN不作负例；标准化仅fit train。确定性配方下三个seed可相同，不视为三次独立证据。

cal逐带近2%/中5%/远10% FREE预算，枚举所有有限分数ties及+∞；最大W、较小F、较高阈值顺序选点。ToF、分带RGB（Uni近/DAV中远）、OR=max原始余量、AND=min原始余量均独立按此规则校准。OR/AND是分数max/min后校准，不是重新组合已校准的两个二值工作点。FREE或POS分母为0则不校准、拒绝全部，禁止借eval补分母。

**W统一为“预测query为正且参考query为POS”**，F为严格FREE上的预测正，U为UNKNOWN上的预测正。此口径不要求旧报告的像素见证交集，不能把本表W与v1.1旧W直接纵向比较；传感器预测和几何读出保持冻结。

HGB为预声明主模型，logistic为独立简单对照，不按eval挑获胜者。最佳单传感器取两个冻结cal点在eval中的较大W（同W较小F，再同则RGB），用于严格要求胜过两者；bootstrap中固定该比较器身份。融合器需在两个来源**相同至少两带**均W超过两单传感器，且eval F/FREE不超该带目标，方可声明双源收益。

## 校准可评价性

在打开eval前已发现：FARO cal仅7帧，近/中/远POS=38/51/12、FREE=16/0/0。中远带为NOT_CALIBRATABLE，无法满足双源至少两带的判读条件。这是独立来源校准覆盖缺口，不能解释成这两带实证无收益；本轮不放宽门槛或改分区。

## 结果与收据

payload：`artifacts.local/work/sync-fusion-v1-dev-20261011/`。

## 一次 eval 结果

W/POS、F/FREE、U/UNKNOWN均为query计数。NA为cal不可校准，预设拒绝全部，不作为有效工作点或实证失败。

| 来源 | 方法 | 带 | W/POS | F/FREE | U/UNKNOWN | W率95%区间(%) | F率95%区间(%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 过门FARO | hgb | 0.3-0.8m | 145/163 | 8/151 | 16/37 | [80.6, 92.0] | [0.0, 7.1] |
| 过门FARO | logit | 0.3-0.8m | 150/163 | 14/151 | 35/37 | [81.5, 97.0] | [2.6, 12.4] |
| 过门FARO | tof | 0.3-0.8m | 85/163 | 16/151 | 0/37 | [42.2, 67.3] | [6.5, 13.5] |
| 过门FARO | rgb | 0.3-0.8m | 146/163 | 2/151 | 32/37 | [77.7, 94.0] | [0.0, 2.2] |
| 过门FARO | or | 0.3-0.8m | 153/163 | 18/151 | 25/37 | [87.8, 95.9] | [6.5, 15.7] |
| 过门FARO | and | 0.3-0.8m | 123/163 | 11/151 | 0/37 | [60.4, 85.7] | [4.5, 12.7] |
| 过门FARO | hgb | 0.8-1.5m | NA/153 | NA/43 | NA/155 | NA | NA |
| 过门FARO | logit | 0.8-1.5m | NA/153 | NA/43 | NA/155 | NA | NA |
| 过门FARO | tof | 0.8-1.5m | NA/153 | NA/43 | NA/155 | NA | NA |
| 过门FARO | rgb | 0.8-1.5m | NA/153 | NA/43 | NA/155 | NA | NA |
| 过门FARO | or | 0.8-1.5m | NA/153 | NA/43 | NA/155 | NA | NA |
| 过门FARO | and | 0.8-1.5m | NA/153 | NA/43 | NA/155 | NA | NA |
| 过门FARO | hgb | 1.5-3m | NA/97 | NA/13 | NA/241 | NA | NA |
| 过门FARO | logit | 1.5-3m | NA/97 | NA/13 | NA/241 | NA | NA |
| 过门FARO | tof | 1.5-3m | NA/97 | NA/13 | NA/241 | NA | NA |
| 过门FARO | rgb | 1.5-3m | NA/97 | NA/13 | NA/241 | NA | NA |
| 过门FARO | or | 1.5-3m | NA/97 | NA/13 | NA/241 | NA | NA |
| 过门FARO | and | 1.5-3m | NA/97 | NA/13 | NA/241 | NA | NA |
| 半循环native | hgb | 0.3-0.8m | 278/371 | 7/690 | 74/91 | [70.6, 85.3] | [0.2, 2.0] |
| 半循环native | logit | 0.3-0.8m | 291/371 | 1/690 | 81/91 | [74.0, 89.0] | [0.0, 0.4] |
| 半循环native | tof | 0.3-0.8m | 209/371 | 43/690 | 1/91 | [36.7, 73.6] | [2.2, 11.6] |
| 半循环native | rgb | 0.3-0.8m | 258/371 | 9/690 | 84/91 | [55.3, 86.2] | [0.0, 2.8] |
| 半循环native | or | 0.3-0.8m | 305/371 | 16/690 | 78/91 | [77.8, 87.8] | [1.6, 3.6] |
| 半循环native | and | 0.3-0.8m | 167/371 | 12/690 | 1/91 | [36.3, 52.2] | [1.1, 2.8] |
| 半循环native | hgb | 0.8-1.5m | 497/542 | 9/286 | 196/324 | [86.7, 96.2] | [1.2, 6.8] |
| 半循环native | logit | 0.8-1.5m | 508/542 | 13/286 | 205/324 | [87.9, 99.0] | [0.6, 10.8] |
| 半循环native | tof | 0.8-1.5m | 393/542 | 28/286 | 65/324 | [44.4, 94.3] | [5.2, 19.5] |
| 半循环native | rgb | 0.8-1.5m | 352/542 | 3/286 | 316/324 | [55.8, 72.2] | [0.0, 2.6] |
| 半循环native | or | 0.8-1.5m | 496/542 | 23/286 | 313/324 | [89.4, 94.8] | [3.1, 16.5] |
| 半循环native | and | 0.8-1.5m | 347/542 | 10/286 | 168/324 | [35.0, 85.5] | [0.8, 5.0] |
| 半循环native | hgb | 1.5-3m | 511/616 | 7/30 | 153/506 | [72.7, 91.9] | [0.0, 46.7] |
| 半循环native | logit | 1.5-3m | 537/616 | 10/30 | 315/506 | [74.0, 97.0] | [0.0, 66.7] |
| 半循环native | tof | 1.5-3m | 463/616 | 2/30 | 296/506 | [54.3, 92.8] | [0.0, 25.0] |
| 半循环native | rgb | 1.5-3m | 485/616 | 0/30 | 398/506 | [72.8, 85.2] | [0.0, 0.0] |
| 半循环native | or | 1.5-3m | 551/616 | 2/30 | 401/506 | [83.8, 96.7] | [0.0, 25.0] |
| 半循环native | and | 1.5-3m | 440/616 | 1/30 | 296/506 | [53.1, 84.3] | [0.0, 6.7] |

## 逐 query 救/损

POS上救/损；FREE列为新增/移除误支持。best_single为本带两单传感器固定点中W较高者；完整UNKNOWN增减及所有基线配对见payload `eval/paired.csv`。

| 来源 | 融合器 | 带 | 对照(角色) | POS救/损 | 净W | FREE增/减 | 净W/POS 95%区间(pp) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 过门FARO | hgb | 0.3-0.8m | rgb (best_single) | 1/2 | -1 | 6/0 | [-2.1, 2.9] |
| 过门FARO | hgb | 0.3-0.8m | or (OR) | 0/8 | -8 | 0/10 | [-7.9, -2.4] |
| 过门FARO | logit | 0.3-0.8m | rgb (best_single) | 5/1 | +4 | 12/0 | [0.7, 7.3] |
| 过门FARO | logit | 0.3-0.8m | or (OR) | 3/6 | -3 | 6/10 | [-7.9, 4.2] |
| 半循环native | hgb | 0.3-0.8m | rgb (best_single) | 45/25 | +20 | 7/9 | [-4.0, 19.6] |
| 半循环native | hgb | 0.3-0.8m | or (OR) | 15/42 | -27 | 7/16 | [-12.1, 0.0] |
| 半循环native | logit | 0.3-0.8m | rgb (best_single) | 44/11 | +33 | 1/9 | [0.4, 22.2] |
| 半循环native | logit | 0.3-0.8m | or (OR) | 9/23 | -14 | 1/16 | [-9.5, 3.7] |
| 半循环native | hgb | 0.8-1.5m | tof (best_single) | 122/18 | +104 | 7/26 | [-4.3, 50.7] |
| 半循环native | hgb | 0.8-1.5m | or (OR) | 21/20 | +1 | 7/21 | [-6.1, 4.7] |
| 半循环native | logit | 0.8-1.5m | tof (best_single) | 126/11 | +115 | 10/25 | [-1.2, 54.5] |
| 半循环native | logit | 0.8-1.5m | or (OR) | 23/11 | +12 | 10/20 | [-2.7, 7.5] |
| 半循环native | hgb | 1.5-3m | rgb (best_single) | 44/18 | +26 | 7/0 | [-5.2, 9.9] |
| 半循环native | hgb | 1.5-3m | or (OR) | 14/54 | -40 | 7/2 | [-18.0, 2.0] |
| 半循环native | logit | 1.5-3m | rgb (best_single) | 68/16 | +52 | 10/0 | [-4.8, 15.4] |
| 半循环native | logit | 1.5-3m | or (OR) | 29/43 | -14 | 9/1 | [-15.7, 7.6] |

## 校准封存表

有限阈值按score≥t；+∞为拒绝全部。ToF/RGB/OR/AND单位为米余量，HGB/logistic为概率。

| 来源 | 方法 | 带 | 阈值 | cal W | cal F | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| 半循环native | hgb | 0.3-0.8m | 0.6686510336287604 | 108 | 6 | ADOPTED |
| 半循环native | logit | 0.3-0.8m | 0.6795376439419151 | 110 | 6 | ADOPTED |
| 半循环native | tof | 0.3-0.8m | -0.11449318896104788 | 85 | 6 | ADOPTED |
| 半循环native | rgb | 0.3-0.8m | -0.0843518495559692 | 124 | 7 | ADOPTED |
| 半循环native | or | 0.3-0.8m | -0.06486876010894771 | 128 | 7 | ADOPTED |
| 半循环native | and | 0.3-0.8m | -0.16414810419082637 | 96 | 6 | ADOPTED |
| 半循环native | hgb | 0.8-1.5m | 0.6461185730943816 | 291 | 7 | ADOPTED |
| 半循环native | logit | 0.8-1.5m | 0.5155544783751843 | 299 | 9 | ADOPTED |
| 半循环native | tof | 0.8-1.5m | -0.11298683905320672 | 230 | 10 | ADOPTED |
| 半循环native | rgb | 0.8-1.5m | -0.201171875 | 287 | 10 | ADOPTED |
| 半循环native | or | 0.8-1.5m | -0.089376537053239 | 297 | 8 | ADOPTED |
| 半循环native | and | 0.8-1.5m | -0.4825156502053798 | 252 | 9 | ADOPTED |
| 半循环native | hgb | 1.5-3m | 0.6703441991227989 | 339 | 5 | ADOPTED |
| 半循环native | logit | 1.5-3m | 0.4053749160311797 | 348 | 4 | ADOPTED |
| 半循环native | tof | 1.5-3m | -1.332122600902642 | 288 | 1 | ADOPTED |
| 半循环native | rgb | 1.5-3m | -0.5860885749710567 | 347 | 4 | ADOPTED |
| 半循环native | or | 1.5-3m | -0.5830690808029735 | 349 | 5 | ADOPTED |
| 半循环native | and | 1.5-3m | -1.332122600902642 | 288 | 1 | ADOPTED |
| 过门FARO | hgb | 0.3-0.8m | 0.8175369322690281 | 37 | 0 | ADOPTED |
| 过门FARO | logit | 0.3-0.8m | 0.7579329350906341 | 35 | 0 | ADOPTED |
| 过门FARO | tof | 0.3-0.8m | -0.05097152488772705 | 27 | 0 | ADOPTED |
| 过门FARO | rgb | 0.3-0.8m | -0.058664691448211626 | 34 | 0 | ADOPTED |
| 过门FARO | or | 0.3-0.8m | -0.05097152488772705 | 35 | 0 | ADOPTED |
| 过门FARO | and | 0.3-0.8m | -0.17443911286465516 | 34 | 0 | ADOPTED |
| 过门FARO | hgb | 0.8-1.5m | +∞ | 0 | 0 | NOT_CALIBRATABLE |
| 过门FARO | logit | 0.8-1.5m | +∞ | 0 | 0 | NOT_CALIBRATABLE |
| 过门FARO | tof | 0.8-1.5m | +∞ | 0 | 0 | NOT_CALIBRATABLE |
| 过门FARO | rgb | 0.8-1.5m | +∞ | 0 | 0 | NOT_CALIBRATABLE |
| 过门FARO | or | 0.8-1.5m | +∞ | 0 | 0 | NOT_CALIBRATABLE |
| 过门FARO | and | 0.8-1.5m | +∞ | 0 | 0 | NOT_CALIBRATABLE |
| 过门FARO | hgb | 1.5-3m | +∞ | 0 | 0 | NOT_CALIBRATABLE |
| 过门FARO | logit | 1.5-3m | +∞ | 0 | 0 | NOT_CALIBRATABLE |
| 过门FARO | tof | 1.5-3m | +∞ | 0 | 0 | NOT_CALIBRATABLE |
| 过门FARO | rgb | 1.5-3m | +∞ | 0 | 0 | NOT_CALIBRATABLE |
| 过门FARO | or | 1.5-3m | +∞ | 0 | 0 | NOT_CALIBRATABLE |
| 过门FARO | and | 1.5-3m | +∞ | 0 | 0 | NOT_CALIBRATABLE |

完整ties、12个joblib权重及SHA见 `calibration/cal_seal.json`。eval仅打开一次，逐query缓存见 `eval/per_query.jsonl`，包含两来源各方法冻结分数、预测、参考状态和visit；不再用其选点。


## 使用边界

本轮是4个eval visit上的描述性query分类实验，非接触事件、行走提醒、安全或实机ToF证据。半循环扰动不消除native参考共源性；FARO门槛筛选后的子集与native全帧分母不同，不能直接横向当传感器优劣。所有bootstrap按原4个visit聚类2000次（seed20261011），零分母抽样不纳入对应区间并报告有效次数；区间不是大样本泛化保证。eval打开后只能复算封存表，不再修改权重或阈值。

## 核验、资源与产物

独立preOPEN核验通过：6份特征表的query唯一性、27格/K2聚合、固定FARO gate、visit隔离；1751个原始观测输入SHA、12份权重配方及train-only标准化、36格全ties切点复算。posteval仅读取不可变缓存，复算4509条query、22842个冻结判定、36格方法指标、48项配对及2000次四visit bootstrap，全部一致；权重与cal seal未变，未再次打开参考或模型前向。禁止源480/test未访问。

本轮无下载、GPU0、无GPU分配或保留进程。CPU command-wall预算1800s，分项与保守计费见最终收据；科学主命令为特征36.765s、train/cal11.105s、eval5.279s。sklearn仅有penalty参数未来弃用warning，无收敛失败，未因warning改变配方。

可审计的[精简收据](SYNC_FUSION_V1_RECEIPTS_DEV_20261011.json)保存全部权重/切点、指标、配对、独立核验及payload哈希。大载荷保留在本地canonical artifacts树，未将数据集图像或权重上传Git。主要路径：

- `features/{source}/{train,cal,eval}.npz`：20维观测输入与四基线分数；`feature_manifest.json`：逐文件SHA。
- `calibration/`：12个joblib、完整cal ties/分数、`cal_seal.json`；seeds取均值。
- `eval/per_query.jsonl`：4509条一次打开记录；`summary.json`、`metrics.csv`、`paired.csv`：全部统计与bootstrap有效次数。
- `primary_open_authorization.json`、`eval/eval_open.json`、`eval/eval_terminal.json`：唯一打开与完成收据；PLAN、执行代码snapshot及独立audit均保留。

PLAN SHA `6f5a790934ca2647d1a46f3b0d1ebe8db968823d1d3c2351102338b28b9c0593`；cal seal SHA `290a100338d7414bea7a67f4bb28a11d3a912fe37f25d450777a672ccbb31c93`；eval cache SHA `821091094048627c8e889324da042b2d9b4cc09d77b5e95c5d39ccaeddceed56`。
