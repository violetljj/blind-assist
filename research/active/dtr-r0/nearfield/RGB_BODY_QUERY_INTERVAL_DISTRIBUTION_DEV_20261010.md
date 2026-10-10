# RGB距离分布与解析查询读出

完成与query无关的距离分布模型和后验点深度诊断。新3RScan有局部增量，但追加ARKit的采样空闲误支持仍明显高于几何基线；解析查询一致性不能解决迁移效果，现有核查也不支持把退化整体归为σ／CDF。下一优先补query级校准负参考及跨相机点深度诊断。

2026-10-10，EXPLORE / COMPLETE_LOCAL_INCREMENT_TRANSFER_GAP。承接[查询外推与负参考](RGB_BODY_QUERY_QUERY_LEVEL_DEV_20261009.md)，用户“继续”授权改变距离／不确定性表征与查询覆盖。旧query head、32通道特征臂及旧校正／阈值保持，不续训这些配方。

本轮先预测与查询无关的log-Z分布，再按公共K与盒边界得到射线进入／离开区间，读出CDF之差。网络不接受query边界；扩大或拆分区间时，分布读出的包含单调与概率质量加法是结构性质，不是检测提升的证据。单Gaussian是否能表达当前RGB距离误差，是本轮最弱假设。分布分数并未认证为校准后的碰撞概率。

预算一次记录于payload `artifacts.local/work/rgb-body-query-interval-distribution-dev-20261010/plan.json`：GPU阶段壁时900s（候选700／保留200），CPU辅助1500s（准备300、候选400、相关核验300、主整合500），下载／新模型下载0B。复用现有真实Development缓存，原train56帧／7环境、cal16帧／2环境与136帧评价集不变，无新Depth Pro骨干推理、采集、ToF或步行实验。

## 预定模型与比较

- `affine_gaussian`：原train拟合的log-affine均值，加原train残差RMS作为全局sigma。
- `depth_ray_gaussian`：输入预测log-Z和公共ray方向，学习log-Z均值及sigma；固定seed7、600步、8192 batch、Adam .002，Gaussian NLL，sigma限定.03–2 log单位，固定最终checkpoint。
- `geometry_gaussian`：相同网络、初始权重／train mean常数base和训练采样，深度输入置零，以检查距离信息相对几何先验的增量。
- `affine_margin_gridcal`：原train log-affine深度直接几何；margin=min(Z−entry,exit−Z)，与前三臂使用相同原cal27查询选择阈值。该强对照在新结果前补入计划，未扩预算。

新臂只在原cal16帧的27查询free query-ray分数上选工作点：实际FPR不高于目标.14190356447903094，选择最接近的可达工作点，处理ties并记录实际cutoff，选值不看召回。几何margin阈值单位米，delta=−cutoff；CDF阈值为分数，目标FPR数字不能当阈值。旧8臂工作点来自原15query cal，作为保存的实用对照；它们与新27cal臂之间不作孤立机制因果比较。学习距离分布的训练目标也不同于旧query BCE。

决策看全部5cohort／环境／距离带的射线召回与free误支持、正query已知TP≥16见证、采样空闲总支持≥16、UNKNOWN支持及配对救回／损失。575个sampledFREE仍相关，只有有限首回波采样参考；UNKNOWN不计为free，包含／加法检查通过也不能替代性能结果。全量报告，不在评价集重选阈值、步数或赢家。

## 准备

72帧1944个原train/cal固定27query全部保留，native数组／K／SHA及观测字段隔离通过；1944格独立XYZ核验一致，train米制配对点仍1591791。train POS764／UNKNOWN746／sampledFREE2，cal POS185／UNKNOWN245／sampledFREE2；校准使用free射线而非只有两格的query计数。Depth Pro72帧仅REUSED、completed_calls0。准备实测14.622s，辅助CPU保守60/300s，GPU／下载0。

## 结果与下一步


## 主实验：预定4臂全量结果

| cohort / arm | 射线召回 / free误支持 | 正见证 | sampledFREE支持 | UNKNOWN支持 |
| --- | --- | --- | --- | --- |
| original_validation24 / affine_gaussian | 0.681079 / 0.131146 | 244/295 | 2/2 | 122/351 |
| original_validation24 / depth_ray_gaussian | 0.746321 / 0.144220 | 258/295 | 2/2 | 145/351 |
| original_validation24 / geometry_gaussian | 0.466850 / 0.131524 | 231/295 | 2/2 | 146/351 |
| original_validation24 / affine_margin_gridcal | 0.618462 / 0.132913 | 213/295 | 1/2 | 79/351 |
| new_3rscan64 / affine_gaussian | 0.601987 / 0.107742 | 656/866 | 1/1 | 367/861 |
| new_3rscan64 / depth_ray_gaussian | 0.666588 / 0.111096 | 701/866 | 1/1 | 397/861 |
| new_3rscan64 / geometry_gaussian | 0.434730 / 0.106711 | 610/866 | 1/1 | 412/861 |
| new_3rscan64 / affine_margin_gridcal | 0.554374 / 0.097548 | 555/866 | 1/1 | 207/861 |
| arkit16 / affine_gaussian | 0.320476 / 0.063672 | 142/228 | 9/105 | 77/99 |
| arkit16 / depth_ray_gaussian | 0.384745 / 0.043060 | 156/228 | 9/105 | 94/99 |
| arkit16 / geometry_gaussian | 0.462782 / 0.076572 | 152/228 | 9/105 | 88/99 |
| arkit16 / affine_margin_gridcal | 0.195219 / 0.019297 | 90/228 | 3/105 | 74/99 |
| arkit_40777060 / affine_gaussian | 0.424992 / 0.024444 | 123/202 | 13/220 | 9/10 |
| arkit_40777060 / depth_ray_gaussian | 0.698398 / 0.164181 | 176/202 | 78/220 | 9/10 |
| arkit_40777060 / geometry_gaussian | 0.395907 / 0.161094 | 166/202 | 87/220 | 9/10 |
| arkit_40777060 / affine_margin_gridcal | 0.433257 / 0.025953 | 119/202 | 12/220 | 9/10 |
| arkit_40777065 / affine_gaussian | 0.517586 / 0.040174 | 134/179 | 21/247 | 4/6 |
| arkit_40777065 / depth_ray_gaussian | 0.765814 / 0.177794 | 170/179 | 89/247 | 6/6 |
| arkit_40777065 / geometry_gaussian | 0.333708 / 0.172929 | 156/179 | 103/247 | 6/6 |
| arkit_40777065 / affine_margin_gridcal | 0.502989 / 0.036901 | 131/179 | 10/247 | 3/6 |

arkit16：depth_ray_gaussian对affine_margin_gridcal正query见证救/损72/6。

arkit16：depth_ray_gaussian对geometry_gaussian正query见证救/损19/15。

arkit_40777060：depth_ray_gaussian对affine_margin_gridcal正query见证救/损58/1。

arkit_40777060：depth_ray_gaussian对geometry_gaussian正query见证救/损19/9。

arkit_40777065：depth_ray_gaussian对affine_margin_gridcal正query见证救/损41/2。

arkit_40777065：depth_ray_gaussian对geometry_gaussian正query见证救/损21/7。

new_3rscan64：depth_ray_gaussian对affine_margin_gridcal正query见证救/损154/8。

new_3rscan64：depth_ray_gaussian对geometry_gaussian正query见证救/损110/19。

original_validation24：depth_ray_gaussian对affine_margin_gridcal正query见证救/损49/4。

original_validation24：depth_ray_gaussian对geometry_gaussian正query见证救/损33/6。

主实验说明：新3RScan的depth-ray分布已知正见证701/866，对匹配27cal几何555/866，救/损154/8；射线召回.666588对.554374，同时free误支持.111096对.097548。相对无深度geometry分布，见证救/损110/19，射线召回.666588对.434730，但free误支持.111096对.106711。这是本组Development上的场景距离信息增量与取舍，不能说同迁移FPR获胜。原val也增加见证；旧ARKit的depth-ray召回低于geometry分布，free误支持较低，仍有工作点取舍。

两追加ARKit capture的learned CDF误支持明显增多：depth-ray free78/220、89/247，geometry分布87/220、103/247；匹配27cal几何12/220、10/247。两learned臂均出现该现象，不能只凭结果归因为sigma、相机、布局、Depth Pro或读出任何一项。全局affine Gaussian较保守，但也没有稳定胜过直接几何；所有四臂和所有切片保留，不在评价集改阈值。

在看到上述取舍后，单独增补后验点深度几何诊断：复用两trained arms的mu与原checkpoint，exp(mu)作为点深度做直接几何margin（是模型Z中位数，非Z算术期望），在相同原cal27选一次保守阈值后冻结到全部136帧。没有再训练、重选原CDF阈值或回改主结果。`mean_probe_amendment.json`与执行source单独留存，这不是预注册主实验或独立确认；同一已学log-Z均值下，读出及其cal阈值的变化可以缩小解释范围，不能单独确定迁移失败原因。


## 后验点深度几何读出：诊断结果

| cohort / arm | 射线召回 / free误支持 | 正见证 | sampledFREE支持 | UNKNOWN支持 |
| --- | --- | --- | --- | --- |
| original_validation24 / depth_ray_mean_margin_gridcal | 0.658492 / 0.139855 | 250/295 | 1/2 | 120/351 |
| original_validation24 / geometry_mean_margin_gridcal | 0.384139 / 0.125918 | 131/295 | 2/2 | 77/351 |
| new_3rscan64 / depth_ray_mean_margin_gridcal | 0.592645 / 0.100255 | 675/866 | 1/1 | 342/861 |
| new_3rscan64 / geometry_mean_margin_gridcal | 0.374711 / 0.102239 | 410/866 | 0/1 | 148/861 |
| arkit16 / depth_ray_mean_margin_gridcal | 0.292958 / 0.030588 | 135/228 | 7/105 | 90/99 |
| arkit16 / geometry_mean_margin_gridcal | 0.448381 / 0.079457 | 113/228 | 9/105 | 21/99 |
| arkit_40777060 / depth_ray_mean_margin_gridcal | 0.646562 / 0.158090 | 171/202 | 73/220 | 9/10 |
| arkit_40777060 / geometry_mean_margin_gridcal | 0.266086 / 0.167875 | 65/202 | 75/220 | 0/10 |
| arkit_40777065 / depth_ray_mean_margin_gridcal | 0.719275 / 0.169717 | 165/179 | 79/247 | 6/6 |
| arkit_40777065 / geometry_mean_margin_gridcal | 0.187420 / 0.178647 | 47/179 | 93/247 | 3/6 |

新增两臂解析已有log-Z均值，不做模型forward：new3RScan点深度见证675/866、召回/free误支持.592645/.100255，geometry点读出410/866、.374711/.102239，匹配27cal affine几何555/866、.554374/.097548。局部距离信息增量仍在，但与强几何的迁移工作点不相同。两追加ARKit点读出free73/220、79/247，仍明显多于affine12/220、10/247；后验读法仅减少5个／10个支持，且重新进行了原cal校准，不能把σ／CDF确定为唯一或主要成因。

## 校准、结论与成本

校准free分母1243275、positive351645个相关query-ray。主4臂实际cutoff：affine CDF .285475670832019、depth-ray CDF .29388670906145503、geometry CDF .3573770101518666、affine margin −.0575793021042712m（δ+.0575793021042712m）。前两CDF／margin FP176425，FPR .1419034405099435；geometry因4同分tie取FP176423、.14190183185538197，下一tie会超预算。后验depth点margin cutoff−.06623158347169666m，FP176425；geometry点margin cutoff+.09735384385732426m，FP176424、FPR .14190263618266272。均是保守最接近可达工作点，没有精确达到目标，不能把目标数字当cutoff。

原cal虽有124万free query-ray，却只有2个严格sampledFREE子盒；射线工作点不能代替query级误报控制，更不能代替步行事件。575个评价sampledFREE、1327个UNKNOWN和1770个POS仍相关、有限采样，完整身体／细障碍／步行／第三硬件证据未增加。新颖性本轮未做文献核查。

当前决定：保留query-independent距离表征及3RScan局部增量，本Gaussian和点深度配方均不升级成稳定跨相机候选。下一优先补足可用于query级校准的负参考，并在跨相机数据上分开核点深度迁移、参考覆盖与读出工作点；现有结果不把问题归于一个硬件、尺度或σ因素。结构一致性可用，但单独不是贡献证明；旧失败／阈值／checkpoint和真实评价硬目标保留。

核验：主实验14688新记录、51408配对、304summary、1064聚合、408真实网格partition／包含检查，以及train56/1591791实际mask、train-only归一化、两个600checkpoint与640点CPU前向spot通过。后验仅新增7344记录、29376新相关pair、152summary及608聚合核验，没有重复旧8臂计数。主审末尾outer-plan SHA guard因临时并入后验字段触发：现场保留；将增补移至独立amendment后，主plan原byte SHA精确恢复，局部身份核验完成而不重跑已通过数值。主审精确wall收尾未写入，明确记gap，不补造计时；相关审计保守250/300s，后验实测30.399s。

候选GPU阶段壁时40.538/900s（含训练及152帧public分布预测、两模型清理，非部署时间），新Depth Pro推理0；主辅助CPU59.012s、后验28.262s、准备14.622s。全run辅助CPU保守660/1500s：准备60、候选及后验150、相关审计250、主摘要/检查/交付200。下载／新模型下载0B；两个新训练checkpoint是本地计算产物，不冒充0训练。全部任务资源释放，模型／分布／来源／校准／全量环境带和pair CSV／失败现场持久保留，owner本run。总`completion_receipt.json`汇集必要输入与交付身份；预算未耗尽，当前诊断已足以改变下一步，结束本轮而不追加调参。

代码：[距离分布与解析query](rgb_body_query_interval_distribution.py)、[后验点深度读出](rgb_body_query_distribution_mean_probe.py)。交付检查为上述真实执行和新增行为复算、scoped diff及docs index；未跑无关Android检查。热文档只替换RGB段，原CNH路线保留。
