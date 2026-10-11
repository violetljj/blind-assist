# 前视障碍感知：当前状态

更新：2026-10-11。主线为盲杖互补前视感知，M3和A+LOCAL保留；原5格局部融合与L2保留。

个人回顾：[10-09至10-11阶段总结](STAGE_SUMMARY_20261011.md)（含v4排序确认、告警取舍与sim→real优先方向）。

RGB [分带混合评价](nearfield/RGB_BAND_HYBRID_DEV_20261010.md)完成六新Validation visit各16帧：Uni近带全局ARKcal仅用已消费240帧/2019严格FREE、全带5%封存cut0.0385859m，DAV中远沿用pooled304 cut0.2440383m。混合近W1→48/159、6/6capture改善，但FREE1→29/694（4.18%）超过预声明绝对2%（最多13），未采用混合，维持DAV2 Indoor Large raw R0、RGB≥0.8m/近带ToF。Uni pooled304漂移对照cut0.0961610为W27/159、FREE11/694（1.59%）；ARKcal放宽带来+21W/+18FREE，不据eval换主切点。中/远W208/634、401/606，FREE0/208、0/36，两个混合与DAV逐query全字段一致（3456次），属结构保留。双模型驻留689.15M参数，热态合计中位0.271s/P950.335s；仅部署参考，GPU已释放。独立native/校准/逐query复算PASS；新近带描述曲线也有交叉，仅Development与近POS覆盖富集，非实机/安全确认。旧[同FREE排序描述](nearfield/RGB_MATCHED_FREE_ARKIT_CAL_DEV_20261010.md)阶段2stop及所有旧失败/冻结对照保留。

[融合确认v4](nearfield/SYNC_FUSION_CONFIRM_V4_DEV_20261011.md)完成预注册，协议`270abc0a`在新数据与交叉拟合前提交并推送；判据明确取自v3同FREE事后分析`7dc9bbba`，原v3 FAIL及数字保留。冻结v3 D′三权重，旧30visit五折15临时模型只生成cal折外分数，加新cal4；六臂混合池15切点/209957ties先封存，新eval16一次OPEN。**主中带低误报排序PASS**：归一化pAUC D′0.76941、RGB0.45021、OR0.56983，D′−RGB+31.920pp CI[16.996,44.803]、D′−OR+19.958pp [5.494,38.464]，两项2000/2000 visit重抽样有效。部署中W/F：D′2538/48、RGB1810/38、OR2178/26（POS2873/FREE866），对RGB/OR未双向占优；半循环cal→eval F率D′4.458→5.543%、RGB5→4.388%，同v4eval上原RGB切点10.970%。OOF中带有限FREE分数比冻结训练内均值高0.04237，不做映射。近带区间跨0，远带及ambient×10中带仅描述；FARO仅92过门帧，中/远FREE14/1。协议/OOF/cal/数据/2000曲线bootstrap/部署漂移三阶段独立核验PASS。CPU保守2034.948/2400command-wall、GPU263.192/600s、下载3488869655B/4GB；特征机械续跑失败和一致性核验保留，总额度不扩展。结论限真实RGB＋半合成ToF新visit Development；不做仲裁器，不接入App/M3/5格，不读保护480/test。

[融合确认v3](nearfield/SYNC_FUSION_CONFIRM_V3_DEV_20261011.md)完成预注册：协议`ec252d68`在建新数据前提交并推送，实现`95f451e6`在OPEN前提交；旧30visit一次训练、新cal6/eval12共576帧，D′六臂等权HGB/B′混合LR/(b)半循环LR均去显式query位置与带别特征，加最低传感器证据OR门控。21阈值/全部ties封存并独立复算后eval一次OPEN。**主中带确认未通过**。半循环中D′ W1696/2077，vs最佳单源rgb净+87，救213/损126，12visit聚类95%CI-0.308–9.155pp，FREE4/689=0.581%；主FAIL。同口径：D_prime净+87、CI-0.308–9.155pp、FREE4/689=0.581%（FAIL）；B_prime净+86、CI-0.519–9.557pp、FREE8/689=1.161%（FAIL）；audit_b净-91、CI-11.354–2.535pp、FREE0/689=0.000%（FAIL）。次判据：faro_rho030_ambient1近带vsRGB净-2、FREE15/105=14.286%，FAIL；faro_rho030_ambient10近带vsRGB净-58、FREE121/1702=7.109%，FAIL。主中带收益按最低证据拆为both净-16、tof_only净+209、rgb_only净-26、neither净-80；全6臂/3带/3模型无证据支持0，属条件分解非因果。FARO仅eval43过门帧，完整贡献visit见报告，远带只报告、不作主张。9权重/hash、六臂manifest、逐query/四类拆分/2000次bootstrap、四阶段独立核验PASS。下载3,889,213,192B/4GB、GPU211.577/600s、CPU保守2209.665/2400 command-wall s，任务进程释放；18新visit已消费，eval后未改模型或阈值，不读保护480/test。仅真实RGB＋半合成ToF Development，不升级App/M3/5格或作实机/安全结论。

历史v2/事后审计/稳健性CV（均已消费）保留：[v2中带净385](nearfield/SYNC_FUSION_CONFIRM_V2_DEV_20261011.md)、[去位置＋门控(b)事后净282](nearfield/SYNC_FUSION_EVIDENCE_AUDIT_DEV_20261011.md)、[含位置六臂HGB探索](nearfield/SYNC_FUSION_ROBUST_DEV_20261011.md)。原结果及失败未回溯改写；本次v3按新visit预注册评价上述组合。

历史v1（已消费）：用户2026-10-11决定进入query级同步融合器阶段，主ToF使用扰动native（**半循环**），固定过门FARO作独立几何检验。[同步融合v1](nearfield/SYNC_FUSION_V1_DEV_20261011.md)已完成train6/cal2/eval4隔离、20维HGB主模型与logistic对照、cal近2%/中5%/远10%选点、封存后一次eval。**仅半循环来源支持，独立几何未复现**：native HGB近/中W278/371、497/542，F7/690、9/286，胜最佳单源净20/104；远W511/616但F7/30超标。FARO近W145/163、F8/151，不及RGB146/163、2/151；中远cal FREE=0不可校准。logistic同判读，不替换主模型；HGB近中净收益visit区间跨0。eval已消费，不据此改模型/阈值；下一先补独立几何cal与新visit来源，未升级默认候选。W为query POS命中，与旧像素交集W不可纵比。

[同步源v1.1](nearfield/SYNC_RGB_TOF_DATASET_V1_1_DEV_20261011.md)的80/80失败（103/384）及几何长尾保留；用户允许半循环小模型探索，不等于合格连续ToF源。FARO train/cal/eval仅57/7/39过门帧，native192/64/128；12权重/36切点封存、4509 eval query/22842判定/四visit bootstrap独立复算PASS。GPU0、无下载，预算及hash见报告。真实RGB+半合成ToF query证据，不是实机/接触/安全效果。

[远带FREE原因诊断](nearfield/FAR_BAND_FREE_DIAGNOSIS_DEV_20261011.md)完成EXPLORE：v2已消费eval与v1.1全部split，a/b/c各176query；a标签可疑116/176（65.9%），b130/176（73.9%），不支持错标共同主因。a几何风险111/176、双源空参考下DAV侵入33/176；v2 a中71/111由logit支持而DAV区间内不足16点，优先查融合支持语义，再查RGB远带深度。confidence239/239补齐、176抽查图、528query独立复算PASS；GPU0、CPU保守400/1500command-wall。未来参考质量合同仅草案，不改旧标签/阈值/模型或远带失败结论，不读保护480/test。

## ToF当前决定

[宿舍孪生全报警判别](../hardware-bringup/DORM_TWIN_DISCRIMINATION_DEV_20261011.md)EXPLORE完成，**窗外尽头时倾向A，近侧杂物单独不足以解释全报；实际几何未测，真实A/B不可确认**。名义幕3.2m/高1.55m，合理−15…0°原5格/S空HEAD/BODY0/108，g20–55cm基本静默；near原全强由幕2.8m进入3m窗驱动，单移回3.2m降至HEAD1/108/BODY0，名义只移幕2.8m则216/216全强。99主cell＋8输出后声明组件诊断，不回改主几何/判据。拼合近bin254/256，v3只换串扰分子；真实4/4b/7仍全强，9 BODY89→24/89，空762→697/762但杆5/6强584→60/584，不采用v3默认。差异含bin0与远幕10/11，单最早bin不足解释组合全强。下一先核实际姿态/raw映射/尽头距离，再补signed仪器与噪声合同、核/gain及窄通道窗内外分布，骨干重训后拟合S仅草案；不先单独微调S。独立7196845项＋CUDA六窗PASS/旧−10°兼容，CPU≤300/1500、GPU≤100/600command-wall s；无重训/模型阈值修改/480访问，几何假设与模板过减局限见报告。

[宿舍十条件实机pilot校准回放](../hardware-bringup/REAL_PILOT_CALIBRATION_REPLAY_DEV_20261011.md)完成EXPLORE，**仍不可迁移**：1507帧/1362满窗，v1继承06d49473与v2在空通道4/4b/9的原5格/S均762/762 query-frames强报警，外侧7也全强；墙/杆/8HEAD报警，无S新增light。用户授权宽松稳定规则，strict失败保留；4/4b观测Var=.145814·positive_mean+.089665·ambient+.081291、R²=.533699，阳台上半弱模板128/1024cell；2fit gain=.749185、核σ=.404742bin，3check能量L2残差11.16%/核逐bin109.98%。旧zone34两bin非所有场景共同异常。仅1–4b拟合、5–9评价，3不回流；模型/阈值/名义−10°/恒等历史保持冻结。独立raw/参数/计数与同CUDAfresh forward PASS，CPU跨后端S分数差留证。CPU保守300/1500、GPU65/300command-wall s；下一全空间bias/单位噪声→核零偏→固定模式/gain，更新模拟器重训骨干再拟合S仅草案；不读480/test、不作检出/误报或安全主张，不升候选/App。

[sim→real输入合同第一步](../hardware-bringup/SIM2REAL_INPUT_CONTRACT_DEV_20261011.md)完成已有真实CNH EXPLORE：13片段12922帧/12726满窗，原M3、5格、S HEAD/BODY为61.88/15.33%、84.83/52.34%、84.86/52.85%；仅减far-selected bias并保留表面的新拟合floor后三模型100%/100%。bias340/1024格覆盖、噪声R²=.0024及ambient系数0边界不支持物理映射成功；新BODY2586无固件弱表面支撑。旧5格04 HEAD144中133对既定两格敏感，但其有柜面支撑；新路径消融0取消不能当FPN消失。独立raw/参数/三类闭合与真实forward核验PASS，CPU保守≤400/1800command-wall、GPU≤150/300s，权重释放。下一先建模拟器噪声/固件处理与空间串扰模板；重训/六条件标注录制仅草案。M3/原5格/S模拟默认保留，无新采集/刷写/调阈值/模型或App修改，不读480/test；只是真实回放描述，不作检出/误报或安全结论。

用户2026-10-10决定：接受S集成的接触收益与轻提醒取舍，**S955/956/957任务成本HGB均值升为ToF模拟默认候选，未接入App**。冻结6e441f9a的6权重/47维，原5格强档逐slot不变，新增slot仅轻档。旧[确认](nearfield/CNH_S_ENSEMBLE_CONFIRM_DEV_20261010.md)净151/1024、a/d通过而b/c失败保留；用户改用取舍曲线与环境波动，不再按成本≤k×原5格二元判定。旧约2.7pp是通知次数归一，实际far+clear有通知clip增量2.214pp。

[池化取舍](nearfield/CNH_S_ENSEMBLE_TRADEOFF_DEV_20261010.md)完成：六组已消费pass/clear负例、14family、7168clip（far+clear5376），全部ties最低阈值tau1.917951703；同池原5格445→606有通知clip，增2.995pp≤3pp，contact不选点。封存后生成1536scene×K2/4新family，与30库存19252键及背景/目标参数无交集。

新hold原5格HEAD372/BODY308各/512；S救86/99、损0，净185/1024=18.07pp，v2加权成本211→325.75。far+clear有通知112→155/1536（7.292%→10.091%，+2.799pp）；family增量1.823–3.646pp。绝对漂移0.195pp，比旧2-family cal的1.042pp缩小；“两family造成漂移”记为用户解释，family数/选点口径/数据同时变动，未隔离因果。描述曲线1–3pp预算S收益高于E/both，零增量E优于S，无全域支配。

near-pass间隙≤10cm轻.25/强1，far/clear任何通知1；gap1逐query、同帧最高档联合计数，contact f3–13及时/f14–15晚。逐档计数、scene/family bootstrap及冻结候选manifest见报告；独立核验697900池ties、77322曲线、49152事件/13600通知PASS。GPU保守100/1200s、评价审计整合500/1200command-wall，权重/特征/保护源不动。

## 继承与边界

原M3/5格保留为对照和强档来源；旧E2E、重训/确认失败与各run停止规则保留。池化已消费负例按本任务授权复用，hold曲线仅描述不选点。有限AABB模拟Development、4family区间仅描述，非实机/安全证明；共向几何不替代外参/时钟核验，方向/PDR、硬件阶段未启动，融合v4当前判读以本页新visit低误报排序预注册结果为准；v3原失败保留，近/远带与稀疏FARO/压力臂仅描述，未构成全面独立实机确认。后续台架/App由候选manifest提供接口，本任务未接入。

本次v3整合前全文见Git `95f451e6`及 sync-fusion-confirm-v3-dev-20261011/ 的 `CURRENT*_integration_before.md`；历史见RUNS/报告。
ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED
