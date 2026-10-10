# 前视障碍感知：当前状态

更新：2026-10-11。主线为盲杖互补前视感知，M3和A+LOCAL保留；原5格局部融合与L2保留。

RGB [分带混合评价](nearfield/RGB_BAND_HYBRID_DEV_20261010.md)完成六新Validation visit各16帧：Uni近带全局ARKcal仅用已消费240帧/2019严格FREE、全带5%封存cut0.0385859m，DAV中远沿用pooled304 cut0.2440383m。混合近W1→48/159、6/6capture改善，但FREE1→29/694（4.18%）超过预声明绝对2%（最多13），未采用混合，维持DAV2 Indoor Large raw R0、RGB≥0.8m/近带ToF。Uni pooled304漂移对照cut0.0961610为W27/159、FREE11/694（1.59%）；ARKcal放宽带来+21W/+18FREE，不据eval换主切点。中/远W208/634、401/606，FREE0/208、0/36，两个混合与DAV逐query全字段一致（3456次），属结构保留。双模型驻留689.15M参数，热态合计中位0.271s/P950.335s；仅部署参考，GPU已释放。独立native/校准/逐query复算PASS；新近带描述曲线也有交叉，仅Development与近POS覆盖富集，非实机/安全确认。旧[同FREE排序描述](nearfield/RGB_MATCHED_FREE_ARKIT_CAL_DEV_20261010.md)阶段2stop及所有旧失败/冻结对照保留。

[融合确认v3](nearfield/SYNC_FUSION_CONFIRM_V3_DEV_20261011.md)完成预注册：协议`ec252d68`在建新数据前提交并推送，实现`95f451e6`在OPEN前提交；旧30visit一次训练、新cal6/eval12共576帧，D′六臂等权HGB/B′混合LR/(b)半循环LR均去显式query位置与带别特征，加最低传感器证据OR门控。21阈值/全部ties封存并独立复算后eval一次OPEN。**主中带确认未通过**。半循环中D′ W1696/2077，vs最佳单源rgb净+87，救213/损126，12visit聚类95%CI-0.308–9.155pp，FREE4/689=0.581%；主FAIL。同口径：D_prime净+87、CI-0.308–9.155pp、FREE4/689=0.581%（FAIL）；B_prime净+86、CI-0.519–9.557pp、FREE8/689=1.161%（FAIL）；audit_b净-91、CI-11.354–2.535pp、FREE0/689=0.000%（FAIL）。次判据：faro_rho030_ambient1近带vsRGB净-2、FREE15/105=14.286%，FAIL；faro_rho030_ambient10近带vsRGB净-58、FREE121/1702=7.109%，FAIL。主中带收益按最低证据拆为both净-16、tof_only净+209、rgb_only净-26、neither净-80；全6臂/3带/3模型无证据支持0，属条件分解非因果。FARO仅eval43过门帧，完整贡献visit见报告，远带只报告、不作主张。9权重/hash、六臂manifest、逐query/四类拆分/2000次bootstrap、四阶段独立核验PASS。下载3,889,213,192B/4GB、GPU211.577/600s、CPU保守2209.665/2400 command-wall s，任务进程释放；18新visit已消费，eval后未改模型或阈值，不读保护480/test。仅真实RGB＋半合成ToF Development，不升级App/M3/5格或作实机/安全结论。

历史v2/事后审计/稳健性CV（均已消费）保留：[v2中带净385](nearfield/SYNC_FUSION_CONFIRM_V2_DEV_20261011.md)、[去位置＋门控(b)事后净282](nearfield/SYNC_FUSION_EVIDENCE_AUDIT_DEV_20261011.md)、[含位置六臂HGB探索](nearfield/SYNC_FUSION_ROBUST_DEV_20261011.md)。原结果及失败未回溯改写；本次v3按新visit预注册评价上述组合。

历史v1（已消费）：用户2026-10-11决定进入query级同步融合器阶段，主ToF使用扰动native（**半循环**），固定过门FARO作独立几何检验。[同步融合v1](nearfield/SYNC_FUSION_V1_DEV_20261011.md)已完成train6/cal2/eval4隔离、20维HGB主模型与logistic对照、cal近2%/中5%/远10%选点、封存后一次eval。**仅半循环来源支持，独立几何未复现**：native HGB近/中W278/371、497/542，F7/690、9/286，胜最佳单源净20/104；远W511/616但F7/30超标。FARO近W145/163、F8/151，不及RGB146/163、2/151；中远cal FREE=0不可校准。logistic同判读，不替换主模型；HGB近中净收益visit区间跨0。eval已消费，不据此改模型/阈值；下一先补独立几何cal与新visit来源，未升级默认候选。W为query POS命中，与旧像素交集W不可纵比。

[同步源v1.1](nearfield/SYNC_RGB_TOF_DATASET_V1_1_DEV_20261011.md)的80/80失败（103/384）及几何长尾保留；用户允许半循环小模型探索，不等于合格连续ToF源。FARO train/cal/eval仅57/7/39过门帧，native192/64/128；12权重/36切点封存、4509 eval query/22842判定/四visit bootstrap独立复算PASS。GPU0、无下载，预算及hash见报告。真实RGB+半合成ToF query证据，不是实机/接触/安全效果。

[远带FREE原因诊断](nearfield/FAR_BAND_FREE_DIAGNOSIS_DEV_20261011.md)完成EXPLORE：v2已消费eval与v1.1全部split，a/b/c各176query；a标签可疑116/176（65.9%），b130/176（73.9%），不支持错标共同主因。a几何风险111/176、双源空参考下DAV侵入33/176；v2 a中71/111由logit支持而DAV区间内不足16点，优先查融合支持语义，再查RGB远带深度。confidence239/239补齐、176抽查图、528query独立复算PASS；GPU0、CPU保守400/1500command-wall。未来参考质量合同仅草案，不改旧标签/阈值/模型或远带失败结论，不读保护480/test。

## ToF当前决定

用户2026-10-10决定：接受S集成的接触收益与轻提醒取舍，**S955/956/957任务成本HGB均值升为ToF模拟默认候选，未接入App**。冻结6e441f9a的6权重/47维，原5格强档逐slot不变，新增slot仅轻档。旧[确认](nearfield/CNH_S_ENSEMBLE_CONFIRM_DEV_20261010.md)净151/1024、a/d通过而b/c失败保留；用户改用取舍曲线与环境波动，不再按成本≤k×原5格二元判定。旧约2.7pp是通知次数归一，实际far+clear有通知clip增量2.214pp。

[池化取舍](nearfield/CNH_S_ENSEMBLE_TRADEOFF_DEV_20261010.md)完成：六组已消费pass/clear负例、14family、7168clip（far+clear5376），全部ties最低阈值tau1.917951703；同池原5格445→606有通知clip，增2.995pp≤3pp，contact不选点。封存后生成1536scene×K2/4新family，与30库存19252键及背景/目标参数无交集。

新hold原5格HEAD372/BODY308各/512；S救86/99、损0，净185/1024=18.07pp，v2加权成本211→325.75。far+clear有通知112→155/1536（7.292%→10.091%，+2.799pp）；family增量1.823–3.646pp。绝对漂移0.195pp，比旧2-family cal的1.042pp缩小；“两family造成漂移”记为用户解释，family数/选点口径/数据同时变动，未隔离因果。描述曲线1–3pp预算S收益高于E/both，零增量E优于S，无全域支配。

near-pass间隙≤10cm轻.25/强1，far/clear任何通知1；gap1逐query、同帧最高档联合计数，contact f3–13及时/f14–15晚。逐档计数、scene/family bootstrap及冻结候选manifest见报告；独立核验697900池ties、77322曲线、49152事件/13600通知PASS。GPU保守100/1200s、评价审计整合500/1200command-wall，权重/特征/保护源不动。

## 继承与边界

原M3/5格保留为对照和强档来源；旧E2E、重训/确认失败与各run停止规则保留。池化已消费负例按本任务授权复用，hold曲线仅描述不选点。有限AABB模拟Development、4family区间仅描述，非实机/安全证明；共向几何不替代外参/时钟核验，方向/PDR、硬件阶段未启动，融合v3当前判读以本页新visit预注册结果为准；半循环主中带与稀疏FARO/压力近带次判据分别报告，远带仅描述，未构成全面独立实机确认。后续台架/App由候选manifest提供接口，本任务未接入。

本次v3整合前全文见Git `95f451e6`及 sync-fusion-confirm-v3-dev-20261011/ 的 `CURRENT*_integration_before.md`；历史见RUNS/报告。
ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED
