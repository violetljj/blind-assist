# 前视障碍感知：当前状态

更新：2026-10-11。主线为盲杖互补前视感知，M3和A+LOCAL保留；原5格局部融合与L2保留。

RGB [分带混合评价](nearfield/RGB_BAND_HYBRID_DEV_20261010.md)完成六新Validation visit各16帧：Uni近带全局ARKcal仅用已消费240帧/2019严格FREE、全带5%封存cut0.0385859m，DAV中远沿用pooled304 cut0.2440383m。混合近W1→48/159、6/6capture改善，但FREE1→29/694（4.18%）超过预声明绝对2%（最多13），未采用混合，维持DAV2 Indoor Large raw R0、RGB≥0.8m/近带ToF。Uni pooled304漂移对照cut0.0961610为W27/159、FREE11/694（1.59%）；ARKcal放宽带来+21W/+18FREE，不据eval换主切点。中/远W208/634、401/606，FREE0/208、0/36，两个混合与DAV逐query全字段一致（3456次），属结构保留。双模型驻留689.15M参数，热态合计中位0.271s/P950.335s；仅部署参考，GPU已释放。独立native/校准/逐query复算PASS；新近带描述曲线也有交叉，仅Development与近POS覆盖富集，非实机/安全确认。旧[同FREE排序描述](nearfield/RGB_MATCHED_FREE_ARKIT_CAL_DEV_20261010.md)阶段2stop及所有旧失败/冻结对照保留。

用户预注册逻辑回归主融合的[同步确认v2](nearfield/SYNC_FUSION_CONFIRM_V2_DEV_20261011.md)已完成：协议cfe21852在新数据前提交，资源修订7674cf8a在追加下载/eval前提交；训练旧train6+已消费eval4，新cal6/eval12共576帧，cal36点/34021ties封存后一次OPEN。**query级融合在中带得到确认（真实RGB+半合成ToF）**：半循环logit中W1767/1923、F54/959=5.631%≤7.5%容许上限（未达严格5%），vs最佳ToF1382净385，12visit95%增益CI15.017–27.495pp；救407/损22，vsOR救267/损37净230。近带净136、CI3.439–23.013pp、F43/2139=2.010%次判据通过；远净368但FREE96/270=35.556%超15%，失败。FARO仅过门eval66帧/6贡献visit，中logit234/284 vsRGB177/284净57、F0/22，近52/303 vsRGB162净−110；RGB近中53.47%/62.32%未饱和，独立几何尚非全面确认。125/576 FARO过门、71缺poseUNKNOWN、4.110GB下载及GPU225.460s边界保留；模型/阈值eval后不改，默认S/M3/5格/RGB不升级App，下一真实ToF中带验证。独立12150query/72900决策/36表/24paired/2000bootstrap PASS，eval12已消费。

历史v1（已消费）：用户2026-10-11决定进入query级同步融合器阶段，主ToF使用扰动native（**半循环**），固定过门FARO作独立几何检验。[同步融合v1](nearfield/SYNC_FUSION_V1_DEV_20261011.md)已完成train6/cal2/eval4隔离、20维HGB主模型与logistic对照、cal近2%/中5%/远10%选点、封存后一次eval。**仅半循环来源支持，独立几何未复现**：native HGB近/中W278/371、497/542，F7/690、9/286，胜最佳单源净20/104；远W511/616但F7/30超标。FARO近W145/163、F8/151，不及RGB146/163、2/151；中远cal FREE=0不可校准。logistic同判读，不替换主模型；HGB近中净收益visit区间跨0。eval已消费，不据此改模型/阈值；下一先补独立几何cal与新visit来源，未升级默认候选。W为query POS命中，与旧像素交集W不可纵比。

[同步源v1.1](nearfield/SYNC_RGB_TOF_DATASET_V1_1_DEV_20261011.md)的80/80失败（103/384）及几何长尾保留；用户允许半循环小模型探索，不等于合格连续ToF源。FARO train/cal/eval仅57/7/39过门帧，native192/64/128；12权重/36切点封存、4509 eval query/22842判定/四visit bootstrap独立复算PASS。GPU0、无下载，预算及hash见报告。真实RGB+半合成ToF query证据，不是实机/接触/安全效果。

## ToF当前决定

用户2026-10-10决定：接受S集成的接触收益与轻提醒取舍，**S955/956/957任务成本HGB均值升为ToF模拟默认候选，未接入App**。冻结6e441f9a的6权重/47维，原5格强档逐slot不变，新增slot仅轻档。旧[确认](nearfield/CNH_S_ENSEMBLE_CONFIRM_DEV_20261010.md)净151/1024、a/d通过而b/c失败保留；用户改用取舍曲线与环境波动，不再按成本≤k×原5格二元判定。旧约2.7pp是通知次数归一，实际far+clear有通知clip增量2.214pp。

[池化取舍](nearfield/CNH_S_ENSEMBLE_TRADEOFF_DEV_20261010.md)完成：六组已消费pass/clear负例、14family、7168clip（far+clear5376），全部ties最低阈值tau1.917951703；同池原5格445→606有通知clip，增2.995pp≤3pp，contact不选点。封存后生成1536scene×K2/4新family，与30库存19252键及背景/目标参数无交集。

新hold原5格HEAD372/BODY308各/512；S救86/99、损0，净185/1024=18.07pp，v2加权成本211→325.75。far+clear有通知112→155/1536（7.292%→10.091%，+2.799pp）；family增量1.823–3.646pp。绝对漂移0.195pp，比旧2-family cal的1.042pp缩小；“两family造成漂移”记为用户解释，family数/选点口径/数据同时变动，未隔离因果。描述曲线1–3pp预算S收益高于E/both，零增量E优于S，无全域支配。

near-pass间隙≤10cm轻.25/强1，far/clear任何通知1；gap1逐query、同帧最高档联合计数，contact f3–13及时/f14–15晚。逐档计数、scene/family bootstrap及冻结候选manifest见报告；独立核验697900池ties、77322曲线、49152事件/13600通知PASS。GPU保守100/1200s、评价审计整合500/1200command-wall，权重/特征/保护源不动。

## 继承与边界

原M3/5格保留为对照和强档来源；旧E2E、重训/确认失败与各run停止规则保留。池化已消费负例按本任务授权复用，hold曲线仅描述不选点。有限AABB模拟Development、4family区间仅描述，非实机/安全证明；共向几何不替代外参/时钟核验，方向/PDR、硬件阶段未启动，同步v2中带query确认仅限半循环来源；FARO中带同向仅描述、近带负增益，未构成全面独立确认。后续台架/App由候选manifest提供接口，本任务未接入。

本次更新前全文见Git `c0fd6849`及 sync-fusion-confirm-v2-dev-20261011/ 的 `integration_before/`；历史见RUNS/报告。
ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED
