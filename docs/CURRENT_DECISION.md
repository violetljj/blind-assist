# 当前研究决定

更新：2026-10-11。主线为盲杖互补前视感知，M3和A+LOCAL保留；原5格局部融合与L2保留。

RGB [分带混合评价](../research/active/dtr-r0/nearfield/RGB_BAND_HYBRID_DEV_20261010.md)完成六新Validation visit各16帧：Uni近带全局ARKcal仅用已消费240帧/2019严格FREE、全带5%封存cut0.0385859m，DAV中远沿用pooled304 cut0.2440383m。混合近W1→48/159、6/6capture改善，但FREE1→29/694（4.18%）超过预声明绝对2%（最多13），未采用混合，维持DAV2 Indoor Large raw R0、RGB≥0.8m/近带ToF。Uni pooled304漂移对照cut0.0961610为W27/159、FREE11/694（1.59%）；ARKcal放宽带来+21W/+18FREE，不据eval换主切点。中/远W208/634、401/606，FREE0/208、0/36，两个混合与DAV逐query全字段一致（3456次），属结构保留。双模型驻留689.15M参数，热态合计中位0.271s/P950.335s；仅部署参考，GPU已释放。独立native/校准/逐query复算PASS；新近带描述曲线也有交叉，仅Development与近POS覆盖富集，非实机/安全确认。旧[同FREE排序描述](../research/active/dtr-r0/nearfield/RGB_MATCHED_FREE_ARKIT_CAL_DEV_20261010.md)阶段2stop及所有旧失败/冻结对照保留。

[同步RGB＋ToF数据源v1](../research/active/dtr-r0/nearfield/SYNC_RGB_TOF_DATASET_V1_DEV_20261011.md)已完成12新visit按6/2/4隔离、384帧/10368query；eval128帧/3456query只完整性封存，未计算方法指标。train+cal近/中/远POS828/1332/1217、FREE1354/536/173；独立FARO时间匹配63帧、有效配准51帧（train14/cal4/eval33），333帧保留ToF UNKNOWN，连续有效FARO窗口仍未建立。主臂K0全网格W31/828、94/1332、38/1217，F2/1354、1/536、0/173；rho/ambient、固定OR/AND及共同18帧对照详表，不选择融合器。所有FARO M3/S窗口NOT_RUN，仅native循环臂可作报警描述；不当ToF能力。384帧双RGB推理及5臂K2合成完成，下载480.49MB，GPU275.468s已释放；[旧可行性试点](../research/active/dtr-r0/nearfield/SYNC_RGB_TOF_FEASIBILITY_DEV_20261010.md)保留。

## ToF当前决定

用户2026-10-10决定：接受S集成的接触收益与轻提醒取舍，**S955/956/957任务成本HGB均值升为ToF模拟默认候选，未接入App**。冻结6e441f9a的6权重/47维，原5格强档逐slot不变，新增slot仅轻档。旧[确认](../research/active/dtr-r0/nearfield/CNH_S_ENSEMBLE_CONFIRM_DEV_20261010.md)净151/1024、a/d通过而b/c失败保留；用户改用取舍曲线与环境波动，不再按成本≤k×原5格二元判定。旧约2.7pp是通知次数归一，实际far+clear有通知clip增量2.214pp。

[池化取舍](../research/active/dtr-r0/nearfield/CNH_S_ENSEMBLE_TRADEOFF_DEV_20261010.md)完成：六组已消费pass/clear负例、14family、7168clip（far+clear5376），全部ties最低阈值tau1.917951703；同池原5格445→606有通知clip，增2.995pp≤3pp，contact不选点。封存后生成1536scene×K2/4新family，与30库存19252键及背景/目标参数无交集。

新hold原5格HEAD372/BODY308各/512；S救86/99、损0，净185/1024=18.07pp，v2加权成本211→325.75。far+clear有通知112→155/1536（7.292%→10.091%，+2.799pp）；family增量1.823–3.646pp。绝对漂移0.195pp，比旧2-family cal的1.042pp缩小；“两family造成漂移”记为用户解释，family数/选点口径/数据同时变动，未隔离因果。描述曲线1–3pp预算S收益高于E/both，零增量E优于S，无全域支配。

near-pass间隙≤10cm轻.25/强1，far/clear任何通知1；gap1逐query、同帧最高档联合计数，contact f3–13及时/f14–15晚。逐档计数、scene/family bootstrap及冻结候选manifest见报告；独立核验697900池ties、77322曲线、49152事件/13600通知PASS。GPU保守100/1200s、评价审计整合500/1200command-wall，权重/特征/保护源不动。

## 继承与边界

原M3/5格保留为对照和强档来源；旧E2E、重训/确认失败与各run停止规则保留。池化已消费负例按本任务授权复用，hold曲线仅描述不选点。有限AABB模拟Development、4family区间仅描述，非实机/安全证明；共向几何不替代外参/时钟核验，方向/PDR、硬件阶段未启动，真实同步RGB融合仍未建立。后续台架/App由候选manifest提供接口，本任务未接入。

更新前全文见Git `96e2b0ca`及本run `integration_before/`；历史见RUNS/报告。
ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED
