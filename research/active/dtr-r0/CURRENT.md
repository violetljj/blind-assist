# 前视障碍感知：当前状态

更新：2026-10-10。主线为盲杖互补前视感知，M3和A+LOCAL保留；原5格局部融合与L2保留。

RGB [同FREE排序描述](nearfield/RGB_MATCHED_FREE_ARKIT_CAL_DEV_20261010.md)完成old6/fresh6冻结R0的全部ties事后描述：old6近/中远、fresh6近在全部共同实际FREE点Uni≥DAV；fresh6中远交叉F0：DAV276/Uni227（−49），F104：1047/1046（−1），未达四曲线全域弱支配，按事前门停止阶段2。F7中远上包络仍Uni641>DAV483，近F0为16>13，收益与24capture带局部交叉完整保留；这是事后最大见证，不是部署选点。未重跑预测/训练/affine、未拟合新ARKcal或下载推理，cal域迁移仍未检验。维持DAV2 Indoor Large raw R0、RGB≥0.8m/近带ToF。23683ties与30240逐query配对独立PASS；仅已消费Development、query相关，不作实机/安全确认。旧[主干确认](nearfield/RGB_BACKBONE_CONFIRM_DEV_20261010.md)的见证与FREE代价、尺度读出混合结果、DepthPro affine/残差/各run stop及同步融合N/E保留。

## ToF当前决定

用户2026-10-10决定：接受S集成的接触收益与轻提醒取舍，**S955/956/957任务成本HGB均值升为ToF模拟默认候选，未接入App**。冻结6e441f9a的6权重/47维，原5格强档逐slot不变，新增slot仅轻档。旧[确认](nearfield/CNH_S_ENSEMBLE_CONFIRM_DEV_20261010.md)净151/1024、a/d通过而b/c失败保留；用户改用取舍曲线与环境波动，不再按成本≤k×原5格二元判定。旧约2.7pp是通知次数归一，实际far+clear有通知clip增量2.214pp。

[池化取舍](nearfield/CNH_S_ENSEMBLE_TRADEOFF_DEV_20261010.md)完成：六组已消费pass/clear负例、14family、7168clip（far+clear5376），全部ties最低阈值tau1.917951703；同池原5格445→606有通知clip，增2.995pp≤3pp，contact不选点。封存后生成1536scene×K2/4新family，与30库存19252键及背景/目标参数无交集。

新hold原5格HEAD372/BODY308各/512；S救86/99、损0，净185/1024=18.07pp，v2加权成本211→325.75。far+clear有通知112→155/1536（7.292%→10.091%，+2.799pp）；family增量1.823–3.646pp。绝对漂移0.195pp，比旧2-family cal的1.042pp缩小；“两family造成漂移”记为用户解释，family数/选点口径/数据同时变动，未隔离因果。描述曲线1–3pp预算S收益高于E/both，零增量E优于S，无全域支配。

near-pass间隙≤10cm轻.25/强1，far/clear任何通知1；gap1逐query、同帧最高档联合计数，contact f3–13及时/f14–15晚。逐档计数、scene/family bootstrap及冻结候选manifest见报告；独立核验697900池ties、77322曲线、49152事件/13600通知PASS。GPU保守100/1200s、评价审计整合500/1200command-wall，权重/特征/保护源不动。

## 继承与边界

原M3/5格保留为对照和强档来源；旧E2E、重训/确认失败与各run停止规则保留。池化已消费负例按本任务授权复用，hold曲线仅描述不选点。有限AABB模拟Development、4family区间仅描述，非实机/安全证明；共向几何不替代外参/时钟核验，方向/PDR、硬件阶段与同步RGB融合未启动。后续台架/App由候选manifest提供接口，本任务未接入。

更新前全文见Git `96e2b0ca`及本run `integration_before/`；历史见RUNS/报告。
ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED
