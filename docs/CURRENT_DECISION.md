# 当前研究决定

本页只记录**避障线**的决定；毕设线见[两条工作线](WORK_LINES.md)。

## 2026-10-13 工程接入状态

[数据再利用](../research/active/hardware-bringup/DATA_POTENTIAL_DEV_20261013.md)及[四轮扩展目录](../research/active/hardware-bringup/DATA_OPPORTUNITY_CATALOG_20261013.md)：第四轮导出8个Hypersim family的393216格原生分布，恢复SANPO50帧人工来源及6条五关键帧pole掩码索引，并实际读取GW-Depth玻璃门raw/人工平面修正配对。优先人工形状与原生几何的表示损失检查；OctoNet18→16的物理合同仍缺，LLRGBD配套深度为尺度未知的8位过滤图。首两轮9本地/27公开来源及第三轮54额外RGB入口保留；没有模型收益或默认升级。

[实机融合工程/几何开发](../research/active/hardware-bringup/REAL_FUSION_ENGINEERING_DEV_20261013.md)完成：手机因果RGB配对/epoch/重试及预检修复现已纳入59/10.19.1安装，原7项JVM、11项预检保持。地面射线仅给条件上界；SIFT/KLT两臂各0/12图像对可采用地面解，不升级测距。[后续工具与数值修复](../research/active/hardware-bringup/FUSION_INPUT_AND_MEASURED_GEOMETRY_DEV_20261013.md)已提供0/20/10cm三视图几何及标线工具，前两图拟合、第三图检验；下一仍需已量横移、物理轴映射与实测杆位置。

[细结构试验](../research/active/hardware-bringup/REAL_THIN_STRUCTURE_DEV_20261013.md)定位17/22；[局部跟踪与几何](../research/active/hardware-bringup/REAL_THIN_LOCAL_GEOMETRY_DEV_20261013.md)的独立RGB链使持续目标9→13/22、侧杆0→3/6，但干扰候选87→116，盒内稳定目标仍3/22且台阶可误入BODY。保留候选跟踪，不升级报警；[距离可信度](../research/active/hardware-bringup/REAL_THIN_CONFIDENCE_DEV_20261013.md)已查明近距锚点外推，硬互证/局部跨度误删细杆，采用独立距离状态记录而不否定视觉候选；仍需候选公制参照。

同日[真实开发尺度/时序比较](../research/active/hardware-bringup/REAL_FUSION_REFINEMENT_DEV_20261013.md)完成：18段1677帧，稳健拟合成功才替换尺度的小候选使低头细杆any65→67/79、首报提前0.568秒，18段any未损；BODY另少15个query帧、无逐query真值，不能判为去误报。[第二轮](../research/active/hardware-bringup/REAL_FUSION_REFINEMENT_V2_DEV_20261013.md)的同质/双mode锚点、运动尺度搬运与地面软回退均未超过旧候选；保留其开发身份，不改变冻结v1/v2或手机默认。下一优先固定安装几何/去杆参考，新头戴确认仍待完成。

本轮 **59 / 10.19.1 已安装，融合实验待佩戴验证**：采集页收流/预览与写盘脱耦，同12秒录制预览12.58→23.74fps、IMU缺序号201→0；另修复float32读出边界与退化拟合。原始数据持久性、模型和阈值保持，完整文件/索引检查通过。用户暂不佩戴，实时RGB分支未验证，不宣称端到端融合或新冻结eval通过。[本轮工程记录](../research/active/hardware-bringup/FUSION_INPUT_AND_MEASURED_GEOMETRY_DEV_20261013.md) · [项目状态](PROJECT_STATE.md) · [入口说明](HARDWARE_OBSTACLE_DEMO.md)。

新头戴冻结检验仍需用户佩戴录制；已准备[标签与 v1/v2 执行入口](../research/active/hardware-bringup/FROZEN_FUSION_EVAL_RUNBOOK.md)，完整性预检不会运行推理或产生新评价成绩。下方研究数字与历史判读保留原文。

**ADT 真值评测支持 v6**：独立的头戴数据上，误报相对下降 23%，近距检出基本不变（81.2%→80.6%），代价是 1.5–2 m 起报变晚。[结果](../research/active/hardware-bringup/ADT_EVAL_RESULT_20261013.md)。

**10-13 晚冻结 v6**：v3＋RGB 像素团深度规则，针对强光窄走廊开头误报（已用数据上 0.96→0.39，无回退，事后选择）；下次录制四版同跑。[阶段总结](../research/active/hardware-bringup/STAGE_SUMMARY_20261013.md)。

**10-13 下午细杆补录**：4 段细杆 ahead 全过且杆由 RGB 看到，但标定失败时窄走廊开头对墙/吊牌误报，quiet 样本尚不足以排除。[结果](../research/active/hardware-bringup/FROZEN_EVAL_RESULT_20261013B.md)。

**10-13 冻结检验已跑**：用户头戴新录 4 段，v1/v2/v3 全部满足总体主张（quiet 1 段 2.7–5.4%，柱/墙 ahead 2 段末段 100%），但样本小、无细杆和标定段。[结果](../research/active/hardware-bringup/FROZEN_EVAL_RESULT_20261013.md)。

录制前已冻结[v3](../research/active/hardware-bringup/FROZEN_FUSION_EVAL_V3_PROTOCOL.md)（v2＋稳健ToF锚定替换）：已消费数据上修掉低头看地面时RGB满屏误报（96%→0%），其余判定不变；新录制三版同跑。[开发记录](../research/active/hardware-bringup/FUSION_V3_DEV_20261013.md)。

更新：2026-10-11。主线为盲杖互补前视感知，M3和A+LOCAL保留；原5格局部融合与L2保留。

RGB [分带混合评价](../research/active/dtr-r0/nearfield/RGB_BAND_HYBRID_DEV_20261010.md)完成六新Validation visit各16帧：Uni近带全局ARKcal仅用已消费240帧/2019严格FREE、全带5%封存cut0.0385859m，DAV中远沿用pooled304 cut0.2440383m。混合近W1→48/159、6/6capture改善，但FREE1→29/694（4.18%）超过预声明绝对2%（最多13），未采用混合，维持DAV2 Indoor Large raw R0、RGB≥0.8m/近带ToF。Uni pooled304漂移对照cut0.0961610为W27/159、FREE11/694（1.59%）；ARKcal放宽带来+21W/+18FREE，不据eval换主切点。中/远W208/634、401/606，FREE0/208、0/36，两个混合与DAV逐query全字段一致（3456次），属结构保留。双模型驻留689.15M参数，热态合计中位0.271s/P950.335s；仅部署参考，GPU已释放。独立native/校准/逐query复算PASS；新近带描述曲线也有交叉，仅Development与近POS覆盖富集，非实机/安全确认。旧[同FREE排序描述](../research/active/dtr-r0/nearfield/RGB_MATCHED_FREE_ARKIT_CAL_DEV_20261010.md)阶段2stop及所有旧失败/冻结对照保留。

[融合确认v4](../research/active/dtr-r0/nearfield/SYNC_FUSION_CONFIRM_V4_DEV_20261011.md)完成预注册，协议`270abc0a`在新数据与交叉拟合前提交并推送；判据明确取自v3同FREE事后分析`7dc9bbba`，原v3 FAIL及数字保留。冻结v3 D′三权重，旧30visit五折15临时模型只生成cal折外分数，加新cal4；六臂混合池15切点/209957ties先封存，新eval16一次OPEN。**主中带低误报排序PASS**：归一化pAUC D′0.76941、RGB0.45021、OR0.56983，D′−RGB+31.920pp CI[16.996,44.803]、D′−OR+19.958pp [5.494,38.464]，两项2000/2000 visit重抽样有效。部署中W/F：D′2538/48、RGB1810/38、OR2178/26（POS2873/FREE866），对RGB/OR未双向占优；半循环cal→eval F率D′4.458→5.543%、RGB5→4.388%，同v4eval上原RGB切点10.970%。OOF中带有限FREE分数比冻结训练内均值高0.04237，不做映射。近带区间跨0，远带及ambient×10中带仅描述；FARO仅92过门帧，中/远FREE14/1。协议/OOF/cal/数据/2000曲线bootstrap/部署漂移三阶段独立核验PASS。CPU保守2034.948/2400command-wall、GPU263.192/600s、下载3488869655B/4GB；特征机械续跑失败和一致性核验保留，总额度不扩展。结论限真实RGB＋半合成ToF新visit Development；不做仲裁器，不接入App/M3/5格，不读保护480/test。

[融合确认v3](../research/active/dtr-r0/nearfield/SYNC_FUSION_CONFIRM_V3_DEV_20261011.md)完成预注册：协议`ec252d68`在建新数据前提交并推送，实现`95f451e6`在OPEN前提交；旧30visit一次训练、新cal6/eval12共576帧，D′六臂等权HGB/B′混合LR/(b)半循环LR均去显式query位置与带别特征，加最低传感器证据OR门控。21阈值/全部ties封存并独立复算后eval一次OPEN。**主中带确认未通过**。半循环中D′ W1696/2077，vs最佳单源rgb净+87，救213/损126，12visit聚类95%CI-0.308–9.155pp，FREE4/689=0.581%；主FAIL。同口径：D_prime净+87、CI-0.308–9.155pp、FREE4/689=0.581%（FAIL）；B_prime净+86、CI-0.519–9.557pp、FREE8/689=1.161%（FAIL）；audit_b净-91、CI-11.354–2.535pp、FREE0/689=0.000%（FAIL）。次判据：faro_rho030_ambient1近带vsRGB净-2、FREE15/105=14.286%，FAIL；faro_rho030_ambient10近带vsRGB净-58、FREE121/1702=7.109%，FAIL。主中带收益按最低证据拆为both净-16、tof_only净+209、rgb_only净-26、neither净-80；全6臂/3带/3模型无证据支持0，属条件分解非因果。FARO仅eval43过门帧，完整贡献visit见报告，远带只报告、不作主张。9权重/hash、六臂manifest、逐query/四类拆分/2000次bootstrap、四阶段独立核验PASS。下载3,889,213,192B/4GB、GPU211.577/600s、CPU保守2209.665/2400 command-wall s，任务进程释放；18新visit已消费，eval后未改模型或阈值，不读保护480/test。仅真实RGB＋半合成ToF Development，不升级App/M3/5格或作实机/安全结论。

[v3同FREE事后分析](../research/active/dtr-r0/nearfield/SYNC_FUSION_V3_MATCHED_FREE_POSTHOC_DEV_20261011.md)满足预写继续规则：半循环中带1%/2%/5% D′见证均≥RGB/OR、2%/5%逐visit为9/12与8/12，建议进入条件仲裁器设计与新visit确认；eval标签oracle仅作排序描述，5%对OR区间跨0，原v3预注册FAIL及数字不改。

历史v2/事后审计/稳健性CV（均已消费）保留：[v2中带净385](../research/active/dtr-r0/nearfield/SYNC_FUSION_CONFIRM_V2_DEV_20261011.md)、[去位置＋门控(b)事后净282](../research/active/dtr-r0/nearfield/SYNC_FUSION_EVIDENCE_AUDIT_DEV_20261011.md)、[含位置六臂HGB探索](../research/active/dtr-r0/nearfield/SYNC_FUSION_ROBUST_DEV_20261011.md)。原结果及失败未回溯改写；本次v3按新visit预注册评价上述组合。

历史v1（已消费）：用户2026-10-11决定进入query级同步融合器阶段，主ToF使用扰动native（**半循环**），固定过门FARO作独立几何检验。[同步融合v1](../research/active/dtr-r0/nearfield/SYNC_FUSION_V1_DEV_20261011.md)已完成train6/cal2/eval4隔离、20维HGB主模型与logistic对照、cal近2%/中5%/远10%选点、封存后一次eval。**仅半循环来源支持，独立几何未复现**：native HGB近/中W278/371、497/542，F7/690、9/286，胜最佳单源净20/104；远W511/616但F7/30超标。FARO近W145/163、F8/151，不及RGB146/163、2/151；中远cal FREE=0不可校准。logistic同判读，不替换主模型；HGB近中净收益visit区间跨0。eval已消费，不据此改模型/阈值；下一先补独立几何cal与新visit来源，未升级默认候选。W为query POS命中，与旧像素交集W不可纵比。

[同步源v1.1](../research/active/dtr-r0/nearfield/SYNC_RGB_TOF_DATASET_V1_1_DEV_20261011.md)的80/80失败（103/384）及几何长尾保留；用户允许半循环小模型探索，不等于合格连续ToF源。FARO train/cal/eval仅57/7/39过门帧，native192/64/128；12权重/36切点封存、4509 eval query/22842判定/四visit bootstrap独立复算PASS。GPU0、无下载，预算及hash见报告。真实RGB+半合成ToF query证据，不是实机/接触/安全效果。

[远带FREE原因诊断](../research/active/dtr-r0/nearfield/FAR_BAND_FREE_DIAGNOSIS_DEV_20261011.md)完成EXPLORE：v2已消费eval与v1.1全部split，a/b/c各176query；a标签可疑116/176（65.9%），b130/176（73.9%），不支持错标共同主因。a几何风险111/176、双源空参考下DAV侵入33/176；v2 a中71/111由logit支持而DAV区间内不足16点，优先查融合支持语义，再查RGB远带深度。confidence239/239补齐、176抽查图、528query独立复算PASS；GPU0、CPU保守400/1500command-wall。未来参考质量合同仅草案，不改旧标签/阈值/模型或远带失败结论，不读保护480/test。

## ToF当前决定

2026-10-12 实机：此前所有真实回放的zone顺序与模型上下颠倒，已修复；模型侧加近距串扰/直流干扰随机化后，宿舍pilot首次呈现合理报警（[实机路径候选](../research/active/hardware-bringup/REAL_NUISANCE_DR_M3_DEV_20261011.md)，仅已消费数据，模拟默认不变，待楼道独立录制）。

2026-10-11 [signed bias／LOCAL v4合同修复](../research/active/hardware-bringup/LOCAL_SIGNED_BIAS_CONTRACT_DEV_20261011.md)预写门槛与独立核验通过，**模拟Development默认候选更新为去bin0＋仪器DR＋新S＋v4合同**；[默认指针](../research/active/dtr-r0/nearfield/CNH_SIM_DEFAULT_CANDIDATE.json)绑定193e4a21既有权重／切点与新合同，不接入App，真实效果待独立录制。主八种确定性控制全链HEAD/BODY各0强0轻/9；相同已消费模拟留出及时净185/1024 vs旧170，far+clear152/1536=9.896% vs168/1536=10.938%，成本291.75 vs344.75，v4成绩不变。实录幕布边界段仍大量报警，不能作误报率或闭合证据；下一尽头>4.5m楼道clear及墙／细杆。以下为旧默认及其原始历史结果，不回溯改193e4a21失败或既有冻结基线。

用户2026-10-10决定：接受S集成的接触收益与轻提醒取舍，**S955/956/957任务成本HGB均值升为ToF模拟默认候选，未接入App**。冻结6e441f9a的6权重/47维，原5格强档逐slot不变，新增slot仅轻档。旧[确认](../research/active/dtr-r0/nearfield/CNH_S_ENSEMBLE_CONFIRM_DEV_20261010.md)净151/1024、a/d通过而b/c失败保留；用户改用取舍曲线与环境波动，不再按成本≤k×原5格二元判定。旧约2.7pp是通知次数归一，实际far+clear有通知clip增量2.214pp。

[池化取舍](../research/active/dtr-r0/nearfield/CNH_S_ENSEMBLE_TRADEOFF_DEV_20261010.md)完成：六组已消费pass/clear负例、14family、7168clip（far+clear5376），全部ties最低阈值tau1.917951703；同池原5格445→606有通知clip，增2.995pp≤3pp，contact不选点。封存后生成1536scene×K2/4新family，与30库存19252键及背景/目标参数无交集。

新hold原5格HEAD372/BODY308各/512；S救86/99、损0，净185/1024=18.07pp，v2加权成本211→325.75。far+clear有通知112→155/1536（7.292%→10.091%，+2.799pp）；family增量1.823–3.646pp。绝对漂移0.195pp，比旧2-family cal的1.042pp缩小；“两family造成漂移”记为用户解释，family数/选点口径/数据同时变动，未隔离因果。描述曲线1–3pp预算S收益高于E/both，零增量E优于S，无全域支配。

near-pass间隙≤10cm轻.25/强1，far/clear任何通知1；gap1逐query、同帧最高档联合计数，contact f3–13及时/f14–15晚。逐档计数、scene/family bootstrap及冻结候选manifest见报告；独立核验697900池ties、77322曲线、49152事件/13600通知PASS。GPU保守100/1200s、评价审计整合500/1200command-wall，权重/特征/保护源不动。

## 继承与边界

原M3/5格保留为对照和强档来源；旧E2E、重训/确认失败与各run停止规则保留。池化已消费负例按本任务授权复用，hold曲线仅描述不选点。有限AABB模拟Development、4family区间仅描述，非实机/安全证明；共向几何不替代外参/时钟核验，方向/PDR、硬件阶段未启动，融合v4当前判读以本页新visit低误报排序预注册结果为准；v3原失败保留，近/远带与稀疏FARO/压力臂仅描述，未构成全面独立实机确认。后续台架/App由候选manifest提供接口，本任务未接入。

本次v3整合前全文见Git `95f451e6`及 sync-fusion-confirm-v3-dev-20261011/ 的 `CURRENT*_integration_before.md`；历史见RUNS/报告。
ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED
