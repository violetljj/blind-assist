# 当前研究决定

更新：2026-10-04。主线：盲杖互补的前视障碍感知。读出试点2完成：V复现、VD小幅增益且成本改善、T2拟合未通过；无候选，M3保留，硬件仍暂停。

Status: `L10_R0_PAUSED / DTR_R2_DYNAMIC_RETAINED`（历史保留）；ToF 阶段：`V5_FROZEN_COMPLETE / HARDWARE_DEFERRED`。

## 当前决定

**V改善复现，VD收益小，T2拟合未通过，保留M3。** 新48场景主AUC M3/V/VD/D=.8740/.8987/.9012/.9519；V−M3+.0247 [.0160,.0346]，补上31.7%，VD补上34.9%。VD−V+.0024 [.0004,.0046]统计为正但未到+.02；自然清晰V/VD=212/193次、浅均26/31、深均163/164，事后配对成本差−.0978/min [−.1449,−.0516]。两臂护栏通过，均未到+.045候选门槛，V_REPLICATED/NO_CANDIDATE。T2硬训练损失.2946>.2756，训练AUC .8762虽达标，仍T2_UNFIT，未跑seed1/2；逐帧表示问题未解决。主计算56.6分钟，无硬时限；全部模拟Development、教师背景/目标特权仍保留。[完整结果、种子及分层](../research/active/dtr-r0/nearfield/CNH_READOUT_PILOT2_20261005.md)。[上轮V/T与失败](../research/active/dtr-r0/nearfield/CNH_TEMPORAL_READOUT_PILOT_20261004.md)。

**完整48场景把下一问题指向远距证据读出。** 同全批清晰首停18/384，视场外M3及时32/96，普通8/12帧56/55，容错8/12帧58/61。容错12的61次首报均在目标仍可见时，60次≥2.1m；不能称出视场后的记忆获益。高组清晰首停10→11，合并宏AUC.8477，不触发旧C；本轮未训练，M3保留。[完整结果、成本和修正判读](../research/active/dtr-r0/nearfield/CNH_POSE_MARGINAL_REFERENCE_COMPLETE_20261004.md)。

前置8帧位姿2×2测量在原48场景发现估计相对位姿D−M3+.0788 [.0575,.1024]，为本轮训练动机；原批分层和帧成本阈值不能直接部署或合并新批结果。[原测量及成本](../research/active/dtr-r0/nearfield/CNH_POSE_FACTORIAL_20261004.md)。

**源漏报与出视场相关，但没有证明是窗口短。** 用户授权先做仿真记忆A/B：原五浅漏报有四个出视场，四个raw8内仍有可见曝光。受控低可见12场景，合并0.9–2.1m估计相对位姿oracle12/16宏AUC为0.8077/0.8092（正768/负1152），真位姿16为0.9272、M3为0.6609；未到≥0.85训练条件，也未到≤0.75停止分支，INTERMEDIATE、C NOT_RUN。M3冻结保留；下一问题建议历史对齐与信息利用，不自动开新训练。真实输入合同暂缓；全场景/UNKNOWN、旧失败和成本保留。[结果与判读](../research/active/dtr-r0/nearfield/CNH_MEMORY_REALITY_20261004.md)

**此前U8/U12测量的条件性余量保留。** 已消费48个直行场景，1.2–2.1m宏AUC M3/U8/U12为0.8708/0.9678/0.9804，U8−M3 +0.0970 [0.0720,0.1225]；匹配外侧成本后浅召回仍有余量，判为REALISTIC_HEADROOM。参照未知目标尺寸/反射率，但背景与姿态仍取真值；有限假设网格不是真实先验观测上界，也不证明现成读出可达到该收益。余量主要在1.6m以外、BODY/panel，不自动训练。M3和旧失败保留。[本轮结果](../research/active/dtr-r0/nearfield/CNH_UNKNOWN_TARGET_REFERENCE_20261004.md)

真实回放定位与评估可行性审计完成；不继续01/04分析。两bin贡献诊断未识别物理成因或修复迁移。真实回放线待设备会话，需串扰/bias标定、ambient、几何和空走廊段。原31浅事件条件只能筛较大改善，3–5pp功效规划不是采样授权；保持负例和误停成本。[审计与复现](../research/active/dtr-r0/nearfield/CNH_REPLAY_EVALUATION_AUDIT_20261003.md)

- **M3优先保留；全物体三级真值审计后浅擦碰优势与条件关系不变。** 固定分数/阈值，不重校准：source评估clear NEAR543/4741=11.45%→M3 519/4741=10.95%，context260/3920=6.63%→254/3920=6.48%。原立杆M3的95次clear报警有30次实际属于背景0–10cm擦身，修正65/560=11.61% [8.83,14.34]；不可当负例盲目抑制。原浅/深检出完全不变，M8 context新clear增2.30pp仍超限。仅描述性旧条件复核，旧判读保留；18432查询几何与144旧指标核对通过。横梁原深净少4实际为丢5补1；末帧差异不能直接当漏停，新的实际轨迹序列结果见上。旧序列结果按原定义保留。[审计](../research/active/dtr-r0/nearfield/CNH_ALL_OBJECT_TRUTH_AUDIT_20261002.md)

- **原目标操作标签、合并清晰预算下，M3新模拟单位比较通过。** 用户授权冻结模型、48校准+96评估新单位，仅把名义伸入扩为−20…+15cm；`b6174d1b`跑前固定主判据。M3浅擦碰145/168对NEAR122/168，差+13.69pp、95%区间[8.28,20.51]；>5cm差+0.12pp、全距离合并清晰误报差−0.52pp，判为`M3_CONFIRMED`。M8单列同规则也通过；M3可接受擦身报警增加，同高度外侧误报略增，整体下降来自另一高度带。新单位仍属已消费Development配方，非保护确认或实机证据。条件序列描述、分母区间及RGB分数见[本轮报告](../research/active/dtr-r0/nearfield/CNH_MARGIN_CONFIRM_RESULTS_20261002.md)。三级真值为伸入必须报、身体外0–10cm只报告、更远或另一高度才计清晰；原外扩guard失败、旧NEAR序列结果和方向扫描`NOT_PATH_LIMITED_AT_<=1DEG`均保留。mask扩展37条0补回/8丢失（`NOT_SUPPORTED`），停止同批15/52条H3单帧归属小试。详见[当前页](../research/active/dtr-r0/CURRENT.md)及[日志](../research/active/dtr-r0/RUNS.md)。
- **本轮用户授权的 v5 冻结复现通过，保留 NN 加固定五分数平滑 A2。** HEAD/BODY AP 0.8444/0.7914；相对 S2 配对95%区间均高于零，两组均64/64单位胜出。A3补回6个强信号漏检，但guard实际误报未通过预设要求，不采用。见[v5结果](../research/active/dtr-r0/nearfield/CNH_V5_RESULTS_20260928.md)。不扩大记忆网络、不追加调参；用户设备暂不方便，交付历史真实输入回放，现场演示延后。02段整段有物体、无负帧，不能判定误报或选择性；主失败对照改为04恢复背景中HEAD左/中均128/128触发。这里“空”仅指无新增物体，64cm柜面仍在名义查询范围内，不能当真实应用假警率；输入定义与查询语义尚未分离，迁移未成立。
- **接受 v4 执行偏差并披露。** 六个主检验及 A1/A2 保留为“正式结果，带已披露偏差”；不写成完全符合冻结流程，不事后修改协议。接受理由及恢复时点见[v4结果追加决定](../research/active/dtr-r0/nearfield/CNH_TRACK_A_SCALE_V4_RESULTS_20260926.md)。
- v1/v2 原失败保留；修复 v2、位置/质量/软先验诊断均为已消费 Development，不追认为正式证据。相机必要性、ToF物理上限与真实效果均未建立。
- 手机保留 A 基线及原首页→手动开始 A+LOCAL→结束返回首页；UNKNOWN 不等于无障碍。City、保护test、新UE采集和硬件第二阶段仍暂停；已有RGB/深度Development缓存可用于已授权的ToF归属探索。

## 保留证据

|证据|关键数字|边界|
|---|---|---|
|v3 / v4 主检验|63 / 64 audit单位，六项均成立；S2−B1-R HEAD/BODY：+0.099/+0.079、+0.092/+0.078|受控仿真相对增益|
|v4 A1/A2|BODY近事件1990；没报减少5.1–5.4pp。HEAD/BODY近事件1936/1990；单帧及时优势7.4–9.2 / 4.8–7.4pp|相同calib预算，audit实际假警不等|
|候选与软先验|修复v2：32calib/63audit；中等组合AUC .9保留HEAD/BODY增益37.1%/17.0%|消费Development；非通用候选规格|

完整主张、分母、来源提交与禁用措辞见[论文主张台账](../research/active/dtr-r0/THESIS_CLAIMS_20260927.md)。

## 未决与入口

RGB逼近选择性首轮失败保留；后续局部边缘小试发现理想正确距离可改善净距，但实际粗格对象/表面归属未解决。产品报警工作点仍未决定；短模拟序列及“序列×查询盒”假警不能换算真实提醒负担。真实计数/串扰/安装标定待硬件。确认畅通距离仅作≥10cm、ρ≥0.5、最坏摆放的附录辅助地图。

[LCSPCData最小样本审查](../research/active/dtr-r0/nearfield/LCSPCDATA_SUITABILITY_20260927.md)已于9月27日完成：下载tall_block JSON/STL及说明、许可共2,281,173 B，128帧TMF8820 3×3×128数据；所查样本无固定视点重复帧、独立背景，时间戳全零，暂不能检验冻结时序报警排名。仅支持有限脉冲形态参考，不作L8CH标定。[公开检索记录](../artifacts.local/work/tof-real-histogram-search-20260927/REPORT.md)与THDR3K入口未核验状态保留。

[路线当前页](../research/active/dtr-r0/CURRENT.md) · [v3结果](../research/active/dtr-r0/nearfield/CNH_TRACK_A_SCALE_V3_RESULTS_20260926.md) · [软先验诊断](../research/active/dtr-r0/nearfield/CNH_SOFT_PRIOR_DEV_20260927.md) · [项目入口](PROJECT_STATE.md)

[封口前全文](operations/snapshots/CURRENT_DECISION_20260927_PRE_TOF_CLOSE.md)逐字节保存；旧快照中的待决状态不覆盖本页。[v5更新前全文](operations/snapshots/CURRENT_DECISION_20260928_PRE_V5.md)保留；后续当前页变更由Git历史保存。
