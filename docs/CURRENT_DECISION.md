# 当前研究决定

更新：2026-10-03。主线：盲杖互补的前视障碍感知。本轮先诊断已有真实回放、审计评估可行性；新的模拟读出变体暂停，包括同场景受控位移难例。

Status: `L10_R0_PAUSED / DTR_R2_DYNAMIC_RETAINED`（历史保留）；ToF 阶段：`V5_FROZEN_COMPLETE / HARDWARE_DEFERRED`。

## 当前决定

**真实回放定位与评估可行性审计完成，保持新模拟读出暂停。** 冻结v5的04分数逐值复现；同一区两个bin均值偏移，四帧z中位8.31/9.74，未平滑NN也左/中128/128。只归零这两格输入后A2左/中降为37/128、12/128，是贡献诊断，不是修复。解码/bin配置一致、无方差floor；整场景背景减法与训练的串扰bias减法合同不同，物理成因与逐query真值仍未识别。31浅事件中M3剩5漏；原bootstrap设置补1–3且零丢仍过不了浅条件，补4(+12.90pp)才可能过，条件可达但只适合筛大幅改善。3/5pp条件功效规划及误停边界见[本轮审计](../research/active/dtr-r0/nearfield/CNH_REPLAY_EVALUATION_AUDIT_20261003.md)。下一步优先让真实观测保留已有障碍语义、区分仪器偏置和场景表面，再决定评估设计与恢复比较；不靠删格或调阈值宣称迁移成功。M3和旧失败保留。

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
