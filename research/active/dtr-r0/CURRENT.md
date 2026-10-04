COVERAGE_RECOVERY_DEV / M3_RETAINED

# 前视障碍感知：当前状态

更新：2026-10-05。已消费Development；`V5_FROZEN_COMPLETE / HARDWARE_DEFERRED`。
Status: `DTR_R2_DYNAMIC_RETAINED`（仅为历史状态）。

## 当前决定

**一帧回正剂量—反应：r*=1.3m，覆盖提示仍待验证。** 同24个mode0场景，原反侧VD/M3及时26/20，1.3m回正85/86（分母88）、全程0°88/88，1m54/51；同侧均104/104。VD与全0°计数差−3，场景配对95%区间[−6,0]，点判据通过非等效证明。目标带清晰合计VD8→7、M3 7→8/192。两臂原R阈值不变、无训练；一帧动作与同期0.4s渐进结果分开，人的反应未测，下一问题为估计位姿+覆盖/观测时效UNKNOWN。[完整剂量、曝光与成本](nearfield/CNH_HEAD_RECENTER_DOSE_20261005.md)。

**受控回正补回漏报，提示策略尚未成立。** 24个mode0×7位移K4，反侧浅及时原20/88；2.5/2.1/1.7m回正均88/88，1.3m为78/88，1m为21/88；同侧全104/104。命中预设恢复分支，最晚网格点1.3m；但联合清晰首停21→25/192，同侧7→10/104，不能宣称护栏通过。1.7m联合14/192，可作后续模拟参照。固定8°/1s估计SE3规则持续偏头96/96触发，扫视也640/640、1280/1280触发，待实际覆盖/观测时效判据。主计算3.25分钟；不训练、不换M3、不自动启动双传感器。[结果与代价](nearfield/CNH_COVERAGE_POLICY_20261004.md)。

覆盖拆分完成：低可见88条均在截止前见过目标，A/B=0，C有曝光未及时62、D及时26；M3/VD/参照D/R_any及时20/26/49/22，C中参照D及时而VD/R均未及时22条。最后可见距离中位数1.93m，高可见296条对照.97m。下一测量以延长曝光、保持近距可见为指标；不重开读出搜索、不自动训练或扫描。[完整结果](nearfield/CNH_FOV_FAILURE_SPLIT_20261005.md)。

R有限参照低分支保留，当前读出及T2优化跟进停止，M3保留；固定.65m/宽先验错配、背景特权及未实机标定失配仍限制极限解释。[R结果](nearfield/CNH_LOCATION_REFERENCE_20261004.md) · [收尾与台账](nearfield/CNH_READOUT_CLOSEOUT_20261004.md)。

固定M3位姿平均未改善报警，固定配方停止，自然候选未运行。[结果与边界](nearfield/CNH_M3_POSE_ENSEMBLE_20261004.md)。

T2诊断MIXED：从零CVR训练优于T2，留出排序高+.0444但损失更差。T2可fit256行、2048行平台未过；优化/泛化与表示未分离。公平V三对M3三改善+.0303 [.0221,.0397]，原试点判定不改。本轮T2优化跟进停止；表示优劣仍未判，无新网络/候选，M3保留。[完整诊断](nearfield/CNH_T2_FIT_DIAGNOSTIC_20261004.md) · [原试点2](nearfield/CNH_READOUT_PILOT2_20261005.md)。

旧批延窗/容错收益主要来自远距仍可见时首报，未证明记忆解决漏报；参照仍知背景/当前真姿态，不是物理界。[容错结果](nearfield/CNH_POSE_MARGINAL_REFERENCE_COMPLETE_20261004.md) · [记忆诊断](nearfield/CNH_MEMORY_REALITY_20261004.md) · [未知目标参照](nearfield/CNH_UNKNOWN_TARGET_REFERENCE_20261004.md)。

真实回放待串扰/bias、ambient、几何和空走廊设备会话；04复现未修复迁移，不继续01/04分析。[回放与评价审计](nearfield/CNH_REPLAY_EVALUATION_AUDIT_20261003.md)

48场景×7位移×K4的源测量与失败均保留；oracle不是物理上界。[源位移测量](../../../artifacts.local/work/cnh-displacement-ceiling-20261003/REPORT.md)

## 保留基线与失败

M3保留，旧浅及时26/31；QMASS/RAY/CCON及SURF/HIST/AGG等原配方失败保留，手机不据此改动。[QMASS](../../../artifacts.local/work/cnh-query-mass-20261003/REPORT.md) · [RAY](nearfield/CNH_RAY_SURFACE_RESULTS_20261002.md) · [CCON](../../../artifacts.local/work/cnh-boundary-contrast-20261003/REPORT.md)。

## 评价与使用边界

原报警96单位7680查询仅1138覆盖0.9m，6542右删失；31浅事件来自26单位。旧浅条件不适合选3–5pp小收益，功效规划与负例成本见审计，不等于采样授权。

三级真值：伸入必须报，身体外0–10cm只报告，更远或其他高度计清晰误报。代理分钟与条件提前量不代表真实提醒负担或安全停止。固定模型条件区间不含重训练/校准不确定性；可见支持为零不证明畅通，密输出不增加8×8物理分辨率。无跨源/实机效果；UNKNOWN不等于安全。City、保护test、新UE和硬件第二阶段暂停。

[RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)。历次完整文字保留在Git历史及原结果中。
