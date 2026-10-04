POSE_ENSEMBLE_NOT_ESTABLISHED_DEV / M3_RETAINED

# 前视障碍感知：当前状态

更新：2026-10-04。已消费Development；`V5_FROZEN_COMPLETE / HARDWARE_DEFERRED`。
Status: `DTR_R2_DYNAMIC_RETAINED`（仅为历史状态）。

## 当前决定

固定M3的24位姿先验平均完成：同清晰首停18/384，浅及时仍309/384；视场外32/96、视场内277/288，补回/丢失0/0与1/1。视场外主AUC .6941→.6999，差区间跨0；成本分层10/8→9/9。固定平均未带来报警收益，停止该配方；自然候选未运行，M3保留。推理30.2min，无训练/新渲染。[结果与边界](nearfield/CNH_M3_POSE_ENSEMBLE_20261004.md)。

读出试点2保留：V复现+.0247，VD对V仅+.0024，均未到候选门槛；自然浅26/31、深163/164，清晰V/VD=212/193。T2拟合未过、无候选。[完整结果](nearfield/CNH_READOUT_PILOT2_20261005.md)。

已知场景位姿容错参照48/48保留：同清晰18/384，低组M3/普通8/12/容错8/12及时32/56/55/58/61（分母96）。61次首报均仍可见、60次≥2.1m；不能称延窗记忆收益。高组成本10→11，宏AUC.8477未触发旧C。[完整结果](nearfield/CNH_POSE_MARGINAL_REFERENCE_COMPLETE_20261004.md)。

另有低可见12场景长窗口诊断：合并0.9–2.1m估计位姿oracle12/16 AUC .8077/.8092，INTERMEDIATE，未触发训练；不证明窗口短。[记忆诊断](nearfield/CNH_MEMORY_REALITY_20261004.md)。此前U8/U12未知目标参照 .9678/.9804（M3 .8708），真背景/姿态条件余量保留。[未知目标结果](nearfield/CNH_UNKNOWN_TARGET_REFERENCE_20261004.md)

本轮长窗口训练条件未触发。真实回放仍待设备会话：需串扰/bias标定、ambient、几何记录和空走廊段。现有04逐值复现及两bin贡献诊断不能识别物理成因，也未修复迁移；不继续分析01/04。[回放与评价审计](nearfield/CNH_REPLAY_EVALUATION_AUDIT_20261003.md)

48场景×7位移×K4的源测量与失败均保留；oracle不是物理上界。[源位移测量](../../../artifacts.local/work/cnh-displacement-ceiling-20261003/REPORT.md)

## 保留基线与失败

M3保留。原浅及时26/31、清晰首停205次；QMASS浅25/31、清晰222次，拟合改善未转化为报警收益。[QMASS结果](../../../artifacts.local/work/cnh-query-mass-20261003/REPORT.md)。RAY浅25/31、清晰217次；CCON浅23/31、补0丢3，均不采用，不继续原配方搜索。[RAY](nearfield/CNH_RAY_SURFACE_RESULTS_20261002.md) · [CCON](../../../artifacts.local/work/cnh-boundary-contrast-20261003/REPORT.md)

SURF/HIST/AGG、共享signed场、NEST/MAX/source yaw、外扩guard与mask失败保留，同批H3单帧小试停止。v5保留NN+A2；正式增益仅在程序化生成器中成立，手机A/A+LOCAL不据此改动。

## 评价与使用边界

原报警96单位7680查询仅1138覆盖0.9m，6542右删失；31浅事件来自26单位。旧浅条件不适合选3–5pp小收益，功效规划与负例成本见审计，不等于采样授权。

三级真值：伸入必须报，身体外0–10cm只报告，更远或其他高度计清晰误报。代理分钟与条件提前量不代表真实提醒负担或安全停止。固定模型条件区间不含重训练/校准不确定性；可见支持为零不证明畅通，密输出不增加8×8物理分辨率。无跨源/实机效果；UNKNOWN不等于安全。City、保护test、新UE和硬件第二阶段暂停。

[RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)。历次完整文字保留在Git历史及原结果中。
