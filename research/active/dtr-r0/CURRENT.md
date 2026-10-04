READOUT_NEAR_LOCATION_FREE_LIMIT / M3_RETAINED

# 前视障碍感知：当前状态

更新：2026-10-04。已消费Development；`V5_FROZEN_COMPLETE / HARDWARE_DEFERRED`。
Status: `DTR_R2_DYNAMIC_RETAINED`（仅为历史状态）。

## 当前决定

未知位置参照R完整48场景/K4完成：主AUC R_any/VD=.8877/.9012，差−.0135 [−.0378,.0095]；同清晰16/384，视场外及时22/26（分母88，补7丢11），辅助R_query25/88、AUC.8988。命中有限参照低分支，结束当前读出搜索，保留M3、不改VD旧采用门槛；下一问题建议视场覆盖几何。预算阶段44.54min，主计算34.28min；半步NOT_RUN_BRANCH。D→R同时移除真实尺寸/rho，none仍知地面/后墙，不能宣称传感器极限。[完整结果](nearfield/CNH_LOCATION_REFERENCE_20261004.md)。

此前固定M3位姿平均未改善报警：同清晰18/384、浅309/384不变，视场外32/96不变；固定配方停止，自然候选未运行。[结果与边界](nearfield/CNH_M3_POSE_ENSEMBLE_20261004.md)。

T2诊断MIXED：从零CVR训练优于T2，留出排序高+.0444但损失更差。T2可fit256行、2048行平台未过；优化/泛化与表示未分离。公平V三对M3三改善+.0303 [.0221,.0397]，原试点判定不改。无新网络/候选，M3保留。[完整诊断](nearfield/CNH_T2_FIT_DIAGNOSTIC_20261004.md) · [原试点2](nearfield/CNH_READOUT_PILOT2_20261005.md)。

已知场景参照旧批低组容错12及时61/96（M3 32/96）；61次首报仍可见，60次≥2.1m，非延窗记忆收益；未触发旧C。[完整结果](nearfield/CNH_POSE_MARGINAL_REFERENCE_COMPLETE_20261004.md)。

旧低可见12场景估计位姿12/16帧AUC .8077/.8092，未触发训练，不能证明窗口短。[记忆诊断](nearfield/CNH_MEMORY_REALITY_20261004.md)。此前U8/U12未知目标参照 .9678/.9804（M3 .8708），真背景/姿态条件余量保留。[未知目标结果](nearfield/CNH_UNKNOWN_TARGET_REFERENCE_20261004.md)

真实回放待串扰/bias、ambient、几何和空走廊设备会话；04复现未修复迁移，不继续01/04分析。[回放与评价审计](nearfield/CNH_REPLAY_EVALUATION_AUDIT_20261003.md)

48场景×7位移×K4的源测量与失败均保留；oracle不是物理上界。[源位移测量](../../../artifacts.local/work/cnh-displacement-ceiling-20261003/REPORT.md)

## 保留基线与失败

M3保留。原浅及时26/31、清晰首停205次；QMASS浅25/31、清晰222次，拟合改善未转化为报警收益。[QMASS结果](../../../artifacts.local/work/cnh-query-mass-20261003/REPORT.md)。RAY浅25/31、清晰217次；CCON浅23/31、补0丢3，均不采用，不继续原配方搜索。[RAY](nearfield/CNH_RAY_SURFACE_RESULTS_20261002.md) · [CCON](../../../artifacts.local/work/cnh-boundary-contrast-20261003/REPORT.md)

SURF/HIST/AGG、共享signed场、NEST/MAX/source yaw、外扩guard与mask失败保留，同批H3单帧小试停止。v5保留NN+A2；正式增益仅在程序化生成器中成立，手机A/A+LOCAL不据此改动。

## 评价与使用边界

原报警96单位7680查询仅1138覆盖0.9m，6542右删失；31浅事件来自26单位。旧浅条件不适合选3–5pp小收益，功效规划与负例成本见审计，不等于采样授权。

三级真值：伸入必须报，身体外0–10cm只报告，更远或其他高度计清晰误报。代理分钟与条件提前量不代表真实提醒负担或安全停止。固定模型条件区间不含重训练/校准不确定性；可见支持为零不证明畅通，密输出不增加8×8物理分辨率。无跨源/实机效果；UNKNOWN不等于安全。City、保护test、新UE和硬件第二阶段暂停。

[RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)。历次完整文字保留在Git历史及原结果中。
