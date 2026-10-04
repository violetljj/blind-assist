TEMPORAL_PILOT_INTERMEDIATE_DEV / M3_RETAINED

# 前视障碍感知：当前状态

更新：2026-10-04。已消费Development；`V5_FROZEN_COMPLETE / HARDWARE_DEFERRED`。
Status: `DTR_R2_DYNAMIC_RETAINED`（仅为历史状态）。

## 当前决定

V/T训练试点完成：新48场景主AUC M3/V/T/D=.8683/.8944/.6031/.9591；V−M3+.0260 [.0169,.0350]、补上28.7%，自然浅26/31、深163/164、清晰212/194.35min，护栏通过但未到+.03。T失败；INTERMEDIATE，保留M3，不追加训练。各3seed/8epoch与480训练场景完整，主计算48.1min。[结果、种子及分层](nearfield/CNH_TEMPORAL_READOUT_PILOT_20261004.md)。已知背景/目标仍是D特权；[前置2×2](nearfield/CNH_POSE_FACTORIAL_20261004.md)。

另一路位姿容错参照按3600s计算预算停止，25/48完成（高19/低6）。低组已完成子集合并宏AUC：普通12帧.7811、容错12帧.7968、容错8帧.7151；仅运行时选择的部分描述，余23缺失，不作全48/误停预算/训练判读。该测量未训练，M3保留。[部分结果与续跑状态](nearfield/CNH_POSE_MARGINAL_REFERENCE_20261004.md)。

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
