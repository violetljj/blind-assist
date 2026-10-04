MEMORY_POSE_INTERMEDIATE_DEV / LONG_WINDOW_C_NOT_RUN / M3_RETAINED

# 前视障碍感知：当前状态

更新：2026-10-04。已消费Development；`V5_FROZEN_COMPLETE / HARDWARE_DEFERRED`。
Status: `DTR_R2_DYNAMIC_RETAINED`（仅为历史状态）。

## 当前决定

用户授权的记忆A/B测量完成，真实输入合同暂缓。源31浅事件中五漏报有四个目标已出视场，但四个raw8内仍有可见曝光；不能简单归因为窗口短。受控低可见12场景，估计相对位姿oracle12/16在合并0.9–2.1m宏AUC为0.8077/0.8092（正768/负1152），真位姿16为0.9272、M3为0.6609。未到0.85训练门槛、也未到≤0.75停止分支，判为INTERMEDIATE，C不运行。下一问题建议聚焦历史对齐与信息利用，M3保留；不自动开新实验。[本轮诊断与分母](nearfield/CNH_MEMORY_REALITY_20261004.md)

此前U8/U12未知目标尺寸/反射率参照发现条件余量：1.2–2.1m M3/U8/U12宏AUC0.8708/0.9678/0.9804；背景/姿态仍取真值，有限网格不是真实先验观测上界，不自动训练。[完整结果](nearfield/CNH_UNKNOWN_TARGET_REFERENCE_20261004.md)

本轮长窗口训练条件未触发。真实回放仍待设备会话：需串扰/bias标定、ambient、几何记录和空走廊段。现有04逐值复现及两bin贡献诊断不能识别物理成因，也未修复迁移；不继续分析01/04。[回放与评价审计](nearfield/CNH_REPLAY_EVALUATION_AUDIT_20261003.md)

此前受控位移测量：48场景×7位移×4噪声，无新训练；1.2–2.1m已知场景oracle8/12 AUC0.969/0.981，M3平滑0.871。两者可辨，未触发新训练分支；这是已知场景的乐观诊断，不能当未知目标或物理分辨率上界。[完整结果与限制](../../../artifacts.local/work/cnh-displacement-ceiling-20261003/REPORT.md)

## 保留基线与失败

M3保留。原浅及时26/31、清晰首停205次；QMASS浅25/31、清晰222次，拟合改善未转化为报警收益。[QMASS结果](../../../artifacts.local/work/cnh-query-mass-20261003/REPORT.md)。RAY浅25/31、清晰217次；CCON浅23/31、补0丢3，均不采用，不继续原配方搜索。[RAY](nearfield/CNH_RAY_SURFACE_RESULTS_20261002.md) · [CCON](../../../artifacts.local/work/cnh-boundary-contrast-20261003/REPORT.md)

SURF/HIST/AGG、共享signed场、NEST/MAX/source yaw、外扩guard与mask失败保留，同批H3单帧小试停止。v5保留NN+A2；正式增益仅在程序化生成器中成立，手机A/A+LOCAL不据此改动。

## 评价与使用边界

原报警96单位7680查询仅1138覆盖0.9m，6542右删失；31浅事件来自26单位。旧浅条件不适合选3–5pp小收益，功效规划与负例成本见审计，不等于采样授权。

三级真值：伸入必须报，身体外0–10cm只报告，更远或其他高度计清晰误报。代理分钟与条件提前量不代表真实提醒负担或安全停止。固定模型条件区间不含重训练/校准不确定性；可见支持为零不证明畅通，密输出不增加8×8物理分辨率。无跨源/实机效果；UNKNOWN不等于安全。City、保护test、新UE和硬件第二阶段暂停。

[RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)。历次完整文字保留在Git历史及原结果中。
