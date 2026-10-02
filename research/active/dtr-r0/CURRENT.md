TRAINING_ONLY_GEOMETRY_DIAGNOSTIC

# 前视障碍感知：当前状态

更新：2026-10-03。已消费Development；`V5_FROZEN_COMPLETE / HARDWARE_DEFERRED`。
Status: `DTR_R2_DYNAMIC_RETAINED`（历史保留）。

## 当前发现与下一问题

冻结RAY已学到全局几何，但训练集浅局部拟合仍很差。96既有训练单位、54912查询帧：全部有效ray径向MAE10.09cm [9.75,10.41]，浅查询内及1cm边带局部89.58cm [83.73,96.51]；2cm内仅2.55% [2.26,2.86]（176905 ray分母），42.07%被预测无效。限定1904/2314有EXACT可见支持的浅查询后，误差仍89.59cm；不可归为看不到表面。

teacher距离/valid替入冻结head，浅logit均值变化−0.07895 [−0.11915,−0.03390]、清晰+0.23654 [0.22071,0.25385]。head有响应但该介入含分布偏移，不是部署收益或上界；TD_PV另含invalid零距离占位干预。

下一步优先检验面向公开身体查询的局部可见表面质量预测，直接监督与任务相交的几何信号。尚未训练；最大的未知是稀少局部监督能否稳定学出并改善报警，仍须对同结构对照及M3守住及时率/误停。

[诊断、分母与区间](nearfield/CNH_RAY_SURFACE_GEOMETRY_PROBE_20261003.md)

## 保留报警基线

RAY原固定配方NOT_ESTABLISHED_SINGLE_SEED_DEV，不扩种子或调参。M3/BCEO/RAY浅及时26/25/25（n31），RAY−M3 −3.23pp [−10.34,0]；整体清晰205/214/217次（194.35代理分钟），外侧69/71/74次（40.69分钟）。M3继续保留。[原结果](nearfield/CNH_RAY_SURFACE_RESULTS_20261002.md)

角距边缘化oracle误差约径向量化8.18倍，但几何表示修正未建立报警收益。SURF/HIST/AGG、共享signed场、NEST/MAX/source yaw、原外扩guard与mask失败保留；同批H3单帧小试停止。

## 必要边界

本轮只有训练拟合、半精度teacher及1cm soft查询；射线区间按整单位bootstrap，不把射线当独立试验。可见质量不是碰撞概率，EXACT零支持不证明畅通；valid非实际SNR或UNKNOWN。密输出依赖先验，不增加8×8物理分辨率。

原报警评估96单位7680查询仅1138覆盖0.9m，6542右删失；浅31来自26单位。三级真值：伸入必须报、身体外0–10cm只报告、更远或其他高度才计清晰。代理分钟含首停后时间，提前量条件于已报，不代表人安全停止；无跨源/实机效果。v5保留A2；手机A/A+LOCAL与UNKNOWN语义不变；City、保护test、新UE、硬件第二阶段暂停。

[前版](CURRENT_HISTORY_20261003_PRE_RAY_GEOMETRY.md) · [RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)
