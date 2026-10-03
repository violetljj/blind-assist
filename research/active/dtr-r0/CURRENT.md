QUERY_MASS_NOT_ESTABLISHED_SINGLE_SEED_DEV

# 前视障碍感知：当前状态

更新：2026-10-03。已消费Development；`V5_FROZEN_COMPLETE / HARDWARE_DEFERRED`。
Status: `DTR_R2_DYNAMIC_RETAINED`（历史保留）。

## 当前发现与下一问题

查询局部可见支持量小试已完成。native CNH加因果姿态、公开变换与查询角点，直接预测16×16微格支持并联合训练报警；同结构QBCE仅报警损失，QMASS加稀疏支持监督，两臂seed0、同初始化/顺序各10轮。训练非零支持80884/14057472格（0.58%）；支持格MAE从QBCE 0.3643降至QMASS 0.1830，零支持格0.07815→0.01795。是训练集归一化积分量拟合，不是厘米距离精度或独立泛化。

48校准/96评估，原M3阈值冻结，新臂各按1清晰首停/代理分钟校准。浅及时M3/QBCE/QMASS为26/24/25（n31、26单位）；QMASS对QBCE补1丢0、对M3补0丢1，后者差−3.23pp [−10.71,0]。深及时162/160/162（n164）。清晰首停205/199/222（194.35代理分钟）；QMASS−M3 +0.0875 [0.0101,0.1655]次/分钟，同高度外10–20cm69/66/79（40.69分钟），差+0.2458 [0.0230,0.4793]。不采用QMASS，M3保留；局部拟合改善尚未转化为报警增益。

下一问题是局部支持信号能否区分浅擦碰与外侧清晰；本轮不凭训练误差下降继续扩种子。旧RAY失败与诊断保留：训练浅局部径向MAE89.58cm；teacher替入冻结head浅分数下降、清晰上升，含分布偏移，不能作部署上界。新小试不关闭其他局部表征机制。

[本轮结果与复现](../../../artifacts.local/work/cnh-query-mass-20261003/REPORT.md) · [训练/推理](nearfield/cnh_query_mass_train.py) · [评价](nearfield/cnh_query_mass_evaluate.py) · [旧RAY诊断](nearfield/CNH_RAY_SURFACE_GEOMETRY_PROBE_20261003.md)

## 保留报警基线

RAY原固定配方NOT_ESTABLISHED_SINGLE_SEED_DEV，不扩种子或调参。M3/BCEO/RAY浅及时26/25/25（n31），RAY−M3 −3.23pp [−10.34,0]；整体清晰205/214/217次（194.35代理分钟），外侧69/71/74次（40.69分钟）。M3继续保留。[原结果](nearfield/CNH_RAY_SURFACE_RESULTS_20261002.md)

角距边缘化oracle误差约径向量化8.18倍，但几何表示修正未建立报警收益。SURF/HIST/AGG、共享signed场、NEST/MAX/source yaw、原外扩guard与mask失败保留；同批H3单帧小试停止。

## 必要边界

本轮复用96训练单位，评价区间按96整单位bootstrap，条件于单种子模型和校准阈值，不含重训练/校准不确定性。teacher为半精度距离投影到1cm soft查询；可见支持不是碰撞概率，零支持不证明畅通；valid非实际SNR或UNKNOWN。密输出依赖先验，不增加8×8物理分辨率。

原报警评估96单位7680查询仅1138覆盖0.9m，6542右删失；浅31来自26单位。三级真值：伸入必须报、身体外0–10cm只报告、更远或其他高度才计清晰。代理分钟含首停后时间，提前量条件于已报，不代表人安全停止；无跨源/实机效果。v5保留A2；手机A/A+LOCAL与UNKNOWN语义不变；City、保护test、新UE、硬件第二阶段暂停。

[前版](CURRENT_HISTORY_20261003_PRE_RAY_GEOMETRY.md) · [RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)
