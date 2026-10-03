BOUNDARY_CONTRAST_NOT_ESTABLISHED_SINGLE_SEED_DEV

# 前视障碍感知：当前状态

更新：2026-10-03。已消费Development；`V5_FROZEN_COMPLETE / HARDWARE_DEFERRED`。
Status: `DTR_R2_DYNAMIC_RETAINED`（历史保留）。

## 当前发现与下一问题

浅擦碰/外侧清晰的分界对比小试完成。CBASE/CCON同结构、seed0各10轮，同初始化/基础顺序/额外样本：每步均加8对训练样本及0.5倍pair BCE，仅CCON再加margin1排序损失。maps仅为latent。954对按运动模式、时刻、查询、高度背景及前距匹配，来自不同场景；负例全13时刻全物体清晰，0–10cm擦身不当负例。不是同物体位移反事实。

48校准/96评估，M3阈值冻结，新臂各按1清晰首停/代理分钟校准。浅及时M3/CBASE/CCON为26/24/23（n31、26单位），深162/159/158（n164）。CCON对M3浅补0丢3，差−9.68pp [−20.70,0]；对CBASE补0丢1。清晰首停205/212/207（194.35代理分钟），同高度外10–20cm69/77/76（40.69分钟）；CCON−M3外侧+0.1720 [−0.1312,0.4814]次/分钟。两新臂均不采用，保留M3。独立raw计数与来源核验188项通过。

训练匹配对M3已正确排序945/954，两新臂均954/954且margin≥1。此训练任务较易，拟合仍未转化为及时报警；不能据此归因所有失败或宣称ToF无信息。下一方向优先同场景受控位移构造真正相邻的难例，检验分界信息；不继续本损失配方的权重搜索。训练循环两臂合计190.56秒，144单位推理31.23秒（RTX5060 Laptop），非端到端成本。

[本轮结果与复现](../../../artifacts.local/work/cnh-boundary-contrast-20261003/REPORT.md) · [配对](nearfield/cnh_boundary_contrast_data.py) · [训练/推理](nearfield/cnh_boundary_contrast_train.py) · [评价](nearfield/cnh_boundary_contrast_evaluate.py)

## 保留报警基线

局部支持QMASS训练非零格MAE0.3643→0.1830，却浅25/31、整体清晰222次、外侧79次，未赢M3。[结果](../../../artifacts.local/work/cnh-query-mass-20261003/REPORT.md)。RAY浅25/31、整体清晰217次、外侧74次，原固定配方失败保留。[原结果](nearfield/CNH_RAY_SURFACE_RESULTS_20261002.md)。RAY浅局部径向MAE89.58cm；teacher替入冻结head含分布偏移，不作部署上界。[诊断](nearfield/CNH_RAY_SURFACE_GEOMETRY_PROBE_20261003.md)

角距边缘化oracle误差约径向量化8.18倍，但几何表示修正未建立报警收益。SURF/HIST/AGG、共享signed场、NEST/MAX/source yaw、原外扩guard与mask失败保留；同批H3单帧小试停止。

## 必要边界

本轮复用96训练单位，评价区间按96整单位bootstrap，条件于单种子模型和校准阈值，不含重训练/校准不确定性。负样本重复使用且物体不同，配对不支持因果归因。可见支持不是碰撞概率，零支持不证明畅通；valid非实际SNR或UNKNOWN。密输出依赖先验，不增加8×8物理分辨率。

原报警评估96单位7680查询仅1138覆盖0.9m，6542右删失；浅31来自26单位。三级真值：伸入必须报、身体外0–10cm只报告、更远或其他高度才计清晰。代理分钟含首停后时间，提前量条件于已报，不代表人安全停止；无跨源/实机效果。v5保留A2；手机A/A+LOCAL与UNKNOWN语义不变；City、保护test、新UE、硬件第二阶段暂停。

[前版](CURRENT_HISTORY_20261003_PRE_RAY_GEOMETRY.md) · [RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)
