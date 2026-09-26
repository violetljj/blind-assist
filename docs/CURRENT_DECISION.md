# 当前研究决定

更新：2026-09-26。主线：盲杖互补的前视障碍感知。

Status: `L10_R0_PAUSED / DTR_R2_DYNAMIC_RETAINED`（历史DTR保留）。

## 当前决定

- 手机继续保留A基线与原首页→手动开始A+LOCAL→结束返回首页。研究组件不自动替换App；UNKNOWN不等于无障碍。
- Python Track A v1.2已停止（失败保留）。v1.3试采（用户授权，现实尺寸+25%小目标层，轨迹重抽）12/12单位完成，G0–G4通过，G5未运行；描述性读出见[v1.3结果](../research/active/dtr-r0/nearfield/CNH_TRACK_A_V13_PILOT_RESULTS_20260926.md)。未放量。
- 旧巷道仅作重复Development诊断。扫描收益已正式复现，但不能据此主张独立泛化、RGB收益或ToF信息上限。City、保护test、UE/RGB与硬件第二阶段均暂停。
- 采用[两级流程](../AGENTS.md)：诊断/预检/工程/试采走快速道，不登记或另写协议，报告≤1页；改变主线、基线或产生论文数字走正式道。v1.3起试采G2计数/组合短缺只报告；G0/G1及放量门槛仍硬停止，v1.2不追改。

## 当前关键数字

|证据|结果与分母|当前用途|
|---|---|---|
|受控A / A OR LOCAL|TP118 / 159，FP7 / 22；224正、352负|增量与误报成本，非自然分布|
|旧巷道扫描raw→H3 S2|AP0.660945；TP190/408，FP5/2472|消费Development方向证据|
|Track A v1.2|105配置、1260几何帧；3完整+1部分单位；传感帧0|生成206秒后停止，G3–G5/audit读出NOT_RUN|
|**Track A 放量 v3（正式，63 独立 audit 单位，M1/SNR6/5Hz/带噪自运动）**|六个 Holm 主检验全部成立：S2−B1-R HEAD/BODY +0.099/+0.079，S1−B0 +0.077/+0.062，S3−S2 +0.005/+0.068；宏 AP S3 0.756/0.749|[结果](../research/active/dtr-r0/nearfield/CNH_TRACK_A_SCALE_V3_RESULTS_20260926.md)；受控仿真，SNR6 经 ZJUL5 粗锚定|
|Track A v1.3试采（M1/SNR6/5Hz/带噪自运动）|audit 4单位宏平均AP HEAD/BODY：B1-R 0.652/0.624，S2 0.759/0.681，S3 0.770/0.733；1081/1051正查询|描述性；S2−B1-R 8条件×两组全为正，非显著性结论|
|v1.2指纹|98非空配置，重复0|已有部分的完整性，不代表12单位完成|
|完整保护test|真实布局/spec 0/164|继续暂停，不用3张候选巷道图替代|

## 待决问题

放量 v3 已完成并通过主判定（v1、v2 失败记录保留）。已决（2026-09-26，用户）：“确认畅通距离”降为论文附录的辅助可确认性地图（承诺 ≥10 cm、ρ≥0.5、最坏摆放），不作核心能力、不做正式检验，见 [CNH_CLEARANCE_RULEFIX_DEV_20260926](../research/active/dtr-r0/nearfield/CNH_CLEARANCE_RULEFIX_DEV_20260926.md)；当前无硬件，默认全部以模拟推进，L8CH 桌面标定待硬件可用时再排（真机 CNH 可配 8×8×16 bin，5468/6160 字节，满足 H3）。待决：头部近处是否启动相机线（可见性诊断：HEAD 视场外正例 93% 在记忆窗口内被照到过，从未照到仅 0.3%，记忆上限约 +0.02 AP；头部差距主要在视场内小目标，AP 0.27，见 [CNH_HEAD_MEMORY_VISIBILITY_DEV_20260926](../research/active/dtr-r0/nearfield/CNH_HEAD_MEMORY_VISIBILITY_DEV_20260926.md)；信号上限诊断（条件性）：S2 在强信号段已接近理想，漏检集中在 z<2 的弱信号段（HEAD 可见正例 25%，小块 54%）；已知模板理想检测器 HEAD 召回 0.80，S2 为 0.62，差距能否收回尚未测量，见 [CNH_SIGNAL_CEILING_DEV_20260926](../research/active/dtr-r0/nearfield/CNH_SIGNAL_CEILING_DEV_20260926.md)；事件级：走到 1 m 内的 HEAD 障碍 76% 及时报警、始终没报 5%（小块 53%/15%），BODY 始终没报 21%，序列假警 33%，见 [CNH_EVENT_CEILING_DEV_20260926](../research/active/dtr-r0/nearfield/CNH_EVENT_CEILING_DEV_20260926.md)；提醒策略：同 10% 假警下单帧提阈优于连续 N 帧，HEAD 报警率 0.78、及时率 0.77，见 [CNH_ALERT_PERSISTENCE_DEV_20260926](../research/active/dtr-r0/nearfield/CNH_ALERT_PERSISTENCE_DEV_20260926.md)）。真实传感器计数/串扰与安装参数仍待标定，SNR 结论按 SNR3/6/12 档报告；算法未达标不能推断物理信息上限。RGB线和实机方案另行决定。

## 证据与历史

[路线当前页](../research/active/dtr-r0/CURRENT.md) · [扫描全表](../research/active/dtr-r0/nearfield/CNH_SCAN_DEVELOPMENT_RESULTS_20260926.md) · [v1.2回执](../research/active/dtr-r0/nearfield/CNH_TRACK_A_V12_PILOT_RESULTS_20260926.md) · [LOCAL对照](../research/active/dtr-r0/nearfield/LOCAL_RESCUE_RESULTS_20260923.md) · [test准备度](../research/active/dtr-r0/nearfield/CNH_FULL_TEST_READINESS_20260925.md)

[压缩前完整原文](operations/snapshots/CURRENT_DECISION_20260926_PRE_FASTLANE.md)逐字节保留此前正文；旧决定与pending按当时日期解读。项目入口见[PROJECT_STATE](PROJECT_STATE.md)。本页仅在决定变化时更新。
