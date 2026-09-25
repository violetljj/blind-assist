# 当前研究决定

更新：2026-09-26。主线：盲杖互补的前视障碍感知。

Status: `L10_R0_PAUSED / DTR_R2_DYNAMIC_RETAINED`（历史DTR保留）。

## 当前决定

- 手机继续保留A基线与原首页→手动开始A+LOCAL→结束返回首页。研究组件不自动替换App；UNKNOWN不等于无障碍。
- Python Track A v1.2已停止：3个完整单位通过G2，单位3/配置9的32个几何候选均规划不可行。保留失败，不换轨迹、不做第三次生成或放量。
- 旧巷道仅作重复Development诊断。扫描收益已正式复现，但不能据此主张独立泛化、RGB收益或ToF信息上限。City、保护test、UE/RGB与硬件第二阶段均暂停。
- 采用[两级流程](../AGENTS.md)：诊断/预检/工程/试采走快速道，不登记或另写协议，报告≤1页；改变主线、基线或产生论文数字走正式道。v1.3起试采G2计数/组合短缺只报告；G0/G1及放量门槛仍硬停止，v1.2不追改。

## 当前关键数字

|证据|结果与分母|当前用途|
|---|---|---|
|受控A / A OR LOCAL|TP118 / 159，FP7 / 22；224正、352负|增量与误报成本，非自然分布|
|旧巷道扫描raw→H3 S2|AP0.660945；TP190/408，FP5/2472|消费Development方向证据|
|Track A v1.2|105配置、1260几何帧；3完整+1部分单位；传感帧0|生成206秒后停止，G3–G5/audit读出NOT_RUN|
|v1.2指纹|98非空配置，重复0|已有部分的完整性，不代表12单位完成|
|完整保护test|真实布局/spec 0/164|继续暂停，不用3张候选巷道图替代|

## 待决问题

下一版生成计划如何处理固定轨迹下的配额不可行；是否授权新试采。当前没有v1.3生成或放量授权。真实传感器计数/串扰与安装参数仍待标定；算法未达标不能推断物理信息上限。RGB线和实机方案另行决定。

## 证据与历史

[路线当前页](../research/active/dtr-r0/CURRENT.md) · [扫描全表](../research/active/dtr-r0/nearfield/CNH_SCAN_DEVELOPMENT_RESULTS_20260926.md) · [v1.2回执](../research/active/dtr-r0/nearfield/CNH_TRACK_A_V12_PILOT_RESULTS_20260926.md) · [LOCAL对照](../research/active/dtr-r0/nearfield/LOCAL_RESCUE_RESULTS_20260923.md) · [test准备度](../research/active/dtr-r0/nearfield/CNH_FULL_TEST_READINESS_20260925.md)

[压缩前完整原文](operations/snapshots/CURRENT_DECISION_20260926_PRE_FASTLANE.md)逐字节保留此前正文；旧决定与pending按当时日期解读。项目入口见[PROJECT_STATE](PROJECT_STATE.md)。本页仅在决定变化时更新。
