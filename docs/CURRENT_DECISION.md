# 当前研究决定

更新：2026-10-06。唯一研究主线：盲杖互补的前视障碍感知。
用户接受既有章节修订；新授权CPU≤1h[mode2支路诊断](../research/active/dtr-r0/nearfield/CNH_MODE2_SEAM_DEV_20261006.md)完成，不支持简单水平交界切分。26个dual漏报中20个single及时；相关目标表面均有一支路覆盖全部水平范围。M3查询得到精确当前head-to-travel信息，部署需估计，已补[章节局限](../research/active/dtr-r0/thesis/CNH_READOUT_COVERAGE_TRISTATE_CHAPTER_20261006.md)。mode2同时转弯与唯一持续头部正视，两因素混淆；single按模式未及时33/8/10、dual2/3/26，不能用静默集中归因转弯。原960事件/3473controls、预算区间及图保留。[事件报告](../research/active/dtr-r0/nearfield/CNH_TRISTATE_EVENT_DEV_20261006.md)
Status: `L10_R0_PAUSED / DTR_R2_DYNAMIC_RETAINED`（历史保留）；ToF阶段：`V5_FROZEN_COMPLETE / HARDWARE_DEFERRED`。

## 当前决定与瓶颈

**覆盖线收口，保留冻结M3。** 现有模拟证据整理成章，不再追加融合、阈值或增强优化。[章节证据链](../research/active/dtr-r0/nearfield/CNH_COVERAGE_CLOSEOUT_20261005.md)。外参增强接续仍判NOT_SUPPORTED：转弯查询首停57/1299对single-M3的24/1299，超过1.5倍上限；其余七项通过。原6/6严格增加的试点FAIL与用户批准事后修订所得PASS均保留，评价门槛未改。完整分母、阈值和来源见[路线当前页](../research/active/dtr-r0/CURRENT.md)及[报告](../research/active/dtr-r0/nearfield/CNH_EXTRINSIC_AUG_20261006.md)。

单传感器近距覆盖和持续曝光不足仍是瓶颈；双路几何覆盖收益伴随转弯代价。软件覆盖提示仅保留UNKNOWN/兜底研究价值，合成几何改善不等于人体可执行或硬件收益。FOV_OUT不等于全程没有信息：低可见88条均曾曝光，重点是近距支持和证据如何使用。[覆盖拆分](../research/active/dtr-r0/nearfield/CNH_FOV_FAILURE_SPLIT_20261005.md)

## 有效授权与停止范围

CPU≤1h查询核实/模式2诊断与章节文字修订完成；原CPU≤3h姿态两臂保留，模型/在线门/报警阈值未改。用户明确不批准按原角速度延长轨迹，曲线两臂未执行并停止：门几乎放行、换门不补及时报警、查询已有精确相对方向、直线接触真值与曲线检查不自洽。本轮及原4h/150min预算各随交付结束，不转作后续额度。新问题、预算和范围明确后自主完成；不能把已看过数据重新称作新鲜确认。

读出R低分支、T2优化、固定M3位姿平均、公开融合搜索及外参接续各按本轮停止规则结束；这些停止不构成所有新机制或训练的永久禁令。原模型、失败、修订前后判读及载荷保留，不按99000结果再调阈值。

City、保护test、新UE采集和硬件第二阶段仍暂停。真实回放待串扰/bias、ambient、几何及空走廊设备会话，不继续01/04分析。手机保持A及手动开始A+LOCAL流程。[设备状态](PROJECT_STATE.md)

## 下一待决问题

当前诊断削弱简单水平交界解释，不确认垂直FOV、信号强度或读出原因。前融合与偏角扫描只是待验证候选，未启动推理/训练；要拆开mode2转弯与正视，可在最终统一确认事前配方加入“直行且持续正视”，当前不渲染。正式证书暂缓、1.5m重标不执行；新确认批先固定方法、事件、controls和独立采样分布/规模，不沿用旧按帧1500unit建议。

增强三seed标为可复用开发候选，M3继续冻结参照；单路浅29/32对28/32、深137/141对138/141、物理首停81/825对104/825，不宣称统计非劣。三态首个方案保留M3单/双对照，不把工作读出升级与语义试算绑定；历史V不能隔离yaw作用。

## 结论边界与入口

当前新结果均为模拟Development；旧角容限重复使用观测，oracle含条件性真值。三级真值为侵入身体走廊必须报、身体外0–10cm擦身只报告、更远或另一高度计清晰误报。代理分钟不等于真实提醒负担，UNKNOWN不证明安全；尚无跨源/实机效果或ToF物理上限结论。v4接受的执行偏差和旧失败不改判。

[路线证据](../research/active/dtr-r0/CURRENT.md) · [主张台账](../research/active/dtr-r0/THESIS_CLAIMS_20260927.md) · [运行日志](../research/active/dtr-r0/RUNS.md) · [工作流](../research/WORKFLOW.md)

整理前全文：Git revision `4f174009da62a8fcd9f219bb6758375f3f1ce2aa` 的 `docs/CURRENT_DECISION.md`；历史结果按该版本链接追溯，旧授权与待决状态不覆盖本页。
