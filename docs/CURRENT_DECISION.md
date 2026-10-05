# 当前研究决定

更新：2026-10-05。唯一研究主线：盲杖互补的前视障碍感知。
最新授权与结果（2026-10-06）：三态r3完成描述性重放；冻结噪声m=3°、自检99.25%、核心97.45%，最大τ single A/B错畅通2067/61560、1880/58765，dual 3829/125317、3369/117230；建议α=5%、A主变体、约1500认证unit+独立审计，400仅试点；无证书、GPU批未启动，r1/r2保留。[开发报告](../research/active/dtr-r0/nearfield/CNH_TRISTATE_DEV_20261006.md)
Status: `L10_R0_PAUSED / DTR_R2_DYNAMIC_RETAINED`（历史保留）；ToF阶段：`V5_FROZEN_COMPLETE / HARDWARE_DEFERRED`。

## 当前决定与瓶颈

**覆盖线收口，保留冻结M3。** 现有模拟证据整理成章，不再追加融合、阈值或增强优化。[章节证据链](../research/active/dtr-r0/nearfield/CNH_COVERAGE_CLOSEOUT_20261005.md)。外参增强接续仍判NOT_SUPPORTED：转弯查询首停57/1299对single-M3的24/1299，超过1.5倍上限；其余七项通过。原6/6严格增加的试点FAIL与用户批准事后修订所得PASS均保留，评价门槛未改。完整分母、阈值和来源见[路线当前页](../research/active/dtr-r0/CURRENT.md)及[报告](../research/active/dtr-r0/nearfield/CNH_EXTRINSIC_AUG_20261006.md)。

单传感器近距覆盖和持续曝光不足仍是瓶颈；双路几何覆盖收益伴随转弯代价。软件覆盖提示仅保留UNKNOWN/兜底研究价值，合成几何改善不等于人体可执行或硬件收益。FOV_OUT不等于全程没有信息：低可见88条均曾曝光，重点是近距支持和证据如何使用。[覆盖拆分](../research/active/dtr-r0/nearfield/CNH_FOV_FAILURE_SPLIT_20261005.md)

## 有效授权与停止范围

本轮只整理覆盖线和三态待审方案，不运行实验或提升运行时模型默认。原试点4小时及接续150分钟预算均随各轮结束，不转作后续额度。后续问题、预算、可调整范围和交付物明确后，范围内自主完成执行、修复、验证与交付；变更冻结指标须明确记录依据和授权，不能把看过的数据重新称作新鲜确认。

读出R低分支、T2优化、固定M3位姿平均、公开融合搜索及外参接续各按本轮停止规则结束；这些停止不构成所有新机制或训练的永久禁令。原模型、失败、修订前后判读及载荷保留，不按99000结果再调阈值。

City、保护test、新UE采集和硬件第二阶段仍暂停。真实回放待串扰/bias、ambient、几何及空走廊设备会话，不继续01/04分析。手机保持A及手动开始A+LOCAL流程。[设备状态](PROJECT_STATE.md)

## 下一待决问题

优先审[三态输出方案](../research/active/dtr-r0/nearfield/CNH_TRISTATE_DRAFT_20261005.md)：先在旧数据描述性比较畅通许可、错误畅通和unknown，不承诺统计保证；正式保证另需确定风险分母、独立校准unit与标签时窗。普通保形边际覆盖不等于“输出畅通时危险率≤α”。

增强三seed标为可复用开发候选，M3继续冻结参照；单路浅29/32对28/32、深137/141对138/141、物理首停81/825对104/825，不宣称统计非劣。三态首个方案保留M3单/双对照，不把工作读出升级与语义试算绑定；历史V不能隔离yaw作用。

## 结论边界与入口

当前新结果均为模拟Development；旧角容限重复使用观测，oracle含条件性真值。三级真值为侵入身体走廊必须报、身体外0–10cm擦身只报告、更远或另一高度计清晰误报。代理分钟不等于真实提醒负担，UNKNOWN不证明安全；尚无跨源/实机效果或ToF物理上限结论。v4接受的执行偏差和旧失败不改判。

[路线证据](../research/active/dtr-r0/CURRENT.md) · [主张台账](../research/active/dtr-r0/THESIS_CLAIMS_20260927.md) · [运行日志](../research/active/dtr-r0/RUNS.md) · [工作流](../research/WORKFLOW.md)

整理前全文：Git revision `4f174009da62a8fcd9f219bb6758375f3f1ce2aa` 的 `docs/CURRENT_DECISION.md`；历史结果按该版本链接追溯，旧授权与待决状态不覆盖本页。
