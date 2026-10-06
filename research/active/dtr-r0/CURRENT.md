COVERAGE_CLOSED_SIM / TRISTATE_WRITING_COMPLETE / M3_RETAINED / HARDWARE_DEFERRED

# 前视障碍感知：当前状态

更新：2026-10-06。已消费模拟Development；`V5_FROZEN_COMPLETE / HARDWARE_DEFERRED`。
Status: `DTR_R2_DYNAMIC_RETAINED`（历史保留）。

## 当前交付

用户确认转向写作。[章节草稿](thesis/CNH_READOUT_COVERAGE_TRISTATE_CHAPTER_20261006.md)已修订：补M3新单位衔接、三项贡献、形式化定义与拒识文献，局限集中一节，来源移附录；[答辩问答](thesis/CNH_DEFENSE_QA_20261006.md)另存。[图清单](thesis/CNH_TRISTATE_FIGURES_20261006.md)含主曲线、预算10模式柱图、方法示意及附录曲线；主张台账保留精确口径。统计图只读CSV，方法图只读配置，无新实验/统计，冻结载荷不变。

[精确事件报告](nearfield/CNH_TRISTATE_EVENT_DEV_20261006.md) `fd55b74d`：0.9 m截止960个episode，single/dual及时909/929；最大τ unknown漏报41/1、静默10/30。3473条采样畅通controls，同≤10静默时unknown时间60.59%/39.77%，差−20.82pp重选区间[−47.25,+0.76]跨零；预算0时87.70%/97.05%，方向反转。旧978代理不混入分母，不能写dual整体更安全。

最大τ残余静默single10/10、dual26/30在转弯；真参考核心gap8/10、25/30，与未检查区代理交叠，不作读出单因果归因。早期S1/S2/S3正式AP比较与后续M3学习读出分开；v4接受的执行偏差仍披露。

## 保留决定

[覆盖线收口](nearfield/CNH_COVERAGE_CLOSEOUT_20261005.md)，M3继续冻结。外参增强接续NOT_SUPPORTED：转弯query首停57/1299对single24/1299，2.375倍超过1.5倍门；原不可达严格增益FAIL和事后修订PASS同时保留。[完整报告](nearfield/CNH_EXTRINSIC_AUG_20261006.md)

读出R、T2、位姿平均、融合及外参各按本轮停止规则结束，不扩大为永久禁训。[读出收尾](nearfield/CNH_READOUT_CLOSEOUT_20261004.md)。低可见组仍曾曝光，近距持续支持是机制问题。[曝光拆分](nearfield/CNH_FOV_FAILURE_SPLIT_20261005.md)

## 下一步与权限边界

本轮写作完成，后续按用户反馈修订章节。转弯的真实姿态替换/路径曲率因素拆分只列未来方案，NOT_RUN；不得继承过去预算自动开算。最终统一确认批须另固定episode、截止、controls及独立采样/模式深度分布，规模NOT_FIXED。正式证书暂缓，1.5 m重标不执行，GPU新批未启动。

City、保护test、新UE采集和硬件第二阶段仍暂停；真实回放待设备会话，不继续01/04分析。UNKNOWN不是安全，代理分钟不是人体负担，oracle不是物理上界，无跨源/实机结论。

[RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)。整理前全文保留于Git revision `fd55b74dcf9567be17903fa527da11f56a94da04` 的本页；历史待决状态不恢复授权。
