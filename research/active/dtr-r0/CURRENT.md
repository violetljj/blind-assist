COVERAGE_CLOSED_SIM / MODE2_SEAM_NOT_SUPPORTED_DEV / QUERY_CONDITION_DISCLOSED / M3_RETAINED / HARDWARE_DEFERRED

# 前视障碍感知：当前状态

更新：2026-10-06。已消费模拟Development；`V5_FROZEN_COMPLETE / HARDWARE_DEFERRED`。
Status: `DTR_R2_DYNAMIC_RETAINED`（历史保留）。

## 当前交付

用户确认转向写作。[章节草稿](thesis/CNH_READOUT_COVERAGE_TRISTATE_CHAPTER_20261006.md)已修订：补M3新单位衔接、三项贡献、形式化定义与拒识文献，局限集中一节，来源移附录；[答辩问答](thesis/CNH_DEFENSE_QA_20261006.md)另存。[图清单](thesis/CNH_TRISTATE_FIGURES_20261006.md)含主曲线、预算10模式柱图、方法示意及附录曲线；主张台账保留精确口径。统计图只读CSV，方法图只读配置，无新实验/统计，冻结载荷不变。

[精确事件报告](nearfield/CNH_TRISTATE_EVENT_DEV_20261006.md) `fd55b74d`：0.9 m截止960个episode，single/dual及时909/929；最大τ unknown漏报41/1、静默10/30。3473条采样畅通controls，同≤10静默时unknown时间60.59%/39.77%，差−20.82pp重选区间[−47.25,+0.76]跨零；预算0时87.70%/97.05%，方向反转。旧978代理不混入分母，不能写dual整体更安全。

mode2（转弯且唯一正视）集中残余静默single10/10、dual26/30；按模式未及时single33/8/10、dual2/3/26。[交界诊断](nearfield/CNH_MODE2_SEAM_DEV_20261006.md)不支持视场交界切分；浅侵入25/26，10/26最后输出时目标未入直线查询，停止追加诊断。报警查询用精确当前head-to-travel，部署需估计。[主动扫视](nearfield/CNH_ACTIVE_SCAN_DEV_20261006.md)（369事件/1255对照）：头部正对行进时单路及时366/静默3/unknown0.15%，优于被动双路354/15/1.66%；闭环提示0.6 s及时365（较被动单路+16[9,23]），负担与提示频率取决于假设的回头行为；直行正视双路未及时2/234、转弯12/135，混淆描述性指向转弯。[查询方向](nearfield/CNH_QUERY_DIRECTION_DEV_20261006.md)：M3对head-to-travel极敏感（±10°单路及时349→188/204）；1 s位移估计直行无损、转弯滞后3.2°使mode2单路及时133→106，滞后补偿（0.58°）恢复到135。[方向不确定性](nearfield/CNH_HEADING_UNCERTAINTY_DEV_20261007.md)：误差RMS 2.3/4.5/9.1°时单路及时349→339/318/287、双路354→352/338/316；查询并集在相同误报下不优于降阈值；并集判畅通把静默压回精确水平（双路7/11/13）。早期S1/S2/S3与M3分开，v4偏差保留。

## 保留决定

[覆盖线收口](nearfield/CNH_COVERAGE_CLOSEOUT_20261005.md)，M3继续冻结。外参增强接续NOT_SUPPORTED：转弯query首停57/1299对single24/1299，2.375倍超过1.5倍门；原不可达严格增益FAIL和事后修订PASS同时保留。[完整报告](nearfield/CNH_EXTRINSIC_AUG_20261006.md)

读出R、T2、位姿平均、融合及外参各按本轮停止规则结束，不扩大为永久禁训。[读出收尾](nearfield/CNH_READOUT_CLOSEOUT_20261004.md)。低可见组仍曾曝光，近距持续支持是机制问题。[曝光拆分](nearfield/CNH_FOV_FAILURE_SPLIT_20261005.md)

## 下一步与权限边界

用户接受既有章节修订，本次CPU≤1h查询核实/支路诊断及模式措辞修订完成；原CPU≤3h姿态两臂保留。用户明确不批准合成路径延长，曲线两臂未执行并停止：门几乎放行、换门不补及时报警、报警查询已有精确相对方向且直线接触真值与曲线检查不同。前融合/偏角扫描、新“直行且持续正视”模式均未启动；后者可列最终统一确认事前配方，但须先固定方法、事件、controls及独立采样/深度分布，规模NOT_FIXED。正式证书暂缓，1.5m重标不执行；本轮预算随交付结束，不启动GPU新批。

City、保护test、新UE采集和硬件第二阶段仍暂停；真实回放待设备会话，不继续01/04分析。UNKNOWN不是安全，代理分钟不是人体负担，oracle不是物理上界，无跨源/实机结论。

[RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)。整理前全文保留于Git revision `fd55b74dcf9567be17903fa527da11f56a94da04` 的本页；历史待决状态不恢复授权。
