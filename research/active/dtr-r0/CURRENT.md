REAL_HEAD_CONFIRMED_SIM / WRITING / M3_RETAINED / HARDWARE_DEFERRED

# 前视障碍感知：当前状态

更新：2026-10-07。保留冻结M3与既有写作；新授权方向置信度加权CPU诊断已完成，未获得推进依据。
Status: `DTR_R2_DYNAMIC_RETAINED`（历史保留，不表示恢复动态研究）。

## 当前结论

- [合成三态](nearfield/CNH_TRISTATE_EVENT_DEV_20261006.md)：960事件，单/双路及时909/929；最大τ静默10/30。3473条对照，同≤10静默预算的unknown时间60.59%/39.77%，差区间跨零；零静默端反转。
- [真实头动确认](nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)：480新模拟单位（96校准、384评估）完成。单路unknown由合成59.45%降至自然8.85%；NAT+E1单/双及时927/923 of1002，差−4 [−16,+7]。2.5%是校准目标，实际误报1.90%/2.78%；不称同等实际误报。E1相对精确少39/51次及时。
- [BlindWays](nearfield/CNH_BLINDWAYS_HEADING_DEV_20261007.md)与[自适应查询](nearfield/CNH_ADAPTIVE_QUERY_DEV_20261007.md)：方向误差在更多轨迹中仍严重；扇形报警未达到收回一半损失的标准。谨慎判畅通可将369事件中的头部方案静默10/23降至1/4，unknown增加约7/20个百分点，及时不变。
- [短时路径](nearfield/CNH_SHORT_HORIZON_DEV_20261007.md)：共同11898锚点躯干角RMS在0.5/1/2s为12.79/11.80/11.52°，1.5m为12.79°；转弯局部改善不足以支持接入，共同集偏易。
- [方向加权](nearfield/CNH_PROBABILITY_QUERY_DEV_20261007.md)：44校准/100评估单元；264评估事件、836对照，实际误报匹配2.5%的描述性比较，头部条件加权相对中心单路143−149=−6 [−15,+3]、双路145−141=+4 [−3,+12]；双路固定权重也145。优于max，未稳健优于中心；仅对称置信度加权，不是左右意图推断。

## 当前交付与继承

[章节](thesis/CNH_READOUT_COVERAGE_TRISTATE_CHAPTER_20261006.md)按方法、合成对照、真实头动确认、方向瓶颈重排；[主张台账](THESIS_CLAIMS_20260927.md)维护证据范围。[图清单](thesis/CNH_TRISTATE_FIGURES_20261006.md)与原载荷保留。

登记`cnh-rhc-20261007`已为`completed`，作为限定模拟结论的论文证据，M3仍为基线；EMA、双路及新阈值不自动继承为在线方法。[覆盖收口](nearfield/CNH_COVERAGE_CLOSEOUT_20261005.md)、读出及外参各轮的失败和停止范围保留，不扩大为所有新机制的禁令。

## 下一步与边界

本次新CPU≤600s诊断实际53s，5项测试及独立重算通过；未启动GPU或改事件/M3。行进意图仍为开放问题，不据本轮扩大实验，不继承已结束预算。

真实轨迹加模拟观测不等于实机效果；P1/P2仍限HEADS-UP一段头动，BlindWays缺头部朝向。UNKNOWN不保证安全。City、保护test、新UE采集及硬件第二阶段仍暂停；设备回放待会话。

[RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)。整理前全文：Git `4c5d5ed6` 同路径；旧待决记录不恢复授权。
