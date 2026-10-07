ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED

# 前视障碍感知：当前状态

更新：2026-10-07。用户持续目标为“持续进行算法探索”。原始距离+HB训练增广成为待验证候选；保留冻结M3，下一步固定模型检查新形状。
Status: `DTR_R2_DYNAMIC_RETAINED`（历史保留，不表示恢复动态研究）。

## 当前证据与判断

- [训练支持与raw组合](nearfield/CNH_TRAINING_SUPPORT_DEV_20261007.md)：2.5%原校准目标下，raw+HB无启动及时275/369 vs原max202，增益73 [52,93]；实际FA329 vs330/12550。HB27/36 vs25，FA6 vs8/360，观察非下降不等于统计非劣。
- 等权单高度重复对照：两分数增广HB16→22/36，FA1→4/360；raw增广20→27，FA2→6，增益7 [3,11.025]、三fold+1/+5/+1。raw单高度及时而HB漏掉9→3，尚未全解决。HB结构明确进入其它训练unit，仍是已消费Development。
- 5%原校准：候选原域300 vs271、FA601 vs644/12550；HB28 vs30、FA11 vs16/360，不能推广为所有阈值均优。两分数HB-only事后控制前沿2.5%同FA9/360时23→26，仅描述性。
- [原始径向历史](nearfield/CNH_RAW_RADIAL_INFORMATION_DEV_20261007.md)独立收益未获支持；此前单调修复和阈值oracle保留。不继承旧浅树、未证明真实行进意图估计解决。

## 下一步与范围

固定raw+HB候选与原校准阈值，检查未用于增广训练的高度/垂直位置变化；先几何标签和可见范围核验，再小规模物理渲染，不按分数挑场景、不用新场景定阈值。新形状是同模拟器结构检查，非独立实机确认。下一检查尚未冻结或运行，M3继续保留。

本轮CPU两分数2.219/300s、raw组合16.141/300s；5测试及独立权重/来源/预测/指标核验通过，HB事后前沿另经穷举校验，二维图已目视检查。无GPU或存活计算；模型/观测/runtime保留。持续授权有效，不继承旧预算、不改失败、不延长已停止路径。

## 既有主线与边界

[合成三态](nearfield/CNH_TRISTATE_EVENT_DEV_20261006.md)保留960事件证据；[真实头动确认](nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)保留480新模拟单位、1002事件的限定结论，登记cnh-rhc-20261007已completed。确认批校准误报目标不等于实际误报相同。EMA、双路和新事件读出均未成为在线基线。

[章节](thesis/CNH_READOUT_COVERAGE_TRISTATE_CHAPTER_20261006.md)及[主张台账](THESIS_CLAIMS_20260927.md)保留，行进意图仍为开放问题。真实误差加模拟观测不等于实机效果，UNKNOWN不保证安全。City、保护test、新UE采集及硬件第二阶段仍暂停；设备回放待会话。

[RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)。此前全文：Git 920eb9e9 同路径；旧待决记录不覆盖持续目标。
