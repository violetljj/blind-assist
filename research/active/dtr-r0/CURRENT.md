ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED

# 前视障碍感知：当前状态

更新：2026-10-07。用户持续目标为“持续进行算法探索”。保留冻结M3；当前原始距离信息有增益迹象，历史独立贡献未获支持，下一步检查组合结构训练支持。
Status: `DTR_R2_DYNAMIC_RETAINED`（历史保留，不表示恢复动态研究）。

## 当前证据与判断

- [原始径向对照](nearfield/CNH_RAW_RADIAL_INFORMATION_DEV_20261007.md)：原校准2.5%目标，当前距离无启动及时271/369 vs旧pair253，实际FA297 vs246/12550。评价控制2.5%上限的描述前沿274 vs265，增益9 [0,18]，实际FA312 vs310；不称稳健同误报增益。
- 历史相对同可插值支持元数据对照：原校准−3 [-14,7]，描述前沿+4 [-5,12.025]、FA同312/12550。没有独立历史内容收益证据，不等于全部时序方法无效。
- 已消费HB结构检查：当前距离21/36、历史18/36、原max25/36，FA1/1/8 of360；工作点不同。当前距离仍有9个anchor出现单高度及时而HB漏掉，旧pair15、原max0。更原始输入缓解但未解决组合泛化，不继承候选。
- [此前单调标定与两阈值oracle](nearfield/CNH_QUERYWISE_CALIBRATION_DEV_20261007.md)保留限定结论，重标度不再优先；[双高度旧反例](nearfield/CNH_DOUBLE_HEIGHT_DEV_20261007.md)仍有效。当前信息实验不解决行进意图估计。

## 下一步与范围

优先检验训练支持：原完整unit folds，仅加入train角色的已有HB物理样本，以同anchor、同新增权重的原单高度重复作对照。先用旧pair定位支持贡献，原cal规则不变，HB仅原OOF fold评价。每fold可用HB训练正例17/19/22，样本少，失败只否定该有限增广。HB明确已消费Development，后续不得称新鲜验证；该试验尚未冻结或运行。

本轮CPU提取16.75/900s、拟合评价87.66/900s；8测试、独立几何算例和结果核验通过，无GPU或存活计算。带噪纯旋转、signed径向值和缺失支持均保留，无travel/未来路径特征。模型/观测/runtime保留；持续授权有效，不继承旧预算、不改失败、不延长已停止路径。

## 既有主线与边界

[合成三态](nearfield/CNH_TRISTATE_EVENT_DEV_20261006.md)保留960事件证据；[真实头动确认](nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)保留480新模拟单位、1002事件的限定结论，登记cnh-rhc-20261007已completed。确认批校准误报目标不等于实际误报相同。EMA、双路和新事件读出均未成为在线基线。

[章节](thesis/CNH_READOUT_COVERAGE_TRISTATE_CHAPTER_20261006.md)及[主张台账](THESIS_CLAIMS_20260927.md)保留，行进意图仍为开放问题。真实误差加模拟观测不等于实机效果，UNKNOWN不保证安全。City、保护test、新UE采集及硬件第二阶段仍暂停；设备回放待会话。

[RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)。此前全文：Git 33c5b228 同路径；旧待决记录不覆盖持续目标。
