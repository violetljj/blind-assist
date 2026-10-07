ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED

# 前视障碍感知：当前状态

更新：2026-10-07。用户持续目标为“持续进行算法探索”。保留冻结M3；单调标定未带来改进，两固定阈值家族余量有限，下一步转原始观测信息。
Status: `DTR_R2_DYNAMIC_RETAINED`（历史保留，不表示恢复动态研究）。

## 当前证据与判断

- [独立单调标定](nearfield/CNH_QUERYWISE_CALIBRATION_DEV_20261007.md)：原369事件无启动及时max202、query标定193、event标定192，实际误报2.629/2.367/2.390%。HB36事件新两臂23/24、max25，误报6/6/8 of360。消除跨高度否决，但没有保住浅树的原域收益，不继承。
- **受检家族oracle只有小幅余量：** 评价真值事后选HEAD/BODY两阈值，每fold误报≤2.5%时无启动及时204/369，matched max195；5%278 vs270。不是部署成绩，不覆盖跨fold误报预算重分配/任意二维边界/新特征。全局重标度不再优先。
- [双高度物理对照](nearfield/CNH_DOUBLE_HEIGHT_DEV_20261007.md)已定位10个不同unit的读出抑制反例，旧浅树HB11/36 vs max25/36；[前轮](nearfield/CNH_EVENT_READOUT_DEV_20261007.md)369事件204→253的收益保留但不能推广。真实结构仍仅同模拟器干预。
- [方向加权](nearfield/CNH_PROBABILITY_QUERY_DEV_20261007.md)未稳健超过中心；[短时路径](nearfield/CNH_SHORT_HORIZON_DEV_20261007.md)只有局部转弯改善。这些失败不证明现有输出已无余量，也不等于行进意图问题已解决。

## 下一步与范围

相同完整unit划分与低容量学习，比较当前M3两分数、加当前原始径向摘要、再加过去径向变化。第二项检验分数压缩损失，第三项才检验接近时序信息。仅有符号H3、环境光和带噪相对旋转，不用真实travel/未来路径/平移真值补偿；HB为已消费结构检查，非新鲜验证。该原始观测实验尚未运行。

本轮单调标定CPU38.16/300s、独立oracle0.422/120s；5测试、525项审计及219例穷举通过，无GPU。前轮GPU已释放，模型/原始观测/runtime保留。持续授权有效，后续明确新子实验范围；不继承旧预算、不改失败、不延长已停止合成路径。

## 既有主线与边界

[合成三态](nearfield/CNH_TRISTATE_EVENT_DEV_20261006.md)保留960事件证据；[真实头动确认](nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)保留480新模拟单位、1002事件的限定结论，登记cnh-rhc-20261007已completed。确认批校准误报目标不等于实际误报相同。EMA、双路和新事件读出均未成为在线基线。

[章节](thesis/CNH_READOUT_COVERAGE_TRISTATE_CHAPTER_20261006.md)及[主张台账](THESIS_CLAIMS_20260927.md)保留，行进意图仍为开放问题。真实误差加模拟观测不等于实机效果，UNKNOWN不保证安全。City、保护test、新UE采集及硬件第二阶段仍暂停；设备回放待会话。

[RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)。此前全文：Git bd8852e2 同路径；旧待决记录不覆盖持续目标。
