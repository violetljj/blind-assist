ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED

# 前视障碍感知：当前状态

更新：2026-10-07。用户持续目标为“持续进行算法探索”。保留冻结M3；双高度物理反例已证实当前双查询浅树的读出抑制。
Status: `DTR_R2_DYNAMIC_RETAINED`（历史保留，不表示恢复动态研究）。

## 当前证据与判断

- [双高度物理对照](nearfield/CNH_DOUBLE_HEIGHT_DEV_20261007.md)：36接触+36畅通anchor，39unit/216变体。浅树H/B/HB及时25/22/11，各36；原max24/15/25。HB实际误报1/360与8/360，不是同误报前沿。15anchor单高度及时而HB漏报；10个不同unit的两个M3分数均不减，浅树却失去报警，max仍及时。5%旧阈值下差距缩至3，不能概括为所有阈值。
- **不继承当前浅树。** [前轮事件读出](nearfield/CNH_EVENT_READOUT_DEV_20261007.md)原369事件及时204→253、误报2.629%→1.960%的开发收益保留，但不能推广至双高度。强反例fold0/1/2为5/5/0，存在模型异质性；没有否定所有学习读出。原“五方向历史优于中心历史”假设仍失败。
- [方向加权](nearfield/CNH_PROBABILITY_QUERY_DEV_20261007.md)未稳健超过中心；[短时路径](nearfield/CNH_SHORT_HORIZON_DEV_20261007.md)只有局部转弯改善。这些失败不证明现有输出已无余量，也不等于行进意图问题已解决。

## 下一步与范围

先检验低成本监督修复：HEAD/BODY分别用各自几何真值单调标定后取并集，区分高度标度问题与跨高度否决；与此前事件标签双输入单调树不同。仅原训练/校准unit拟合，双高度作已消费结构检查。若无收益，再用原始有符号径向直方图、环境光和带噪相对旋转检验接近变化；不得把真实travel或未来路径放入特征。后续均尚未运行。

双高度GPU59.72/1800s、几何CPU88.58/600s、分析9.91s；原场景hist到logits逐值复现，6测试及全部分数/物理真值独立重算通过，GPU已释放。模型、原始观测和runtime保留。持续授权有效，后续明确新子实验范围；不继承旧预算、不改失败、不延长已停止合成路径。

## 既有主线与边界

[合成三态](nearfield/CNH_TRISTATE_EVENT_DEV_20261006.md)保留960事件证据；[真实头动确认](nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)保留480新模拟单位、1002事件的限定结论，登记cnh-rhc-20261007已completed。确认批校准误报目标不等于实际误报相同。EMA、双路和新事件读出均未成为在线基线。

[章节](thesis/CNH_READOUT_COVERAGE_TRISTATE_CHAPTER_20261006.md)及[主张台账](THESIS_CLAIMS_20260927.md)保留，行进意图仍为开放问题。真实误差加模拟观测不等于实机效果，UNKNOWN不保证安全。City、保护test、新UE采集及硬件第二阶段仍暂停；设备回放待会话。

[RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)。此前全文：Git 7e4f1871 同路径；旧待决记录不覆盖持续目标。
