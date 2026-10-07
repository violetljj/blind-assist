ALGORITHM_EXPLORATION / M3_RETAINED / HARDWARE_DEFERRED

# 前视障碍感知：当前状态

更新：2026-10-07。用户持续目标为“持续进行算法探索”。保留冻结M3，当前检查事件级双查询读出的收益及支持范围捷径。
Status: `DTR_R2_DYNAMIC_RETAINED`（历史保留，不表示恢复动态研究）。

## 当前证据与判断

- [事件读出](nearfield/CNH_EVENT_READOUT_DEV_20261007.md)：144模拟unit三折训练/校准/评价，369事件、1255对照。原中心max阈值及时204、实际误报2.629%；保留HEAD/BODY两分数的浅树253、1.960%，差+49 [26,72]。仅重学max为202，加入宽度为257，固定mean/min为91/40。不增加M3查询的组合结构值得继续查。
- **尚不继承候选：** 校准点HEAD-only126→121/177，BODY-only78→132/192；双高度接触0例。模型压低两高度共同高分，可能利用单高度目标与背景的模拟规律。none↔corner迁移仍有描述性排序收益，未排除双高度/新结构失败；原“五方向历史优于中心历史”假设失败。
- [方向加权](nearfield/CNH_PROBABILITY_QUERY_DEV_20261007.md)未稳健超过中心；[短时路径](nearfield/CNH_SHORT_HORIZON_DEV_20261007.md)只有局部转弯改善。这些失败不证明现有输出已无余量，也不等于行进意图问题已解决。

## 下一步与范围

优先物理渲染配对HEAD-only、BODY-only、贯穿双高度实体与旁侧畅通对照，冻结已训练读出检查反例。[接线计划](../../../artifacts.local/work/cnh-center-readout-ablation-dev-20261007/NEXT_DOUBLE_HEIGHT_PLAN.md)已准备，尚未运行。不能用分数相加代替观测，不能延长此前已停止的合成路径。若收益保留，再核对新单位、误差来源及扰动；若不保留，转向原始径向变化的信息检验。

本轮CPU方向/历史探针52.84s、中心消融46.53s、固定算子58.53s已完成，因果/分组测试及独立重算通过；无GPU、无M3更新。持续探索授权有效，各已完成子实验载荷和停止规则保留，不把其预算自动继承到后续阶段。主分析进程已释放，模型及runtime留作本目标复核。

## 既有主线与边界

[合成三态](nearfield/CNH_TRISTATE_EVENT_DEV_20261006.md)保留960事件证据；[真实头动确认](nearfield/CNH_REAL_HEAD_CONFIRM_RESULTS_20261007.md)保留480新模拟单位、1002事件的限定结论，登记cnh-rhc-20261007已completed。确认批校准误报目标不等于实际误报相同。EMA、双路和新事件读出均未成为在线基线。

[章节](thesis/CNH_READOUT_COVERAGE_TRISTATE_CHAPTER_20261006.md)及[主张台账](THESIS_CLAIMS_20260927.md)保留，行进意图仍为开放问题。真实误差加模拟观测不等于实机效果，UNKNOWN不保证安全。City、保护test、新UE采集及硬件第二阶段仍暂停；设备回放待会话。

[RUNS](RUNS.md) · [总决定](../../../docs/CURRENT_DECISION.md)。本次整理前全文：Git 5567be8a 同路径；旧待决记录不覆盖持续目标。
