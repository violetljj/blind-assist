# walk-alert-r0 current

更新：2026-11-16。状态：`DESIGN_ONLY`（EXPLORE）。

## 方向

用户 2026-11-16 提出：硬件改用主流平台（手机/眼镜 RGB+IMU），创新放在算法上，尽量不自采数据。
文献地图（本地 `artifacts.local/work/deep-research/reports/助盲避障文献地图.md`）认为，
空白在模块之间的衔接处，最有希望的组合是：

- G1：把自监督学到的可走区域转成给盲人的提醒；
- G4：带代价、可个性化的提醒策略；
- G6：用视障者步态校准行走者模型，做闭环评估。

三项证据强度都只是“中”，不能声称首个；最大的抢先风险来自 Ohn-Bar 组（PING、BlindWays、TIMELI）。

## 当前步骤

[可行性实验计划](FEASIBILITY_PLAN_20261116.md)：在 SANPO synthetic 加 BlindWays 上，检验代价敏感提醒策略
相对规则基线的误报余量。继续或放弃的判据见计划，执行前冻结。

这条路线不替代避障线既有决定（见 [当前决定](../../../docs/CURRENT_DECISION.md)），也没有改动 App 或默认模型。
