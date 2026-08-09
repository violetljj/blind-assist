# Assistive Geometry B1-A0 program closure

状态：`B1_A0_PROGRAM_PERMANENTLY_CLOSED_NEGATIVE_TERMINAL`

B1-A0 及其条件式 A1–A4 ladder 现正式永久关闭。三个冻结 seed 都完成 `20 epochs / 6,000
steps`，训练、checkpoint、数据角色和 evaluator 前门完整；但 clearance MAE、false-block 和
geometry transition agreement 均为 `0/3` seed 通过。该结果不是“接近成功”，也不给换 seed、
改 loss、增加 epoch、放宽 threshold 或选择最好 checkpoint 的许可。

## 永久保存

- 三 seed 的四个 retained checkpoint、训练 result 与 seed 29 OOM / full-restart receipts；
- 固定四 parent / 1,200 帧 Development Selection target manifest；
- 三 seed / 3,600 帧 observation package；
- 正式 Development evaluation r1、其 pre-metric integrity failure r0；
- failure-anatomy governed r2 及其被 supersede 的诊断 attempts。

这些工件是可复现负证据，不是可继续训练的 warm start。Development Selection 已消费，只能作为
历史诊断；Development Calibration 与 Confirmation 仍未打开，且永远不转作 R2 reserve。

## Post-mortem 终点

[Failure anatomy](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_FAILURE_ANATOMY_RESULT_2026-08-09.md)
已完成，终态为 `B1_A0_FAILURE_ANATOMY_COMPLETE_NOT_ELIGIBLE_FOR_PROMOTION`。它确认 false-block
来自 predicted clearance 在 deterministic threshold 前的系统性保守偏差，且三个 seed 的错误掩码
高度相似；但全部 truth-clear 支持集中在 parent `464241`，聚合 observation 也不能因果区分 depth
scale、ground/support 数值误差或其他上游几何误差。因此不作全场景或单模块罪因主张。

## 独立新路线

[Geometry R2 factorized hypothesis](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_FACTORIZED_GEOMETRY_HYPOTHESIS_PROTOCOL_2026-08-09.md)
是实质不同的 pre-outcome 假设，不是 B1 修复：学习图只输出 metric-ish depth、support surface、
obstacle boundary/evidence 及 uncertainty，最终 clearance / occupancy / UNKNOWN 只能由确定性 reducer
产生。它不继承 B1 threshold、Selection outcome 或 execution authority。

当前唯一 successor 是：

`BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F0_SYNTHETIC_FACTOR_GEOMETRY_CANARY_PROTOCOL_AND_FIXTURES`

该 successor 尚无执行权限，只允许另行冻结 synthetic factor/reducer 协议与 fixtures；不授权训练、
真实数据、teacher、时序、移动、Calibration、Confirmation、默认 App、产品或 safety。
