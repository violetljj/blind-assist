# Assistive Geometry B1 A0 Development 评估协议

状态：`FROZEN_IMPLEMENTED_PENDING_ALL_THREE_A0_TRAIN_SEEDS`

本协议把 A0 的真实 Development 评估实现在打开任何 Development frame 前冻结。激活门是
seed `17 / 29 / 43` 均完成 20 epochs、6000 optimizer steps 和四个留存检查点；物化器会在
读取首帧前再次验证该门。

评估只打开冻结的 `DEVELOPMENT_SELECTION` 四个 parent，共 1200 帧。Calibration 与
Confirmation 保持封存。A0 使用 `预测 dense depth → 冻结 gravity/geometry reader`，不会读取
未训练的 assistive heads，也不会挑 best seed。

任务指标覆盖 known coverage、valid→UNKNOWN、ground recovery、clearance coverage/MAE、
false-clear、false-block、temporal clearance delta 与 geometry transition agreement。clearance
truth/pred validity 独立记录；`UNKNOWN` 永不作为 negative。无入侵但三段均观察为 clear 的预测
只映射为 2.0m task-horizon lower bound，不伪造远场精确距离。

通过后唯一 successor 是 A1 `PLUS_GROUND` 的正式训练协议；本协议及未来结果都不具有
Confirmation、部署、默认 App、产品或 safety 权限。
