# Assistive Geometry B1 A0 Development 评估协议

状态：`FROZEN_EXECUTED / B1_A0_DEVELOPMENT_EVALUATION_FAIL_TASK_GATES`

结果：[B1 A0 Development evaluation result](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_DEVELOPMENT_EVALUATION_RESULT_2026-08-09.md)

本协议把 A0 的真实 Development 评估实现在打开任何 Development frame 前冻结。激活门是
seed `17 / 29 / 43` 均完成 20 epochs、6000 optimizer steps 和四个留存检查点；物化器会在
读取首帧前再次验证该门。
seed 29 的 Attempt 01 OOM 失败收据永久保留；只有从共同 DepthART 初始化完整重跑的 Attempt 02
r2 result 可满足 seed 29 激活门，未落盘的 297 steps 不拼接、不计入 6000 steps。

评估只打开冻结的 `DEVELOPMENT_SELECTION` 四个 parent，共 1200 帧。Calibration 与
Confirmation 保持封存。A0 使用 `预测 dense depth → 冻结 gravity/geometry reader`，不会读取
未训练的 assistive heads，也不会挑 best seed。

任务指标覆盖 known coverage、valid→UNKNOWN、ground recovery、clearance coverage/MAE、
false-clear、false-block、temporal clearance delta 与 geometry transition agreement。clearance
truth/pred validity 独立记录；`UNKNOWN` 永不作为 negative。无入侵但三段均观察为 clear 的预测
只映射为 2.0m task-horizon lower bound，不伪造远场精确距离。

通过后唯一 successor 是 A1 `PLUS_GROUND` 的正式训练协议；本协议及未来结果都不具有
Confirmation、部署、默认 App、产品或 safety 权限。

执行时首次 evaluator 在读取 observation 前发现实现错误：它把 seed 29 的已冻结 Attempt 02
协议 SHA 当成原始三 seed 协议漂移。完整性修正只按 seed 验证协议与全部 checkpoint，并复用
SHA-bound observation package；身份、阈值、观察值和聚合未改。正式结果未通过，故上述条件
successor 没有激活。
