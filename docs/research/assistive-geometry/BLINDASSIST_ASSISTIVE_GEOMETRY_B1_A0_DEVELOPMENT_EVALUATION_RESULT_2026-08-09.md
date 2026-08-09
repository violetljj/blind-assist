# Assistive Geometry B1 A0 Development evaluation result

终态：`B1_A0_DEVELOPMENT_EVALUATION_FAIL_TASK_GATES`

A0 depth-only 三种子均完整训练至 `20 epochs / 6,000 optimizer steps`，每 seed 的四个
checkpoint、模型状态 SHA、训练协议和数据防火墙均通过完整性校验。seed 29 Attempt 01 的 CUDA OOM
保留失败收据；Attempt 02 没有续跑或挑选中间状态，而是从共同 DepthART 初始化完整重跑。

Development Selection 固定四个 parent、1,200 帧，仅物化和观察一次。三 seed 共 3,600 帧观察完成，
没有打开 Development Calibration 或 Confirmation，也没有选择最佳 seed。

## 冻结门结果

| 指标 | 三 seed 平均 | 门槛 | 通过 seed 数 | 结论 |
| --- | ---: | ---: | ---: | --- |
| known coverage | 0.9808 | >= 0.90 | 3/3 | PASS |
| valid to UNKNOWN | 0.0192 | <= 0.10 | 3/3 | PASS |
| ground recovery | 0.9901 | >= 0.90 | 3/3 | PASS |
| clearance coverage | 0.9808 | >= 0.90 | 3/3 | PASS |
| clearance MAE | 0.3152 m | <= 0.20 m | 0/3 | FAIL |
| false-clear / all known | 0.0241 | <= 0.08 | 3/3 | PASS |
| false-block / truth clear | 0.7501 | <= 0.02 | 0/3 | FAIL |
| temporal clearance-delta MAE | 0.0324 m | <= 0.15 m | 3/3 | PASS |
| geometry transition agreement | 0.7728 | >= 0.90 | 0/3 | FAIL |
| worst-parent false-clear | 0.0501 | <= 0.12 | 3/3 | PASS |

前门完整性为 PASS，但任务门为 FAIL。失败集中在 clearance 数值精度、极高的 false-block 和状态转移
一致性，且三项都是 `0/3` seed 通过；它不是可用 best-seed 选择掩盖的随机波动。A0 对 false-clear
保持保守，但代价是把大量真实可通行空间判成阻塞，不能晋级为 Assistive Geometry 基线。

## 完整性修正

首次 evaluator 调用在读取 observation 之前停止，因为旧实现错误要求 seed 29 Attempt 02 的训练结果
匹配原始三 seed 协议 SHA。冻结 Development 协议本身已提前绑定 Attempt 02；修正只让 checkpoint
完整性校验按 seed 读取已授权协议，并继续检查全部 12 个 checkpoint。身份、阈值、观察值、聚合规则
和数据角色均未改变；原 `r0` INTERNAL_FAILURE 保留，正式统计写入独立 `r1`。

## 决策与边界

冻结协议只在 A0 通过时激活 A1。该条件未满足，因此 A1-A4、M0 移动执行、C0 教师执行和 D0 时序
执行均未授权；不会用当前负结果事后放宽门槛或盲目追加训练。Development Selection 已消费，今后不得
再次用于选模或调参；Development Calibration 与 Confirmation 继续封存。

如果重开主线，必须先提出实质不同的任务几何假设、冻结新的 pre-outcome 协议，并使用未消费的选择
证据。当前结论只具有 Development-only 负证据权限，不是部署、产品或安全结论。
