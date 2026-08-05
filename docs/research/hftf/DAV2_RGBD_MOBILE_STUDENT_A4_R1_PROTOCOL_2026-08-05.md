# DA V2 RGB-D 轻量 student A4 R1

日期：2026-08-05（R0 中止后的独立冻结执行）

A4 R0 因固定 roster 中存在零可用真值帧而在首个 epoch 内 invalid，且没有 checkpoint。R1
仅修正输入可评估性：在模型初始化前保留至少有一个 `confidence==2`、深度 `[0.25,6] m`
像素的帧，得到 train `2374`、validation `590`。该规则与候选输出及 P1 结果无关。

架构、ImageNet 权重、RGB-D/teacher 损失、seed、5 epochs、batch、优化器、checkpoint 选择和
P1 R1 门全部与 R0 相同。R0 保持 `TRAINING_INVALID`，不得被 R1 覆盖。
