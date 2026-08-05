# DA V2 RGB-D 轻量 student A4 R0 结果

日期：2026-08-05

## 终点

`A4_RGBD_MOBILE_STUDENT_TRAINING_INVALID`

唯一执行在首个 epoch 内遇到 `47333053_60131.388`：该帧没有任何满足冻结
`confidence == 2` 且深度在 `[0.25,6] m` 的像素，原 trainer 抛出 `ValueError`。
没有生成 checkpoint、training result 或 P1 cache；P1 真值未打开。

因为模型已经初始化并处理过部分 batch，不能把这次执行描述为“训练前 preflight repair”。R0
永久保留 invalid 终点。后继 A4 R1 在模型初始化前扫描同一固定 roster，以纯输入可评估性规则
排除零真值帧；模型、损失、seed、epochs 和 P1 门均不改变。

只读统计：train `2374/2400` 可监督、validation `590/600` 可监督，共 36 个零真值帧。
