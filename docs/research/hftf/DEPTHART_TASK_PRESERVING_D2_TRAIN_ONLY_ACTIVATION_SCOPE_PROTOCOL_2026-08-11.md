# DepthART D2 TRAIN-only base-output and head-training scope

状态：`PRE_OUTCOME / FROZEN / AWAITING_EXPLICIT_TRAINING_SCOPE`

该门只允许解码 4 个 D2 TRAIN identity 的 1,200 帧源文件，使用冻结的 `608×448` saved-context
DepthART 在 fresh `SM-S9280 / SM8650 / HTP v75` 上生成 base outputs，再按固定 seed/recipe 训练唯一
277 参数 task-evidence head，并只锁 step-500 checkpoint。

4 个 D2 DEVELOPMENT identity 在整个训练阶段继续 sealed：不得解码、派生 truth、运行 base model
或用来 early-stop/选 checkpoint。训练完成后必须先锁 head SHA 和训练 receipt，才可另行申请
Development quality activation。本门不授权 R2、性能、默认 App、production 或 safety。
