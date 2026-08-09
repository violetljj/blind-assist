# Assistive Geometry B1 A0 seed 29 正式训练 Attempt 02

状态：`SEED_29_FULL_RETRY_FROZEN_AFTER_RECEIPTED_CUDA_OOM`

Attempt 01 在 epoch 7 backward 的 2097 steps 处收到 CUDA OOM；guard、failure 和 progress
收据均已保留，最后一个可独立加载的原子边界为 epoch 6 / 1800 steps。runner 没有冻结的
partial-epoch recovery 入口，因此 Attempt 02 不把 297 个未落盘 step 拼接进结果，也不从旧
checkpoint 继续；它以相同 seed、数据、模型、optimizer、schedule 和 DepthART 初始化完整重跑。

唯一运行时变化是预先冻结 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，用于减轻 CUDA
allocator 碎片，不修改科学参数。输出改写到新的 `...formal-train-20260809-r2/seed-29`；旧失败
目录不可覆盖。Development/Confirmation、A1–A4、teacher、部署、默认 App 与 safety 仍关闭。
