# BlindAssist Assistive Geometry B1 A0 host performance preflight

终态：`B1_A0_TRAIN_HOST_PERFORMANCE_PILOT_PASS_WORKERS_1_SELECTED`

正式训练尚未启动。本轮只用 seed 17 的冻结 TRAIN 输入执行三档 20-step 性能 pilot，
没有读取 Development/Confirmation，也没有运行 A1–A4 或 teacher。

| DataLoader workers | 优化净时长 | optimizer step/s | mean TRAIN loss | peak CUDA allocation |
|---:|---:|---:|---:|---:|
| 0 | 41.206 s | 0.4854 | 1.255872 | 2,054,216,704 B |
| **1** | **36.676 s** | **0.5453** | **1.255773** | **2,054,216,704 B** |
| 4 | 41.775 s | 0.4788 | 1.255480 | 2,054,216,704 B |

`workers=1` 比 0/4 档分别快 12.35%/13.90%，因此 guarded launcher 固定注入 1。
三档前 8 个 CPU 输入批次摘要逐项相同；mean TRAIN loss 跨档跨度为 `0.0003923`。
CUDA 最终权重摘要并非逐位相同，所以这里只签署 loader/scientific-parameter 等价，
不签署 CUDA bit-exact determinism；科学判断仍必须使用冻结的三个 seed。

按净吞吐外推每 seed 6,000 steps 约 11,003 秒（3.06 小时），正式诊断上界冻结为
4 小时。启动要求至少 4 GiB 空闲 VRAM、4 GiB 系统 RAM reserve、AC 电源、每 10 steps
更新进度；连续两个窗口低于 `0.4635 step/s`、两次无推进或超过 4 小时触发诊断。

下一步唯一允许动作：为 seed 17 建立 seed-specific guarded preflight 并执行 TRAIN-only
正式训练。它仍不产生模型质量、Development、Confirmation、部署、产品或 safety authority。
