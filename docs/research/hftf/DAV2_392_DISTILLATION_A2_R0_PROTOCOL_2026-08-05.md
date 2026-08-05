# DA V2 392 蒸馏 A2 R0

日期：2026-08-05，在 teacher cache 与 student training 前冻结。

A1 证明单纯把 input 从 518 降到 392 会破坏 metric scale、clearance 和 false-clear。A2 只
回答一个问题：不读取 P1 sensor truth，固定 teacher-only distillation 能否把这些损失拉回
canonical 非劣化包络。

训练集使用既有 ARKitScenes Training RGB identity roster：2,400 train、600 validation，按
video/parent 分离。teacher 为冻结的 518 FP16 DA V2，student 从同 checkpoint 初始化、输入
392、全参数训练三轮。固定 loss 为 log-depth SmoothL1 + log-depth gradient + median log-scale；
固定 AdamW、`2e-5`、batch 2、accumulation 4、seed `20260805`。checkpoint 只按 600 张
teacher-only validation loss 选择，不读取 ARKit sensor depth、affine target 或 TUM P1 truth。

A2 仍是同一 ViT-S 参数量，只减少 token；它是“分辨率 + 蒸馏”门，不是假装已经得到轻量
backbone。只有一次性通过 P1 全部工程非劣化门，才允许转换到 Android 并测固定 APK P95。
