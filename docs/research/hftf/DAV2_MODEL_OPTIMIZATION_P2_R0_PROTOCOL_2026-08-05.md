# DA V2 模型侧优化 P2 R0

日期：2026-08-05，冻结于首个 P2 candidate cache 之前。

## 路由

逐算子 profile 已把约 88% 成本定位在 Transformer encoder，reshape+transpose 仅 3.68%。
因此 P2 不再继续打磨 JNI/layout，也不以全图 INT8 起步。执行顺序为：

1. `A1`：固定 `392` input、同 checkpoint、FP16、无蒸馏，仅做 resolution-only control；
2. `A2`：另行冻结训练合同后，做一个固定 `392` distilled student；
3. `A3`：若 A2 仍不足，再做轻量 student backbone/head；
4. `A4`：固定一种 token/attention 结构降本；
5. quantization 只做选择性 mixed precision，不先做全图 INT8。

A1 的相关旧结果已经看过，所以它不是 fresh Confirmation，只用于验证 P1 门和量化“单纯降
分辨率”的损失。其结果不能反过来改 P1 threshold。

## 执行防火墙

模型先在不读取 sensor truth 的 materializer 中生成完整 aligned-depth cache，锁定 SHA 后，
再由 P1 evaluator 打开注册深度。工程非劣化失败立即停止，不做 Android 转换和深 profile；
通过后才看固定 APK 的 QNN P95 与 full-chain P95，并要求 full-chain 至少 `1.15x` speedup。

## 多速率目标

- student depth：`8–12 Hz`；
- DA V2 teacher：`1–3 Hz`；
- YOLO/segmentation 快环：`15–30 Hz`。

teacher 可以做周期校正、disagreement、student confidence supervision；只有被独立量距或外部
标定绑定时才可提供 metric scale anchor。teacher 与 student 同源 RGB 时，二者一致不等于尺度
正确。teacher stale、scale invalid、disagreement 过大或 student 不确定时必须输出 `UNKNOWN`。
