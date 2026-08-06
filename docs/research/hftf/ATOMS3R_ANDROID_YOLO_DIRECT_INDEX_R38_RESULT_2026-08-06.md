# AtomS3R Android YOLO direct-index R38 结果（2026-08-06）

## 结论

保留为当前最佳 `AI_REALTIME` 性能实现。YOLO 输出解码将 channels-first 与 channels-last 分成两条直索引循环，移除每个预测、每个类别读取时的布局分支与局部函数调用；模型、阈值、框映射、标签和 NMS 语义保持不变。

## 验证

- `:core:vision:testDebugUnitTest` 通过，现有测试覆盖 channels-first/channels-last 输出布局；
- Debug APK 与 AndroidTest APK 构建通过；
- SM-S9280 真机 10 秒和 1 分钟测试通过。

## 1 分钟结果

- AtomS3R：QVGA `320×240` / JPEG Q10 / ToF-on；
- Android：`decodeSampleSize=1`、`maxFrameAgeMs=65`；
- native QVGA padding R37 同时启用；
- 1659 帧，0 error，0 reconnect；
- source packets `1664`，latest overwrite `0`，stale dropped `5`；
- capture→risk P50/P95/P99：`52.81 / 62.50 / 66.47 ms`；
- detector total P50/P95：`3.98 / 8.96 ms`；
- postprocess P50/P95/P99：`0.67 / 3.85 / 4.50 ms`；
- preprocess P50/P95：`0.28 / 0.73 ms`；
- PSS：`190448 → 184482 KiB`。

相对 R37 的 QVGA 1 分钟结果 `54.64 / 66.96 / 70.93 ms`，R38 的 capture→risk P50/P95/P99 改善约 `1.83 / 4.46 / 4.46 ms`。Detector P50/P95 从 `5.75 / 13.32 ms` 降至 `3.98 / 8.96 ms`。

## 边界

原始证据：`artifacts.local/evidence/atoms3r-yolo-direct-index-r38-1min/`。

本结果为 development-only 性能证据。现有单测证明解码布局和样例输出行为未回归，但不能替代真实检测质量、小目标召回、风险一致率和 false-clear 对照。
