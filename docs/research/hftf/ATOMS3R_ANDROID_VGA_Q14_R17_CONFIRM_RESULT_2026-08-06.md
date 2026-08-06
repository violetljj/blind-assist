# AtomS3R Android VGA/Q14 R17 确认结果（2026-08-06）

## 结论

在稳定 r11 双 framebuffer 固件、当前 Android native 预处理/Bitmap 池/NMS 优化与真实 QNN HTP 路线上，VGA 640×480 + JPEG quality 14 再次取得当前最低稳定端到端延迟。

它正式成为 `AI_REALTIME` 性能候选，但在检测一致率、小目标召回和 false-clear 对照完成前，不替换 SVGA/Q10 默认。

## 1 分钟结果

- 1662 帧，约 27.7 fps
- 0 error，0 reconnect，1 sequence gap，1 overwrite
- JPEG size：P50 `7077 B`，P95 `7228 B`，P99 `7319 B`
- capture→JPEG complete：P50 `48.92 ms`，P95 `56.74 ms`，P99 `60.79 ms`
- frame age at first byte：P50 `42.43 ms`，P95 `46.88 ms`，P99 `50.92 ms`
- first byte→JPEG complete：P50 `6.27 ms`，P95 `13.14 ms`，P99 `17.38 ms`
- JPEG decode：P50 `2.66 ms`，P95 `7.43 ms`，P99 `8.75 ms`
- preprocess：P50 `1.21 ms`，P95 `6.08 ms`，P99 `7.06 ms`
- QNN execute：P50 `2.86 ms`，P95 `3.65 ms`，P99 `3.98 ms`
- detector total：P50 `6.42 ms`，P95 `15.35 ms`，P99 `16.70 ms`
- capture→risk：P50 `59.85 ms`，P95 `74.02 ms`，P99 `80.75 ms`
- PSS：`189089 KB → 187909 KB`

## 判定

相对 SVGA/Q10 约 `67.5/83.2/104.5 ms`，VGA/Q14 同时压低普通分位与尾延迟，主要来自更小 JPEG、更短完整帧传输和更短解码。该结果达到阶段目标 `capture→risk P95 < 100 ms`，并进入 75–85 ms 进阶区间。

仍缺少的晋升证据：

- 与 SVGA/Q10 相同场景逐帧检测一致率；
- 小目标/远距离目标召回；
- 风险等级一致率和 false-clear；
- 画面主观可用性与截图/OCR 边界。

结果为 development-only 性能证据，不代表准确率、安全或物理反馈起点。
