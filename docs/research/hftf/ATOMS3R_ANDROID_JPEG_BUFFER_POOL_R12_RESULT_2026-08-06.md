# AtomS3R Android JPEG 缓冲池 R12 结果（2026-08-06）

## 结论

JPEG `ByteArray` 复用池通过了编译、10 秒短测和 1 分钟真机回归，但没有形成可确认的端到端收益，暂不保留。已撤回 JPEG 池，仅保留已验证的 Bitmap 池（R10）与 MJPEG header scratch（R11）。

## 实验配置

- 设备：SM-S9280 / SM8650 / Android SDK 36
- AtomS3R：`http://192.168.5.11`
- 当前设备参数：SVGA 800×600，JPEG quality 10，自动曝光
- 运行链路：QNN HTP production route，latest-frame 接收

## 10 秒短测

- 269 帧，0 error，0 reconnect，2 gap，2 overwrite
- decode：P50 3.46 ms，P95 8.73 ms，P99 11.64 ms
- capture→risk：P50 67.75 ms，P95 86.75 ms，P99 144.59 ms

## 1 分钟回归

- 1630 帧，0 error，0 reconnect，1 gap，1 overwrite
- decode：P50 3.66 ms，P95 10.43 ms，P99 12.11 ms
- capture→risk：P50 67.49 ms，P95 83.22 ms，P99 104.45 ms
- preprocess：P50 1.22 ms，P95 5.45 ms，P99 6.15 ms
- QNN execute：P50 2.86 ms，P95 3.61 ms，P99 4.17 ms
- detector total：P50 6.40 ms，P95 14.66 ms，P99 15.98 ms
- PSS：188798 KB → 178299 KB

## 判定

与 R10/R11 的 decode 约 3.65–3.67 ms、capture→risk 约 67–68 ms 基本一致，差异落在测量噪声范围内。该实验增加了 packet 所有权和生命周期复杂度，却没有压缩主链路尾延迟，因此不晋升。

## 后续

继续优先分析设备/Wi-Fi 的 `frame_age_at_first_byte_ms` 和 `android_first_byte_to_jpeg_complete_ms` 长尾；不再盲目扩大 Android ByteArray 池化。所有结果为 development-only 性能证据，不代表准确率、安全或物理反馈起点。
