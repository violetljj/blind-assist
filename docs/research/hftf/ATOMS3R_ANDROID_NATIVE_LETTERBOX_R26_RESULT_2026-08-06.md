# AtomS3R Android native letterbox R26 结果（2026-08-06）

## 结论

拒绝“从解码后的 Bitmap 直接由手写 native 双线性 resize + letterbox + RGB Float”候选，并已撤回实现。该路径避开了 Kotlin Canvas 和中间 `320×320` Bitmap，但在 SM-S9280 上明显慢于当前 Skia Canvas + native RGB Float 路径。

## 10 秒短测

冻结配置：

- AtomS3R：VGA `640×480` / JPEG Q14 / 自动曝光；
- Android：`decodeSampleSize=2`、`maxFrameAgeMs=65`；
- 手机：SM-S9280 / SM8650 / Android 16；
- QNN：现有真实 HTP 路径；
- 固件：`atoms3r_m12_tof4m_stream_r11_per_frame_copy_buffer`。

结果：

- 264 帧，0 error，0 reconnect；
- stale packets dropped `6`；
- native letterbox + Float：P50 `3.02 ms`，P95 `10.38 ms`，P99 `11.98 ms`；
- 总 preprocess：P50 `3.05 ms`，P95 `10.44 ms`，P99 `12.05 ms`；
- detector total：P50 `8.19 ms`，P95 `19.55 ms`，P99 `21.51 ms`；
- frame age at first byte：P50 `44.90 ms`，P95 `51.27 ms`，P99 `56.43 ms`。

当前保留路线在 R20 的 1 分钟结果中 preprocess 为 P50 `0.40 ms`、P95 `1.08 ms`、P99 `1.31 ms`。虽然测试时长不同，不应把分位数差当严格 A/B，但 R26 的普通分位和尾部均出现数量级明确的退化，没有继续跑 1 分钟的价值。

## 归因与边界

手写逐像素 CPU 双线性采样比 Android/Skia 的过滤缩放更慢；同时手写采样不能保证与 Skia bit-equivalent，因此既没有性能收益，也会增加模型输入等价验证成本。

本结果仅为 development-only 性能诊断，不是检测准确率、风险一致率、false-clear、安全或物理反馈证据。测试结束后设备已恢复稳定的 SVGA/Q10 配置。

原始证据：`artifacts.local/evidence/atoms3r-native-letterbox-r26/`。
