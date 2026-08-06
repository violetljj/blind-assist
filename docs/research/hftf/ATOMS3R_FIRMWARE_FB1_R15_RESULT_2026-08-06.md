# AtomS3R 固件单 framebuffer R15 结果（2026-08-06）

## 结论

`fb_count=1` 已完成真实刷写和 Android 端到端验证，明确拒绝。单 framebuffer 没有降低帧年龄，反而使相机采集与网络写出强耦合，吞吐和尾延迟显著恶化。设备已回滚到双 framebuffer 基线。

## 实验边界

- Arduino CLI `1.5.1`
- M5Stack ESP32 `3.3.8`
- FQBN `m5stack:esp32:m5stack_atoms3r`
- 设备：AtomS3R-M12，COM5，ESP32-S3-PICO-1
- Android：SM-S9280 / SM8650 / Android SDK 36
- 设备网络：`192.168.5.11`
- 相机条件：XGA / quality 10 / 自动曝光（刷写重启后的默认配置）
- 手机解码：完整 Bitmap 解码，`decodeSampleSize=1`

## fb_count=1，10 秒

- 126 帧，约 12.6 fps
- 0 error，0 reconnect，0 sequence gap，0 overwrite
- capture→JPEG complete：P50 `65.48 ms`，P95 `116.19 ms`，P99 `129.67 ms`
- frame age at first byte：P50 `44.23 ms`，P95 `95.38 ms`，P99 `102.22 ms`
- capture→risk：P50 `79.78 ms`，P95 `127.83 ms`，P99 `140.82 ms`

## 双 framebuffer 恢复后，10 秒

- 236 帧，约 23.6 fps
- 0 error，0 reconnect，0 sequence gap，0 overwrite
- `/api/status` 明确报告：`frame_buffer_count=2`、`grab_mode=LATEST`、固件 `r11_per_frame_copy_buffer`
- capture→JPEG complete：P50 `65.37 ms`，P95 `103.90 ms`，P99 `113.67 ms`
- frame age at first byte：P50 `43.90 ms`，P95 `81.13 ms`，P99 `93.06 ms`
- capture→risk：P50 `78.90 ms`，P95 `119.88 ms`，P99 `172.23 ms`

短测受 Wi-Fi/调度噪声影响，恢复后的单次 P99 不用于宣称收益；但帧率从约 12.6 fps 恢复到约 23.6 fps，且 fb_count=1 的 P95 frame age 明显变差，足以拒绝单 framebuffer。

## 当前设备状态

回滚后已通过 `/api/status` 恢复：

```text
firmware = atoms3r_m12_tof4m_stream_r11_per_frame_copy_buffer
resolution = SVGA 800×600
jpeg_quality = 10
brightness = 1
auto_exposure = true
frame_buffer_count = 2
grab_mode = LATEST
```

## 下一步

不要再扫描 `fb_count`、核心绑定或简单 buffer 复用。若继续压设备端尾延迟，应进入真正的采集—发送生产者/消费者结构实验，并保留 newest-ready 淘汰、SENDING 不覆盖和显式 superseded 计数；每次先独立编译，再刷写 COM5，按 10 秒→1 分钟验证。
