# AtomS3R-M12 + ToF4M 相机 PSRAM DMA 对照 R3

## 结论

`CAMERA_DELIVERY_PERIOD_OBSERVED / PSRAM_DMA_REJECTED_INCOMPATIBLE_STREAM_ROUTE`

R1 账本的后验分层显示，所有慢帧的 camera capture timestamp 都早于
`esp_camera_fb_get()` 调用；正常帧 capture→framebuffer return P50/P95 为
`36.55/36.75 ms`，慢帧为 `72.50/83.32 ms`。因此主现象不是应用过早调用后等待
下一帧，而是相机驱动交付单帧时偶发跨越两个或更多约 36 ms 周期。

作为唯一变量启用 ESP32-S3 的 camera PSRAM DMA 后，设备只交付 1 帧，随后五分钟
出现 59 次 stream reconnect 和 59 个 stream error。该路线对当前
AtomS3R-M12/OV3660/XGA/JPEG/double-buffer/LATEST 配置不兼容，禁止晋升；正式固件
保持 PSRAM DMA 关闭。

## 依据与实验合同

Espressif 官方驱动说明：JPEG 使用两个或更多 framebuffer 时进入连续采集；
`CAMERA_GRAB_LATEST` 管理队列中的最近帧；ESP32-S3 可通过
`esp_camera_set_psram_mode()` 在运行时切换 PSRAM DMA。参考：
[esp32-camera README](https://github.com/espressif/esp32-camera)、
[esp_camera.h](https://github.com/espressif/esp32-camera/blob/master/driver/include/esp_camera.h)。

R6 为状态 API 和逐帧账本增加：

- `psram_dma_enabled` / `X-Camera-Psram-Dma-Enabled`；
- 固定 `frame_buffer_count=2`、`grab_mode=LATEST` 的身份；
- `device_capture_to_fb_return_us`；
- `device_capture_minus_acquire_start_us`。

实验保持 XGA `1024×768`、quality 10、自动曝光、两个 framebuffer、LATEST、ToF
开启、Wi-Fi、主机 latest-frame 队列、参考 pipeline 和 300 秒时长不变；唯一变量为
PSRAM DMA `false→true`。

## 失败结果

- 时长 `300.766 s`，仅 1 帧，`0.0033 fps`；
- 59 reconnect，59 error；主要错误为设备在 HTTP response 前主动断开，另有 1 次
  MJPEG EOF；
- 60 个状态样本持续成功，首尾均显示 Wi-Fi connected、camera ready、ToF valid；
- camera `total_frames` 从 0 只增至 1，证明不是主机解码或模型推理吞吐问题；
- free heap 首尾仅 `-136 B`，状态侧未显示设备重启或 Wi-Fi 重连。

因此这不是可比较的性能样本，而是 fail-closed 的兼容性失败。失败 summary：
`artifacts.local/evidence/atoms3r-e2e/20260805T105602.273417Z/summary.json`，SHA-256
`3b8aa6ca9d57523bf95c60173ac89043b1eab052910d403d2a239af74edfe081`。

实验同时暴露旧主机工具只要收到至少一帧就返回成功。R6 已收紧为：有帧、0 reconnect
且 0 error 才 `run_accepted=true`/返回成功；否则保留完整证据并以非零状态退出。

## 恢复与边界

已恢复并刷入 `atoms3r_m12_tof4m_slow_frame_r6`，PSRAM DMA=false、ToF sampling=true
且 valid。最终 20 帧带模型验收为 0 reconnect、0 error，所有帧 DMA=false，退出后
`stream_clients=0`。最终 program/RAM 为 `1,077,911 B (32%) / 62,608 B (19%)`，
app bin SHA-256 为
`027f2142df98706e9cdf8d63464ba3abe16f7b2af75eaf207c2d9499cfd215b6`。

本结果只拒绝当前硬件和固件路线下的运行时 PSRAM DMA，不证明其他相机、分辨率、
ESP-IDF 配置或驱动版本不支持该功能，也不构成图像质量、模型、人体、产品或安全证据。
下一合法单变量可固定当前实际曝光值 490，检查 36/72 ms 交付双峰是否来自自动曝光
控制；不应增加 framebuffer 数量来换取可能更旧的画面。
