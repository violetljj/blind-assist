# AtomS3R Android ToF-off 归因 R29 结果（2026-08-06）

## 结论

拒绝把 ToF 关闭作为延迟优化。ToF-off 固件成功刷写并确认 `sampling_enabled=false`、`NOT_READY`，但 10 秒真机结果没有显示端到端或传输收益。

## 测试

- 固件：`atoms3r_m12_tof4m_stream_r11_per_frame_copy_buffer`，唯一变量为 `kEnableTofSampling=false`；
- 设备：AtomS3R-M12，COM5，ESP32-S3-PICO-1；
- 手机：SM-S9280 / SM8650 / Android 16；
- 相机：VGA `640×480` / JPEG Q14 / 自动曝光；
- Android：`decodeSampleSize=2`、`maxFrameAgeMs=65`；
- 时长：10 秒。

## 结果

- 265 帧，0 error，0 reconnect；
- source packets `272`，latest overwrite `1`，stale dropped `6`；
- capture→risk P50/P95/P99：`63.39 / 75.14 / 83.35 ms`；
- capture→JPEG complete P50/P95/P99：`52.82 / 61.36 / 64.70 ms`；
- frame age at first byte P50/P95/P99：`45.09 / 52.73 / 54.98 ms`；
- ToF 状态：`sampling_enabled=false`、`valid=false`、`NOT_READY`。

没有获得可归因的收益，且 10 秒结果不能替代长测；因此不再延伸 ToF-off 路线。

## 恢复与边界

测试后已重新编译、刷回 ToF-on 固件，并确认 ToF `VALID`、Wi-Fi connected。设备配置随后恢复为 SVGA/Q10。

原始证据：`artifacts.local/evidence/atoms3r-tof-off-r29/`。

本记录为 development-only 性能诊断，不是检测准确率、风险一致率、false-clear、安全或物理反馈证据。
