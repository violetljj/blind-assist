# AtomS3R 固件 newest-ready 异步采集/发送 R16 结果（2026-08-06）

## 结论

实现并真实刷写验证了双 slot newest-ready 生产者/消费者候选：相机采集任务与 HTTP 发送任务解耦，READY 帧可被新帧淘汰，SENDING 帧不覆盖。路径编译和运行稳定，但在当前 SVGA/Q10 链路下没有形成可确认的 P95/P99 端到端收益，因此不晋升，默认开关已关闭，设备已恢复 r11 基线。

## 实验实现

- 2 个 PSRAM JPEG slot
- `WRITING → READY → SENDING` 生命周期
- 只淘汰未开始发送的 READY slot
- SENDING slot 不覆盖
- 相机生产者与 `/stream` handler 分离
- 默认 `kEnableAsyncStreamProducer=false`
- 固件标识：`atoms3r_m12_tof4m_stream_r16_async_latest_ready`

## 10 秒（修复启动竞态后）

- 267 帧，约 26.7 fps
- source packets 269，2 overwrite，3 sequence gap，0 error，0 reconnect
- capture→JPEG complete：P50 `53.94 ms`，P95 `66.93 ms`，P99 `107.64 ms`
- frame age at first byte：P50 `42.23 ms`，P95 `47.06 ms`，P99 `79.88 ms`
- capture→risk：P50 `66.62 ms`，P95 `84.07 ms`，P99 `126.01 ms`

## 1 分钟

- 1624 帧，0 error，0 reconnect，0 sequence gap，0 overwrite
- capture→JPEG complete：P50 `53.66 ms`，P95 `67.23 ms`，P99 `93.21 ms`
- frame age at first byte：P50 `42.22 ms`，P95 `48.25 ms`，P99 `79.08 ms`
- JPEG decode：P50 `3.66 ms`，P95 `10.33 ms`，P99 `11.60 ms`
- preprocess：P50 `1.25 ms`，P95 `5.08 ms`，P99 `6.16 ms`
- QNN execute：P50 `2.84 ms`，P95 `3.59 ms`，P99 `4.03 ms`
- detector total：P50 `6.40 ms`，P95 `14.34 ms`，P99 `15.75 ms`
- capture→risk：P50 `66.08 ms`，P95 `84.77 ms`，P99 `108.85 ms`
- PSS：`188755 KB → 185338 KB`

## 对照与判定

此前双 framebuffer r11 1 分钟基线约为 capture→risk `67.49/83.22/104.45 ms`。异步路径 P50 轻微下降，但 P95/P99 没有稳定改善，且引入更多 FreeRTOS 生命周期复杂度；不足以晋升。

## 当前设备状态

已刷回并确认：

```text
firmware = atoms3r_m12_tof4m_stream_r11_per_frame_copy_buffer
resolution = SVGA 800×600
jpeg_quality = 10
frame_buffer_count = 2
grab_mode = LATEST
```

结果均为 development-only 性能证据，不代表准确率、安全或物理反馈起点。
