# AtomS3R-M12 + ToF4M 端到端时间基线 R0

## 结论

`DEVELOPMENT_E2E_TIMING_BASELINE_AVAILABLE / PHYSICAL_FEEDBACK_NOT_EVALUATED`

在 AtomS3R-M12、Unit ToF4M、XGA `1024×768`、JPEG quality 10、同一局域网及
Windows host CPU YOLO11n 参考链路上，30 分钟运行完整收口。设备未重启，43,230 帧
中没有 frame-sequence 缺口、流重连或记录错误；但 P95 端到端时间达到 265.8 ms，
长时尾延迟是下一项性能诊断重点。

## 时间合同

- 所有设备时间都属于 `esp32_boot_monotonic:<boot-sequence>`。
- `capture_timestamp_us` 是 `esp32-camera` 返回的首个 DMA buffer 时间，不是曝光起点。
- `jpeg_ready_timestamp_us` 是相机 framebuffer/JPEG 可用时间。
- ToF 使用历史环中与 capture 绝对时间差最小的样本，同时保留有符号 skew。
- 主机经 3333/UDP 周期对时，以最小 RTT midpoint 建立 clock offset；逐帧保留 RTT
  与半 RTT 误差上界。
- 主机接收时间是 MJPEG parser 的 body read boundary，不是 NIC 硬件时间戳。

## 30 分钟结果

运行时间 `1802.422 s`，有效接收约 `23.98 fps`。

| 指标 | P50 | P95 | P99 / max |
| --- | ---: | ---: | ---: |
| capture → host JPEG complete | 99.1 ms | 225.5 ms | P99 259.2 ms |
| capture → decode complete | 101.9 ms | 228.4 ms | P99 262.2 ms |
| host inference | 32.7 ms | 46.1 ms | P99 52.9 ms |
| capture → risk complete | 137.2 ms | 265.8 ms | P99 300.7 ms |
| capture → feedback record complete | 137.2 ms | 265.8 ms | P99 300.7 ms |
| host inter-arrival | 38.1 ms | 65.2 ms | max 166.2 ms |
| absolute ToF—capture skew | 23.3 ms | 51.5 ms | max 59.7 ms |
| clock-sync error bound | 1.45 ms | 2.20 ms | max 2.22 ms per selected sample |

稳定性账本：

- `43,230` frames；`0` reconnect；`0` error；`0` sequence-gap event；ToF valid fraction `1.0`。
- status `355` samples；单一 frame/status boot sequence，未观察到设备重启。
- free heap 首尾 `153,288 → 153,288 B`，全程最小 `146,364 B`；未观察到单调内存下降。
- ESP32 internal sensor 首尾 `67.1 → 71.1 °C`、最高 `72.1 °C`；这不是相机或外壳温度。
- Wi-Fi RSSI P50 `-37 dBm`，范围 `-39..-35 dBm`。

## 证据与边界

本地 summary：
`artifacts.local/evidence/atoms3r-e2e/20260805T090231.009682Z/summary.json`，SHA-256
`c91218b37d22d82e3e6d707677d902f61e7f16e6d16fc9d824d6d83283fac1e5`。
最终固件编译占用 program/RAM `1,076,407 B (32%) / 62,600 B (19%)`，app bin
SHA-256 为 `fef05a3ab307f498bc14ab9c60dc8833dbdde7cd9c0b59bda5e8976aff1ceade`。

参考 pipeline 身份为 `HOST_REFERENCE_YOLO11N_RAW_SCORE_RISK_R0_NOT_PRODUCTION`。
风险计算只用于计时，feedback 阶段只写记录且明确 `physical_output_emitted=false`。
因此本结果不支持手机端性能、真实语音/震动执行时间、风险准确率、RGB-ToF 空间标定、
人体使用、产品可靠性或安全结论。
