# AtomS3R-M12 + ToF4M stream task priority R9 result

## 结论

将 MJPEG HTTP server task priority 从默认 5 提高到 6 的候选不晋升：

`STREAM_PRIORITY6_NOT_PROMOTED / NO_THROUGHPUT_GAIN / SMALL_LATENCY_REGRESSION`

相邻五分钟 A/B 的吞吐几乎相同：`23.940` 与 `23.957 fps`。priority 6 的 device
response write P50/P95 从 `27.930/35.089 ms` 恶化到 `28.627/35.835 ms`，
JPEG-ready→host read start P50/P95 从 `3.445/6.883 ms` 恶化到
`3.737/7.274 ms`，capture→feedback P50/P95 从 `83.717/121.438 ms` 恶化到
`84.800/122.407 ms`。幅度虽小，但方向一致且没有吞吐或 ToF 收益，正式保留默认
priority 5，并停止继续扫描更高优先级。

## 冻结条件

- XGA、JPEG quality 10、brightness 1、自动曝光开启
- double buffer、LATEST、PSRAM DMA 关闭
- ToF 连续采样开启
- `TCP_NODELAY=true`，preamble split
- stream server no-affinity，正式五分钟两臂实际 handler core 均为 `[1]`
- host TFLite 4 threads、latest-frame queue
- 每臂 300 秒；唯一变量为 HTTP stream server task priority 5 或 6

状态 API 记录 configured priority，逐帧新增实际 `X-Stream-Handler-Priority`，主机
缺失时 fail closed，并在 summary 汇总实际 priority。priority-6 canary 曾在短运行中
观察到 core `[0,1]`，但正式五分钟全部为 core 1，因此正式 A/B 未受 core migration
混杂。

## A/B 结果

| 指标 | priority 5 | priority 6 | 变化 |
| --- | ---: | ---: | ---: |
| 帧数 / fps | 7,195 / 23.940 | 7,202 / 23.957 | 等价 |
| reconnect/error/overwrite/gap | 0/0/0/0 | 0/0/0/0 | 均通过 |
| response write P50 | 27.930 ms | 28.627 ms | +0.697 ms |
| response write P95 | 35.089 ms | 35.835 ms | +0.746 ms |
| response write P99 | 38.916 ms | 40.147 ms | +1.231 ms |
| JPEG ready→host read start P50 | 3.445 ms | 3.737 ms | +0.291 ms |
| JPEG ready→host read start P95 | 6.883 ms | 7.274 ms | +0.391 ms |
| host JPEG read P50/P95 | 26.007/33.137 ms | 26.635/33.949 ms | 略慢 |
| capture→feedback P50 | 83.717 ms | 84.800 ms | +1.083 ms |
| capture→feedback P95 | 121.438 ms | 122.407 ms | +0.968 ms |
| capture→feedback P99 | 129.448 ms | 132.683 ms | +3.235 ms |
| capture→feedback max | 195.248 ms | 197.061 ms | 相近 |
| absolute ToF skew P50/P95 | 23.181/51.885 ms | 23.378/51.468 ms | 相近 |
| ToF age P50/P95 | 51.400/93.583 ms | 51.367/92.998 ms | 相近 |

两臂 slow-frame fraction 为 `20.16%/22.36%`，camera capture P50/P95 几乎相同
（`36.610/72.592` 与 `36.621/72.597 ms`）；RSSI P50 为 `-35/-36 dBm`。因此本轮
小幅回归不是由明显相机节拍或信号强度优势掩盖。priority 6 没有获得可推广收益。

## 证据与边界

- priority 5：`artifacts.local/evidence/atoms3r-e2e/20260805T130648.054486Z/`
- priority-5 summary SHA-256：
  `bac05123dc7e5fcba5da5bf55293f1f4e112eba70e71ce76cc0fa770ab521203`
- priority 6：`artifacts.local/evidence/atoms3r-e2e/20260805T131359.809007Z/`
- priority-6 summary SHA-256：
  `ab4a1dcfd0fc739ac7d2d6ba384688ba4af7921fa480acf0ad76f1766febf3a2`

## 发布终态

- 固件版本：`atoms3r_m12_tof4m_stream_r10_priority5`
- program/RAM：`1,078,467/62,608 bytes`（`32%/19%`）
- 固件 SHA-256：
  `9230ba67004c793fe1711cfd52582856e1092b0b9ca82ddf0990dbd4bf8b3c54`
- 20 帧 release smoke：0 reconnect/error/overwrite/gap，实际 handler
  core/priority 为 `[1]/[5]`
- smoke 退出后设备状态：`stream_clients=0`、自动曝光开启、ToF sampling/valid、
  Wi-Fi reconnect attempts 为 0

本结果只约束当前硬件、工具链与网络的 Development 调度路线。停止 priority >6
扫描，避免消费更多同环境参数尝试或提高对 ToF/loop/系统任务的抢占风险。不授权
画质、ToF 精度、风险准确率、人体、产品或安全结论。
