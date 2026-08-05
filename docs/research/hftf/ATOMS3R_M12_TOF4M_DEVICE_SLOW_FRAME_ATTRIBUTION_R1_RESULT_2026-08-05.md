# AtomS3R-M12 + ToF4M 设备慢帧归因 R1

## 结论

`DEVICE_SLOW_FRAME_ATTRIBUTION_AVAILABLE / CAMERA_CADENCE_PRIMARY / WIFI_WRITE_SECONDARY`

冻结 XGA `1024×768`、JPEG quality 10、自动曝光和 ToF 行为，不扫描参数。五分钟
R4 账本显示，慢 frame-ready interval 的主机制是 stream loop 等待下一份 camera
framebuffer/相机节拍，少数尖峰由前一帧 HTTP/MJPEG 写出阻塞直接传导。JPEG 大小和
实际曝光值在 slow/normal 间没有支持主因的变化；ToF 相关性仍有等待窗口长度混杂。

## 合同

设备逐帧记录：

- frame sequence、frame-ready interval、mutex + `esp_camera_fb_get` acquire duration；
- JPEG/metadata prepare duration；
- 由下一帧 header 回报并按 sequence 回填的 response write duration；
- JPEG bytes、分辨率、quality、曝光模式/实际值、RSSI、free heap；
- ToF age、acquire 内更新数、相邻 frame-ready 间更新数。

主机记录 first-byte observed、full-frame received、decode complete、latest queue wait；
处理帧固定 `frame_overwritten=false`，容量 1 队列覆盖事件进入
`overwritten_frames.jsonl`。主机时间仍是 parser/OS observation，不是 NIC 硬件时间戳。

冻结慢帧定义：

```text
frame_ready_interval > median + 3 × MAD
OR
frame_ready_interval > 2 × median
```

## 五分钟结果

- 运行 `300.578 s`，处理 7,071 帧，`23.52 fps`；0 reconnect、0 error。
- 7,070 个可评 interval 的 median/MAD 为 `36.047/0.091 ms`。
- 阈值为 `36.320 ms` 和 `72.094 ms`；slow `1,252/7,070 = 17.71%`。
- 其中 1,095 帧只越过 median+3×MAD，157 帧越过 2×median。
- latest queue 覆盖 105 帧；这是 host 新鲜度取舍，不解释设备 interval 分类。

| 指标 | Slow P50 | Normal P50 | Slow P95 | Normal P95 |
| --- | ---: | ---: | ---: | ---: |
| frame-ready interval | 71.87 ms | 36.02 ms | 83.29 ms | 36.25 ms |
| frame acquire | 48.42 ms | 13.74 ms | 54.82 ms | 17.85 ms |
| preceding response write | 22.78 ms | 21.44 ms | 50.04 ms | 30.67 ms |
| JPEG/metadata prepare | 0.74 ms | 0.75 ms | 1.27 ms | 0.86 ms |
| JPEG size | 31,360 B | 31,371 B | 31,902 B | 31,857 B |
| ToF age at JPEG ready | 68.13 ms | 47.25 ms | 116.18 ms | 89.20 ms |

实际 exposure value 在 slow/normal 全部为 `490`；Wi-Fi RSSI 平均值为
`-34.92/-33.95 dBm`，free heap P50 均为 `149,056 B`。这不支持曝光变化、JPEG
体积或内存下降为主机制。

## 机制分层

对 1,252 个 slow frame 作不改变冻结 slow 标签的诊断分层：

- `981`（78.4%）：`acquire >= 30 ms` 且 preceding write `<40 ms`，由 camera
  framebuffer/帧节拍等待主导；
- `130`（10.4%）：preceding write `>=40 ms`，网络写阻塞直接占用下一 interval；
- `141`（11.3%）：不落入上述两个简化诊断桶，保留为 mixed/other。

当 preceding write `40–80 ms` 或 `>=80 ms` 时，本次样本的 slow rate 均为 100%。
最大 interval `1,280.338 ms` 的 preceding write 为 `1,278.869 ms`，是明确的 Wi-Fi/
socket write 尖峰；另一些约 `144 ms` interval 则由 `115–126 ms` acquire 等待形成。

ToF update 出现在 acquire 内时，slow acquire P50 为 `51.27 ms`，未出现时为
`44.32 ms`；normal 对应 `15.20/13.41 ms`。因为更长 acquire 本身更容易覆盖 ToF
更新时刻，这一观察不能证明 ToF 是原因。下一单变量实验只能关闭 ToF 读取，保持其他
配置与五分钟合同不变。

## 边界与证据

本地 summary：
`artifacts.local/evidence/atoms3r-e2e/20260805T102420.808660Z/summary.json`，SHA-256
`e2d542665bbea7b7c808c321295675c5f72611141978475c9569e3b813782b11`。
最终固件 program/RAM 为 `1,077,639 B (32%) / 62,608 B (19%)`，app bin SHA-256
为 `713973e77c79f4f4c50508da6e07bc37490121c8fc2eb32711c206e4f0d2642a`。

测试结束后 `stream_clients=0`。本结果是 Development 性能机制证据，不是摄像头
图像质量、ToF 精度、风险准确率、真实语音/震动、人体、产品或安全证据。
