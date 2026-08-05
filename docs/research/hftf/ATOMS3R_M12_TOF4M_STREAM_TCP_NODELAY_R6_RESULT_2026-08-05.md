# AtomS3R-M12 + ToF4M MJPEG TCP_NODELAY R6 result

## 结论

当前 AtomS3R-M12 MJPEG 正式路线启用 `TCP_NODELAY`，终态为：

`TCP_NODELAY_PROMOTED_FOR_TAIL_LATENCY / MEDIAN_WRITE_COST_ACCEPTED / EXTREME_STALL_NOT_PROVEN_ELIMINATED`

在相邻的五分钟单变量 A/B 中，开启后设备 `response_write` P50/P95/P99
由 `22.13/30.83/35.80 ms` 变为 `26.46/34.63/38.82 ms`，常态写出约慢
`4 ms`；但主机完整 JPEG 读取 P95/P99 从 `62.48/72.02 ms` 降到
`32.86/37.23 ms`，最大值从 `1090.78 ms` 降到 `59.29 ms`。对需要最新画面
而不是历史帧吞吐的实时链路，尾延迟收益高于这项中位成本。

这只证明当前设备、网络、冻结配置和相邻两次运行下的 Development 性能结果。
一次 A/B 不能证明以后不会再次出现秒级网络阻塞，也不授权准确率、人体、产品或
安全结论。

## 冻结条件

- 设备：AtomS3R-M12 + Unit ToF4M，局域网地址 `192.168.5.11`
- 相机：XGA、JPEG quality 10、brightness 1、自动曝光开启且实测 exposure 490
- 相机缓冲：double buffer、LATEST、PSRAM DMA 关闭
- ToF：连续采样开启且正式运行全部有效
- 主机：18 logical CPU，参考 TFLite pipeline 固定 4 threads
- 日常运行：300 秒；未执行 30–60 分钟压力测试
- 唯一实验变量：MJPEG stream socket 的 `TCP_NODELAY`

固件在建立 stream 后对具体 socket 执行 `setsockopt(TCP_NODELAY)`，随后用
`getsockopt` 回读；应用或回读失败时 stream fail closed。状态 API 记录配置值，
每帧 `X-Stream-Tcp-Nodelay` header 记录具体连接的回读值，主机 summary 汇总实际值。

## 五分钟结果

| 指标 | TCP_NODELAY off，R5 基线 | TCP_NODELAY on，R6 | 解读 |
| --- | ---: | ---: | --- |
| 时长 | 300.609 s | 300.547 s | 等价 |
| 处理帧 / fps | 6,927 / 23.043 | 6,938 / 23.085 | 等价 |
| reconnect / error | 0 / 0 | 0 / 0 | 均通过 |
| overwrite / sequence gap | 0 / 0 | 0 / 0 | 主机无积压 |
| device response write P50 | 22.129 ms | 26.461 ms | +4.332 ms |
| device response write P95 | 30.830 ms | 34.631 ms | +3.801 ms |
| device response write P99 | 35.802 ms | 38.821 ms | +3.019 ms |
| device response write max | 1095.318 ms | 59.978 ms | 本轮无秒级尖峰 |
| host JPEG read P50 | 24.147 ms | 24.507 ms | 基本相同 |
| host JPEG read P95 | 62.482 ms | 32.862 ms | -29.621 ms |
| host JPEG read P99 | 72.016 ms | 37.234 ms | -34.782 ms |
| host JPEG read max | 1090.784 ms | 59.294 ms | 本轮无秒级尖峰 |
| capture→完整 JPEG P50 | 67.132 ms | 66.367 ms | 基本相同 |
| capture→完整 JPEG P95 | 111.982 ms | 105.129 ms | -6.853 ms |
| capture→完整 JPEG P99 | 147.625 ms | 135.429 ms | -12.196 ms |
| capture→完整 JPEG max | 1205.109 ms | 179.030 ms | 尾部显著收窄 |
| capture→feedback P50 | 82.979 ms | 82.301 ms | 基本相同 |
| capture→feedback P95 | 128.937 ms | 121.366 ms | -7.571 ms |
| capture→feedback P99 | 164.421 ms | 151.237 ms | -13.184 ms |
| capture→feedback max | 1221.089 ms | 197.787 ms | 尾部显著收窄 |

开启运行的 latest queue wait P50/P95/P99 为 `0.065/0.099/0.203 ms`，推理
为 `12.596/14.677/16.113 ms`；0 overwrite 和 0 sequence gap 说明网络尾部改善
不是由主机丢弃大量旧帧制造的。

## 反证与边界

- `jpeg_ready→host read start` P50/P95/P99 从 `3.401/6.385/9.723 ms`
  变为 `3.350/6.883/10.720 ms`，没有同步改善；收益主要位于完整 JPEG 传输尾部。
- slow-frame fraction 从 `19.72%` 变为 `22.50%`，相机交付双周期噪声仍在，不能把
  本轮解释为相机变快。
- 两次运行顺序固定且环境不可能完全相同；特别是 off 基线中的 `1.095 s` 写出尖峰
  可能是偶发网络事件。晋升依据是实时系统对尾延迟和冻结风险的偏好，而不是声称
  TCP_NODELAY 必然消灭所有未来尖峰。
- 未做固定曝光优化；正式配置继续保持自动曝光。

## 证据

- off 基线：`artifacts.local/evidence/atoms3r-e2e/20260805T112314.111945Z/`
- off summary SHA-256：
  `066fcea70270657db40a227603477bcd65407144498a96ed8d6c4917ac23f6a2`
- on 正式运行：`artifacts.local/evidence/atoms3r-e2e/20260805T114340.682967Z/`
- on summary SHA-256：
  `ada9f563f5a45136f48e0c4782c6d7f0bc2ded358bfd81f7cca9270779d4f540`
- 正式固件身份：`atoms3r_m12_tof4m_stream_r7_tcp_nodelay`
- 正式发布 app bin SHA-256：
  `84f059606efa0cb0560a8f7fe7110c38d8df30b22ae7a66a188ee3a608cd1d3f`
- 正式发布 program/RAM：`1,078,167 B (32%) / 62,608 B (19%)`
- 发布后 20 帧回归：0 reconnect/error/overwrite/gap，全部帧
  `stream_tcp_nodelay=true`、ToF sampling/valid、pipeline threads=4；退出后
  `stream_clients=0`
