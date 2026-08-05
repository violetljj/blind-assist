# AtomS3R-M12 + ToF4M stream preamble coalescing R7 result

## 结论

将每帧 MJPEG boundary 与 metadata header 合并为一个 HTTP chunk 的候选不晋升：

`STREAM_PREAMBLE_COALESCE_NOT_PROMOTED / TYPICAL_GAIN_TOO_SMALL / EXTREME_WRITE_STALL_OBSERVED`

相邻五分钟单变量 A/B 中，合并候选的 device response write P50/P95 从
`25.531/35.616 ms` 降至 `24.193/33.504 ms`，但 P99 只从 `39.244 ms` 降至
`38.988 ms`。候选同时出现一次 `1563.568 ms` 的设备写出阻塞，对应 host JPEG read
`1565.344 ms` 和 capture→feedback `1620.721 ms`；split 基线三项最大值分别只有
`96.897/95.620/187.649 ms`。实时路线不接受用约 1 ms 的常态收益交换一次已观测到的
1.6 秒冻结，正式配置恢复为 split preamble，并继续保留 R6 的 `TCP_NODELAY`。

一次相邻 A/B 不能证明 coalescing 必然造成该尖峰，也不能证明 split 永远不会产生
极端阻塞；因此终态是“不晋升”，而不是机制性定罪。该候选的正常分位收益过小，
不足以支持承担额外不确定性或继续消费同一环境进行追参。

## 冻结条件

- 相机：XGA、JPEG quality 10、brightness 1、自动曝光开启
- 缓冲：double buffer、LATEST、PSRAM DMA 关闭
- ToF：连续采样开启
- 网络：`TCP_NODELAY` 开启并逐连接回读
- 主机：18 logical CPU，reference TFLite pipeline 4 threads，latest-frame queue
- 每臂 300 秒；唯一变量为 boundary 与 metadata header 分两次还是一次 chunk 写出

固件状态 API 增加 `stream_preamble_coalesced_configured`，逐帧增加
`X-Stream-Preamble-Coalesced`，主机缺失字段时 fail closed，并在 summary 汇总实际值。

## A/B 结果

| 指标 | split 基线 | coalesced 候选 | 变化 |
| --- | ---: | ---: | ---: |
| 帧数 / fps | 7,223 / 24.025 | 7,358 / 24.481 | +0.456 fps |
| reconnect / error / overwrite / gap | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 均通过 |
| slow-frame fraction | 20.69% | 15.51% | 相机节拍有利但非目标变量 |
| response write P50 | 25.531 ms | 24.193 ms | -1.338 ms |
| response write P95 | 35.616 ms | 33.504 ms | -2.112 ms |
| response write P99 | 39.244 ms | 38.988 ms | -0.256 ms |
| response write max | 96.897 ms | 1563.568 ms | +1466.671 ms |
| host JPEG read P50 | 23.700 ms | 22.804 ms | -0.896 ms |
| host JPEG read P95 | 33.503 ms | 31.978 ms | -1.525 ms |
| host JPEG read P99 | 37.316 ms | 36.849 ms | -0.466 ms |
| host JPEG read max | 95.620 ms | 1565.344 ms | +1469.724 ms |
| capture→feedback P50 | 81.129 ms | 79.543 ms | -1.586 ms |
| capture→feedback P95 | 119.066 ms | 116.650 ms | -2.416 ms |
| capture→feedback P99 | 128.825 ms | 126.253 ms | -2.572 ms |
| capture→feedback max | 187.649 ms | 1620.721 ms | +1433.071 ms |

异常 frame `3879` 的 capture→JPEG ready 为 `36.604 ms`、JPEG 为 `31,938 B`、
RSSI `-32 dBm`、free heap `149,048 B`，但 device response write 与 host JPEG read
分别为 `1563.568/1565.344 ms`。相机、图像大小、RSSI 和 heap 没有同时异常，阶段
账本将冻结定位在设备写出/网络接收窗口；它不是主机 latest queue 或模型排队造成。

## 证据与边界

- split：`artifacts.local/evidence/atoms3r-e2e/20260805T115807.269377Z/`
- split summary SHA-256：
  `a4dd9d2fde42e3a568ecc57bfb60279571a871d8ddac36e31ff8b84b72bb289a`
- coalesced：`artifacts.local/evidence/atoms3r-e2e/20260805T120423.528527Z/`
- coalesced summary SHA-256：
  `782e9e74c8a78e93a99db9bebd510f990eae391a997b944f8d4a834b597304b4`

两臂 RSSI P50 均为 `-33 dBm`，ToF 全部 valid，pipeline threads 均为 4，且均无
reconnect/error/overwrite/gap。候选较低的 slow fraction 表明相机节拍环境并未更差，
但也禁止把吞吐变化完全归因于 chunk 合并。结果只约束当前 Development 传输路线，
不授权画质、准确率、人体、产品或安全结论。

## 正式恢复状态

- 固件身份：`atoms3r_m12_tof4m_stream_r8_preamble_split`
- program/RAM：`1,078,267 B (32%) / 62,608 B (19%)`
- app bin SHA-256：
  `a9f265e6db715b106438b6dfffb1e05d8680f7514cd3a5bcf4195de1d1a68a73`
- 刷入 COM5 后 20 帧带模型验收：0 reconnect/error/overwrite/gap；全部帧
  `TCP_NODELAY=true`、`preamble_coalesced=false`、ToF sampling/valid、pipeline
  threads=4；退出后 `stream_clients=0`
