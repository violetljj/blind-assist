# AtomS3R-M12 + ToF4M stream copy-buffer reuse R10 result

## 结论

将每帧 PSRAM JPEG 暂存区分配改为每连接复用的候选不晋升：

`STREAM_COPY_BUFFER_REUSE_NOT_PROMOTED / DIRECT_COST_NOT_IMPROVED / CORE_MIGRATION_CONFOUNDED`

候选没有改善预期的直接成本。JPEG copy/metadata prepare P50/P95 从
`802/970 us` 变为 `823/996 us`，response write P50/P95 从
`24.849/33.244 ms` 变为 `24.807/33.651 ms`。因此不能把候选臂更高吞吐和更低慢帧
比例归因为缓冲复用。

## 冻结条件与实现

- XGA、quality 10、brightness 1、自动曝光开启
- double framebuffer、LATEST、PSRAM DMA 关闭、ToF 开启
- `TCP_NODELAY=true`、split preamble、no-affinity、priority 5
- host TFLite 4 threads、latest-frame queue、每臂 300 秒
- 唯一预期变量：每帧 `heap_caps_malloc/free`，或每连接按需
  `heap_caps_realloc` 后复用

状态 API 与逐帧 header 新增 `stream_frame_copy_buffer_reused` 身份，主机缺失时
fail closed，summary 汇总实际值。

## 相邻 A/B

| 指标 | per-frame baseline | reuse candidate |
| --- | ---: | ---: |
| 帧数 / fps | 7,016 / 23.341 | 7,487 / 24.903 |
| reconnect/error/overwrite/gap | 0/0/0/0 | 0/0/0/0 |
| slow fraction | 20.68% | 14.96% |
| copy/metadata prepare P50/P95 | 802/970 us | 823/996 us |
| response write P50/P95/P99 | 24.849/33.244/38.786 ms | 24.807/33.651/38.515 ms |
| JPEG-ready to host-read-start P50/P95 | 3.553/7.208 ms | 3.337/7.343 ms |
| capture to feedback P50/P95/P99 | 80.927/119.549/137.365 ms | 80.309/117.145/125.116 ms |
| free-heap first to last | 0 B | -1,604 B |
| actual handler core | `[0,1]` | `[1]` |

吞吐、慢帧和端到端尾部在候选臂较好，但 baseline 发生 core migration，而候选全在
core 1；同时 JPEG 场景内容和相机双周期比例也不同。直接分配/复制与网络写出指标没有
出现对应改善，故这些下游差异不能授权晋升。

## 证据与边界

- baseline：`artifacts.local/evidence/atoms3r-e2e/20260805T135513.027251Z/`
- baseline summary SHA-256：
  `accd812aff6342dd7a105062582824d0651d70a15106e91db4be3e6e97ed70b0`
- candidate：`artifacts.local/evidence/atoms3r-e2e/20260805T140143.137268Z/`
- candidate summary SHA-256：
  `ccecd78191c19955822c24e0e9c9885005e007868a25f1e945d12202877a7913`

## 发布终态

- 固件版本：`atoms3r_m12_tof4m_stream_r11_per_frame_copy_buffer`
- program/RAM：`1,078,575/62,608 bytes`（`32%/19%`）
- application binary SHA-256：
  `5dd4afc81d880674a2e6dd0fe560f42644a85992766b1ed4088335220eb0c732`
- 20 帧 release smoke：0 reconnect/error/overwrite/gap，实际 handler
  core/priority `[1]/[5]`，buffer-reused `[false]`
- 退出后：`stream_clients=0`、自动曝光开启、ToF sampling/valid、Wi-Fi 0 重连

本结果只关闭当前实现的 PSRAM 暂存区复用微优化。工具链同时确认 `SO_SNDBUF` 标为
unimplemented，默认 TCP send buffer 为 5744 B，因此未烧录无效的 socket send-buffer
扫描。结果不授权画质、精度、人体、产品或安全结论。
