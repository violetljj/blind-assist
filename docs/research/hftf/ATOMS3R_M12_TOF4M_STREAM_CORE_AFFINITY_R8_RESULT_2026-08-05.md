# AtomS3R-M12 + ToF4M stream server core affinity R8 result

## 结论

将 MJPEG HTTP server 从 no-affinity 固定到 core 0 的候选不晋升：

`STREAM_CORE0_AFFINITY_NOT_PROMOTED / NETWORK_START_AND_WRITE_MEDIAN_REGRESSED`

本地工具链确认 Wi-Fi task 与 lwIP TCP/IP task 均固定 core 0，Arduino loop 和 UDP
对时固定 core 1；HTTP server 默认 priority 5、no-affinity。新增逐帧实际 handler core
后发现 no-affinity 基线全部运行在 core 1，因此“固定 core 1”不是有效单变量。唯一候选
改为固定 core 0，测试与网络栈同核是否能减少跨核交互。

五分钟结果显示 core 0 虽将吞吐从 `23.419` 提高到 `24.603 fps`，但
JPEG-ready→host read start P50/P95 从 `3.561/7.587 ms` 恶化到
`6.770/10.431 ms`，device response write P50/P95 从 `26.287/35.132 ms`
恶化到 `31.439/39.009 ms`，capture→feedback P50 从 `82.549 ms` 恶化到
`86.798 ms`。这支持与 Wi-Fi/lwIP 同核争用超过跨核交互收益，正式配置恢复
no-affinity；当前实测 handler 落在 core 1。

## 冻结条件

- XGA、JPEG quality 10、brightness 1、自动曝光开启
- double buffer、LATEST、PSRAM DMA 关闭
- ToF 连续采样开启
- `TCP_NODELAY=true`，preamble split
- stream server task priority 固定 5
- host TFLite 4 threads、latest-frame queue
- 每臂 300 秒；唯一变量为 HTTP stream server `core_id`

状态 API 增加 configured core 与 task priority，逐帧增加实际
`X-Stream-Handler-Core`，主机缺失时 fail closed，并在 summary 汇总实际核心。

## A/B 结果

| 指标 | no-affinity，实际 core 1 | 固定 core 0 | 解读 |
| --- | ---: | ---: | --- |
| 帧数 / fps | 7,040 / 23.419 | 7,397 / 24.603 | core 0 吞吐较高 |
| reconnect/error/overwrite/gap | 0/0/0/0 | 0/0/0/0 | 均通过 |
| response write P50 | 26.287 ms | 31.439 ms | +5.152 ms |
| response write P95 | 35.132 ms | 39.009 ms | +3.877 ms |
| response write P99 | 42.301 ms | 43.308 ms | +1.007 ms |
| response write max | 79.671 ms | 72.217 ms | 均无极端尖峰 |
| JPEG ready→host read start P50 | 3.561 ms | 6.770 ms | +3.208 ms |
| JPEG ready→host read start P95 | 7.587 ms | 10.431 ms | +2.844 ms |
| host JPEG read P50/P95 | 24.466/33.352 ms | 24.689/32.803 ms | JPEG 本体相近 |
| capture→feedback P50 | 82.549 ms | 86.798 ms | +4.249 ms |
| capture→feedback P95 | 121.248 ms | 122.050 ms | +0.802 ms |
| capture→feedback P99 | 148.166 ms | 128.578 ms | 候选较好但受相机节拍混杂 |
| capture→feedback max | 194.256 ms | 160.727 ms | 两臂均无冻结 |

core 1 基线的 camera capture P99/max 为 `108.413/144.687 ms`，core 0 候选只有
`72.736/108.748 ms`；两臂 RSSI P50 也为 `-32/-35 dBm`。因此 core 0 较好的端到端
P99/max 不能干净归因于 affinity。与相机节拍较不相关的网络起始和设备写出指标方向
一致地恶化，足以拒绝候选。

## 证据与边界

- no-affinity：`artifacts.local/evidence/atoms3r-e2e/20260805T124417.386012Z/`
- no-affinity summary SHA-256：
  `667b90f98136b91d5022ce51c4b222bc139945b80d2e6ba87ca8d6f210f349b1`
- core 0：`artifacts.local/evidence/atoms3r-e2e/20260805T125119.177054Z/`
- core 0 summary SHA-256：
  `e5daa4f707d282aa5df4a01b920a40a5c7dc675fefec4e40e5604457aaa5deec`
- 两臂实际 handler core 分别固定为 `[1]` 与 `[0]`

本结果只约束当前硬件、工具链与网络的 Development 调度路线；不授权画质、ToF
精度、风险准确率、人体、产品或安全结论。task priority 未扫描，避免和 affinity 混杂。

## 正式恢复状态

- 固件：`atoms3r_m12_tof4m_stream_r9_no_affinity`
- configured core：`tskNO_AFFINITY`，task priority：5
- program/RAM：`1,078,375 B (32%) / 62,608 B (19%)`
- app bin SHA-256：
  `928af057147198a233cb10db1c5e464307321e705f4c2b6647b1f056ea847bc8`
- 刷入 COM5 后 20 帧验收：实际 handler core 全为 `[1]`，0 reconnect/error/
  overwrite/gap，TCP_NODELAY=true、preamble split、自动曝光、ToF sampling/valid；
  退出后 `stream_clients=0`
