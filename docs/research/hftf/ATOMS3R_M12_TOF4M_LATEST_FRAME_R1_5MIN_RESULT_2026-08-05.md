# AtomS3R-M12 + ToF4M latest-frame R1 五分钟结果

## 结论

`DEVELOPMENT_5MIN_LATENCY_REGRESSION_PASS / LONG_STRESS_NOT_RERUN`

旧 30 分钟账本显示，端到端 P95 `265.8 ms` 中的主导异常不是 YOLO 推理，而是
`jpeg_ready → host_read_start` P95 `179.4 ms`。原主机脚本串行执行“读下一帧→解码
→推理→风险”，处理期间 TCP 缓冲持续积累旧 JPEG。

R1 将 MJPEG socket 读取移入独立线程，并在 reader 与处理器之间使用容量 1 的
latest-frame 队列。消费者落后时覆盖旧帧、增加明确计数，不允许无限 backlog。

## 五分钟回归

配置保持 XGA `1024×768`、JPEG quality 10、同一设备/网络和 host CPU YOLO11n
参考 pipeline。运行 `300.297 s`，处理 7,158 帧，有效处理速率 `23.84 fps`。

| 指标 | P50 | P95 | P99 / max |
| --- | ---: | ---: | ---: |
| capture → host JPEG complete | 75.7 ms | 112.6 ms | P99 147.5 ms |
| JPEG ready → host read start | 3.7 ms | 7.2 ms | max 39.7 ms |
| host latest queue wait | 0.15 ms | 12.2 ms | max 39.3 ms |
| capture → decode complete | 78.5 ms | 115.4 ms | P99 149.9 ms |
| host inference | 30.3 ms | 33.3 ms | P99 39.0 ms |
| capture → feedback record | 109.3 ms | 146.8 ms | P99 180.3 ms |
| absolute ToF—capture skew | 23.2 ms | 51.3 ms | max 57.4 ms |

- stream reconnect `0`，error `0`。
- latest queue overwrite `2`，对应 sequence gap `2`；约占 reader 到达帧的 `0.028%`。
- ToF valid fraction `1.0`。
- free heap 首尾增加 `5,520 B`，全程最小 `145,996 B`；没有五分钟单调下降迹象。
- ESP32 internal sensor `68.1..71.1 °C`；Wi-Fi RSSI P50 `-36 dBm`。
- 测试结束后 `/api/status` 报告 `stream_clients=0`，未遗留流连接或测量进程。

相对旧 30 分钟基线，capture→feedback P95 从 `265.8 ms` 降至 `146.8 ms`
（约 `44.8%`），JPEG ready→host read start P95 从 `179.4 ms` 降至 `7.2 ms`
（约 `96.0%`）。持续时间不同，因此这些百分比是机制诊断和日常回归对照，不替代
同长度压力测试结论。

## 失败片段与边界

用户将默认回归时长改为五分钟后，外层命令被终止，但旧批处理包装器留下了 30 分钟
Python 子进程，占用设备唯一 stream handler。紧接的一次五分钟尝试得到 0 帧并被
排除；精确结束该已识别进程树后，`stream_clients` 从 1 回到 0，随后正式回归通过。
今后日常运行直接使用项目虚拟环境 Python；30–60 分钟仅在用户明确要求压力测试时运行。

本地正式 summary：
`artifacts.local/evidence/atoms3r-e2e/20260805T095101.730543Z/summary.json`，SHA-256
`ac6bd0ddf72f85d7bb282cc79000036d6c16cd99da89cc1853a1be4b215ac854`。

本结果仍只覆盖 host reference 的 feedback record，不覆盖真实语音/震动执行、风险
准确率、手机端性能、RGB-ToF 空间标定、人体使用、产品或安全结论。
