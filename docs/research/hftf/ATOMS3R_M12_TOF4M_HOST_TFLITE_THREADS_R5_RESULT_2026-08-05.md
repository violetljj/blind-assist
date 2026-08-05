# AtomS3R-M12 + ToF4M 主机 TFLite 4 线程 R5

## 结论

`HOST_TFLITE_THREADS_4_PROMOTED / HOST_BACKLOG_ELIMINATED_IN_5MIN / DEVICE_TAIL_UNCHANGED`

在设备保持自动曝光、XGA/quality 10、PSRAM DMA-off、ToF-on 的正式配置下，将主机
参考 pipeline 从 TFLite 默认单线程改为显式 4 线程，五分钟内把推理 P50/P95/P99 从
`32.47/42.52/48.13 ms` 降至 `12.92/15.00/16.43 ms`。latest queue wait P95 从
`28.67 ms` 降至 `0.18 ms`，覆盖旧帧从 105 降至 0。该主机上的 4 线程配置晋升为
参考 pipeline 默认值，同时保留 `--pipeline-num-threads` 覆盖能力。

本轮设备侧反而出现更多相机慢周期及一次约 1.1 秒 Wi-Fi/socket write 尖峰，因此不
声称设备得到优化。即使输入侧条件略差，capture→反馈记录 P50/P95/P99 仍从
`114.48/149.38/178.37 ms` 改善为 `82.98/128.94/164.42 ms`，支持收益来自主机
推理与排队阶段。

## 预筛选

在当前 18 logical CPU 主机上，对同一实时 XGA JPEG、相同 resize/RGB/float32
预处理和 YOLO11n invoke 做 40 次微基准：

| TFLite threads | P50 | P95 |
| ---: | ---: | ---: |
| 1 | 29.84 ms | 30.79 ms |
| 2 | 16.95 ms | 17.53 ms |
| 4 | 11.89 ms | 12.58 ms |
| 8 | 17.73 ms | 20.17 ms |

8 线程已出现过度并行，不继续扫描线程数；正式实验预先选择 4。

## 五分钟结果

| 指标 | legacy/default single-thread R1 | explicit 4 threads R5 |
| --- | ---: | ---: |
| 时长 | 300.578 s | 300.609 s |
| processed frames/fps | 7,071 / 23.52 | 6,927 / 23.04 |
| reconnect/error | 0/0 | 0/0 |
| pipeline identity | legacy R0 | `TFLITE_THREADS_4` |
| inference P50/P95/P99 | 32.47/42.52/48.13 ms | 12.92/15.00/16.43 ms |
| latest queue wait P50/P95/P99 | 7.90/28.67/35.39 ms | 0.07/0.18/0.27 ms |
| overwritten/gap frames | 105/105 | 0/0 |
| capture→完整 JPEG P50/P95/P99 | 65.01/111.19/140.05 ms | 67.13/111.98/147.63 ms |
| capture→inference P50/P95/P99 | 114.37/149.29/178.28 ms | 82.89/128.82/164.32 ms |
| capture→feedback P50/P95/P99 | 114.48/149.38/178.37 ms | 82.98/128.94/164.42 ms |

R5 共 6,927 帧，`run_accepted=true`，0 reconnect、0 error、0 overwrite、0 sequence
gap。主机 queue wait P95 下降 99.4%，说明 4 线程提供了足够余量，使主机不再因模型
推理贴近相机周期而积压。

R5 的设备 slow fraction 为 19.72%，高于 R1 的 17.71%；最大 frame interval
`1,096.797 ms` 对应前一帧 response write `1,095.318 ms`。capture→完整 JPEG
P95/P99 也略差于 R1。这些输入侧反例使“主机改善”结论更保守：结果不是靠设备本轮
更快伪造的，但也不能用 R5 评价相机或 Wi-Fi 优化。

## 实现与边界

`measure_e2e_latency.py` 新增 `--pipeline-num-threads`，范围 `1..64`，默认 4；
`host_reference_pipeline.py` 将线程数显式传给 LiteRT Interpreter。pipeline identity、
逐帧账本和 summary 均记录线程数，避免默认值漂移。summary 还记录 host logical CPU
count，便于后续跨主机解释。

R5 summary：
`artifacts.local/evidence/atoms3r-e2e/20260805T112314.111945Z/summary.json`，SHA-256
`066fcea70270657db40a227603477bcd65407144498a96ed8d6c4917ac23f6a2`。

这是当前 Windows 主机、当前 LiteRT/XNNPACK 和模型的 Development 性能证据。4 线程
默认值可在较小主机上通过 CLI 下调；本结果不证明手机端线程最优，也不构成模型准确率、
风险、物理反馈、人体、产品或安全证据。
