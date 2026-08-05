# AtomS3R-M12 + ToF4M Android 延迟分解 R2

## 结论

R2 在冻结的 XGA `1024×768`、JPEG quality 10、当前 AtomS3R 固件和正式
`com.linnan.blindassist` QNN HTP 路由上完成了逐帧延迟账本。主要瓶颈不在
QNN HTP，而在**设备 JPEG/网络到达之前的帧龄和完整 JPEG 传输**：

- 首字节到达时帧龄 P50/P95/P99：`59.71/81.61/89.94 ms`；
- 首字节到完整 JPEG P50/P95/P99：`33.10/44.66/52.00 ms`；
- 手机收到完整 JPEG 后到 detector 调用只需约 `0.36/1.25/1.39 ms`；
- 稳态预处理 P50/P95/P99：`12.46/16.09/17.42 ms`；
- 真正 host `Interpreter.run` / QNN HTP P50/P95/P99：`2.12/2.36/3.28 ms`；
- 输出读取 `0.03/0.03/0.10 ms`，YOLO decode/NMS `0.81/0.92/1.01 ms`，风险
  计算 `0.08/0.17/0.22 ms`；
- capture→risk P50/P95/P99：`115.39/139.98/154.17 ms`。

因此，“NPU 只占约 13%–14%”的方向判断正确，但 R2 进一步证明：上一版
`detector_stage` 的 15–20 ms 不是纯 NPU，主要是 320×320 预处理；稳定态 HTP
本身约 2–3 ms。

本结果为 Development-only 同设备部署/性能/诊断证据，不证明准确率、安全性、
功耗、跨设备泛化、真实用户效果或物理反馈时延。

## 逐帧账本

每个 JSONL row 以同一 `frame_sequence` 绑定：

```text
设备：capture -> jpeg_ready -> device_send_start
手机：first_byte -> jpeg_complete -> decode_start -> decode_complete
      -> preprocess_start -> preprocess_complete -> qnn_enqueue
      -> qnn_complete -> output_read_complete -> postprocess_complete
      -> risk_ready
```

`qnn_enqueue` 的冻结语义是 host 进入 `Interpreter.run`；QNN delegate 内部
enqueue 不由公开 LiteRT API 单独暴露，因此不能把该时间点描述成 DSP 硬件入队。
所有 Android 阶段都统一使用 `SystemClock.elapsedRealtimeNanos()`；首轮 smoke
发现并修正了混用 `System.nanoTime()` 与 `elapsedRealtimeNanos()` 的时钟域错误，
错误 smoke 未纳入正式结果。

每帧同时保存：

- `frame_age_at_first_byte_ms`
- `frame_age_at_decode_start_ms`
- `frame_age_at_preprocess_start_ms`
- `frame_age_at_qnn_enqueue_ms`
- `frame_age_at_postprocess_complete_ms`
- `frame_age_at_risk_ready_ms`

## 5 分钟正式短测

配置保持：XGA `1024×768`、quality 10、当前亮度/自动曝光、ToF sampling 开启、
endpoint `http://192.168.5.11`、SM-S9280/SM8650/Android 16、正式 App QNN HTP。

| 项目 | 结果 |
| --- | ---: |
| 请求/实际时长 | 300 s / 300.868 s |
| 处理帧 | 7232 |
| 有效处理帧率 | 约 24.03 fps |
| source packets read | 7233 |
| latest overwrite / sequence gap | 1 / 1 |
| 重连 / 流错误 / 记录错误 | 0 / 0 / 0 |
| clock sync | 10 成功 / 0 失败 |
| ToF | 逐帧记录；标定仍暂停 |
| QNN execute | 7232/7232 `status 0x0` |
| CPU fallback | 0 |
| PSS | 189655 -> 189973 KB |

| 阶段 | P50 | P95 | P99 | 最大值 |
| --- | ---: | ---: | ---: | ---: |
| capture -> JPEG complete | 94.08 | 117.75 | 131.43 | 200.10 ms |
| first byte -> JPEG complete | 33.10 | 44.66 | 52.00 | 115.41 ms |
| JPEG complete -> decode start | 0.30 | 1.13 | 1.20 | 6.37 ms |
| JPEG decode | 4.44 | 8.35 | 11.54 | 22.07 ms |
| decode complete -> detector call | 0.06 | 0.12 | 0.19 | 0.44 ms |
| preprocess | 12.46 | 16.09 | 17.42 | 21.33 ms |
| QNN `Interpreter.run` | 2.12 | 2.36 | 3.28 | 6.65 ms |
| output read | 0.03 | 0.03 | 0.10 | 0.69 ms |
| postprocess | 0.81 | 0.92 | 1.01 | 34.52 ms |
| risk | 0.08 | 0.17 | 0.22 | 2.83 ms |
| detector total | 15.51 | 19.21 | 20.59 | 61.17 ms |
| capture -> risk | **115.39** | **139.98** | **154.17** | **223.89 ms** |

## Perfetto 验证

使用 package-filtered Perfetto trace 验证自定义 slices 真实进入正式 App 的
`pool-4-thread-2`，可见：

```text
BlindAssist.AtomS3rJpegDecode
BlindAssist.YoloPreprocess
BlindAssist.QnnExecute
BlindAssist.YoloOutputRead
BlindAssist.YoloPostprocess
BlindAssist.RiskDecision
```

一个长样本为 JPEG decode 5.065 ms、preprocess 24.192 ms、QNN 5.235 ms、
output read 0.325 ms、postprocess 29.581 ms、risk 2.653 ms；随后普通样本为
3.238/9.819/2.151/0.033/0.874/0.173 ms。这说明偶发长尾可发生在 CPU 预处理或
postprocess 调度/执行，不能归因给 HTP。线程状态查询未建立应用线程持续 `D`
状态的 I/O 阻塞证据。

Perfetto 证据目录：
`artifacts.local/evidence/atoms3r-android-latency-decomposition-r2-perfetto-20260806/`

## 实现变更

- `DetectorStageTiming`：拆分 preprocess、host QNN execute、output read、postprocess；
- `Trace.beginSection`：JPEG decode、YOLO preprocess、QNN execute、output read、
  postprocess、risk；
- instrumentation schema 升级为
  `blindassist_atoms3r_android_latency_decomposition_r2_*`；
- 保留旧 external timing 与 ToF 逐帧绑定；未修改模型、风险逻辑、曝光、协议或
  设备固件。

## 下一步授权

只授权一个下一候选：

> **AI_REALTIME：SVGA 源分辨率单变量对照**

暂不做 VGA、质量大扫描、TCP 二进制协议、设备采集/发送多任务、INT8、固定曝光
或 QNN further tuning。SVGA 对照必须保持同一手机 App、同一模型/风险逻辑，并先
执行 10 秒 smoke，再执行 1 分钟短测；若通过再做 5 分钟正式 A/B。比较 capture→
JPEG ready、JPEG bytes、first byte/full frame、decode、preprocess、QNN、capture→
risk、帧龄、检测/风险一致性和稳定性。

## 证据

- 正式 5 分钟目录：`artifacts.local/evidence/atoms3r-android-latency-decomposition-r2-5min-20260806/`
- frames SHA-256：`5798E9269748E5BA0234CDBFADB9A5AA77304EFBEAEC1A8C617341595FC81904`
- summary SHA-256：`24EEB3DC192AC685D6D68D50B779C407E677D5CA23D4E66C1410C0C1EE869EF2`
- logcat SHA-256：`BD3AE6BB95BF5B41547F212CC8AFEC65620B7CE9F8EDAAF8A2AF56B198ECFEAB`
- R2 smoke（统一时钟修正后）：`artifacts.local/evidence/atoms3r-android-latency-decomposition-r2-smoke-fixed-20260806/`
- Perfetto package-filtered trace：`atoms3r-r2-app.perfetto-trace`，位于上述 Perfetto 证据目录。

本地 evidence 位于 ignored `artifacts.local`；复现实验必须重新绑定 Git、APK、
固件、设备、endpoint 和 instrumentation 输出。
