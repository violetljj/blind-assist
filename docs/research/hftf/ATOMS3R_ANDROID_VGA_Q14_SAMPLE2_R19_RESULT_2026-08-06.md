# AtomS3R Android VGA/Q14 + decodeSampleSize=2 R19 结果（2026-08-06）

## 结论

在 VGA 640×480 / JPEG Q14 下，Android JPEG 解码期 `inSampleSize=2` 稳定降低解码和 letterbox 成本，并改善端到端尾延迟。它是 `AI_REALTIME` 候选参数，默认仍为完整解码，直到检测质量对照完成。

## 1 分钟结果

- 1660 帧，0 error，0 reconnect，1 gap，1 overwrite
- capture→JPEG complete：P50 `49.74 ms`，P95 `55.76 ms`，P99 `59.84 ms`
- frame age at first byte：P50 `42.78 ms`，P95 `47.14 ms`，P99 `50.01 ms`
- first byte→JPEG complete：P50 `6.70 ms`，P95 `11.74 ms`，P99 `15.19 ms`
- decode：P50 `1.79 ms`，P95 `5.71 ms`，P99 `6.22 ms`
- preprocess：P50 `0.41 ms`，P95 `1.10 ms`，P99 `1.33 ms`
- QNN execute：P50 `2.89 ms`，P95 `4.14 ms`，P99 `4.69 ms`
- detector total：P50 `5.67 ms`，P95 `13.02 ms`，P99 `14.01 ms`
- capture→risk：P50 `59.43 ms`，P95 `70.77 ms`，P99 `75.83 ms`
- PSS：`111169 KB → 191043 KB`（短测进程启动/运行基线差异，未见测试错误）

完整解码 VGA/Q14 对照为 `59.85/74.02/80.75 ms`，下采样使 P95 下降约 `3.25 ms`、P99 下降约 `4.92 ms`。

## 边界

结果为 development-only 性能证据，不代表检测准确率、安全或物理反馈起点。必须和 SVGA/Q10 做同场景检测一致率、小目标召回、风险一致率与 false-clear 对照。
