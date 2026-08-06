# AtomS3R Android VGA/Q14 + decodeSampleSize=4 R24 结果（2026-08-06）

## 结论

`inSampleSize=4` 在 VGA/Q14 下不能继续降低全链路延迟，正式拒绝。虽然 JPEG decode 更快，但输出约 160×120，后续必须放大/letterbox 到 320×320，导致预处理和 detector total 反弹。

## 10 秒结果

- 274 帧，0 error，0 reconnect，1 gap，1 overwrite
- stale dropped：0
- frame age at first byte：P50 `42.65 ms`，P95 `48.07 ms`，P99 `54.21 ms`
- capture→risk：P50 `60.67 ms`，P95 `73.71 ms`，P99 `78.50 ms`
- decode：P50 `1.67 ms`，P95 `5.05 ms`，P99 `6.09 ms`
- preprocess：P50 `1.34 ms`，P95 `6.28 ms`，P99 `7.55 ms`
- detector total：P50 `6.55 ms`，P95 `15.86 ms`，P99 `17.90 ms`

sampleSize=2 的 1 分钟组合结果为 capture→risk `59.27/70.80/74.43 ms`，因此 sampleSize=4 不晋升。

## 边界

结果为 development-only 性能证据，不代表检测准确率、安全或物理反馈起点。冻结 `sampleSize=2` 作为 VGA/Q14 AI 实时候选。
