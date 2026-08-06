# AtomS3R Android VGA/Q18 R21 结果（2026-08-06）

## 结论

在当前最佳组合 `VGA + decodeSampleSize=2 + maxFrameAgeMs=65` 上，将 JPEG quality 从 14 单变量调整为 18。Q18 的 JPEG 略小，但端到端收益相对 Q14 落在测量噪声范围，且会进一步损失画质，因此不晋升，继续保留 Q14 作为更保守候选。

## 1 分钟结果

- 1654 帧，0 error，0 reconnect
- source packets 1660，stale dropped 6，5 sequence gap
- JPEG size：P50 `7033 B`，P95 `7098 B`，P99 `7120 B`
- capture→JPEG complete：P50 `49.28 ms`，P95 `54.74 ms`，P99 `57.85 ms`
- frame age at first byte：P50 `42.46 ms`，P95 `47.16 ms`，P99 `50.47 ms`
- decode：P50 `1.74 ms`，P95 `5.64 ms`，P99 `6.16 ms`
- preprocess：P50 `0.40 ms`，P95 `1.06 ms`，P99 `1.25 ms`
- detector total：P50 `5.63 ms`，P95 `13.17 ms`，P99 `14.14 ms`
- capture→risk：P50 `58.77 ms`，P95 `70.31 ms`，P99 `74.47 ms`

Q14 组合对照为 `59.27/70.80/74.43 ms`。Q18 的普通分位略低、P99相同，差异不足以抵消画质风险。

结果为 development-only 性能证据，不代表检测准确率、安全或物理反馈起点。
