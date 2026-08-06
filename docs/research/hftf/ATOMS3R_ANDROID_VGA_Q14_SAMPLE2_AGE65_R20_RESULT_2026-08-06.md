# AtomS3R Android VGA/Q14 + decodeSampleSize=2 + stale gate R20 结果（2026-08-06）

## 结论

将 `inSampleSize=2` 与 `maxFrameAgeMs=65` 组合，得到当前最低稳定端到端性能；准入门只跳过明显过期帧，正常帧不增加处理成本。

## 1 分钟结果

- 1660 帧，0 error，0 reconnect，3 sequence gap
- source packets 1664，stale packets dropped `4`（约 0.24%）
- frame age at first byte：P50 `42.58 ms`，P95 `46.49 ms`，P99 `49.87 ms`
- capture→risk：P50 `59.27 ms`，P95 `70.80 ms`，P99 `74.43 ms`
- decode：P50 `1.79 ms`，P95 `5.75 ms`，P99 `6.22 ms`
- preprocess：P50 `0.40 ms`，P95 `1.08 ms`，P99 `1.31 ms`
- detector total：P50 `5.67 ms`，P95 `13.07 ms`，P99 `14.12 ms`

相对完整解码 VGA/Q14，组合策略 P95 约下降 `3.22 ms`、P99 约下降 `6.32 ms`。

## 边界

该策略仅作为 `AI_REALTIME` 性能候选，默认关闭。阈值和源分辨率必须经过检测质量、风险一致率和 false-clear 对照后才能晋升；结果为 development-only 性能证据。
