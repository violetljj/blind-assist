# AtomS3R Android 过期帧准入门 R18 结果（2026-08-06）

## 结论

新增可配置 `maxFrameAgeMs`：Android 在 JPEG 解码前使用统一设备→Android 时间映射检查帧龄，超过阈值直接丢弃并计数，避免为已失去实时价值的旧帧执行解码和推理。默认阈值 `0` 表示关闭。

VGA/Q14 + `maxFrameAgeMs=65` 完成 10 秒和 1 分钟真机验证。它不增加正常帧成本，只丢弃极少数过期帧；P99 有约 1.4 ms 小幅改善，但 P95 无显著变化，因此保留为 AI 实时策略候选，不改变默认。

## 1 分钟结果

- 1657 帧，0 error，0 reconnect
- source packets 1662，1 overwrite，5 sequence gap
- stale packets dropped：4（约 0.24%）
- frame age at first byte：P50 `42.44 ms`，P95 `46.25 ms`，P99 `49.25 ms`
- capture→risk：P50 `59.76 ms`，P95 `74.27 ms`，P99 `79.31 ms`
- JPEG decode：P50 `2.55 ms`，P95 `7.33 ms`，P99 `8.76 ms`
- detector total：P50 `6.44 ms`，P95 `15.48 ms`，P99 `17.00 ms`
- PSS：`188245 KB → 188009 KB`

## 对照

VGA/Q14 无准入门 1 分钟：capture→risk `59.85/74.02/80.75 ms`。因此准入门主要改善极少数尾部旧帧，不改变稳定主分位。

## 边界

结果为 development-only 性能证据，不代表检测准确率、安全或物理反馈起点。阈值需要在同场景检测质量和风险输出对照后才可进入产品策略。
