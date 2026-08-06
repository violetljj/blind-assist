# AtomS3R Android QVGA/Q10 + age55 R34 结果（2026-08-06）

## 结论

拒绝 `maxFrameAgeMs=55`。在当前 QVGA 实时档中，55 ms 准入门没有改善实际风险结果年龄，反而略微恶化端到端延迟并增加丢帧。

## 10 秒结果

- AtomS3R：QVGA `320×240` / JPEG Q10 / ToF-on；
- Android：`decodeSampleSize=1`、`maxFrameAgeMs=55`；
- 262 帧，0 error，0 reconnect；
- source packets `275`，latest overwrite `0`，stale dropped `13`（约 4.7%）；
- capture→risk P50/P95/P99：`57.29 / 68.07 / 70.97 ms`；
- frame age at first byte P50/P95：`45.98 / 52.30 ms`；
- frame age at risk-ready P50/P95：`57.29 / 68.07 ms`；
- preprocess P50/P95：`0.47 / 1.23 ms`。

已确认的 QVGA/age65 1 分钟结果为 capture→risk `54.20 / 66.48 / 71.92 ms`，stale dropped 仅 `2/1663`。因此保留 age65，不采用 age55。

原始证据：`artifacts.local/evidence/atoms3r-qvga-age55-r34/`。

测试后设备恢复 SVGA/Q10；本结果为 development-only 性能诊断，不是质量、安全或 false-clear 证据。
