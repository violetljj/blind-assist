# AtomS3R Android QVGA/Q10 + age60 R35 结果（2026-08-06）

## 结论

拒绝 `maxFrameAgeMs=60`。相较 QVGA/age65，60 ms 门限没有降低实际结果年龄或端到端尾延迟；10 秒结果的 P95/P99 反而略高，不值得继续做 1 分钟确认。

## 10 秒结果

- AtomS3R：QVGA `320×240` / JPEG Q10 / ToF-on；
- Android：`decodeSampleSize=1`、`maxFrameAgeMs=60`；
- 271 帧，0 error，0 reconnect；
- source packets `274`，latest overwrite `0`，stale dropped `3`（约 1.1%）；
- capture→risk P50/P95/P99：`54.54 / 68.47 / 77.48 ms`；
- frame age at first byte P50/P95：`45.69 / 54.35 ms`；
- frame age at risk-ready P50/P95：`54.54 / 68.47 ms`。

QVGA/age65 的 1 分钟结果为 capture→risk `54.20 / 66.48 / 71.92 ms`，stale dropped `2/1663`。因此保留 age65。

原始证据：`artifacts.local/evidence/atoms3r-qvga-age60-r35/`。

测试后设备恢复 SVGA/Q10；本结果为 development-only 性能诊断，不是质量、安全或 false-clear 证据。
