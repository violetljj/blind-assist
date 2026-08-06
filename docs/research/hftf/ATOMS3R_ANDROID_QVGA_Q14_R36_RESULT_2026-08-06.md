# AtomS3R Android QVGA/Q14 R36 结果（2026-08-06）

## 结论

拒绝把 QVGA JPEG quality 从 Q10 改为 Q14。10 秒结果与 Q10 接近，没有稳定端到端收益；已确认的 QVGA/Q10 1 分钟结果更好。

## 10 秒结果

- AtomS3R：QVGA `320×240` / JPEG Q14 / ToF-on；
- Android：`decodeSampleSize=1`、`maxFrameAgeMs=65`；
- 274 帧，0 error，0 reconnect；
- source packets `275`，latest overwrite `0`，stale dropped `1`；
- capture→risk P50/P95/P99：`55.33 / 69.05 / 72.80 ms`；
- capture→JPEG complete P50/P95：`46.03 / 53.22 ms`；
- JPEG decode P50/P95：`1.37 / 3.51 ms`；
- JPEG size P50/P95：`2241 / 2262 bytes`。

QVGA/Q10 的 1 分钟结果为 capture→risk `54.20 / 66.48 / 71.92 ms`，因此保留 Q10。

原始证据：`artifacts.local/evidence/atoms3r-qvga-q14-r36/`。

测试后设备恢复 SVGA/Q10；本结果为 development-only 性能诊断，不是质量、安全或 false-clear 证据。
