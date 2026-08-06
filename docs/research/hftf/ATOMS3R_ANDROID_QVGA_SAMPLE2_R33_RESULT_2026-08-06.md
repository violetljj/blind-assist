# AtomS3R Android QVGA/Q10 + sample2 R33 结果（2026-08-06）

## 结论

拒绝 `QVGA + decodeSampleSize=2`。QVGA 源图只有 `320×240`，下采样到约 `160×120` 后再放大到 320×320，虽然 JPEG 解码略快，却显著增加预处理成本，端到端没有收益。

## 10 秒结果

- AtomS3R：QVGA `320×240` / JPEG Q10 / ToF-on；
- Android：`decodeSampleSize=2`、`maxFrameAgeMs=65`；
- 270 帧，0 error，0 reconnect；
- source packets `275`，latest overwrite `1`，stale dropped `4`；
- capture→risk P50/P95/P99：`56.83 / 71.48 / 76.82 ms`；
- capture→JPEG complete P50/P95：`46.40 / 55.48 ms`；
- JPEG decode P50/P95：`1.19 / 2.85 ms`；
- preprocess P50/P95：`1.31 / 7.36 ms`；
- detector total P50/P95：`6.57 / 17.02 ms`；
- JPEG size P50/P95：`2203 / 2252 bytes`。

已确认的 QVGA/sample1 1 分钟结果为 capture→risk `54.20 / 66.48 / 71.92 ms`，因此 QVGA 必须使用完整解码 `decodeSampleSize=1`。

原始证据：`artifacts.local/evidence/atoms3r-qvga-sample2-r33/`。

测试后设备恢复 SVGA/Q10；本结果为 development-only 性能诊断，不是质量、安全或 false-clear 证据。
