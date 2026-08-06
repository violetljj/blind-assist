# AtomS3R Android 解码下采样 R13 结果（2026-08-06）

## 结论

`BitmapFactory.inSampleSize=2` 能稳定降低 JPEG 解码耗时约 1.36 ms（P50），但没有显著压低 capture→risk；它暂不切换为默认配置，保留为 AI 实时档候选参数。

## 1 分钟结果

- 1631 帧，0 error，0 reconnect，0 gap，0 overwrite
- decode：P50 `2.30 ms`，P95 `7.20 ms`，P99 `8.52 ms`
- capture→risk：P50 `66.80 ms`，P95 `82.45 ms`，P99 `101.74 ms`
- capture→JPEG complete：P50 `55.26 ms`，P95 `64.16 ms`，P99 `91.20 ms`
- preprocess：P50 `1.19 ms`，P95 `6.06 ms`，P99 `7.07 ms`
- detector total：P50 `6.42 ms`，P95 `15.66 ms`，P99 `17.12 ms`
- frame age at first byte：P50 `42.41 ms`，P95 `47.70 ms`，P99 `79.68 ms`
- PSS：`189928 KB → 194329 KB`

与完整解码 R12/R11（decode P50 约 3.66 ms、capture→risk P50 约 67.5 ms）相比，主要收益只在解码局部；设备端 capture→JPEG 段仍是最大主段。

## 判定

保留 `decodeSampleSize` 可配置能力，默认仍为 `1`，避免在检测质量未完成对照前改变主链路。后续 AI_REALTIME 档可结合 SVGA/VGA 源分辨率与检测质量一起评估 `sampleSize=2`。
