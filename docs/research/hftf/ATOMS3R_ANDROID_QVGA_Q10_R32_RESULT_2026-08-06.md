# AtomS3R Android QVGA/Q10 R32 结果（2026-08-06）

## 结论

保留为当前最佳 `AI_REALTIME` 性能候选，但不直接晋升默认。QVGA `320×240` 明显降低了设备 JPEG 生成、传输和 Android 解码成本；检测质量、小目标召回、风险一致率和 false-clear 尚未完成对照。

## 实现

固件新增可选分辨率：

```text
QVGA = 320×240
```

默认分辨率、网页控制和恢复配置保持不变；默认仍为 XGA，测试后恢复为 SVGA/Q10。

## 1 分钟真机结果

- 固件：`atoms3r_m12_tof4m_stream_r11_per_frame_copy_buffer`，ToF-on；
- AtomS3R：QVGA `320×240` / JPEG Q10 / 自动曝光；
- Android：完整解码 `decodeSampleSize=1`、`maxFrameAgeMs=65`；
- 手机：SM-S9280 / SM8650 / Android 16；
- 1660 帧，0 error，0 reconnect；
- source packets `1663`，latest overwrite `1`，stale dropped `2`；
- capture→risk P50/P95/P99：`54.20 / 66.48 / 71.92 ms`；
- capture→JPEG complete P50/P95：`45.00 / 51.85 ms`；
- JPEG decode P50/P95：`1.37 / 3.51 ms`；
- preprocess P50/P95：`0.41 / 1.14 ms`；
- detector total P50/P95：`5.64 / 13.90 ms`；
- JPEG size P50/P95：`2222 / 2266 bytes`；
- PSS：`188701 → 185117 KiB`，没有出现短测内存上升信号。

相对 VGA/Q14 + sample2 + age65 的 R20（capture→risk `59.27 / 70.80 / 74.43 ms`），QVGA 在本次 1 分钟确认中改善约 `5.07 / 4.32 / 2.51 ms`。这是性能对照，不是质量或安全结论。

## 边界与证据

原始证据：`artifacts.local/evidence/atoms3r-qvga-r32-1min/`。

该结果为 development-only 性能证据。只有在检测质量、小目标召回、风险一致率和 false-clear 对照通过后，QVGA 才能成为产品默认实时档；否则保留 VGA/SVGA 作为质量优先档。
