# AtomS3R Android VGA/Q16 R31 结果（2026-08-06）

## 结论

拒绝 Q16 作为当前实时档默认候选。相较已验证的 VGA/Q14，Q16 在本次 10 秒短测中没有稳定端到端收益，且 JPEG 大小几乎没有下降；不值得继续做 1 分钟确认。

## 测试配置

- 固件：`atoms3r_m12_tof4m_stream_r11_per_frame_copy_buffer`，ToF-on；
- AtomS3R：VGA `640×480` / JPEG Q16 / 自动曝光；
- Android：`decodeSampleSize=2`、`maxFrameAgeMs=65`；
- 手机：SM-S9280 / SM8650 / Android 16；
- 时长：10 秒。

## 结果

- 262 帧，0 error，0 reconnect；
- source packets `275`，latest overwrite `0`，stale dropped `13`；
- capture→risk P50/P95/P99：`62.84 / 76.05 / 80.16 ms`；
- capture→JPEG complete P50/P95：`52.62 / 62.12 ms`；
- JPEG decode P50/P95：`1.77 / 5.61 ms`；
- preprocess P50/P95：`0.43 / 1.06 ms`；
- detector total P50/P95：`5.67 / 12.76 ms`；
- JPEG size P50/P95：`7088 / 7215 bytes`。

R20 VGA/Q14 的 1 分钟结果为 capture→risk `59.27 / 70.80 / 74.43 ms`，Q16 没有显示改善。因此不晋升 Q16。

## 恢复与边界

测试结束后设备已恢复 SVGA/Q10，ToF 保持 `VALID`。原始证据：`artifacts.local/evidence/atoms3r-vga-q16-r31/`。

本结果为 development-only 性能诊断，不是准确率、风险一致率、false-clear、安全或物理反馈证据。
