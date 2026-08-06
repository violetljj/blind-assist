# AtomS3R Android QVGA native padding R37 结果（2026-08-06）

## 结论

保留为 QVGA `AI_REALTIME` 性能候选。对于源 Bitmap 恰好为 `320×240`、模型输入为 `320×320` 的情况，Android 端跳过 Canvas 和中间 letterbox Bitmap，由 native 直接写入 RGB Float，并将上下黑边清零。该特例没有缩放或插值，保持当前 letterbox 几何语义。

## 实现

- 新增 `NativeBitmapPreprocessor.writePaddedArgbToFloat()`；
- 仅当 `bitmap.width == inputSize`、`scale == 1` 且源高度小于输入高度时启用；
- 使用 Bitmap stride，按 Android RGBA_8888 写 RGB Float；
- 整个输出先清零，避免黑边和前帧残留；
- 失败自动回退到原有 Canvas + native 路径；
- VGA/SVGA/XGA 路径不变。

## 1 分钟真机结果

- AtomS3R：QVGA `320×240` / JPEG Q10 / ToF-on；
- Android：`decodeSampleSize=1`、`maxFrameAgeMs=65`；
- 手机：SM-S9280 / SM8650 / Android 16；
- 1658 帧，0 error，0 reconnect；
- source packets `1662`，latest overwrite `0`，stale dropped `4`；
- capture→risk P50/P95/P99：`54.64 / 66.96 / 70.93 ms`；
- preprocess P50/P95/P99：`0.27 / 0.71 / 0.91 ms`；
- input write P50/P95：`0.24 / 0.63 ms`；
- detector total P50/P95：`5.75 / 13.32 ms`；
- PSS：`190344 → 184590 KiB`。

原 QVGA/sample1 1 分钟基线为 capture→risk `54.20 / 66.48 / 71.92 ms`、preprocess `0.41 / 1.14 ms`。R37 的端到端处于同一波动范围，但预处理下降稳定，故保留候选；不宣称检测质量、安全或 false-clear 收益。

原始证据：`artifacts.local/evidence/atoms3r-qvga-native-r37-1min/`。
