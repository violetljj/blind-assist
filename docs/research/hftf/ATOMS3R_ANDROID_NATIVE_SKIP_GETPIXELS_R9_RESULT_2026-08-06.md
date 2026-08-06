# AtomS3R Android native 跳过 getPixels R9（2026-08-06）

## 结论

native Bitmap→RGB float 成功时不再无条件调用 `Bitmap.getPixels()`；只有 native 不可用或失败时才读取 IntArray 并走 Kotlin fallback。该优化通过构建、10 秒 smoke 和 1 分钟 SVGA/quality10 真机回归。

## 1 分钟结果

证据目录：`artifacts.local/evidence/atoms3r-android-native-r9-skip-getpixels-20260806/`

- 1627 帧，约 27.1 fps；
- 0 gap、0 overwrite、0 错误、0 重连；
- 时钟同步 2/2 成功；
- PSS：190085 → 180137 KB；
- QNN HTP 路由保持成功。

| 阶段 | R5 native 基线 | R9 skip getPixels |
|---|---:|---:|
| preprocess P50/P95/P99 | 2.12/5.23/6.15 ms | **1.21/4.33/5.36 ms** |
| getPixels P50/P95/P99 | 约 0.89/1.45/1.92 ms | **0/0/0 ms** |
| native input write P50/P95/P99 | 0.25/0.57/0.72 ms | 0.27/0.50/0.65 ms |
| capture→risk P50/P95/P99 | 68.61/84.86/103.58 ms | 67.77/85.52/116.44 ms |

端到端 P50 改善约 0.84 ms；P95/P99 仍主要受设备 framebuffer 和网络长尾支配，不能把单次尾分位波动归因于该手机侧改动。

## 边界

- native 失败时仍保留原 Kotlin `getPixels` fallback；
- 模型输入、QNN 路由、分辨率和 JPEG 参数均未改变；
- Development-only 性能证据，不是准确率或安全证据。
