# AtomS3R Android native 预处理 R5（2026-08-06）

## 结论

将 SVGA 端到端链路中的 Bitmap→RGB float 写入移到 Android NDK native 路径后，预处理耗时显著下降；已通过 10 秒 smoke 和 1 分钟真机回归。当前代码保留 Kotlin fallback，但本次 arm64 真机实际走 native 路径。

## 证据

目录：`artifacts.local/evidence/atoms3r-android-preprocess-r4-20260806/`

- 1 分钟：1630 帧，约 27.1 fps；
- 0 错误、0 重连、1 次 gap、1 次 latest-frame 覆盖；
- 时钟同步 2/2 成功；
- PSS：190288 → 180266 KB；
- QNN HTP 路由保持成功。

| 阶段 | Kotlin 基线 10 秒 | native 10 秒 | native 1 分钟 |
|---|---:|---:|---:|
| preprocess P50 | 12.51 ms | 2.05 ms | 2.12 ms |
| input write P50 | 11.56 ms | 0.25 ms | 0.25 ms |
| capture→risk P50 | 76.30 ms | 67.83 ms | 68.61 ms |
| capture→risk P95 | 99.91 ms | 83.69 ms | 84.86 ms |

## 实现边界

- native 使用 `AndroidBitmap_lockPixels`，按 `AndroidBitmapInfo.stride` 逐行读取 RGBA_8888；
- 输入仍是 320×320、RGB、FLOAT32、0..1，模型和 QNN HTP 路由不变；
- native 失败时回退原 Kotlin 写入路径；
- 这是性能/稳定性 Development-only 证据，不代表检测准确率或安全性能提升；
- 暂不自动改变 XGA/SVGA 默认选择，也未推送或提交。
