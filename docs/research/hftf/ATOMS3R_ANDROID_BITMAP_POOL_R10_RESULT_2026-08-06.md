# AtomS3R Android Bitmap 解码复用池 R10（2026-08-06）

## 结论

为 AtomS3R MJPEG 解码增加单槽 `inBitmap` 复用池，避免每帧分配/回收同尺寸 ARGB_8888 Bitmap。native/Bitmap 复用失败时保留安全回退，并在 source shutdown 时回收池内 Bitmap。该改动通过 core:device 单元测试、APK 构建、10 秒 smoke 和 1 分钟真机回归。

## 1 分钟证据

目录：`artifacts.local/evidence/atoms3r-android-bitmap-pool-r10-20260806/`

- 1628 帧，约 27.1 fps；
- 0 gap、0 overwrite、0 错误、0 重连；
- 时钟同步 2/2 成功；
- PSS：189871 → 178638 KB；
- QNN HTP 路由保持成功。

| 指标 | R9 无 Bitmap 池 | R10 Bitmap 池 |
|---|---:|---:|
| JPEG decode P50/P95/P99 | 5.04/11.89/14.72 ms | **3.67/10.07/11.89 ms** |
| preprocess P50/P95/P99 | 1.21/4.33/5.36 ms | 1.24/4.82/5.99 ms |
| capture→risk P50/P95/P99 | 67.77/85.52/116.44 ms | **67.15/83.99/108.27 ms** |

## 实现边界

- 仅解码 Bitmap 存储复用；JPEG 字节、模型输入、QNN 路由和设备配置不变；
- `inBitmap` 不兼容时捕获 `IllegalArgumentException`，释放候选并重新分配；
- 这是 Development-only 性能/稳定性证据，不代表准确率或安全证据。
