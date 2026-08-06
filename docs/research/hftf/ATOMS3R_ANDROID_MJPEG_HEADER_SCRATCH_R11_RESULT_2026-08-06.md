# AtomS3R Android MJPEG header scratch R11（2026-08-06）

## 结论

将 MJPEG header 行读取从每行新建 `ByteArrayOutputStream` 改为复用单个固定大小 scratch byte array，保持 header 字符串、Content-Length、JPEG 字节和时间戳语义不变。该改动通过 core:device 单元测试、APK 构建、10 秒 smoke 和 1 分钟真机回归。

## 1 分钟证据

目录：`artifacts.local/evidence/atoms3r-android-mjpeg-scratch-r11-20260806/`

- 1637 帧，约 27.3 fps；
- 0 错误、0 重连、1 gap、1 latest-frame 覆盖；
- 时钟同步 2/2 成功；
- QNN HTP 路由保持成功；
- PSS：188756 → 194973 KB，未据此单独作内存稳定结论。

| 阶段 | Bitmap 池 R10 | header scratch R11 |
|---|---:|---:|
| JPEG decode P50/P95/P99 | 3.67/10.07/11.89 ms | 3.65/9.95/12.34 ms |
| native preprocess P50/P95/P99 | 1.24/4.82/5.99 ms | 1.23/5.01/5.96 ms |
| capture→risk P50/P95/P99 | 67.15/83.99/108.27 ms | 67.84/83.78/100.97 ms |

## 边界

- 主要价值是减少 header 解析的短生命周期分配和潜在 GC 压力；单次分位数收益较小，端到端改善不显著；
- JPEG 字节仍按协议 Content-Length 分配，Bitmap 复用池保持不变；
- Development-only 性能/稳定性证据，不代表准确率或安全证据。
