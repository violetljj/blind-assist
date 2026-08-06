# AtomS3R Android JPEG quality 单变量 R7（2026-08-06）

## 结论

在 SVGA、自动曝光、native RGB→Float、QNN HTP 和 latest-frame 均不变时，将 ESP32-CAM `jpeg_quality` 从 10 调到 14（数值越大压缩越强）减少了约 25% 的 JPEG 字节数，并小幅降低传输/端到端延迟。quality=6 反向验证更慢，已排除。quality=14 通过 10 秒 smoke 和 1 分钟回归；设备随后恢复 SVGA/quality=10。

## quality=14 1 分钟证据

目录：`artifacts.local/evidence/atoms3r-android-jpegq6-r7-20260806/`

- 1642 帧，约 27.3 fps；
- 0 错误、0 重连、2 gap、2 latest-frame 覆盖；
- 时钟同步 2/2 成功；
- QNN HTP 路由保持成功。

| 指标 | quality 10 SVGA native | quality 14 SVGA native |
|---|---:|---:|
| JPEG size P50 | 约 14531 B | **10886 B** |
| capture→JPEG P50/P95/P99 | 约 57.63/70.68/96.33 ms | **54.01/62.29/87.60 ms** |
| first byte→JPEG P50/P95/P99 | — | 11.04/18.05/21.28 ms |
| native preprocess P50/P95/P99 | 约 2.12/5.23/6.15 ms | 2.15/5.07/5.97 ms |
| capture→risk P50/P95/P99 | 68.61/84.86/103.58 ms | **67.58/82.83/100.83 ms** |

quality=6 的 10 秒 smoke 中，JPEG P50 为 15937 B、capture→risk P50 为 74.33 ms，因此不进入 1 分钟。

## 边界

- 这是性能/稳定性 Development-only 证据，不代表画质、检测准确率或安全性能；
- quality=14 是否采用需要同场景画质/检测质量对照；
- 设备已恢复到 SVGA、quality=10；未提交或推送。
