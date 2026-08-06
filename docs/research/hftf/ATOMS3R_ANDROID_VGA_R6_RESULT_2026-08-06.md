# AtomS3R Android VGA + native 预处理 R6（2026-08-06）

## 结论

在 native RGB→Float 预处理基础上，将 AtomS3R 从 SVGA 改为 VGA（640×480）进一步降低了设备采集、JPEG 传输和端到端延迟。VGA 已通过 10 秒 smoke 和 1 分钟回归；建议保留为低延迟可选档，暂不替换默认配置，需补充画质/检测质量对照。

## 1 分钟证据

目录：`artifacts.local/evidence/atoms3r-android-vga-r6-20260806/`

- 1642 帧，约 27.3 fps；
- 0 错误、0 重连、1 次 gap、1 次 latest-frame 覆盖；
- 时钟同步 2/2 成功；
- QNN HTP 路由保持成功；
- PSS：109994 → 196105 KB（短测期间未以此单独作内存稳定结论，需结合基线进程状态复核）。

| 阶段 | SVGA native 1 分钟 | VGA native 1 分钟 |
|---|---:|---:|
| JPEG size P50 | — | 7513 B |
| capture→JPEG complete P50/P95/P99 | 约 57 / 71 / 96 ms | 50.14 / 57.00 / 88.42 ms |
| first byte→JPEG complete P50/P95/P99 | — | 6.94 / 12.41 / 16.29 ms |
| JPEG decode P50/P95/P99 | — | 3.72 / 8.55 / 11.15 ms |
| native preprocess P50/P95/P99 | 2.12 / 5.23 / 6.15 ms | 2.14 / 5.71 / 6.92 ms |
| QNN execute P50/P95/P99 | 2.74 / 3.33 / 4.08 ms | 2.80 / 3.46 / 3.86 ms |
| capture→risk P50/P95/P99 | 68.61 / 84.86 / 103.58 ms | **62.47 / 76.53 / 99.56 ms** |

## 边界

- 这是性能/稳定性 Development-only 证据，不是画质、准确率或安全证据；
- 默认仍不自动切换 XGA/SVGA/VGA；
- VGA 画面更小，下一步若要晋升需在同一场景补画质和检测质量对照；
- 30–60 分钟压力测试仍按既定规则暂缓。
