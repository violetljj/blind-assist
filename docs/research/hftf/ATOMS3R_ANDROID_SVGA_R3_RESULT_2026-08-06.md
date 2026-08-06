# AtomS3R Android SVGA 单变量延迟对照 R3（2026-08-06）

## 结论

在保持 JPEG quality=10、自动曝光、Android latest-frame、模型与正式 Qualcomm QNN HTP 路由不变的条件下，将 AtomS3R 从 XGA（1024×768）改为 SVGA（800×600）显著降低了设备采集/JPEG 与端到端延迟。SVGA 通过 10 秒 smoke、1 分钟回归和 5 分钟正式测试；建议作为可选低延迟配置保留，当前不自动替换默认 XGA，需另行确认画质/检测质量影响。

## 5 分钟稳定性

证据目录：`artifacts.local/evidence/atoms3r-android-svga-r3-20260806/`

- 8056 帧，约 26.8 fps；
- source packets 8057，1 次 latest-frame 覆盖，1 个 sequence gap；
- 0 错误、0 重连、0 流错误；
- QNN HTP 路由保持正式路径（测试通过，未出现 CPU fallback）；
- 时钟同步 10/10 成功；
- PSS：189545 → 185611 KB（测试期间未见持续增长）。

## 阶段分位数（ms；JPEG 大小为 bytes）

| 指标 | XGA R2 5 分钟 | SVGA R3 5 分钟 | 变化 |
|---|---:|---:|---:|
| JPEG size P50 | — | 14531 | — |
| capture→JPEG complete P50/P95/P99 | 94.08 / 117.75 / 131.43 | 57.63 / 70.68 / 96.33 | -36.45 / -47.07 / -35.10 |
| first byte→JPEG complete P50/P95/P99 | 33.10 / 44.66 / 52.00 | 14.71 / 22.90 / 29.03 | -18.39 / -21.76 / -22.97 |
| JPEG decode P50/P95/P99 | 4.44 / 8.35 / 11.54 | 2.87 / 5.36 / 8.37 | -1.57 / -2.99 / -3.17 |
| preprocess P50/P95/P99 | 12.46 / 16.09 / 17.42 | 13.35 / 18.17 / 19.73 | +0.89 / +2.08 / +2.31 |
| QNN execute P50/P95/P99 | 2.12 / 2.36 / 3.28 | 2.09 / 2.33 / 3.22 | 近似不变 |
| postprocess P50/P95/P99 | 0.81 / 0.92 / 1.01 | 0.82 / 0.97 / 1.06 | 近似不变 |
| capture→risk P50/P95/P99 | 115.39 / 139.98 / 154.17 | 78.48 / 92.98 / 118.51 | -36.91 / -47.00 / -35.66 |

XGA 数值来自 `ATOMS3R_ANDROID_LATENCY_DECOMPOSITION_R2_RESULT_2026-08-06.md`；SVGA 原始摘要和逐帧账本见上述证据目录。

## 边界与后续

- 本结果是性能/稳定性 Development-only 证据，不等同于画质、检测准确率或安全证据。
- 不改变默认 XGA；若要采用 SVGA，需要补充同场景画质和检测质量对照。
- 当前主要剩余瓶颈已从设备帧龄转移到 Android preprocess（约 13–18 ms）；下一优化候选应围绕 Bitmap→RGBA 全帧复制和预处理路径，而不是继续调 QNN execute。
- 30–60 分钟压力测试按既定规则暂缓；日常回归优先 10 秒 smoke、1 分钟短测。
