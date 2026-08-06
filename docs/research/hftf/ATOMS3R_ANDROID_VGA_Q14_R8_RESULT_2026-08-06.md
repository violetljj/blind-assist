# AtomS3R Android VGA + JPEG quality 14 组合 R8（2026-08-06）

## 结论

VGA（640×480）与 JPEG quality 14 组合在 native 预处理、QNN HTP 和 latest-frame 不变的条件下，通过 10 秒 smoke 和 1 分钟回归，取得当前最低的端到端延迟候选。它是性能候选，不自动晋升默认，因为画质和检测质量尚未对照。

## 1 分钟证据

目录：`artifacts.local/evidence/atoms3r-android-vgaq14-r8-20260806/`

- 1662 帧，约 27.7 fps；
- 0 错误、0 重连、1 gap、1 latest-frame 覆盖；
- 时钟同步 2/2 成功；
- PSS：188004 → 193963 KB，短测未见需要单独判定的持续异常；
- QNN HTP 路由保持成功。

| 指标 | VGA quality 10 | VGA quality 14 |
|---|---:|---:|
| JPEG size P50 | 约 7513 B | **7040 B** |
| capture→JPEG P50/P95/P99 | 50.14/57.00/88.42 ms | **49.10/56.17/59.38 ms** |
| first byte→JPEG P50/P95/P99 | 6.94/12.41/16.29 ms | 6.48/12.35/15.56 ms |
| native preprocess P50/P95/P99 | 2.14/5.71/6.92 ms | 2.14/5.89/6.91 ms |
| QNN execute P50/P95/P99 | 2.80/3.46/3.86 ms | 2.82/3.44/3.84 ms |
| capture→risk P50/P95/P99 | 62.47/76.53/99.56 ms | **61.89/75.29/81.59 ms** |

## 边界

- Development-only 性能/稳定性证据，不是画质、检测准确率或安全证据；
- 当前设备已恢复 SVGA/quality 10；
- VGA+quality14 仅作为低延迟实验档保留，若要采用需补同场景画质/检测质量对照。
