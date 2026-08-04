# DA V2 geometry equivalent optimization R0 result

Decision: `GEOMETRY_EQUIVALENT_OPTIMIZATION_R0_SUPPORTED_DEVICE_ONLY`. The optimized implementation replaces the allocation-heavy benchmark path without changing sampling, seed, RANSAC iterations, thresholds, crop, scale, features, or rejection semantics. Sparse sampling, fewer iterations, and early stop remain separate unrun experiment arms.

The implementation caches pixel rays per camera contract, stores candidates as reusable structure-of-arrays buffers, reuses inlier/residual/finite-depth memory, and obtains exact median/quantile order statistics without full sorting. `ThreadLocal` confines each workspace to a fixed worker. The frozen reference remains callable for parity tests.

On the frozen clean HTP depth output, 100 repetitions on `SM-S9280 / Android 16` produced:

| Measure | Frozen reference | Equivalent optimized |
|---|---:|---:|
| wall P50 / P95 | 119.87 / 121.00 ms | 64.04 / 64.43 ms |
| thread CPU P50 / P95 | 117.55 / 118.55 ms | 63.50 / 63.80 ms |
| allocated bytes / iteration | 23,655,998.4 | 3,276.8 |
| GC count / time over 100 | 90 / 970 ms | 0 / 0 ms |

P50 speedup was `1.87x`. Status and every geometry output field matched; maximum absolute field error was `6.94e-18`. JVM synthetic noisy/invalid parity tests also pass. This tiny floating difference is well below the frozen `1e-12` implementation-parity tolerance and does not change downstream state, scale, direction, or refusal.

## Evidence

- Runner: `scripts/research/hftf/run_geometry_equivalent_optimization_r0.ps1`
- Bundle: `artifacts.local/evidence/hftf/geometry-equivalent-optimization-r0-20260804-193721/result.json`
- Bundle SHA-256: `79AC53B7726B48897819E8FF3F376EC59AA3002ECE4FF98A09F75F1F214BDFE7`

Next gate: include this equivalent geometry in the 2 Hz CameraX pipeline and complete a bright-screen ten-minute sustained run.

That gate subsequently passed; see [CameraX sustained ten-minute R0](DAV2_CAMERAX_SUSTAINED_10MIN_R0_RESULT_2026-08-04.md).
