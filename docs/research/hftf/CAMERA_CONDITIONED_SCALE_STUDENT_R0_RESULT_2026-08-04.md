# Camera-Conditioned Lightweight Scale Student R0 Result

Date: 2026-08-04

Decision: `PROMOTE_TO_FRESH_REAL_PHONE_MECHANISM_CANDIDATE_NOT_DEFAULT_APP`

## Answer

The aggressive software successor produced a reproducible positive synthetic Development signal without ToF. A fixed ten-feature, closed-form ridge student learned a DA scale correction from known camera height, R0 plane geometry, and DA depth quantiles.

In five leave-one-parent-out folds (165 records), all frozen parent-macro gates passed: coverage `0.9636`, clearance MAE `0.1461 m`, envelope agreement `0.9942`, false-clear `0.00329`, and temporal-delta MAE `0.1032 m`. Every fold trained on four parents and tested the fifth; train/test parent overlap was zero.

The model family and features were then frozen. One final student trained on the five Development parents and was evaluated on ten additional student-unseen TartanGround parents (330 records) from historically consumed corpora. Training/external parent overlap was zero, and all 10 parents were jointly better than raw DA. Parent-macro replication metrics were:

| metric | raw DA | fixed student |
|---|---:|---:|
| known coverage | 0.2676 | 0.9909 |
| clearance MAE (m) | 1.2704 | 0.1046 |
| envelope agreement | 0.4025 | 0.9899 |
| false-clear rate | 0.5975 | 0.0101 |
| temporal-delta MAE (m) | 0.5026 | 0.0518 |

Median DA inference was `51.78 ms` and p95 was `59.61 ms` on the local RTX 5060 Laptop GPU. This is not phone latency; the ridge computation is negligible relative to DA but was not isolated as a device benchmark.

## Why this is not a product or safety result

Both evaluations are synthetic and historically consumed. Exact `robot_height`, fixed `640×640` intrinsics, simulator poses, and metric sensor depth provide much cleaner receipts and labels than a real phone. The external pixel-median scale diagnostic still has `13.6%` median and `30.9%` p90 relative error. JapaneseAlley has a per-parent false-clear rate of `0.0673`, and the cross-validation winter-night parent has temporal-delta MAE `0.1541`; macro gates hide those local misses.

The existing fresh ARKit source cohort remains held at 2/4 qualified parents, and the current ARCore SM-S9280 capture remains `NOT_EVALUABLE` for exact-timestamp raw depth. Therefore the evidence authorizes only a separately frozen real-phone fixed-height/intrinsics shadow test. It does not authorize ToF purchase, default-App integration, live reminders, independent assistance, or a safety claim.

Cross-validation result SHA-256: `E6766A023EEE45D056E918AAAC59A2545AD561DDD4A2FEDA5311130ECAB2317E`.

External replication result SHA-256: `D2A8A1E091CB946078B1E8F6857749290088E247597A2BBC8F52E96D40BCCD43`.

Full records and prediction caches remain under ignored `artifacts.local/evidence/hftf/` roots.
