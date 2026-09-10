# MZ16 visual detail: more retrieval, no reliable attribution upgrade

2026-09-10 · EXPLORE · consumed Development · completed, not promoted.

The matched high-detail arm adds 80 far true-positive output bits and 16 false-positive bits over MZ5 on the 3000 placement Development frames; low detail adds 61 and 22. High detail retrieves 49/49 clean and 48/49 stressed pole opportunities, versus 47/49 and 46/49. However its actual query-contributor macro AP is lower on BOTH placement cohorts. Both arms fail the frozen false-alert budget. Keep MZ5/MZ9 unchanged and stop these fits. A source-localization benefit from simply retaining more detail is not established by this formulation.

## Controlled intervention and exposure

See [frozen protocol](MZ16_VISUAL_DETAIL_PROTOCOL_20260910.md). Both arms crop the same original pixels (208,68)-(432,292), cover the same ToF field, use 224x224 input, frozen encoder, 64x28x28 features, unchanged physical rays, and the same 10740-parameter QUERY readout. LOW first BOX-downsamples 640x360 to 256x144 and bilinearly restores 640x360; HIGH keeps native detail. The detail intervention includes this specified resampling. It is not a full-resolution encoder comparison.

Weights at initialization, saved 1200x16 batch IDs, normalization, optimizer and losses match. Training uses 19200 draws per arm, 7562 unique frames from the existing 10200-frame pool, including the 200 diagnostic sequence frames. Paired feature caching covers 11562 unique RGB frames, including oldDEV1000 and consumed placementDEV3000. PlacementDEV is excluded from gradients and cutoff calibration, but has been inspected in earlier experiments and is not independent confirmation. No collection or protected EVAL access occurred.

Comparisons to historical MZ15 change crop/context, grid shape and absolute coordinate representation as well; only HIGH versus LOW isolates detail in this ROI pipeline. Actual return-contributor labels are supervision/evaluation, never inference inputs. Validity gaps remain unknown. Stress labels track the foreground deletion and background-slot movement.

## Obstacle outputs

Order in vectors: BODY_NEAR, BODY_FAR, HEAD_NEAR, HEAD_FAR. Counts are output bits unless marked exact frames. Each arm freezes its zero-added-oldDEV-FP alert cutoffs and adds positives to MZ5. This preserves all baseline positives, including existing false near alerts; it does not repair MZ5's premature HEAD_NEAR behavior. The zero-added-FP guarantee is calibration-set-only and failed to transfer.

| Cohort | Arm | Exact frames | TP vector | FP vector | Added far TP | Added FP |
|---|---|---:|---|---|---:|---:|
| DEV | MZ5 | 912 | [187, 180, 183, 165] | [6, 11, 10, 7] | 0 | 0 |
| DEV | LOW_DETAIL | 934 | [195, 186, 192, 179] | [6, 11, 10, 7] | 20 | 0 |
| DEV | HIGH_DETAIL | 943 | [196, 195, 193, 181] | [6, 11, 10, 7] | 31 | 0 |
| clean | MZ5 | 143 | [2, 6, 9, 0] | [5, 3, 19, 0] | 0 | 0 |
| clean | LOW_DETAIL | 173 | [9, 32, 9, 27] | [5, 3, 19, 0] | 53 | 0 |
| clean | HIGH_DETAIL | 176 | [9, 36, 9, 32] | [5, 3, 19, 0] | 62 | 0 |
| stress | MZ5 | 143 | [2, 6, 9, 0] | [5, 3, 19, 0] | 0 | 0 |
| stress | LOW_DETAIL | 172 | [9, 32, 9, 26] | [5, 3, 19, 0] | 52 | 0 |
| stress | HIGH_DETAIL | 175 | [9, 36, 9, 31] | [5, 3, 19, 0] | 61 | 0 |
| relation10000 | MZ5 | 1883 | [382, 357, 388, 375] | [2, 5, 26, 14] | 0 | 0 |
| relation10000 | LOW_DETAIL | 1893 | [388, 372, 398, 387] | [3, 6, 27, 29] | 27 | 18 |
| relation10000 | HIGH_DETAIL | 1914 | [389, 383, 400, 393] | [5, 6, 26, 19] | 44 | 9 |
| distance5000 | MZ5 | 920 | [0, 0, 489, 446] | [0, 0, 25, 9] | 0 | 0 |
| distance5000 | LOW_DETAIL | 947 | [0, 0, 499, 480] | [4, 0, 25, 9] | 34 | 4 |
| distance5000 | HIGH_DETAIL | 944 | [0, 0, 499, 482] | [7, 0, 25, 9] | 36 | 7 |

Cohort sizes are DEV1000, clean200, stress200, relation10000 DEV2000, distance5000 DEV1000; dataset names are not the evaluated denominators. Placement aggregate exact frames: MZ5 2803/3000, LOW 2840/3000, HIGH 2858/3000. HIGH versus LOW adds 19 more far true-positive bits and has 6 fewer false-positive bits in aggregate, but distanceDEV BODY_NEAR false alerts rise from 4 to 7. This is not equal achieved test-set FP and cannot be advertised as an upgrade at the same false-alert budget.

All 22 LOW and all 16 HIGH new placement false alerts have no actual selected-return source at the winning angular cell. More detail therefore has not eliminated the failure established in MZ14/MZ15.

## Local attribution: direct mechanism test

AP is computed on known, valid, geometrically eligible return/cell/query candidates, against actual query contributor presence. This is conditional candidate localization, not whole-field segmentation or obstacle recall. Many candidates are positive; AP magnitude alone is not proof of accurate localization. Macro averages omit queries with no positives (distanceDEV has only HEAD positives).

| Cohort | LOW macro AP | HIGH macro AP | HIGH minus LOW |
|---|---:|---:|---:|
| DEV | 0.822567 | 0.816296 | -0.006271 |
| clean | 0.768053 | 0.720636 | -0.047416 |
| stress | 0.777440 | 0.727301 | -0.050138 |
| relation10000 | 0.811330 | 0.806391 | -0.004939 |
| distance5000 | 0.951801 | 0.942514 | -0.009287 |

Every per-query AP with positives on the two placement cohorts also decreases in HIGH. Thus improved task counts are not evidence that the actual return location is better resolved. The paired result limits this frozen encoder/readout/training recipe; it does not show native detail is inherently useless or prove the representation lacks the information.

Local cutoffs below are separate diagnostics, selected on oldDEV to permit at most 1% negative-cell FPR, and never used to select task outputs. They are then frozen. Precision/recall are percentages; local candidate cells are correlated, not independent obstacle events.

| Cohort | Arm | Query | Positive/negative cells | AP | Precision % | Recall % | FPR % |
|---|---|---|---|---:|---:|---:|---:|
| DEV | LOW_DETAIL | BODY_NEAR | 31663/14738 | 0.708665 | 63.250 | 0.799 | 0.997 |
| DEV | LOW_DETAIL | BODY_FAR | 22016/10464 | 0.730941 | 81.462 | 2.076 | 0.994 |
| DEV | LOW_DETAIL | HEAD_NEAR | 41219/8036 | 0.940253 | 97.974 | 9.386 | 0.996 |
| DEV | LOW_DETAIL | HEAD_FAR | 12648/3456 | 0.910411 | 98.126 | 14.073 | 0.984 |
| DEV | HIGH_DETAIL | BODY_NEAR | 31663/14738 | 0.717972 | 75.211 | 1.409 | 0.997 |
| DEV | HIGH_DETAIL | BODY_FAR | 22016/10464 | 0.724190 | 78.601 | 1.735 | 0.994 |
| DEV | HIGH_DETAIL | HEAD_NEAR | 41219/8036 | 0.930441 | 92.734 | 2.477 | 0.996 |
| DEV | HIGH_DETAIL | HEAD_FAR | 12648/3456 | 0.892581 | 95.479 | 5.677 | 0.984 |
| clean | LOW_DETAIL | BODY_NEAR | 894/943 | 0.608296 | NA | 0.000 | 0.000 |
| clean | LOW_DETAIL | BODY_FAR | 1770/5568 | 0.591338 | 73.630 | 12.147 | 1.383 |
| clean | LOW_DETAIL | HEAD_NEAR | 1240/104 | 0.992987 | 100.000 | 0.242 | 0.000 |
| clean | LOW_DETAIL | HEAD_FAR | 1232/2410 | 0.879589 | 90.155 | 42.370 | 2.365 |
| clean | HIGH_DETAIL | BODY_NEAR | 894/943 | 0.541824 | NA | 0.000 | 0.000 |
| clean | HIGH_DETAIL | BODY_FAR | 1770/5568 | 0.522138 | 57.349 | 22.486 | 5.316 |
| clean | HIGH_DETAIL | HEAD_NEAR | 1240/104 | 0.991060 | 100.000 | 0.806 | 0.000 |
| clean | HIGH_DETAIL | HEAD_FAR | 1232/2410 | 0.827522 | 84.906 | 10.958 | 0.996 |
| stress | LOW_DETAIL | BODY_NEAR | 894/943 | 0.608296 | NA | 0.000 | 0.000 |
| stress | LOW_DETAIL | BODY_FAR | 1636/4656 | 0.614407 | 78.680 | 9.474 | 0.902 |
| stress | LOW_DETAIL | HEAD_NEAR | 1240/104 | 0.992987 | 100.000 | 0.242 | 0.000 |
| stress | LOW_DETAIL | HEAD_FAR | 1152/1965 | 0.894069 | 92.250 | 42.361 | 2.087 |
| stress | HIGH_DETAIL | BODY_NEAR | 894/943 | 0.541824 | NA | 0.000 | 0.000 |
| stress | HIGH_DETAIL | BODY_FAR | 1636/4656 | 0.536658 | 58.437 | 20.110 | 5.026 |
| stress | HIGH_DETAIL | HEAD_NEAR | 1240/104 | 0.991060 | 100.000 | 0.806 | 0.000 |
| stress | HIGH_DETAIL | HEAD_FAR | 1152/1965 | 0.839664 | 85.211 | 10.503 | 1.069 |
| relation10000 | LOW_DETAIL | BODY_NEAR | 66050/30178 | 0.709831 | 66.524 | 0.707 | 0.779 |
| relation10000 | LOW_DETAIL | BODY_FAR | 42742/20322 | 0.709414 | 74.789 | 2.908 | 2.062 |
| relation10000 | LOW_DETAIL | HEAD_NEAR | 86500/17618 | 0.929209 | 97.999 | 6.681 | 0.670 |
| relation10000 | LOW_DETAIL | HEAD_FAR | 24238/6256 | 0.896866 | 93.443 | 20.579 | 5.595 |
| relation10000 | HIGH_DETAIL | BODY_NEAR | 66050/30178 | 0.707776 | 74.029 | 0.837 | 0.643 |
| relation10000 | HIGH_DETAIL | BODY_FAR | 42742/20322 | 0.703209 | 76.252 | 2.637 | 1.727 |
| relation10000 | HIGH_DETAIL | HEAD_NEAR | 86500/17618 | 0.920113 | 96.825 | 2.785 | 0.448 |
| relation10000 | HIGH_DETAIL | HEAD_FAR | 24238/6256 | 0.894468 | 93.359 | 4.930 | 1.359 |
| distance5000 | LOW_DETAIL | BODY_NEAR | 0/34 | NA | NA | NA | 0.000 |
| distance5000 | LOW_DETAIL | BODY_FAR | 0/7 | NA | NA | NA | 0.000 |
| distance5000 | LOW_DETAIL | HEAD_NEAR | 122866/36945 | 0.932913 | 97.473 | 11.082 | 0.955 |
| distance5000 | LOW_DETAIL | HEAD_FAR | 34796/2488 | 0.970689 | 97.477 | 30.087 | 10.892 |
| distance5000 | HIGH_DETAIL | BODY_NEAR | 0/34 | NA | NA | NA | 0.000 |
| distance5000 | HIGH_DETAIL | BODY_FAR | 0/7 | NA | NA | NA | 0.000 |
| distance5000 | HIGH_DETAIL | HEAD_NEAR | 122866/36945 | 0.921424 | 94.722 | 4.279 | 0.793 |
| distance5000 | HIGH_DETAIL | HEAD_FAR | 34796/2488 | 0.963604 | 99.436 | 5.070 | 0.402 |

For example distanceDEV HEAD_FAR local recall drops from 30.09% to 5.07%, while FPR drops from 10.89% to 0.40%. These are different transferred operating points; higher precision is not free localization gain. AP avoids choosing a single cutoff and also declines. Saved per-frame local counts and sample labels preserve all denominators.

## Pole and wrong correspondence

Pole outcomes are one trained 25-frame configuration, 25 BODY_FAR plus 24 HEAD_FAR opportunities, not 49 independent obstacles. LOW clean/stress is 47/46, HIGH is 49/48 out of49. Wrong-zone RGB yields 0/49 in both arms and both conditions.

On the placement cohorts, wrong-zone RGB removes every added far TP: LOW correct61 versus wrong0; HIGH correct80 versus wrong0. Wrong-zone placement added FP is LOW1, HIGH6. Thus this recipe uses the supplied RGB-zone relationship for task recovery, but the disruption also changes local content and distribution. It cannot establish that winning cells correspond to the true echo source, which the AP and false-winner diagnostics contradict.

## Configuration and site aggregation

These units, rather than frame counts alone, expose correlated outcomes. Relation has 400 groups across100 sites; distance has500 near/far pairs across100 sites. The two datasets share sites and must not be summed as200 independent sites.

| Cohort | Unit | Arm | All frames exact / units |
|---|---|---|---:|
| relation10000 | group | MZ5 | 325/400 |
| relation10000 | group | LOW_DETAIL | 325/400 |
| relation10000 | group | HIGH_DETAIL | 341/400 |
| relation10000 | site | MZ5 | 47/100 |
| relation10000 | site | LOW_DETAIL | 46/100 |
| relation10000 | site | HIGH_DETAIL | 58/100 |
| distance5000 | group | MZ5 | 424/500 |
| distance5000 | group | LOW_DETAIL | 447/500 |
| distance5000 | group | HIGH_DETAIL | 445/500 |
| distance5000 | site | MZ5 | 57/100 |
| distance5000 | site | LOW_DETAIL | 64/100 |
| distance5000 | site | HIGH_DETAIL | 65/100 |

| Cohort | Family | Arm | Added TP vector | Added FP vector |
|---|---|---|---|---|
| relation10000 | crossbar | LOW_DETAIL | [1, 8, 6, 3] | [0, 0, 0, 0] |
| relation10000 | cabinet | LOW_DETAIL | [3, 1, 3, 2] | [0, 0, 1, 14] |
| relation10000 | oblique_rod | LOW_DETAIL | [2, 4, 1, 7] | [0, 0, 0, 1] |
| relation10000 | hanging_sign | LOW_DETAIL | [0, 2, 0, 0] | [1, 1, 0, 0] |
| relation10000 | crossbar | HIGH_DETAIL | [2, 12, 8, 10] | [0, 0, 0, 0] |
| relation10000 | cabinet | HIGH_DETAIL | [2, 1, 3, 2] | [0, 0, 0, 4] |
| relation10000 | oblique_rod | HIGH_DETAIL | [3, 8, 1, 6] | [0, 0, 0, 1] |
| relation10000 | hanging_sign | HIGH_DETAIL | [0, 5, 0, 0] | [3, 1, 0, 0] |
| distance5000 | crossbar | LOW_DETAIL | [0, 0, 7, 11] | [0, 0, 0, 0] |
| distance5000 | cabinet | LOW_DETAIL | [0, 0, 0, 6] | [0, 0, 0, 0] |
| distance5000 | oblique_rod | LOW_DETAIL | [0, 0, 3, 15] | [0, 0, 0, 0] |
| distance5000 | hanging_sign | LOW_DETAIL | [0, 0, 0, 2] | [4, 0, 0, 0] |
| distance5000 | crossbar | HIGH_DETAIL | [0, 0, 7, 13] | [0, 0, 0, 0] |
| distance5000 | cabinet | HIGH_DETAIL | [0, 0, 0, 8] | [0, 0, 0, 0] |
| distance5000 | oblique_rod | HIGH_DETAIL | [0, 0, 3, 13] | [0, 0, 0, 0] |
| distance5000 | hanging_sign | HIGH_DETAIL | [0, 0, 0, 2] | [7, 0, 0, 0] |

## Verification, compute and disposition

- CUDA on RTX5060 Laptop GPU. Cache PASS:11562 paired frames,211.63s total. Dynamic extraction reproduces legacy 256x144 features exactly on16 checked inputs (max error0); input hashes, finite outputs and shapes verified. Failed cache-v1 stopped on a manifest-key mismatch before extracting features and is preserved; only cache-v2 is used.
- LOW/HIGH fits completed exactly1200 steps,23.40/25.09s. Full training/evaluation/artifact run67.65s. No extra fitting or cutoff rescue.
- Encoder-only synchronized timing, excluding warmup and PIL/transfers/write: LOW1.053ms/frame and HIGH0.931ms/frame across11546 frames each. Identical compute architecture; this one-run difference is timing variation, not evidence HIGH is cheaper. No end-to-end device latency claim.
- Two focused tests pass (ROI geometry and tied AP/FPR behavior). Independent audit recomputes62400 task output bits and1324062 local candidate bits, validates initial-state and batch identity, checks saved cutoffs/receipts, and matches AP against sklearn. The float64 threshold comparison explicitly preserves nextafter boundaries.
- Neither candidate meets the fixed false-alert budget; LOW also misses the pole retention threshold. HIGH fails the preregistered localization criterion on both placement cohorts. Preserve this pair as NEGATIVE_CONTROL for this detail-only intervention, not a ban on other representations. MZ5/MZ9 remain the baseline; no temporal restart or new data collection.

The useful decision is to stop treating input detail alone as the proven missing piece. A future proposal must explain how it will improve actual echo assignment at the error budget, and distinguish that from recovering final positives by max aggregation. This report does not authorize or start a successor experiment.

## Evidence

Durable local root: `artifacts.local/work/mz16-visual-detail-20260910/`.
`cache-v2/receipt.json`, `parity.json`, `timing.json`; `run-v1/start.json`,
`receipt.json`, `result.json`, `predictions.npz`, `local_samples.npz`, both initial/final checkpoints,
alert/local cutoffs, shared `batches.npy`, and independent `audit.json`.
Delivery verification and resource-release receipts reside in `delivery/`.
These are consumed synthetic/controlled Development results, not real-hardware or product-safety evidence.
