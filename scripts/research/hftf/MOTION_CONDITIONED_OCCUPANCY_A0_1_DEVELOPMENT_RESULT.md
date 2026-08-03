# Motion-conditioned occupancy A0.1 Development result

Date: 2026-08-03

Terminal: `MOTION_CONDITIONED_OCCUPANCY_A0_WINDOW_LOSO_PASS`

Ten complete 3-second windows from consumed TUM `walking_static`,
`walking_xyz`, and `walking_rpy` produced 2,505 known band×horizon
opportunities. Each fold held out one complete window.

The unweighted model passed five of six gates; occupied recall was 84.62%
against the frozen 85% gate. No probability threshold was changed. The single
successor fixed positive cross-entropy weight at 1.25 and reran the identical
window folds.

| Measure | Weighted result | Gate | Pass |
|---|---:|---:|:---:|
| Brier reduction vs deterministic | 22.81% | >=15% | yes |
| Log-loss reduction | 62.19% | >=20% | yes |
| Expected calibration error | 0.04024 | <=0.10 | yes |
| High-confidence-clear false-clear | 2.00% | <=5% | yes |
| High-confidence-clear coverage | 27.98% | >=10% | yes |
| Occupied recall at P>=0.50 | 86.53% | >=85% | yes |

This is the first HFTF/M3D-CF candidate in the branch to pass probability
quality, calibration, conservative-clear coverage, false-clear, and occupied
recall gates simultaneously under whole-window isolation. It authorizes one
fresh `walking_halfsphere` confirmation only; it does not authorize A1,
Android, reminders, mainline promotion, or safety claims.

The frozen final fit is stored in
`MOTION_CONDITIONED_OCCUPANCY_A0_1_FROZEN_MODEL.json`.
