# Corridor-conditioned future field A2.1 Development result

Date: 2026-08-03

Terminal: `CORRIDOR_CONDITIONED_FUTURE_FIELD_A2_1_WINDOW_LOSO_PASS`

The single-source successor added A1's class-free 2D corridor evidence to the
closed A2 future field. Across 17 complete-window folds and 3,522 known
0.5-second future opportunities, all six frozen gates passed.

| Measure | Result | Gate | Pass |
|---|---:|---:|:---:|
| Brier reduction vs best fixed arm | 25.07% | >=15% | yes |
| Log-loss reduction vs best fixed arm | 28.68% | >=20% | yes |
| ECE | 0.02786 | <=0.10 | yes |
| Occupied recall | 87.86% | >=85% | yes |
| False-positive rate | 13.90% | <=15% | yes |
| MCC | 0.73981 | greater than every fixed arm | yes |

Relative to A2, the 2D evidence lowered FPR from 16.28% to 13.90% while
preserving 87.86% recall. This supports a complementary representation result:
ground-separated 3D geometry, ground-inclusive 2D near-depth, and causal motion
features answer different parts of the future occupancy problem.

The all-Development fit is frozen for one official TUM `sitting_halfsphere`
execution only. It does not authorize external-camera, runtime, alert, safety,
or mainline claims.

Machine report SHA-256:
`4279EF695BC5C1F351724F7F4B18479BEDD783949E38470F4730E20E58A5C469`.
