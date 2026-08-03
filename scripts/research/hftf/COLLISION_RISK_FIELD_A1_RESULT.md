# Collision risk field A1 result

Date: 2026-08-03

Terminal: `COLLISION_RISK_FIELD_A1_DEVELOPMENT_FAIL`

The frozen comparison used 1,716 known band x horizon opportunities from the
now-consumed TUM `walking_halfsphere` cohort. Four of five continuation gates
passed. The strict non-oracle MCC dominance gate failed: the simple 2D depth
corridor reached MCC 0.76141, versus 0.75579 for the motion probability field.
No threshold or gate was changed after seeing the arm report.

| Arm | Brier | Recall | FPR | F1 | MCC |
|---|---:|---:|---:|---:|---:|
| YOLO centre box | 0.33196 | 81.37% | 82.47% | 0.68482 | -0.01399 |
| bbox + UniDepth | 0.37657 | 41.37% | 1.01% | 0.58247 | 0.45773 |
| UniDepth 2D corridor | 0.12679 | 80.98% | 3.45% | 0.88342 | **0.76141** |
| UniDepth 3D envelope | 0.15005 | 87.16% | 18.25% | 0.87328 | 0.68848 |
| motion probability field | **0.08936** | **88.43%** | 12.36% | **0.89841** | 0.75579 |
| sensor-depth oracle | 0 | 100% | 0% | 1.0 | 1.0 |

The probability field reduced Brier score by 40.45% versus deterministic 3D,
exceeded the 85% recall gate, and stayed below the 15% FPR gate. It was the
best non-oracle arm on Brier, recall, and F1, but not on MCC. Therefore the
claim that it unconditionally dominates simpler current-frame risk rules is
not supported.

The result also exposes useful structure rather than closing M3D-CF: detector
centres without distance massively over-alerted; box depth was precise but
missed most occupied opportunities; ground-separated deterministic 3D retained
recall but over-alerted; and an unseparated 2D depth corridor was a surprisingly
strong hard-decision baseline. Any successor must treat that 2D signal as a
real comparator, not hide it through threshold selection.

This A1 cohort is consumed. It may diagnose a separately frozen successor but
may not provide another fresh confirmation.

Ignored machine report SHA-256:
`B07E1B9193775415D049FC04FD7BFD62DFE980BF0AF1E061C1C5D81C868560AF`.
