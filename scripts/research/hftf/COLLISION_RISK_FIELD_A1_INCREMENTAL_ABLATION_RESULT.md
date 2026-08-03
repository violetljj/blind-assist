# Collision risk field A1 consumed incremental ablation result

Date: 2026-08-03

Terminal: `A1_CONSUMED_MOTION_INCREMENT_NOT_SUPPORTED`

Preserved predecessor terminal:
`COLLISION_RISK_FIELD_A1_DEVELOPMENT_FAIL`

## Result

The frozen two-arm ablation used all 1,716 opportunities in the already
consumed A1 `walking_halfsphere` cohort, with seven complete windows held out
one at a time. No fresh source, threshold search, feature search, seed search,
or hyperparameter change was used.

| Arm | Brier | Log loss | ECE | Recall | FPR | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| geometry only | **0.11198** | **0.35826** | 0.06356 | 88.73% | **20.83%** | **0.87440** | **0.68406** |
| geometry + motion | 0.11639 | 0.37129 | **0.06048** | **89.31%** | 24.28% | 0.86762 | 0.66114 |

Adding the frozen ten-feature motion block changed:

- Brier by `+0.004408`, a `3.94%` relative worsening;
- log loss by `+0.013030`;
- recall by `+0.005882` (`+0.59` percentage points);
- F1 by `-0.006777`;
- MCC by `-0.022919`;
- FPR by `+0.034483` (`+3.45` percentage points).

Motion improved held-out-window Brier in 4/7 windows, so the strict-majority
consistency condition passed. The pooled Brier, log-loss, and F1 conditions
failed. The exact incremental claim is therefore not supported.

## Interpretation boundary

The original frozen A1 arm comparison remains valid: its probability field was
the best non-oracle arm on Brier, F1, and recall, while the 2D corridor retained
the best MCC. This new ablation asks a different question by refitting equal
logistic heads within the now-consumed A1 cohort. It shows that the original
descriptive advantage cannot be attributed to a robust independent increment
from the ten motion features.

Do not remove motion from the shipped candidate solely from this consumed
diagnostic, and do not search motion subsets or thresholds to rescue it. A
future causal-motion claim requires a separately frozen independent cohort.

Ignored machine report SHA-256:
`194907F41D8FB15AC4BD4924E0889C2EC5B1ADDD66B89B1E23465C584202308C`.

Bound input SHA-256 values:

- source report: `51018CA9576E4728EE76716C716F229DE231C27A9C32705BD9E12E98D953E3B2`;
- RAFT checkpoint: `01064C6DBA73B0FC9FC8EDF772248560A00A3ACFD62AC6677E9EEEBAD9680E27`.
