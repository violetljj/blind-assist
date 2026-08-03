# Motion-conditioned occupancy A0.1 recall-preserving successor

Date: 2026-08-03

Status: `FROZEN_BEFORE_WEIGHTED_WINDOW_LOSO_OUTCOME`

A0 passed five of six window-LOSO probability gates. Occupied recall at the
unchanged 0.50 probability boundary was 84.622%, below the frozen 85% gate;
all calibration, false-clear, coverage, and score gates passed strongly.

A0.1 changes one variable only: positive occupancy examples receive fixed
cross-entropy weight `1.25`. Negative examples remain weight 1.0. This is a
predeclared 25% penalty for missed occupied space, not a searched class weight.

Features, RAFT checkpoint, whole-window folds, standardization, L2 `0.01`,
probability thresholds, six gates, and all data remain unchanged. If all six
gates pass, freeze the all-Development fit before opening
`walking_halfsphere`. Otherwise stop this model family; do not search class
weights or probability thresholds.
