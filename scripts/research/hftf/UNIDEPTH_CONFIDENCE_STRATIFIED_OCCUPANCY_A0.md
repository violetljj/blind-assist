# UniDepth confidence-stratified occupancy A0

Date: 2026-08-03

Status: `FROZEN_BEFORE_CONFIDENCE_STRATIFIED_WALKING_RPY_OUTCOME`

## Question

Can UniDepthV2's native pixel confidence make residual-based occupancy
probabilities transfer better than the unconditioned Metric3D empirical field?

For each band, the confidence attached to a clearance observation is the median
`log1p(pixel_confidence)` among obstacle points no more than 0.10 m behind the
2nd-percentile clearance support. On the fixed-world-floor calibration cohort,
each band is divided by its own 25/50/75% confidence quantiles. Residual CDFs
are calibrated independently in the resulting 12 fixed band×confidence strata,
using the same half-count smoothing as Metric3D probabilistic occupancy.

Evaluation uses the consumed 180-frame `walking_rpy` UniDepth report. Missing
confidence remains unknown. No quartile, support width, smoothing, probability
threshold, band, or horizon may be changed from evaluation outcomes.

The six continuation gates are identical to
`METRIC3D_PROBABILISTIC_OCCUPANCY_A0.md`: Brier reduction, log-loss reduction,
ECE, high-confidence-clear false-clear and coverage, and occupied recall. All
must pass before any fresh sequence is opened.
