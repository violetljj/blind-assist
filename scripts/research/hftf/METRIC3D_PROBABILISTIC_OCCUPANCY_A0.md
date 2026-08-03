# Metric3D probabilistic occupancy A0

Date: 2026-08-03

Status: `FROZEN_BEFORE_WALKING_RPY_PROBABILITY_OUTCOME`

## Question

Can uncertainty calibrated on independent dense-clearance residuals turn an
imperfect Metric3D geometry field into a useful probability of collision-space
occupancy, without pretending that one clearance estimate is exact?

## Frozen candidate

- Calibration: fixed-world-floor reports from consumed TUM `walking_static`
  and `walking_xyz` (119 paired-valid frames).
- Development evaluation: consumed `walking_rpy` fixed-world-floor report.
- RGB candidate: original depth-only Metric3D/RANSAC field; no pose, sensor
  depth, normal, outcome, or future frame.
- For each left/centre/right band, calibrate empirical residuals
  `truth_clearance - predicted_clearance`.
- For predicted clearance `c` and collision horizon `h`, output
  `P(occupied) = P(residual <= h-c)` using the empirical CDF with fixed
  half-count smoothing `(count+0.5)/(n+1)`.
- Horizons remain 1.0, 1.5, and 2.0 m. Unknown inputs remain unknown.
- Deterministic comparator is the original `c <= h` field on identical known
  pairs.

No residual binning, band pooling, probability temperature, threshold, or
temporal filter may be selected from `walking_rpy` outcomes.

## Development continuation gates

All must pass before opening a new fresh sequence:

1. Brier-score reduction `>=15%` versus deterministic occupancy;
2. log-loss reduction `>=20%` versus deterministic probabilities clipped to
   `[0.001, 0.999]`;
3. 10-bin expected calibration error `<=0.10`;
4. among high-confidence clear outputs `P(occupied)<=0.05`, false-clear rate
   `<=0.05`;
5. high-confidence clear coverage `>=0.10` of known opportunities;
6. occupied recall at `P(occupied)>=0.50` is `>=0.85`.

Passing authorizes a separately frozen `walking_halfsphere` fresh confirmation.
Failure closes this exact empirical-residual probability candidate, not all
probabilistic or learned occupancy fields.
