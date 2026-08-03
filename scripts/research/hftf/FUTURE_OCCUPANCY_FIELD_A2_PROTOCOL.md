# Future occupancy field A2 protocol

Date: 2026-08-03

Status: `FROZEN_BEFORE_A2_WINDOW_LOSO_OUTCOME`

## Question

Can a causal, low-capacity probability field predict whether the user-relative
corridor will be occupied 0.5 seconds later better than fixed HOLD, constant
velocity, constant acceleration, and IMM-style geometric baselines?

This is the first M3D-CF stage whose label is future sensor-depth occupancy.
A0/A1 predicted current occupancy and must not be described as future-risk
evidence.

## Development data and isolation

- Consumed TUM `walking_static`, `walking_xyz`, `walking_rpy`, and
  `walking_halfsphere`: 17 complete three-second windows at 10 FPS.
- The target is the sensor-depth occupancy label exactly five sampled frames
  after the causal input frame.
- Leave one complete window out. Standardization and Logistic fitting use only
  the other 16 windows.
- Histories never cross a window boundary and contain at most the current plus
  six prior frames.

## Frozen candidate

The candidate is weighted Logistic Regression with L2 `0.01` and occupied
example weight `1.25`, inherited from A0.1. There is no hyperparameter,
threshold, feature, or operating-point search.

Its inputs are the frozen 18 current-frame A0.1 features plus:

- available-history fraction;
- OLS clearance slope and quadratic acceleration;
- CV and CA in-history residual RMS;
- HOLD, CV, and CA future clearance margins to the tested horizon;
- a fixed IMM-style mixture occupancy probability.

The fixed geometric probabilities use sigmoid clearance scales 0.20 m for
HOLD, 0.25 m for CV, and 0.35 m for CA. IMM mode priors are 0.30/0.50/0.20;
mode evidence is `exp(-RMSE/0.15 m)`. CA falls back to CV with fewer than five
history values, and CV falls back to HOLD with fewer than two.

## Baselines and gates

Compare the learned future field with the frozen HOLD, CV, CA, and IMM
probability scores. All decisions use `P(occupied)>=0.50`.

All six gates must pass:

1. at least 3,000 known future opportunities;
2. Brier reduction at least 15% versus the best geometric baseline;
3. log-loss reduction at least 20% versus the best geometric baseline;
4. ECE at most 0.10;
5. occupied recall at least 85% and false-positive rate at most 15%;
6. MCC strictly greater than every geometric baseline.

Pass authorizes freezing one all-Development model before opening official TUM
`sitting_halfsphere`. Failure closes this exact feature/model family without
threshold or polynomial rescue.
