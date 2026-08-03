# Corridor-conditioned future field A2.1 protocol

Date: 2026-08-03

Status: `FROZEN_BEFORE_A2_1_WINDOW_LOSO_OUTCOME`

## Single successor change

A2.1 adds one independently motivated evidence source to the closed A2 model:
the class-free, non-ground-separated UniDepth 2D corridor that achieved 3.45%
FPR and the best non-oracle MCC in A1.

For each current band x horizon row, append exactly three values:

- 2D corridor clearance;
- 2D corridor clearance minus horizon;
- fixed sigmoid occupancy probability with scale 0.20 m.

The 2D clearance definition is unchanged from A1: valid UniDepth pixels from
image rows 10% through 90%, projected into the fixed metric lateral bands, with
the second depth percentile as clearance. It uses RGB and intrinsics only.

All A2 data, 0.5-second lead, seven-frame causal history, RAFT features, HOLD,
CV, CA, IMM construction, window folds, L2 `0.01`, occupied weight `1.25`, and
decision threshold `0.50` remain unchanged. The 2D hold probability is also
reported as a fifth fixed geometric baseline.

## Gates

The same six A2 gates apply against the best of five fixed baselines:

1. at least 3,000 known future opportunities;
2. Brier reduction at least 15%;
3. log-loss reduction at least 20%;
4. ECE at most 0.10;
5. occupied recall at least 85% and FPR at most 15%;
6. MCC strictly greater than every fixed baseline.

All must pass to freeze one model before opening `sitting_halfsphere`. Failure
stops further TUM-internal feature, weight, or threshold rescue.
