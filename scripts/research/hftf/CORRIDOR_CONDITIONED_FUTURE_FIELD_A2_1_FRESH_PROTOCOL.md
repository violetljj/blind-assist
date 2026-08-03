# Corridor-conditioned future field A2.1 fresh protocol

Date: 2026-08-03

Status: `FROZEN_BEFORE_SITTING_HALFSPHERE_ARCHIVE_OPEN`

## Source and sampling

- Unopened official TUM Freiburg 3 `sitting_halfsphere` archive.
- Seven 3-second windows beginning at 0, 5, 10, 15, 20, 25, and 30 seconds.
- 10 FPS, at most 210 paired input frames.
- The future target is exactly five sampled frames, or 0.5 seconds, later.
- No failed window or frame is replaced.

This source changes the human-motion regime from walking to sitting while
retaining camera translation and rotation. It is one same-dataset-family fresh
test, not independent-camera evidence.

## Frozen candidate and reference

- UniDepthV2-S resolution level 0 produces the 3D clearance and 2D corridor.
- RAFT-small uses the frozen official checkpoint at `224x128`.
- The exact 30-feature Logistic mean, scale, coefficients, L2 `0.01`, and
  occupied weight `1.25` are in
  `CORRIDOR_CONDITIONED_FUTURE_FIELD_A2_1_FROZEN_MODEL.json`.
- Sensor depth plus TUM pose and fixed world-floor z
  `0.003324743488025139 m` provides future labels only.
- All decisions retain `P(occupied)>=0.50`.

## Fresh gates

All must pass on the first conforming execution:

1. at least 1,200 known future opportunities;
2. Brier reduction at least 15% versus the best fixed HOLD/CV/CA/IMM/2D arm;
3. log-loss reduction at least 20% versus the best fixed arm;
4. ECE at most 0.10;
5. occupied recall at least 85% and FPR at most 15%;
6. MCC strictly greater than every fixed arm.

Pass terminal:
`CORRIDOR_CONDITIONED_FUTURE_FIELD_A2_1_FRESH_SUPPORTED_DEVELOPMENT_ONLY`.

Otherwise:
`CORRIDOR_CONDITIONED_FUTURE_FIELD_A2_1_FRESH_NOT_SUPPORTED`.

No retraining, recalibration, threshold adjustment, or successor is allowed on
this source.
