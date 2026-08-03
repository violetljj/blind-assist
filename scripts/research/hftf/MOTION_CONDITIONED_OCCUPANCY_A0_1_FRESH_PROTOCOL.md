# Motion-conditioned occupancy A0.1 fresh protocol

Date: 2026-08-03

Status: `FROZEN_BEFORE_WALKING_HALFSPHERE_ARCHIVE_OPEN`

## Source and sampling

- Fresh source: official TUM Freiburg 3 `walking_halfsphere`.
- Fixed 3-second windows begin at 0, 5, 10, 15, 20, 25, and 30 seconds.
- Sampling is 10 FPS; expected maximum is 210 paired frames.
- No failed window may be replaced. No frame rate, duration, band, horizon,
  geometry, model feature, coefficient, normalization, probability threshold,
  gate, or reference parameter may change after archive open.

The source contains two walking people and camera translation plus rotation on
a half-sphere, making it distinct from static, xyz-only, and rpy-dominant
Development windows.

## Frozen candidate and reference

- UniDepthV2-S resolution level 0 receives RGB and Freiburg 3 intrinsics.
- RAFT-small receives only previous and current RGB at `224x128`.
- The exact 18-feature Logistic model, mean, scale, coefficients, L2, positive
  weight, and source hashes are frozen in
  `MOTION_CONDITIONED_OCCUPANCY_A0_1_FROZEN_MODEL.json`.
- The sensor reference uses registered depth and the fixed TUM world floor z
  `0.003324743488025139 m`, expressed by mocap pose. Neither enters the
  candidate.
- Unknown geometry or confidence remains unknown.

## Fresh gates

All six Development gates must pass on all known fresh opportunities:

1. Brier reduction `>=15%` versus deterministic clearance;
2. log-loss reduction `>=20%`;
3. ECE `<=0.10`;
4. high-confidence-clear false-clear `<=5%`;
5. high-confidence-clear coverage `>=10%`;
6. occupied recall at `P>=0.50` `>=85%`.

Pass terminal:
`MOTION_CONDITIONED_OCCUPANCY_A0_1_FRESH_SUPPORTED_DEVELOPMENT_ONLY`.

Otherwise:
`MOTION_CONDITIONED_OCCUPANCY_A0_1_FRESH_NOT_SUPPORTED`.

No retraining, recalibration, or threshold adjustment is allowed after this
one fresh execution.
