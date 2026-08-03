# Metric3D clearance field A0 fresh protocol

Date: 2026-08-03

Status: `FROZEN_BEFORE_WALKING_RPY_ARCHIVE_OPEN`

## Data and sampling

- Fresh source: official TUM Freiburg 3 `walking_rpy` archive.
- Rationale fixed before download: two people move while the camera is rotated
  in roll, pitch, and yaw, directly challenging ground recovery and a
  camera-relative collision envelope.
- Fixed 3-second windows begin at 0, 5, 10, 15, 20, and 25 seconds.
- Sampling is 10 FPS; expected maximum is 180 unique paired RGB/depth frames.
- No failed window may be replaced. No start, duration, frame rate, intrinsics,
  plane rule, band, percentile, height, horizon, or model checkpoint may change
  after the archive is opened.

Metric3D receives RGB and published Freiburg 3 intrinsics only. Registered
sensor depth is read solely by the evaluator to construct the reference field.

## Fixed geometry

The construction is exactly `METRIC3D_CLEARANCE_FIELD_A0.md`: deterministic
ground-plane recovery, class-free obstacle points, left/centre/right bands,
robust 2nd-percentile clearance, and 1.0/1.5/2.0 m collision-envelope probes.
Missing ground or insufficient obstacle support remains `UNKNOWN`.

## Fresh support gates

All must pass:

1. paired-valid field fraction `>=0.90`;
2. pooled clearance MAE `<=0.25 m`;
3. pooled collision agreement `>=0.90`;
4. pooled false-clear rate `<=0.05`;
5. temporal clearance-delta MAE `<=0.15 m`;
6. centre-band collision agreement `>=0.90`;
7. centre-band false-clear rate `<=0.08`.

Pass terminal:
`METRIC3D_CLEARANCE_FIELD_A0_FRESH_SUPPORTED_DEVELOPMENT_ONLY`.

Otherwise:
`METRIC3D_CLEARANCE_FIELD_A0_FRESH_NOT_SUPPORTED`.

A pass authorizes A1 static collision-risk comparison design only. It does not
change the research mainline, default App, reminder behavior, or safety claim.
