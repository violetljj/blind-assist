# Motion occupancy A0.1 Bonn cross-dataset protocol

Date: 2026-08-03

Status: `FROZEN_BEFORE_BONN_A0_1_OUTCOME`

## Evidence role

This is a cross-dataset and different-RGB-D-sensor proxy for the unavailable
final external camera. The Bonn `person_tracking` sources were consumed by an
earlier metric-depth study, so this is not a fresh-source claim. The frozen
A0.1 occupancy model has not been fitted or recalibrated on Bonn outcomes.

## Sources and fixed reference calibration

- `rgbd_bonn_person_tracking`
- `rgbd_bonn_person_tracking2`
- Bonn intrinsics: `542.822841, 542.576870, 315.593520, 237.756098`
- The first 30 RGB-D frames of each source calibrate reference world-floor z
  only and are excluded from evaluation windows.
- Frozen floor z values:
  - tracking: `0.12734546155016813 m` (MAD `0.0027142836720802643 m`)
  - tracking2: `0.10736298788463228 m` (MAD `0.005585494644797517 m`)

For each source, five non-overlapping three-second windows begin at 2, 5, 8,
11, and 14 seconds and are sampled at 10 FPS. Expected total: 300 frames. No
window or failed frame may be replaced.

## Frozen candidate and admission

UniDepthV2-S resolution level 0, depth-RANSAC geometry, RAFT-small at `224x128`,
the A0.1 18-feature model, all bands, horizons, coefficients, normalization,
and probability thresholds remain unchanged.

Cross-dataset support requires:

- exactly two sources;
- paired-valid fraction at least 90% in each source;
- at least 900 known band x horizon opportunities in each source;
- all six original pooled A0.1 probability gates pass.

Per-source probability metrics are reported but are diagnostic because their
smaller opportunity counts were not powered as independent admission tests.

Pass remains Development-only proxy evidence. It cannot replace the final
external-camera canary or authorize alerts, Android, or safety claims.
