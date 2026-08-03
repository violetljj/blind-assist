# Metric3D clearance field A0 mechanism result

Date: 2026-08-03

Terminals:

- `DEPTH_ONLY_DETERMINISTIC_CLEARANCE_FIELD_NOT_SUPPORTED`
- `GROUND_ORIENTATION_ONLY_SUCCESSORS_NOT_SUFFICIENT`
- `M3D_CF_PROBABILISTIC_OCCUPANCY_QUESTION_REMAINS_OPEN`
- `RESEARCH_MAINLINE_UNCHANGED / DEFAULT_APP_UNCHANGED`

## Why the reference was corrected

The original registered-depth reference independently fitted a ground plane
each frame. A nominally static camera produced median inferred heights of 1.39,
0.79, and 0.77 m across three windows; the RPY reference also switched between
approximately 0.8 and 1.5 m horizontal surfaces. Sensor depth was accurate, but
the derived semantic choice of which plane was the floor was not an oracle.

A separate `walking_static` 0-second calibration window fixed the TUM world
floor at z `0.003324743488025139 m` from 30 frames (MAD `0.00271 m`). The final
reference uses mocap pose only to express that fixed plane in each camera frame;
registered depth supplies obstacle geometry. No pose, gravity, floor, or depth
reference enters any RGB candidate.

## Clean fixed-reference comparison on consumed walking_rpy

These are Development mechanism diagnostics because the 180 RPY outcomes were
already opened. All arms use the same fixed world-floor sensor reference.

| RGB candidate ground frame | Valid | Clearance MAE | Collision agreement | False-clear | Temporal delta MAE |
|---|---:|---:|---:|---:|---:|
| Depth-only per-frame RANSAC | 96.11% | 0.25103 m | 84.17% | 7.29% | 0.18496 m |
| Metric3D predicted-normal guided | 88.33% | 0.20871 m | 87.34% | 6.77% | 0.13861 m |
| Mocap-gravity ceiling | 98.33% | 0.22235 m | 86.00% | 6.49% | 0.13306 m |

Predicted normals and exact gravity both reduced temporal error below the A0
0.15 m gate, demonstrating that orientation stabilization is useful. Neither
reached 90% collision agreement or the 5% false-clear limit. The normal arm
also missed the 90% valid-field requirement. Even the non-deployable exact
gravity ceiling did not solve collision classification.

The remaining error is concentrated in centre/right collision structure rather
than ground orientation alone. For the gravity ceiling, centre agreement was
83.33% with 7.45% false-clear; pooled agreement was only 86.00%.

## Decision

Stop deterministic clearance from a single per-frame point estimate and stop
additional ground-normal, gravity, plane-threshold, band, percentile, or horizon
search on these consumed outcomes. Do not open `walking_halfsphere` for any of
the three failed arms.

This does not close M3D-CF. It changes the next scientific question exactly as
the collision-field proposal intended: can a conservative uncertainty-aware
occupancy field, calibrated on independent Development residuals, control
false-clear risk without collapsing coverage or flooding the corridor? The
next candidate should output occupied probability or a lower-clearance bound,
not another deterministic depth percentile.
