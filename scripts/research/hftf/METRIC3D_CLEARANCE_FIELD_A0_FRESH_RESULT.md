# Metric3D clearance field A0 fresh result

Date: 2026-08-03

Terminals:

- `METRIC3D_CLEARANCE_FIELD_A0_FRESH_NOT_SUPPORTED`
- `DEPTH_ONLY_PER_FRAME_GROUND_FRAME_ROTATION_SENSITIVE`
- `RESEARCH_MAINLINE_UNCHANGED / DEFAULT_APP_UNCHANGED`

## Fresh source

The protocol and implementation were committed and pushed before the archive
was downloaded. The official TUM Freiburg 3 `walking_rpy` archive had SHA-256
`8F92FEFE6F67D9A47DCBB07924A7371EC5DEF6B432C52393B46EE7A03865D1C7` and
contained 910 RGB and 872 registered-depth frames.

All six fixed 3-second windows at 0, 5, 10, 15, 20, and 25 seconds materialized
at 10 FPS, giving 180 unique frames. No window, geometry parameter, model, or
gate was replaced or adjusted.

## Result

| Measure | Fresh result | Gate | Pass |
|---|---:|---:|:---:|
| Paired valid fields | 162/180 (90.00%) | >=90% | yes |
| Pooled clearance MAE | 0.24915 m | <=0.25 m | yes |
| Pooled collision agreement | 83.93% | >=90% | no |
| Pooled false-clear rate | 6.03% | <=5% | no |
| Temporal clearance-delta MAE | 0.22692 m | <=0.15 m | no |
| Centre collision agreement | 83.86% | >=90% | no |
| Centre false-clear rate | 5.87% | <=8% | yes |

Only three of seven fresh gates passed. The terminal is therefore
`METRIC3D_CLEARANCE_FIELD_A0_FRESH_NOT_SUPPORTED`.

Per-band results were:

| Band | Clearance MAE | Collision agreement | False-clear rate |
|---|---:|---:|---:|
| Left | 0.24488 m | 84.44% | 3.56% |
| Centre | 0.27802 m | 83.86% | 5.87% |
| Right | 0.22073 m | 83.45% | 8.87% |

## Failure localization

The fixed windows varied strongly. The 10-second window achieved 94.07%
agreement, 1.85% false-clear, and 0.09945 m temporal-delta MAE. The 0- and
25-second windows fell to 76.00% and 70.31% agreement, with temporal-delta MAE
of 0.30976 m and 0.38080 m. Recovered camera-height error was only 0.06548 m in
the strong 10-second window but 0.41740 m in the weak 25-second window.

This pattern is consistent with the depth-only, independently fitted ground
frame becoming unstable under parts of the roll/pitch/yaw motion. It is a
diagnostic inference, not proof that orientation is the only cause.

## Decision

Do not promote this exact per-frame depth-only RANSAC clearance field to A1.
Do not tune bands, percentiles, plane thresholds, or horizons on
`walking_rpy`.

The broader M3D-CF question remains open because dense geometry was strong on
the Development cohort and one fresh RPY window. A single successor may change
the causal variable: stabilize the ground coordinate with Metric3D's predicted
surface normals or an independent gravity estimate, then test on a new fresh
motion sequence. It must not reuse `walking_rpy` as fresh evidence.
