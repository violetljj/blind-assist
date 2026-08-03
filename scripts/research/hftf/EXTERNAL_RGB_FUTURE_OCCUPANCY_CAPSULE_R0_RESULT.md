# External RGB future occupancy capsule R0 result

Date: 2026-08-03

Terminals:

- `FUTURE_OCCUPANCY_CAPSULE_NOT_SUPPORTED`
- `METRIC3D_PROPAGATION_NOT_RUN_ORACLE_GATE_FAILED`
- `RESEARCH_MAINLINE_UNCHANGED / DEFAULT_APP_UNCHANGED`

## Question and frozen boundary

This experiment tested whether a horizontal capsule joining the current person
position to the seven-frame OLS endpoint could represent both stopping and
continued motion more compactly than independently calibrated current-static
and OLS-endpoint disks.

The construction, one-second horizon, 90% split-conformal target, fixed TUM
Freiburg 3 `walking_xyz` windows, and all seven support gates were frozen in
`EXTERNAL_RGB_FUTURE_OCCUPANCY_CAPSULE_R0.md` before the fresh archive was
downloaded or its trajectory outcomes were read.

## Fresh cohort admission

The official TUM Freiburg 3 `walking_xyz` archive had SHA-256
`1459E9488AC0E61A2EC80DFBC35CFB77942F6D8EABDED1C8D26A70BE650D0E1D` and
contained 859 RGB frames and 833 registered-depth frames.

Seven fixed 3-second windows starting at 0, 4, 8, 12, 16, 20, and 24 seconds
were attempted at 10 FPS. Only the 24-second window contained a person track
present in every sampled frame under the frozen pose-torso ByteTrack rule. It
admitted two tracks and 60 rows. The other six windows closed with
`segment has no complete admissible person track`; no replacement windows or
threshold changes were used.

This yielded 28 fresh forecast opportunities. The result is therefore a valid
fixed-cohort rejection of this candidate, but not a precise population estimate
for all TUM motion or the final external camera.

## Sensor-depth oracle result

Radii were calibrated only on the consumed TUM `walking_static` cohort with 70
opportunities. Registered sensor depth supplied both calibration tracks and the
fresh oracle tracks.

| Arm | Radius | Coverage | Mean area | Median area | Mean excess distance |
|---|---:|---:|---:|---:|---:|
| Current-static disk | 0.19511 m | 10.71% | 0.11960 m^2 | 0.11960 m^2 | 0.10393 m |
| OLS-endpoint disk | 0.42450 m | 60.71% | 0.56611 m^2 | 0.56611 m^2 | 0.07403 m |
| Current-to-OLS capsule | 0.19511 m | 57.14% | 0.24935 m^2 | 0.23945 m^2 | 0.07527 m |

The capsule reduced mean area by 55.95% versus the OLS disk, but increased mean
area by 108.49% versus the current-static disk. Its median area showed the same
conflict. Coverage was 57.14%, below the frozen 85% gate, and mean uncovered
excess distance was slightly worse than the OLS disk.

Only three of seven gates passed:

- mean and median area at least 20% below the OLS disk;
- mean excess distance no worse than the current-static disk.

Coverage, both area comparisons against the static disk, and excess distance
against the OLS disk failed.

## Decision

Stop this exact `current -> OLS endpoint` conformal capsule. The failure occurs
with registered-depth oracle tracks, so running Metric3D on the fresh RGB frames
cannot establish support for the geometric forecasting mechanism and was not
performed.

This does not invalidate Metric3D as the supported Development-only RGB metric
depth source. It shows that uncertainty inflation around one fixed
constant-velocity segment is not enough to handle cross-sequence human motion.
Any successor must pose a genuinely different question, such as a causal
multi-state or multi-modal future set with explicit stationary, translating,
turning, and start/stop alternatives, and must use a new calibration/evaluation
boundary rather than tune this consumed result.
