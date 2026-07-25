# USTRF Egomotion-Compensated Looming Signal R1 — Bonn Static-Surface Truth Goal

Date: `2026-07-25`
Status: `PREREGISTERED / DISCOVERY_ONLY / SIGNAL_CLOSED`

## 1. Narrow question

This audit asks only:

> Can the independent Bonn Leica static map provide a usable camera-to-visible-static-surface
> distance trajectory for fixed image-grid units inside the two already-frozen discovery
> windows?

It does **not** establish obstacle semantics, traversability, collision risk, route relevance,
event lifecycle, alert timing, human safety, or product authority. A wall, floor, ceiling, or
other visible static surface remains a `STATIC_SURFACE`, never an automatically inferred
`OBSTACLE`.

## 2. Frozen inputs

Only the following discovery sequences and pre-existing, non-sliding 10-second windows may be
read:

- `rgbd_bonn_person_tracking2`, starting at `1548265919.50097`;
- `rgbd_bonn_balloon`, starting at `1548266469.85281`.

The validation and holdout sequences remain sealed. RGB image members remain unopened. No
candidate signal arm may run.

One earlier metadata-only receipt froze exactly six discovery registered-depth members at
offsets `{0.0, 5.0, 9.9}` seconds in the two windows. Those six members, and no others, may be
decoded solely to test the published map-transform interpretation. They cannot define a
window, cell, threshold, truth outcome, or signal.

The static-map input is the official subsampled Leica map:

```text
rgbd_bonn_groundtruth_1mm_section.zip
SHA-256 1ce515267759537eb534ba14327f81e98a3459c7d956bd4b23a9964f69467d35
PLY vertices 54,676,774
```

The already-completed stream audit retained a deterministic coordinate-hash sample with:

- quantization: `0.001 m`;
- modulus: `64`;
- selected points: `856,075`;
- selection independent of RGB, depth, pose, truth outcome, and candidate signal.

That stream audit preceded this projection contract, but it did not project points, inspect
per-window surface distances, choose image regions, or compute a candidate signal.

## 3. Coordinate and calibration contract

The only admitted map transform is the Bonn-published chain:

```text
T_map_from_camera(t) = inverse(T_ROS) * T_pose(t) * T_ROS * T_m
```

where `T_pose(t)` is the source-native ground-truth pose row and:

```text
T_ROS =
[-1  0  0  0
  0  0  1  0
  0  1  0  0
  0  0  0  1]

T_m =
[ 1.0157  0.1828 -0.2389  0.0113
  0.0009 -0.8431 -0.6413 -0.0098
 -0.3009  0.6147 -0.8085  0.0111
  0       0       0       1]
```

`T_ROS` is self-inverse. `T_m` is treated as the published similarity calibration, not forced
to a rigid transform. Point projection therefore uses a full matrix inverse.

The only admitted RGB calibration is the official Bonn calibration:

```text
width  = 640
height = 480
fx = 542.822841
fy = 542.576870
cx = 315.593520
cy = 237.756098
```

No alternative transform direction, axis flip, calibration, or post-hoc alignment may be
selected after projection outcomes are visible. If this exact chain fails its geometry
canaries, Bonn abstains for this endpoint.

For the six frozen canary frames, the source-native registered depth uses the TUM-compatible
scale `raw / 5000 = meters`. At pixels where both sources have support, the fixed comparison
is between registered depth and the nearest projected static-map depth in a `4 × 4` pixel
bin. Dynamic foreground and absent static-map surfaces may create outliers; therefore this
comparison is a transform plausibility canary, not a mutual accuracy proof.

## 4. Frozen units

Each 10-second window is sampled on the source-independent `500 ms` grid:

```text
start + {0.0, 0.5, ..., 9.5} seconds
```

The nearest source-native pose must be within `40 ms`; otherwise that anchor abstains.

The image plane is divided into a fixed `3 × 3` grid:

```text
row-major cells r0c0 ... r2c2
```

Every cell is reported. No cell may be deleted because it sees floor, ceiling, a difficult
surface, weak closing, or an unfavorable future signal result.

The statistical unit is:

```text
source / discovery sequence / frozen window / image-grid cell / 500ms anchor
```

## 5. Static-map projection

For each anchor:

1. build the exact published `T_map_from_camera(t)`;
2. invert it and transform the deterministic map sample into the RGB optical frame;
3. retain points with finite coordinates and optical depth `z` in `(0.10 m, 20.0 m]`;
4. project with the frozen intrinsics and retain points inside the `640 × 480` image;
5. assign each projected point to the fixed `3 × 3` cell;
6. estimate cell range as the `10th percentile` of Euclidean camera-to-point range.

The percentile is fixed before projection to reduce sensitivity to isolated retained points.
It is not a semantic z-buffer and may not be described as the nearest obstacle.

## 6. Eligibility and abstention

An anchor-cell is geometry-supported only when at least `64` sampled map points project into
the cell.

A window-cell is eligible for a static-surface distance trajectory only when:

- at least `18/20` anchors are geometry-supported;
- all supported ranges are finite and in `(0.10 m, 20.0 m]`;
- the maximum joined-pose delta is no more than `40 ms`.

Otherwise it is retained as `abstained` with one of:

- `POSE_JOIN_MISSING_OR_LATE`;
- `PROJECTED_MAP_SUPPORT_INSUFFICIENT`;
- `RANGE_NONFINITE_OR_OUT_OF_DOMAIN`;
- `TRANSFORM_GEOMETRY_CANARY_FAILED`.

No missing range is filled with zero or interpolated across an unsupported anchor.

## 7. C2 static-closing mechanics

Eligibility above only establishes a distance trajectory. A window-cell becomes a
`C2_STATIC_SURFACE_CLOSING_MECHANICS_CANDIDATE` when, on supported consecutive anchors:

- endpoint range reduction is at least `0.25 m`;
- endpoint fractional reduction is at least `10%`;
- at least `60%` of adjacent supported pairs have positive
  `-d(log(range))/dt`;
- median `-d(log(range))/dt` is strictly positive.

These are discovery mechanics filters, not signal acceptance gates. They do not prove that
the surface is an obstacle, that looming is observable, or that an estimator succeeds.

## 8. Geometry canaries

The source abstains for this endpoint if any of the following holds:

- the published first-pose matrix does not match the official formula numerically;
- fewer than `1%` of the sampled map points project into the image at every anchor of a
  sequence;
- no image-grid cell reaches the frozen support threshold at any anchor of a sequence;
- across the six pre-frozen depth frames, fewer than four frames have at least `1,000`
  common-support `4 × 4` bins;
- across those usable frames, median absolute map-depth versus registered-depth difference
  exceeds `0.50 m` or median absolute relative difference exceeds `20%`;
- matrix inversion is singular or non-finite;
- the official-source hashes, archive inventory, or frozen window identities differ.

Canary failure closes this transform interpretation. It does not authorize testing a reversed
or tuned transform in the same round.

## 9. Evidence grade and allowed claim

The Leica map is an independent static-environment measurement. The six registered-depth
frames are only an independent sensor-chain plausibility check; neither chain is used to
declare the other perfectly accurate. The per-frame projection still depends on source-native
camera pose, published marker-to-sensor calibration, timestamp joining, and deterministic
`1/64` map sampling.

Therefore a passing unit is recorded as:

```text
evidence_grade = B
claim_scope = STATIC_VISIBLE_SURFACE_DISTANCE_ONLY
```

It may support a future C2 comparison only on common support. It cannot supply dynamic-target
truth, association, obstacle semantics, route truth, lifecycle truth, or a safety claim.

## 10. Legal terminals

```text
BONN_C2_STATIC_SURFACE_TRUTH_UNITS_AVAILABLE
BONN_C2_STATIC_SURFACE_TRUTH_ABSTAINED_INSUFFICIENT_SUPPORT
BONN_C2_STATIC_SURFACE_TRANSFORM_CANARY_FAILED
```

All are valid outcomes. No outcome opens RGB decoding or any signal arm automatically.
