# Exploiting motion for angular attribution: first ideal result

2026-09-10 EXPLORE. [Protocol](TOF_MOTION_SCAN_20260910.md).
The user's intended question is whether jitter can supply a scan, rather than
whether independent frame decisions withstand jitter. This experiment jointly
uses pose and range; it does not replace original B alerts or train RGB.

## Result

Under the predeclared nearest-supported return hypothesis, pose-aware motion
resolves HEAD membership in271/300 positive controlled cases. Single observations
and stationary repetitions resolve0/300. This is a useful information gain in the
explicit ideal model, not proof of actual VL53L1X response or real-world accuracy.

| Method, nearest-supported response | Guaranteed HEAD /300 positives | False HEAD /200 negatives | UNKNOWN /500 | Empty model fits |
| --- | --- | --- | --- | --- |
| One initial observation | 0 | 0 | 500 | 0 |
| 25 stationary repeats | 0 | 0 | 500 | 0 |
| 25 moving, joint zero/one/two-object inference | 271 | 0 | 214 | 0 |
| Moving, incorrectly restrict model to at most one object | 270 | 0 | 171 | 25 |

The lower UNKNOWN count in the incomplete model is not superiority:25 sequences
cannot be explained by that model at all. Empty fits remain UNKNOWN. Complete
model median feasible scene count decreases35065 to14276. Many geometries remain
possible; the result establishes body-region membership, not a unique 3D map.

Both greatest-area and farthest-supported return controls stay UNKNOWN500/500,
with0/300 positive resolutions. Background wins under those assumptions. Motion
cannot recover spatial information that the output stream never carries.

## Why motion helps here

Each range observation constrains which patches could lie inside its particular
field of view. As the cone rotates, near/background transitions impose different
constraints. The solver retains every zero/one/two-patch scene consistent with
the full valid sequence. It asserts HEAD only when all remaining explanations
intersect the physical HEAD query. It never assumes consecutive returns share
one object, intersects all returns into a fabricated point, or reads true target
identity to choose the returning patch.

The complete hypothesis family contains64010 scenes; the500 examples are selected
from it before evaluation. Consequently zero false positives is a consistency
property of the matched family, not an empirical safety estimate. The meaningful
contrast is increased resolution with motion versus identical stationary repeats.
The same calibrated pose and scene-family assumptions are supplied to both.

## Limits and decision

Fixed origin, known exact pose, static scene, integer angular-grid rectangles,
two possible patch planes, stipulated response law, and synthetic noise/dropout.
There is no translational walking, new occlusion, reflectance model, IMU error,
arbitrary multi-object reconstruction, fresh UE validation or physical sensor
measurement. The input is25 hypothetical scalar readings over2seconds, not a
16x16 depth camera. The500 cases are in-family controlled evidence.

Retain motion-coded joint feasibility as a candidate spatial-information method.
The next useful test must challenge observation/scene assumptions and preserve
multiple-return-source alternatives; merely smoothing outputs would not test
this mechanism. Do not promote an actual sensor integration from this result.

## Engineering evidence

`artifacts.local/work/tof-motion-scan-20260910/run-v1/` contains result.json,
rows.json and observations.npz, with operator/input hashes. CUDA on local
RTX5060 Laptop completed geometry and feasibility in3.65seconds. Footprints use
the existing exact rotation/raster projection and solid-angle weighting. Maximum
near coverage is8.0675%; histogram bounds and background-dominance assumptions
are explicitly checked. CPU is used only for scalar orchestration/serialization.

Independent audit passes:6000 saved rows recomputed using NumPy/Python integer
bitsets, without calling the inference function; physical query labels and seeds
also verified.32 artificial scene/pose combinations across three return laws
(96 checks) agree with the original native-depth histogram measurement operator.
See run-v1/independent-audit.json. These checks validate implementation, not the
realism of the stipulated sensor or scene family.

Separate verification of the previous jitter study's rotated-box approximation:
exact analytic cone extrema recover only4 of1000 lost near-support observations
per law, removing no old positives. Thus99.6% of the old support loss persists
under exact bounds. This verifies the approximation issue; it does not address
joint inference. Three independent SciPy geometry tests pass. Receipts reside at
`artifacts.local/work/body-query-tof-fusion-sim-20260909/exact-bounds-audit-v1/`.
The frozen old operator/results are unchanged.
