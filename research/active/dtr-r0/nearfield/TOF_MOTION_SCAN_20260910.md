# Motion-coded angular attribution: ideal feasibility experiment

2026-09-10 EXPLORE. User clarifies exploiting natural jitter as a scan, not merely
resisting it. Test joint pose/range constraints without assuming successive returns
come from the same object. This is an explicit ideal scene-family experiment,
not fresh UE validation or measured VL53L1X response.

Fixed camera/sensor origin, eye1.7m, native640x360 and100degree horizontal FoV.
Use the existing25-step2-second seed23 angular trajectory and15degree diagonal
square cone. Exact camera-ray solid-angle weights and rotated footprint determine
coverage. Hypothetical scene contains zero, one or two opaque2degree angular-square
patches at axial2.5m, with background at3.5m. Patch centers lie on the integer
degree grid[-9,9] in each angular axis. Two patches must have disjoint interiors.
BODY/HEAD intersection labels come from physical patch rectangles at x=2.5m.

All near-plane directions occupy radial histogram bin[2.5,2.6)m and background
directions within these footprints occupy[3.5,3.6)m; verify these geometric claims.
Nearest-supported return chooses the near bin when summed near-plane solid-angle
coverage is>=2%, otherwise background. Greatest-area and farthest-supported laws
are fixed negative controls: they are expected to select background in this scene
family; verify rather than assume. Signal/reflection behavior is uncalibrated.

Inference enumerates every permitted empty/one/two-patch scene and retains all
whose predicted returns agree with valid measurements. It reports positive HEAD
only if EVERY feasible scene intersects HEAD; multiple feasible geometries remain
multiple, never a fabricated single reconstructed obstacle. Labels are read only
after observation compatibility, by the evaluator/query of feasible physical
geometry. Empty compatibility means model mismatch/UNKNOWN, not CLEAR.

Compare one initial observation,25 stationary repeats, and25 pose-aware jitter
observations. Noise +/-0.05m and20% dropout use seed41, shared by methods per
case/time; fixed bin decision midpoint3.05m. No observation filtering in the
headline denominator. Preselect100 single-HEAD,100 single-nonHEAD,100 two-HEAD,
100 two-nonHEAD,100 mixed two-patch cases with seed37, before evaluation. These
500 cases are controlled in-family examples, not held-out scene generalization.
Include a deliberately incomplete one-patch inference model on the same data to
expose errors from incorrectly assuming one return source.

Report guaranteed-positive HEAD coverage on positive cases, false positives on
negative cases, UNKNOWN and empty feasible sets, feasible-scene counts and
stationary-to-moving changes. Success means joint moving observations resolve
more positive cases than stationary repetitions without false HEAD assertions
under the complete stated family. No claim of unique mapping, actual sensor
capability, gait reconstruction, or original alert changes. Stop after this
fixed simulation and its falsifiers; do not fit motion amplitudes or scene grids.
