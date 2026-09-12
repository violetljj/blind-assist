# MZ101: full ToF versus computed stereo spatial support

Date: 2026-09-12. Mode: EXPLORE, one fresh curated simulation panel.

## Decision and hypothesis

MZ98 separated route relevance from actual future entry; MZ99 showed an angular
information opportunity without achieving calibration; MZ100 could not supply
reliable observable anchors often enough. This run asks a different capability
question: can an actual rendered stereo pair supply useful missing lateral and
height support for forward BODY/HEAD obstacles, beyond a full 64-zone ToF expert?
It tests the value of sensor inputs, not same-input algorithm superiority.
No learned detector, risk head, Radar patch, native-depth completion, or hardware
validation. Prior retained-core authority is unchanged.

## Fixed acquisition and inputs

- Seed 101013, 12 geometry families, each paired textured/flat appearance, 12
  posed frames at 0.25 s: 24 episodes, 288 frames. These appearance pairs share
  geometry and are not 24 independent scene families. No train/test fit.
- Head bar/overhead clearance, two independently positioned thin poles in path,
  lateral pole, wall/open corridor, crossing hit/miss, receding object, partial
  occlusion and head rotation. Sub-zone phase is fixed before ToF hits are read.
  Cuboids approximate poles; all actors and background are synthetic.
- UE rendered parallel stereo: 640x360, horizontal FOV 70 deg, baseline 0.10 m;
  right camera lies on left camera's local +Y. Fixed exposure and disabled motion
  blur. Posed synchronized sampling is not real-time sensor capture.
- Known camera/body pose and rig intrinsics/extrinsics are controlled inputs;
  this does not establish estimated IMU or real calibration performance.
- ToF has all 64 equal-angle centre rays in a 45x45 deg array, native nearest
  collision range up to 4 m. It is a point-sampling surrogate, not zone-wide
  integration, VL53L8CX physics or an observation of real sensor performance.
- Primary strong baseline uses ideal ray ranges. A disclosed secondary proxy
  adds Gaussian 0.02 m range error, independent 15% dropout, and two missing
  packets at frames 5/6 of every episode, seed101031. This is a stress proxy.
- Native left axial depth is evaluator-only. It cannot fill stereo holes or
  choose thresholds, pixels, points, labels or sensor branches. Predictor consumes
  only left/right RGB, range/valid and a whitelist of known pose/time fields.
- Generated image texture patches have no collision and may sit 1.5 mm ahead of
  the base surface. This small native visual/collider distinction is disclosed.
  No specular/refraction physics claim; flat texture and occlusion are the fixed
  image difficulty controls in this first run.

## Frozen spatial and alert methods

OpenCV4.10 StereoSGBM, minDisparity0/numDisparities96/block5, P1=200/P2=800,
uniqueness10, left-right consistency1pixel, connected valid region>=8pixels,
no large-speckle deletion, filling or GT-guided crop. Disparity divided by16;
axial depth fB/d. CPU `GPU_BACKEND_UNAVAILABLE` for this installed OpenCV backend;
record stereo timing. UE rendering uses its actual graphics backend.

All three arms use the same 0.5--4m usable depth/range interval and pose transform:
ToF points; computed stereo points; their per-frame support union. Primary
comparison limits camera-coordinate azimuth to +/-22.5 deg and elevation to
+/-20 deg. Native view is a secondary contrast, not pooled into the primary.

Coordinates forward/right/up. BODY box [0.18,-0.28,0.65] to [3.18,0.28,1.4];
HEAD [0.13,-0.18,1.4] to [3.13,0.18,1.85], metres relative to wearer floor point.
Check original points against closed boxes first, then deduplicate 0.05m voxels.
At least one measured occupied voxel supplies per-frame support. An actor spanning
the 1.4m boundary may support both labels. Zero support is UNKNOWN/no supported
alert, never measured clearance. Each arm uses identical causal 2-on/2-off state,
reset between episodes; union occurs before state, not after separate hysteresis.

## Truth, metrics and interpretation

GT is intersection of the same query boxes with all task actor AABBs, not object
classes or the predictor's inferred points. Fixed background at10m and floor at0m
cannot intersect the query boxes for the specified trajectories; validate that
capture geometry matches that assumption. Occluders are also truth actors.
Shared pose is ideal controlled geometry, not per-target oracle information.

Report per-frame TP/FP/FN/F1 by BODY/HEAD and case/appearance, directly unsupported
queries, event misses, entirely false warning sessions, false-positive segments
and duration, within-event fragmentation and first correct alert delay from GT
query entry. Mark first/last-frame censored events. This is **current forward
corridor awareness**, not one-second contact prediction or proven warning lead
before collision. Approaches ending in hazard are right-censored for exit timing.
Frame scores from older generic Radar task labels are not comparable.

Useful integrated component criterion (primary ideal/common-FOV union vs ToF):
at least one newly detected ToF-missed predefined BODY/HEAD event; no lost detected
ToF event; no increase in entirely false sessions or FP duration; no more than
one sample (0.25s) additional delay on retained events. Report paired raw changes
even if this gate fails. A local spatial gain alone remains component evidence.
No thresholds selected from outcomes, no frame-wise bootstrap independence claim.

## Execution, checks and stop

Before formal capture, use separate engineering planes to validate actual RGB
parallax, nominal disparity/depth, ToF units and pose. Synthetic known-shift tests
check SGBM and invalid-pair behavior; geometry and state tests check boundaries.
Freeze code/protocol before main capture. Register one run, <=288 formal frames,
no learning. Save input hashes and seal predictions before evaluator scoring and
native-depth diagnostics. Save video and fixed frame7 per episode contact sheets.

Mechanical capture/format failures may be repaired with preserved logs; any
re-capture is disclosed and uses unchanged design. Outcome-driven reruns, new
scenes or method changes require a separate decision. Stop after this panel and
finish report, inheritance, scoped commit/push and owned UE-process release.
If stereo has no reliable useful support, record that failure rather than using
native depth as its input. If acquisition is invalid, call it NOT_EVALUABLE.
