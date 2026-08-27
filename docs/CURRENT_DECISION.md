# Current decisions: L10-R0 Goal-Lock active; Dynamic Travel Risk R0 advancing

Status: `L10_R0_ACTIVE / CONTROLLED_MECHANISM_POSITIVE` and
`DTR_R0_ACTIVE / FIXED_KNOWN_HEIGHT_RGB_SOURCE_SELECTED /
ANDROID_EXPERIMENT_BUILD_READY / LIVE_DEVICE_UNVERIFIED`

## Parallel product lines

Ten-meter goal completion and dynamic obstacle/risk guidance are separate
lines. L10-R0 does not wait for DTR-R0, and neither line supplies evidence for
the other. The DTR result and next action do not change the L10-R0 route.

L10-R0 deliberately starts with readable goals such as room signs, exits,
named entrances, elevator buttons, and service desks. It replaces the former
GRAIL dependency with a goal-conditioned evidence controller: long-term
text/appearance/structure identity memory, short-term target motion, explicit
LOST -> REACQUIRE search, and independent near/completion evidence.

In the first 250-episode controlled closed-loop Development benchmark, L10-R0
reached 87.5% task completion, 81.2% post-occlusion reacquisition among
prior-locked episodes, 93.1%
direction accuracy, 1.7% wrong-lock frames, and 0/50 false completions when the
target was absent. Against the sticky local tracker this is +8.5pp completion,
+13.3pp reacquisition, +13.6pp direction accuracy, and a reduction from 67.1%
to 1.7% wrong-lock frames. These are synthetic controller/mechanics numbers,
not real-camera or product evidence. The runnable route is
`research/active/l10-r0/`.

With the controller frozen, a second 250-episode seed reproduced 88.5% task
completion, 82.4% reacquisition, 91.9% direction accuracy, 1.8% wrong-lock
frames, and another 0/50 absent false completions.

## DTR-R0 strategic transition

GRAIL owner orientation is no longer the daily algorithm mainline. R1C-V,
R1C-P, R1C-L, G0, and G1 form a preserved negative-result chain: the available
RGB/masks, simple pose transport, and fixed multiview representation did not
reliably recover ProcTHOR owner-canonical sign. The terminal G1 evidence is
recoverable at commit `4db9a11964ff9af9b5b500d59a60d8bb6fc0213b`.

G1 specifically closed this claim:

> On fresh house-disjoint synthetic ProcTHOR Development data, a fixed
> camera-lateral anchor/left/right scan, with no view-role, scan-geometry,
> camera-pose, owner-pose, or canonical-sign input and a shared pair encoder plus
> permutation-invariant `mean+max` aggregation, did not recover more stable
> owner-canonical PRESERVE/FLIP authority than its matched single-view model.

The apparent accuracy gain was a stronger PRESERVE tendency under a 988/148
class imbalance. Balanced-accuracy uplift was -0.05pp and -0.49pp. Doorway FLIP
collapsed to `0/24` in both seeds, taking Doorway balanced accuracy from
53.09% to 46.18% and from 50.32% to 47.35%. Across the 17 owner groups that
contain both modes, macro balanced accuracy remained near chance:
52.57% to 53.64% and 50.68% to 51.29%.

This does not establish that every active multiview RGB formulation is
impossible. A future GRAIL successor would need either a different task
representation, such as a reference-anchored dense correspondence field, or a
genuinely different information source. Neither is active now.

## DTR-R0 exact native-track answer

Can short causal target tracks, ego-motion compensation, and intersection of
future target occupancy with the wearer's short-horizon route reduce
non-actionable alerts while preserving truly crossing or oncoming events?

Yes at the narrow privileged ceiling, with important uncertainty. The primary
run uses the locally available 19-session THÖR-MAGNI Pupil subset. QTM supplies
the camera wearer's and other people's helmet centroids in one global metric
frame. Both arms receive only current-and-past centroids; the wearer route yaw
comes from the past 0.5 seconds of ego motion. Future synchronized centroids are
opened only by the evaluator.

At 10 Hz the run covered 461,182 source rows, 132 wearer-target identities, 357
evaluable track segments, 520.0 target-track seconds, and 10 non-left-censored
geometric critical events.

## Public privileged comparison

Both compared arms use the same causal observation ledger and the same
`ONSET / HOLD / CLEAR / UNKNOWN` lifecycle:

- B2: radial time-to-collision crosses a fixed horizon;
- C: an ego-compensated short track predicts target
  occupancy from now through a frozen horizon in 1.5--3.0 seconds and
  intersects it with the wearer route tube.

The primary comparison remained `C_ROUTE_INTERSECTION vs B2_RADIAL_TTC`, so an
increment can be attributed to route relevance rather than merely to tracking.

| Metric | B2 radial TTC | C route intersection |
| --- | ---: | ---: |
| Critical-event recall | 80.0% (8/10) | 90.0% (9/10) |
| Lateral-crossing recall | 85.7% (6/7) | 85.7% (6/7) |
| Oncoming-corridor recall | 0.0% (0/1) | 100.0% (1/1) |
| False alert segments | 96 | 55 |
| Median first-alert lead | 1.85 s | 2.90 s |
| Mean fragments / event | 1.3 | 1.5 |
| Post-event CLEAR | not evaluable | not evaluable |

## DTR decision

DTR-R0's core ceiling line required, relative to B2:

- critical-event recall does not decrease;
- non-actionable alerts decrease by at least 40%;
- event lifecycle is scored as ONSET segments rather than positive frames.

C raised recall by 10 percentage points, preserved the crossing partition,
recovered the one oncoming event, and reduced target-level false alert segments
by 42.7%. It passed the core line. No post-event critical sample remained
evaluable for CLEAR, so this native ceiling alone is not a completed lifecycle,
product, or safety result.

The authorized public-data RGB detector/tracker bridge has now run on fixed
JRDB train sequence `packard-poster-session-2019-03-20_1`, frames `115..257`.
YOLO11n plus the existing causal tracker was sealed to a truth-blind ledger
before native labels were opened. Same-frame IoU then supplied an evaluator-only
identity/metric binding; future geometry remained evaluator-only.

It scored all 52 evaluable target segments in the window and found three
critical events (two lateral crossings, one oncoming):

| Metric | B2 radial TTC | C route intersection |
| --- | ---: | ---: |
| Critical-event recall | 100% (3/3) | 100% (3/3) |
| False alert segments | 7 | 4 |
| Median first-alert lead | 3.94 s | 2.20 s |
| Mean fragments / event | 1.00 | 1.33 |
| Post-event CLEAR | 100% (2/2) | 50% (1/2) |

Thus the narrow false-alert effect survived real RGB detection and causal
tracking: no critical event was lost and false alert segments fell 42.9%.
This is directionally positive, not a new advancement gate. Its shorter lead,
one missed CLEAR, `45.97%` known prediction coverage across all targets, and
fragmented tracker identities remain explicit defects.

Current metric geometry has now also been replaced. The final fixed public
sensor bridge uses the latest causal upper and lower Velodyne scans, compensates
each scan through bag odometry to image time, projects them with official JRDB
calibration, and keeps only points inside a fixed YOLO11n-seg person mask. Its
truth-blind sensor ledger was sealed before evaluator identity/future truth was
opened. Current observations use a fixed `0.30 m` person radius; native body
extent remains evaluator-only future event truth.

On the same 52 target segments and three critical events:

| Metric | B2 radial TTC | C route intersection |
| --- | ---: | ---: |
| Critical-event recall | 100% (3/3) | 100% (3/3) |
| False alert segments | 18 | 13 |
| Median first-alert lead | 3.67 s | 1.91 s |
| Mean fragments / event | 1.00 | 1.33 |
| Post-event CLEAR | 100% (2/2) | 100% (2/2) |

The mask-gated dual-lidar source covered `90.41%` of detector-track
occurrences. Against evaluator-only native centers, matched geometry had
`0.106 m` median and `0.284 m` p90 position error. The actionable-risk effect
therefore remains real but weaker: C preserved all three events and removed
five false alert segments (`27.8%`), not the frozen `40%` strong-effect line.
This is `DIRECTIONALLY_POSITIVE / STRONG_EFFECT_NOT_REPRODUCED`, not an Android
or safety promotion.

The next fixed source replaced lidar with one phone-transferable rule: current
RGB person-box height, a fixed `1.70 m` upright-person prior, and camera focal
length. It covered all `4,826/4,826` detector-track occurrences, with `0.386 m`
median / `1.016 m` p90 position error against evaluator-only native centers.
On the same three events, both arms recalled `3/3` and cleared `2/2`; C reduced
false alert segments from `17` to `9` (`47.1%`) with `2.06 s` median lead. No
height, boundary, tracker, horizon, or threshold sweep was used. Because this
is the same already-opened curated window, it selects the Android information
source but is not a new generalization gate.

That fixed source is now wired into an isolated `dtrKnownHeight` Android build:
Camera2 calibration, CameraX frame binding, causal multi-person tracking,
known-height projection, relative-motion route intersection, and an explicit
`ONSET / HOLD / CLEAR / UNKNOWN` decision input to the shared feedback effect
boundary. It bypasses the legacy temporal stabilizer, so UNKNOWN cannot become
negative evidence. Compilation and one focused crossing-versus-lateral
lifecycle check pass; live-device behavior is still unverified.

Do not record the superseded 24/120 local RGB cohorts, widen this window, or
tune detector, tracker, IoU, horizon, smoothing, aggregation, or lifecycle
against these opened outcomes. The full-box upper-lidar source and the fixed
mask-gated dual-lidar source are now consumed observations. The metric-motion
source has now changed and the Android experiment exists; do not rescue any
opened result with a matcher/test matrix. The next useful DTR increment is only
a bounded live-device demonstration with the camera aligned to the walking
direction, not the superseded 24/120 recording cohorts.

The larger JRDB processed-label diagnostic remains a warning, not route
authority: over 175 events C raised recall from 75.43% to 92.00%, but false
alert segments rose from 191 to 260. JRDB's processed labels/timestamps have no
synchronized ego pose and are mostly interpolated, so this tests relative
closure rather than the exact wearer route.

## Boundaries

- G1 remains `STOP_G1_ACTIVE_MULTIVIEW_APPEARANCE /
  DEVELOPMENT_GATE_NOT_MET / NO_FINAL_TEST`; no extra views, seed/epoch/loss,
  view selector, larger DINO, G0 fusion, or consumed-subset search is open.
- Prior USTRF route-target source searches, causal route-intrusion signals, and
  HFTF selected-box projection outcomes remain historically closed. DTR-R0
  changes the event representation and requires a new controlled cohort; it is
  not a renamed rerun.
- `UNKNOWN` is not CLEAR, and silence is never presented as evidence that the
  route is safe.
- All four arms share a frozen 0.50-second clear grace. A single known-negative
  frame cannot fragment an active event, and UNKNOWN cannot complete a clear.
- Evaluator-only side/overhead video or ground markers may define wearer and
  target 2-D trajectories and route entry/exit times, but none of that truth
  may enter the DTR observation ledger.
- No Android/default-App, natural-distribution, user-benefit, product, or safety
  claim follows from route scaffolding or synthetic trajectories.
- Semantic Anchor to Marker Pose remains a separate live-device demonstration
  closure, not DTR-R0 algorithm evidence.
- The THÖR-MAGNI result is
  `PUBLIC_REAL_EXACT_GLOBAL_TRACK_AND_EGO_PRIVILEGED_CEILING_ONLY`: helmet
  centroids, a fixed person radius, and controlled laboratory motion are not
  body collision, intended route, natural walking, product, or safety truth.
- The RGB-only JRDB bridge is
  `CURATED_PUBLIC_REAL_RGB_DETECTOR_TRACKER_BRIDGE_ONLY`; its current range used
  native annotations. The later dual-lidar bridge removes that current-range
  dependency but still uses evaluator-only annotations for identity and future
  event truth. The bag supplies dynamic odometry but not a self-contained
  static TF tree.
- The causal dual-lidar bridge is one curated Development observation. Its
  `27.8%` false-alert reduction does not cross the `40%` strong-effect line.
  Learned trajectory models, Transformers, VLM fusion, and DTR-R1 were not
  opened to improve that number.
- The fixed-height RGB source restores the strong line only on that same
  curated Development window. The Android build assumes an upright full-body
  person and a camera aligned with the short route; it has no live-device,
  default-App, independent-walking, user-benefit, or safety authority.
