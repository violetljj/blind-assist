# Current decisions: L10-R0 active; Dynamic Travel Risk R2 established

Status: `L10_R0_ACTIVE / SC14_CAUSAL_MICRO_MOTION_ACTION_BELIEF_MECHANICS_SIGNAL` and
`DTR_R2_PUBLIC_REAL_PRIVILEGED_CEILINGS_ESTABLISHED /
NO_LOCAL_RECORDING / DETECTOR_RUNTIME_DOWNSTREAM`

## Parallel product lines

Ten-metre goal completion and obstacle-risk guidance are separate lines. L10
does not wait for DTR, DTR does not change L10, and evidence from one line does
not count for the other.

L10-R0 remains the goal-conditioned controller for readable destinations. Its
controlled Development benchmark reached 87.5% task completion, 81.2%
post-occlusion reacquisition, 93.1% direction accuracy, 1.7% wrong-lock frames,
and `0/50` target-absent false completions. A second seed reproduced 88.5%
completion, 82.4% reacquisition, 91.9% direction accuracy, 1.8% wrong-lock
frames, and another `0/50` false completions. These remain controller/mechanics
results, not real-camera or product evidence.

SC8 established the proposal-conditional `exact instance -> task functional
part set` selector. SC9, SC9T, and SC10 then removed evaluator masks from the
proposal source. Generic-handle RGB, ray-triangulated RGB, and generic-handle
native-depth routes all failed the frozen source gate; their best proposal
recall was 50% and their best legal-task result was `2/6`. These are consumed
mechanism negatives, not invitations to tune thresholds, clustering, or the
opened cohort.

The current increment is SC11. On the same SceneFun3D Development scene, a
truth-blind task-grounded Grounding DINO provider plus native depth, authorized
parent geometry, and frozen multi-view consensus raised functional-proposal
recall from `8/10` single-view to `10/10`, while precision remained `84.62%`.
Across the six evaluable task descriptions, legal commits improved from `3/6`
to `4/6`, mean target-set recall from 50% to 83.33%, and wrong parts fell from
eight to two. Four descriptions remain `NOT_EVALUABLE_PARENT_BINDING`; two
evaluable tasks still contain a wrong in-parent functional cluster.

Accept
`SC11_TASK_GROUNDED_RGBD_MULTIVIEW_FUNCTIONAL_PROPOSAL_DEVELOPMENT_SIGNAL` as
evidence that task-conditioned semantics plus metric multi-view consistency can
replace privileged functional masks at this narrow source layer. Do not promote
it to open-vocabulary or phone-camera coverage, exact-instance acquisition,
reachability, orientation, approach, arrival, completion, product, user-benefit,
or safety evidence. The next L10 source must add independently motivated
candidate-integrity or action geometry on separately versioned evidence; do not
rescue the two opened failures by detector, prompt, DBSCAN, seed, or fusion
sweeps, and do not restore parent-box center or visual scale as completion truth.

The next claim layer is action geometry. SC12 task-signed parent geometry
regressed against its static baseline on source-disjoint SceneFun3D 421013
(`7/9 -> 6/9` signed translation hits; 20.40 -> 30.39 degree mean error).
SC13 separated the approach-facing axis from a local contact-surface action axis
on fresh scene 421010, but retained `4/6` hits and improved mean error only
30.39 -> 28.18 degrees. Both static routes are closed on their consumed scenes;
do not sweep task mappings, PCA radii, OBB faces, thresholds, seeds, or fusion.

SC14 changes the source to causal before/after functional-point motion. On
421013, eight regions had enough evaluator-paired points. Relative to static
task/parent geometry, motion-type correctness improved `7/8 -> 8/8`; all seven
translations were within 15 degrees (`4/7 -> 7/7`) and mean error fell from
38.89 to 0.91 degrees. The single rotation axis passed and its estimated pivot
line was 2.22 cm from evaluator truth. Accept
`SC14_CAUSAL_MICRO_MOTION_ACTION_BELIEF_MECHANICS_SIGNAL` as evidence that
action kinematics should remain uncertain until causal motion closes them.

This is simulated paired-point mechanics evidence: the perturbation and point
correspondences are evaluator-authoritative, with no real user action, RGB
tracking, or safety authority. It cannot emit `HANDOFF_READY` and does not prove
reachability, body orientation, arrival, user completion, product benefit, or
safety. The next legal source is passive natural before/after RGB-D motion or an
explicitly authorized benign micro-interaction protocol; do not return to
static action-axis inference.

## DTR-R2 decision

Accept R2 as the current dynamic-track algorithm. It combines robust
finite-horizon route-occupancy consensus with a fixed 1.5-second imminent
route-intersection guard and produces stable
`ONSET / HOLD / ESCALATE / CLEAR` events.

- On route-authoritative THÖR-MAGNI, R2 recalls `10/10` events with 42 false
  alert segments versus R0's `9/10` and 55.
- On 27 JRDB test sequences, R2 recalls `164/175` with 256 false alerts versus
  R0's `161/175` and 260.
- On CODa development plus rainy holdout, R2 recalls `119/122` pedestrian
  events with 285 false alerts versus R0's `122/122` and 286. The three-event
  cost is retained rather than threshold-tuned away.

Accept S3 as the current static-obstacle ceiling. Its causal curved route plus
bounded vertical occupancy recalls all `12/12` CODa barrier, fixed-structure,
and temporary-obstacle path contacts while reducing 3 m proximity false alerts
from 104 to 10 (`90.4%`) and clearing `4/4` eligible events.

The decision is algorithm success at a privileged public-data ceiling, not
detector or product completion. CODa supplies positive dynamic events only for
pedestrians and positive static events only for the observed
barrier/fixed/temporary classes. Bicycle, vehicle, vegetation, thin-branch,
drop-off, and head-clearance positive recall remain unproved. GOOSE does not
supply synchronized route and ground-clearance truth for a trustworthy hanging
branch event, so head-clearance is `NOT_EVALUABLE`, not a negative result.

## 2026-08-28 source admission

The next-source canaries are complete without changing either frozen algorithm.
For L10, DSText V2 public media and the RRC-authoritative V2 training annotations
are now admitted; the annotation archive contains 90 XML ground-truth files.
This opens a frozen track-gap audit, but is not itself semantic-reacquisition
evidence. HierText independently confirms the steering
geometry gap: line center and word center disagree on the frozen coarse
direction for `20,866/85,380` (`24.439%`) legible words in multi-word validation
lines. This is static geometry evidence only.

For DTR, RoboSense metadata are compatible but its published raw
LiDAR/occupancy train-validation stream requires concatenating roughly 239.4 GB,
so it is not admitted as a minimal residual source. Argoverse 2 Sensor is
admitted at exact-log granularity: a deterministic 32-sweep, 3.100-second
validation shard produced a truth-separated current-LiDAR/pose adapter with
native boxes kept evaluator-only. Its simple straight 12 m admission tube has
`0` candidate native boxes, so it is not an event cohort. This is source and
adapter evidence, not a residual-occupancy algorithm gain. If that route is
opened, freeze a multi-log Development roster before comparing a
current/past-only residual source against unchanged R2.

The same-window R6 direct-metric falsifier is now complete. Calibrated
perspective zero-shot metric depth and a current/past raw-LiDAR ceiling both
recover `0/9` induced dropout windows with 12 false segments, but the result is
`R6_DIRECT_METRIC_SINGLE_FACTOR_NOT_EVALUABLE_STATIC_OCCUPANCY_MATCHER_UNREACHABLE`,
not a negative metric-depth finding. R5 residual occupancy derives closing
only from ego motion; the three event windows remain below `0.000316 m/s`
versus the frozen `0.05 m/s` minimum, so metric geometry alone cannot activate
the matcher. Do not rescue this cohort with threshold, tube, lifecycle,
imputation, or depth-backbone changes. A successor requires an independent
causal spatiotemporal occupancy-flow signal and must not use evaluator identity
for temporal association.

That R7-P successor is now complete on the same consumed Development window.
Its full flow ledger is produced from latest-at-or-before causal ego poses plus
current/past raw upper/lower Velodyne before labels are loaded; temporal
association is voxel-component correspondence, never evaluator physical ID.
It recovers all `9/9` induced windows versus `0/9` for both R2 and R6-P and
keeps critical-event recall at `3/3`. However, original false segments increase
`12 -> 20` (`+66.7%`), one-to-one event F1 remains 22.22%, and global flow
route-risk is active in `123/143` frames. The frozen false-segment gate fails:
`R7_P_CAUSAL_OCCUPANCY_FLOW_DEVELOPMENT_GATE_NOT_MET_NO_R8`.

Interpret this as existence of an independent spatiotemporal signal, not as a
detector-independent dynamic-occupancy algorithm success. Do not train R8, and
do not sweep voxel/history/speed/overlap/attribution/tube/lifecycle settings on
the opened cohort. Any successor must change the motion information source or
use a separately frozen fresh protocol.

DTR-M0 has now completed the permitted read-only attribution on that sealed
R7-P result. Of the 20 R7 false segments, 11 are unchanged R2 inheritance,
eight are flow-new, and one is flow-extended; the last two groups are the nine
flow-caused or flow-modified segments. At their first flow-only frame, all nine
R7 route-entry claims are unsupported by the scorer-side target's same-history
linear velocity. Five targets move below R7's already frozen `0.25 m/s`
minimum, while the other four are moving but have `1.15--3.50 m/s`
flow-to-target velocity error. M0 therefore localizes the immediate failure to
motion selectivity/attribution, with component discontinuity remaining only a
suspicion because R7 component IDs are frame-local. It does not support a
route-extrapolation negative or prove a specific split/merge mechanism.

The next admissible successor is a fresh frozen DTR-M1 motion-source
selectivity ceiling: replace only BEV component-centroid pseudo-flow with
point-wise 3-D scene flow or direct-velocity evidence that exposes confidence
and temporal consistency. Keep R2, the route tube, 0--3 s horizon, lifecycle,
and one-to-one evaluator fixed. Route-conditioned future occupancy and R8 RGB
training remain closed unless a better independent motion source first retains
dropout recovery while reducing false segments and increasing event F1.

That DTR-M1-O ceiling is now complete and negative under the frozen downstream.
AV2 native point flow could not honestly test R7's `9/9` or its nine modified
false segments because the admitted AV2 shard is a different zero-event cohort.
M1-O instead used current raw JRDB LiDAR plus causal native current/past 3-D
boxes to construct piecewise-rigid point velocity, then robustly aggregated it
before the unchanged R7 route-risk. It suppressed the first diagnostic risk in
`8/9` M0 flow-caused segments, showing markedly better motion selectivity, but
recovered only `6/9` dropout trials. The same oncoming event failed at all three
dropout durations. Original false segments were 17, including five new
point-velocity-induced segments; one-to-one true positives fell `3 -> 2` and
event F1 fell `22.22% -> 16.00%`.

The terminal is
`DTR_M1_O_POINT_VELOCITY_ORACLE_CEILING_NOT_MET_CLOSE_SCENE_FLOW_ROUTE`.
Correct point velocity is not sufficient under the frozen hard
point/cell-to-route aggregation. Do not run TeFlow, DeltaFlow, another
scene-flow estimator, R8, body-dilation/tube/lifecycle tuning, or
route-conditioned forecasting on this consumed protocol. Any successor needs
a newly authorized representation-level question and fresh evidence rather
than a different estimator.

## What stops here

- Do not record the superseded 24 canary or 120 staged local RGB clips.
- Do not widen the public cohorts or tune tracker, support, tube, horizon,
  urgency, lifecycle, or guard thresholds against the opened outcomes.
- Do not create separate per-class test matrices to make the numbers look more
  complete.
- Do not treat `UNKNOWN` or `NOT_EVALUABLE` as safe.

The next increment, only when deployment evidence is wanted, is to replace
privileged boxes/tracks with a real RGB/LiDAR detector and tracker behind the
already frozen metric-frame adapters. Phone recording and live-device testing
are not the current blocker or current priority. Full methods, receipts, and
claim limits are in [the DTR route README](../research/active/dtr-r0/README.md).
