# DTR: route-conditioned obstacle-risk events

Status: `DTR_R3_GATE_NOT_MET / R2_DYNAMIC_RETAINED /
S4_CONTINUOUS_GEOMETRY_VALIDATED_NO_PUBLIC_GAIN /
R5_RGB_DROPOUT_CANARY_GATE_NOT_MET /
R6_DIRECT_METRIC_SINGLE_FACTOR_NOT_EVALUABLE_STATIC_OCCUPANCY_MATCHER_UNREACHABLE /
R7_P_CAUSAL_OCCUPANCY_FLOW_DEVELOPMENT_GATE_NOT_MET_NO_R8 /
DTR_M1_O_POINT_VELOCITY_ORACLE_CEILING_NOT_MET_CLOSE_SCENE_FLOW_ROUTE /
DTR_M2_D_EXTENT_GAP_NOT_SUPPORTED_NO_FRESH_M2_O /
DTR_M3_D_EVALUATOR_CIRCLE_OBB_SEMANTICS_MISMATCH_NO_FRESH_M3_O /
DTR_C0_GLOBAL_ORIENTED_RISK_CONTRACT_NOT_EVALUABLE_ALWAYS_CONTACT_WINDOW /
DTR_C1_FRESH_GLOBAL_OBB_COHORT_ADMITTED_METADATA_ONLY /
DTR_C2_M1_CTB_CONFIDENCE_TRACK_GAP_BRIDGE_FRESH_MECHANICS_SIGNAL /
DTR_C3_M1_HYBRID_RAW_POINT_GAP_BRIDGE_FRESH_MECHANICS_SIGNAL /
DTR_C4_M1_CT_DETECTOR_INDEPENDENT_GLOBAL_RISK_DEVELOPMENT_SIGNAL /
DTR_C5_CROSS_ESTIMATOR_CONSENSUS_DEVELOPMENT_NO_GAIN /
DTR_C9_SELF_SUSTAINING_GLOBAL_RISK_BELIEF_DEVELOPMENT_SIGNAL /
DTR_C10_FIXED_C9_ALGORITHM_FRESH_CONFIRMATION_SIGNAL /
DTR_C11_ROUTE_REGION_OCCUPANCY_FRESH_SIGNAL /
DTR_C12_C13_ROUTE_TIME_REPRESENTATIONS_DEVELOPMENT_GATE_NOT_MET /
DTR_C14_STOCHASTIC_ROUTE_CONFLICT_DEVELOPMENT_GATE_NOT_MET /
DTR_C15_COMPONENT_VELOCITY_MIXTURE_DEVELOPMENT_GATE_NOT_MET /
DTR_C16_EMPIRICAL_VELOCITY_MODES_DEVELOPMENT_GATE_NOT_MET /
DTR_C17_TEMPORAL_ROUTE_CONSENSUS_DEVELOPMENT_GATE_NOT_MET /
DTR_C18_THREE_FRAME_MOTION_CONFIDENCE_DEVELOPMENT_GATE_NOT_MET /
DTR_C19_JOINT_MOTION_CONFIDENCE_DEVELOPMENT_GATE_NOT_MET /
DTR_C20_LOCAL_MOTION_VOTING_DEVELOPMENT_GATE_NOT_MET /
DTR_C21_SCENE_BIAS_RESIDUAL_MOTION_DEVELOPMENT_GATE_NOT_MET /
DTR_C22_EGO_RIGID_VISUAL_MOTION_DEVELOPMENT_GATE_NOT_MET /
DTR_C23_RIGID_TEMPORAL_OBJECT_FLOW_DEVELOPMENT_GATE_NOT_MET /
DTR_C24_CYCLE_CONSISTENT_POINT_FLOW_DEVELOPMENT_GATE_MET /
DTR_X0_MOTION_SOURCE_ATTRIBUTION_COMPLETE /
DTR_X3_FULL_LAG_FLOXEL_GATE_NOT_MET /
DTR_X3_FULL_REPLAY_FAILURE_ATTRIBUTION_COMPLETE /
DTR_X4_DETERMINISTIC_CLUSTER_VOTE_REPEATABILITY_GATE_NOT_MET /
DTR_X5_OVERLAP_CYCLE_SOURCE_FALSIFIER_GATE_NOT_MET /
DTR_X6_STATIC_WORLD_PERSISTENCE_FALSIFIER_GATE_MET /
DTR_X7_FULL_STATIC_WORLD_ANCHOR_GATE_NOT_MET /
DTR_X8_RGB_STATIC_VETO_FALSIFIER_GATE_MET /
DTR_X9_FULL_RGB_STATIC_VETO_GATE_NOT_MET /
DTR_X10_CROSS_FITTED_MOTION_AUTHORITY_GATE_NOT_MET /
R4_NOT_OPENED`

## Result first

DTR asks a narrower and more useful question than object detection:

```text
wearer future route tube
        intersect
target or obstacle future occupancy
        within 3 seconds
        -> ONSET / HOLD / ESCALATE / CLEAR
```

The public-data R3 ceiling is complete. It compared three fixed successors with
the retained dynamic R2 on THÖR-MAGNI, JRDB test, and CODa 16+18+20. The frozen
gate required at least 95% pooled critical-event recall and at least 30% fewer
false-alert segments; the headline target was 97% and 40%.

| Dynamic arm, pooled over 307 events | Recall | False segments | False change from R2 | Decision |
| --- | ---: | ---: | ---: | --- |
| **R2 guarded robust occupancy** | **`293/307` (95.44%)** | **583** | baseline | **retained** |
| R3-A curved CTRV + robust target CV | `285/307` (92.83%) | 673 | 15.44% worse | reject |
| R3-B straight + distributional occupancy | `280/307` (91.21%) | 554 | 4.97% better | reject |
| R3-C curved + distributional + R2 imminent guard | `293/307` (95.44%) | 637 | 9.26% worse | reject |

R3-C preserves R2's pooled recall but adds 54 false segments, so
`R3_GATE_NOT_MET`. R3-B deletes only 29 false segments while losing 13 recalled
events. No arm approaches the 30% false-reduction gate, and the conditional
learned stochastic R4 / three-source LOSO stage is therefore not opened. There
was no threshold, support, seed, backbone, or cohort sweep after seeing the
result.

## R5 RGB dropout canary

The first detector-independent residual canary is complete and did not pass.
It freezes R2, the three-second route tube, and the event lifecycle, then removes
the evaluator-associated true RGB track for the final `0.2 / 0.4 / 0.8 s`
before contact in each of the three events in the existing curated 143-frame
JRDB RGB window.

| Arm | Dropout-window alert recall at 0.2 / 0.4 / 0.8 s | Original event recall | Original one-to-one event F1 | Original false segments |
| --- | ---: | ---: | ---: | ---: |
| R2 track-only | `0/3 / 0/3 / 0/3` | `3/3` | 22.22% | 12 |
| R2 + bounded imputation | `2/3 / 2/3 / 2/3` | `3/3` | 16.00% | 19 |
| R2 + RGB semantic residual occupancy | `0/3 / 0/3 / 0/3` | `3/3` | 22.22% | 12 |

The intervention creates nine dropout-window evidence misses for track-only R2;
the RGB residual recovers `0/9`. This is not a detector-visibility failure. The
fixed ADE20K semantic head emits person pixels in `143/143` frames, and a
residual component is evaluator-associated in every stress trial with maximum
IoU `0.67-0.93`. The failure is metric occupancy: the fixed-height projection
puts the nearest residual component at roughly `1.69-1.75 m`, and no residual
frame intersects the frozen route-risk geometry. Bounded imputation recovers
window evidence in `6/9` trials but raises original false segments `12 -> 19`
(`58.3%`), so it is not retained.

Complete-event recall stays `3/3` because missing observations remain `UNKNOWN`
and cannot manufacture `CLEAR`; short dropout therefore does not by itself
erase the already active event. Track-only fragmentation is already zero, so a
fragmentation reduction is `NOT_EVALUABLE`, and this window has no eligible
CLEAR event. Occupancy calibration is also `NOT_EVALUABLE`: the source exposes
a hard semantic argmax and JRDB supplies boxes rather than pixel-occupancy
probabilities/truth.

This is source-level visual presence without a functional route-risk gain.
`R5_RGB_DROPOUT_CANARY_GATE_NOT_MET`; a learned RGB residual head is not opened
from this result. Do not rescue the fixed-height proxy with component, IoU,
distance, route, or lifecycle threshold sweeps. A successor would need a new
metric current/past occupancy source, such as the admitted AV2 raw-sensor route
or a separately justified direct metric BEV representation.

## R6 direct metric occupancy falsifier

The proposed metric-only successor has been executed on the exact same 143
frames and nine induced-dropout trials. It does **not** support the earlier
interpretation that fixed-height distance was the only remaining bottleneck.

R6 preserves the sealed R5 semantic mask, residual components,
evaluator-only current 2-D association, R2, lifecycle, route horizon/width, and
the `0.2 / 0.4 / 0.8 s` intervention. It replaces only the metric source:

- R6-RGB runs the frozen Depth Anything V2 Hypersim metric checkpoint, without
  scale/shift alignment, on the five calibrated and undistorted JRDB
  `752x480` perspective cameras. It never feeds the stitched panorama to the
  pinhole model.
- R6-P projects the latest current/past-only upper and lower raw Velodyne
  sweeps into the same semantic residual through official JRDB calibration.
  It is a privileged metric-source ceiling, not a product dependency.

| Arm | Dropout-window recovery | Original critical-event recall | Original one-to-one event F1 | Original false segments |
| --- | ---: | ---: | ---: | ---: |
| R2 track-only | `0/9` | `3/3` | 22.22% | 12 |
| R6-RGB direct metric occupancy | `0/9` | `3/3` | 22.22% | 12 |
| R6-P raw-LiDAR metric occupancy | `0/9` | `3/3` | 22.22% | 12 |

This is **not** a metric-depth negative. The privileged raw-LiDAR arm moved the
nearest associated occupied surface to `0.68-0.98 m`, versus `1.16-1.67 m` for
zero-shot RGB metric depth, yet both arms produced zero residual-risk frames.
The reason is structural: the frozen R5 residual matcher treats the residual
as static in the world and derives closing velocity only from ego motion. The
maximum ego speed in the three 0.8-second event windows is only
`0.000247 / 0.000316 / 0.000095 m/s`, while the frozen matcher requires
`0.05 m/s`. Thus `0/3` events are mechanically reachable by *any* static
metric-occupancy source on this cohort.

Terminal:
`R6_DIRECT_METRIC_SINGLE_FACTOR_NOT_EVALUABLE_STATIC_OCCUPANCY_MATCHER_UNREACHABLE`.
Do not treat `0/9` as evidence against direct metric depth or LiDAR occupancy,
and do not rescue it by lowering closing speed, widening the tube, changing
ONSET/CLEAR, adding imputation, or choosing a different depth backbone after
seeing the outcome.

The next information layer, only if separately opened, is detector-independent
**spatiotemporal metric occupancy** with its own causal closing signal (for
example raw-sensor occupancy flow or scene flow), not another static depth map.
A privileged current/past-only raw-sensor arm should establish that mechanism
before any learned RGB occupancy-flow head is trained. Temporal association
must not use evaluator identity.

## R7-P causal occupancy-flow ceiling

R7-P has now executed the separately opened information-bearing successor on
the same 143 frames and nine induced-dropout trials. It does not use the R5
semantic mask or detector boxes. Before labels are opened, current/past upper
and lower raw Velodyne sweeps are transformed with latest-at-or-before causal
ego poses, voxelized into BEV, and linked by component correspondence. Each
matched dynamic occupied cell carries `(x, y, vx, vy)` and is extrapolated for
0--3 seconds against the unchanged 0.65 m route tube. Evaluator physical IDs
never enter temporal association; native 3-D centers/radii are used only after
the truth-blind flow ledger is hash sealed to attribute current cells for
scoring.

| Arm | Dropout-window recovery | Original critical-event recall | Original one-to-one event F1 | Original false segments |
| --- | ---: | ---: | ---: | ---: |
| R2 track-only | `0/9` | `3/3` | 22.22% | 12 |
| R6-P static raw-LiDAR occupancy | `0/9` | `3/3` | 22.22% | 12 |
| R7-P causal raw-LiDAR occupancy flow | **`9/9`** | `3/3` | 22.22% | **20** |

This is a real information gain but not a functional-gate pass. R7-P raises
one-to-one true positives from 2 to 3, while evaluable alert segments grow from
15 to 24; event F1 therefore remains 22.22%. False segments increase `12 -> 20`
(`+66.7%`), above the frozen `13.2` limit. The truth-blind ledger contains
dynamic cells in `139/143` frames and its un-attributed global route-risk signal
is active in `123/143` frames, so the result is consistent with an overly broad
motion field rather than a clean detector-independent risk source.

Terminal:
`R7_P_CAUSAL_OCCUPANCY_FLOW_DEVELOPMENT_GATE_NOT_MET_NO_R8`. The controlled
result supports the narrow mechanism statement that temporal raw-sensor flow
can restore route-entry evidence that static distance cannot. It does not
support promotion from track-based risk to detector-independent dynamic
occupancy, and it does not authorize R8 RGB student training. Do not tune
voxel size, history, speed, overlap, tube, attribution, or lifecycle against
this opened cohort. A successor needs genuinely better independent motion
information or a fresh frozen protocol, not a sweep over this result.

## DTR-M0 read-only R7 error attribution

DTR-M0 replays the sealed R7 ledger and scorer-side JRDB target trajectories
without changing the flow source, matcher, route tube, lifecycle, evaluator,
gate, or verdict. It first separates inherited R2 false alerts from changes
actually caused by R7 flow:

| False-segment provenance | Count |
| --- | ---: |
| Unchanged R2 inheritance | 11 |
| Flow-new | 8 |
| Flow-extended | 1 |
| Flow-merged/split | 0 |

The nine flow-caused or flow-modified segments contain 23 flow-only risk
frames. Their first flow-only frame gives this mutually exclusive primary
attribution:

| Flow-caused primary attribution | Segments |
| --- | ---: |
| `STATIC_PSEUDO_MOTION` | 5 |
| `ATTRIBUTION_OR_FRAGMENTATION` | 4 |
| `REAL_MOVER_BUT_NONCRITICAL` | 0 |
| `BAD_ROUTE_EXTRAPOLATION` | 0 |
| `NOT_EVALUABLE` | 0 |

All nine R7 route-entry claims are unsupported by the scorer-side target's
same-history linear velocity. Five target speeds are below the already frozen
R7 `0.25 m/s` minimum. The other four targets are moving, but their R7 flow
velocity differs from target velocity by `1.15--3.50 m/s`; this supports a
motion-attribution mismatch, not the stronger claim that R7 correctly saw a
real but noncritical mover. Five segments share a responsible frame-local
component across multiple targets, four show temporal-component discontinuity
suspicions, and five show velocity discontinuity suspicions. Because component
IDs reset every frame, none of those flags proves a particular split or merge.

This diagnostic selects the next information source rather than reopening R7.
A fresh frozen DTR-M1 should replace only component-centroid pseudo-flow with
point-wise 3-D scene flow or direct-velocity evidence carrying confidence and
temporal consistency. R2, the route tube, 0--3 s horizon, lifecycle, and
one-to-one evaluator stay fixed. Route-conditioned future occupancy is not yet
the localized bottleneck, and R8 remains closed.

Evidence:

- `artifacts.local/evidence/dtr-m0-r7-error-attribution/result.json`, SHA-256
  `a495ff1c3921ca3617f9ae7a9d40a2b938f889032445835c69a38bd1cba65c92`.
- `artifacts.local/evidence/dtr-m0-r7-error-attribution/false-segments.csv`,
  SHA-256
  `1913b0115ff15689de53bcdf3798e9503b68f8c596fe58437dc9f9e3cd97f1c8`.
- `artifacts.local/evidence/dtr-m0-r7-error-attribution/timeline.svg`, SHA-256
  `097d06ef1c9a2dc38e0e874608aa2ba6a753868c3059b097035d240ee04a670d`.

## DTR-M1-O causal point-velocity oracle terminal

M1-O has now executed the frozen same-window ceiling. The originally proposed
AV2 native-flow arm cannot directly adjudicate R7's `9/9` or its nine modified
false segments: those denominators belong to the JRDB window, while the already
admitted 32-sweep AV2 shard is a different cohort with zero native boxes in the
frozen 12 m admission tube. Mixing those results would change the cohort and
the denominator at the same time as the motion source.

The valid ceiling therefore used the same construction principle as AV2 native
flow on the sealed JRDB window: current raw LiDAR points inside a native 3-D
box receive the causal piecewise-rigid velocity implied by that box and its
latest frozen-history past instance. `flow_support=1.0` denotes oracle support,
not a calibrated probability. Point velocities are reduced with a per-instance
BEV-cell median; R2, route tube, 0--3 s horizon, hard cell propagation,
lifecycle, induced dropout, and one-to-one evaluator remain unchanged.

| Arm | Dropout recovery | Critical-event recall | One-to-one event F1 | False segments | Motion-source-induced false segments |
| --- | ---: | ---: | ---: | ---: | ---: |
| R2 track-only | `0/9` | `3/3` | 22.22% | 12 | 0 |
| R7-P coarse occupancy flow | **`9/9`** | `3/3` | 22.22% | 20 | 9 |
| M1-O native-box point velocity oracle | **`6/9`** | `3/3` | **16.00%** | **17** | **5** |

M1-O suppresses the first diagnostic risk in eight of the nine M0
flow-caused/modified segments. That confirms the narrower mechanism: direct
velocity is much more selective than component-centroid pseudo-flow. It still
fails the functional ceiling. The same `pedestrian:35` oncoming event is not
recovered under any of the three `0.2/0.4/0.8 s` dropout interventions, so the
three misses are one event repeated at three stress durations, not three
independent failures. The oracle also creates five new target-aware false
segments, reduces one-to-one true positives from three to two, and lowers event
F1 from 22.22% to 16.00%.

The oncoming miss localizes the remaining boundary further. Native point
velocities near the target agree with its scorer-side motion, but the observed
hard surface points do not enter the frozen route tube in the 0--3 s point-cell
propagation even when center-plus-body-extent future truth becomes positive.
Thus a correct selective velocity source is not sufficient under the present
hard point/cell aggregation. This is not permission to tune the tube, add body
dilation, change the lifecycle, or reopen route forecasting on the consumed
cohort.

Terminal:
`DTR_M1_O_POINT_VELOCITY_ORACLE_CEILING_NOT_MET_CLOSE_SCENE_FLOW_ROUTE`.
Do not run TeFlow, DeltaFlow, another scene-flow estimator, R8 training, or
route-conditioned forecasting under this frozen protocol. A successor would
need a newly authorized representation-level question and fresh protocol; it
cannot rescue this result by changing the estimator.

Evidence:

- `artifacts.local/evidence/dtr-m1/point-velocity-oracle/result.json`, SHA-256
  `81ecf662c784208c14d5d2af451e324866cf4240ab884449d50997f3ab1b3e83`.
- `artifacts.local/evidence/dtr-m1/point-velocity-oracle/result.point-velocity-oracle.npz`,
  SHA-256
  `0dc135c90c0f23cb92aa5d9fdc83dfdf9d7cf584bb5b8451d60338a5de7bc906`.
- `artifacts.local/evidence/dtr-m1/point-velocity-oracle/result.point-velocity-oracle.json`,
  SHA-256
  `a57a4984dc9d6581be213a693d523ea9bd9154c8f85e5f58c4325930ed40571a`.

## DTR-M2-D read-only extent-gap terminal

M2-D tested the newly authorized representation-level hypothesis without
changing any prediction, threshold, route tube, lifecycle, or evaluator gate.
It replayed the exact sealed M1 ledger and compared each attributed
zero-radius cell trajectory with the continuous translation of that same
component's current native oriented box. The box used the robust median M1
direct velocity, fixed current yaw, the same 0--3 s horizon, and analytic
Minkowski entry against the frozen 0.65 m route body.

The proposed extent gap is not supported. All three M1 dropout misses--the
same `pedestrian:35` event repeated at `0.2/0.4/0.8 s`--are
`POINT_MISS_FOOTPRINT_MISS`, not `POINT_MISS_FOOTPRINT_HIT`. Every diagnostic
frame contains the target's own native component and box. The closest swept
footprint still misses the route body by **0.0374 m**; the closest point path
misses by 0.0292 m. Conversely, all five M1 new/modified false segments are
`FOOTPRINT_HIT_TRUTH_NEGATIVE`, so the extent primitive retains rather than
removes those false contacts.

This resolves the apparent M1 asymmetry narrowly. The README's positive
center-plus-body-extent future uses the realized future native box, whereas
M2-D translates the current box with the same frozen M1 constant velocity.
Those are different information contracts. Static current extent alone does
not reproduce the future event truth here. The remaining discrepancy may
involve time-varying future pose/shape, route/contact semantics, or both; this
audit does not distinguish them and does not reopen forecasting.

Terminal: `DTR_M2_D_EXTENT_GAP_NOT_SUPPORTED_NO_FRESH_M2_O`. Do not run the
fresh M2-O swept-footprint oracle from this hypothesis, and do not reopen
TeFlow/DeltaFlow, another scene-flow estimator, R8, or route-conditioned
forecasting. A successor would need a newly frozen information contract that
can distinguish realized future occupancy from a constant-velocity current
footprint, plus fresh evidence.

Evidence:

- `artifacts.local/evidence/dtr-m2/extent-gap-audit/result.json`, SHA-256
  `3c0c276ebc9ff5c8f3a4d36dd993332552b0f8d4c87e58d918a235cd3f320a76`.
- `artifacts.local/evidence/dtr-m2/extent-gap-audit/result.extent-gap.csv`,
  SHA-256
  `a415535f9ccbcc563351648f106782925df3c320fd19f461c7f6740c110a7ef9`.
- `artifacts.local/evidence/dtr-m2/extent-gap-audit/result.extent-gap.svg`,
  SHA-256
  `908bb989f550263e7930933329668e3dc4da15275f996ed0ecaab16f4ad8fdb9`.

## DTR-M3-D realized-future truth-contract terminal

M3-D corrects the remaining M2 interpretation before any forecasting work. It
is a scorer-side, read-only decomposition on the same consumed eight rows. At
each M2 diagnostic origin it evaluates the exact future-discrete evaluator
circle, realized future OBB, realized center with current OBB, and M1
constant-velocity center with realized future shape. A discrete CV/current-OBB
control and the original continuous M2 control are retained. No arm changes a
prediction, event, threshold, lifecycle, or gate.

All three repeated `pedestrian:35` dropout rows are
`EVAL_CIRCLE_HIT_REALIZED_OBB_MISS`. At the evaluator's first contact frame
185, the circularized contract is 0.0209 m inside its threshold, while the
realized native OBB remains 0.0374 m outside the 0.65 m route body. The
realized-center versus M1-CV center residual is only 0.0103 m at that frame.
The contact flip is therefore explained by circle-versus-OBB semantics, not by
demonstrated future-dynamics headroom. The longer 0.8 s counterfactual
realized-center/current-shape arm does hit later, but the actual realized OBB
never hits; that hybrid alone cannot authorize a dynamics model.

The five M1 new/modified false segments also split cleanly. Three target-owned
risks (`pedestrian:26`, `pedestrian:3`, and `pedestrian:5`) remain realized-OBB
misses and are genuine constant-transport false positives. Two are attribution
errors: the `pedestrian:32` row is driven by `pedestrian:26`, and the
`pedestrian:9` row by `pedestrian:34`. The latter source component is genuinely
positive under both evaluator-circle and realized-OBB geometry, but was bound
to the wrong target row.

Terminal:
`DTR_M3_D_EVALUATOR_CIRCLE_OBB_SEMANTICS_MISMATCH_NO_FRESH_M3_O`. Do not open
M3-O, learned/residual future occupancy, route-conditioned forecasting, R8, or
another scene-flow estimator. First choose and freeze the intended event
semantics: circularized proximity or oriented-body contact. If oriented-body
collision is the intended claim, the current evaluator must be revised and a
fresh cohort must be rescored before any dynamics ceiling. M3-D does not claim
that circular proximity is intrinsically wrong; it proves that it is not the
same truth contract as OBB collision on the decisive event.

Evidence:

- `artifacts.local/evidence/dtr-m3/realized-future-contract-decomposition/result.json`,
  SHA-256
  `2cdd82e4a1677c1d03042635aaeac06cac7ec3a07e3d14318945791efc04e297`.
- `artifacts.local/evidence/dtr-m3/realized-future-contract-decomposition/result.contract-decomposition.csv`,
  SHA-256
  `e4b815e5e936dcc9893ffc1b750f7461921728db6f100806e7d04961a860e79a`.
- `artifacts.local/evidence/dtr-m3/realized-future-contract-decomposition/result.contract-decomposition.svg`,
  SHA-256
  `306856d96f41dcb9e81415f530f9398da4ebf03664e7b91f1621c1bbdc394858`.

## DTR-C0 global oriented-risk contract terminal

C0 adopts the corrected task contract without changing any model or frozen
prediction. Primary `CONTACT` is the wearer-level union of all realized future
native OBB intersections with the 0.65 m route body. Legacy circles remain a
simultaneous secondary `PROXIMITY` component set. Per-target R2, R3-C, R7-P,
and M1-O alert timelines are unioned for global correctness; target/component
identity remains diagnostic only. This makes false segments per actual wearer
timeline minute well-defined on a cohort containing both bounded contacts and
known non-contact time.

The consumed M1/M3 window cannot evaluate that metric. Its OBB futures form an
overlapping `pedestrian:33 -> pedestrian:34 -> pedestrian:36` chain, so all
`143/143` frames are primary `CONTACT`. There is one left- and right-censored
global contact interval, zero bounded CONTACT events, and exactly 0.0 minutes
of known non-CONTACT authority. All four arms descriptively overlap that
always-positive interval, yielding a vacuous 1.0 overlap F1 and zero unmatched
segments. C0 records those values only as descriptive diagnostics; primary
CONTACT recall/F1 and false segments per wearer minute are `NOT_EVALUABLE`,
not 100%/zero-false results.

The legacy nine dropout rows reset to six OBB-CONTACT rows: all three repeated
`pedestrian:35` circle-only trials are excluded. On those six non-independent
rows, R7-P and M1-O each have target-component raw motion evidence in `6/6`.
R2 and R3-C produce a global alert in `5/6`, but the stressed target contributes
in `0/6`; unrelated simultaneous targets create those alerts. These are
consumed-cohort mechanism diagnostics, not recovery performance or a fresh
100% ceiling. Secondary circle-only contribution segments are `1 / 1 / 2 / 2`
for R2 / R3-C / R7-P / M1-O respectively.

Terminal:
`DTR_C0_GLOBAL_ORIENTED_RISK_CONTRACT_NOT_EVALUABLE_ALWAYS_CONTACT_WINDOW`.
The contract is retained, but this window cannot rank algorithms or estimate
false alerts per wearer minute. The next admissible experiment is a fresh,
frozen global-OBB cohort with at least one bounded CONTACT event and known
non-CONTACT wearer time. Forecasting, R8, scene-flow competition, and a
deployable direct-motion estimator remain closed until that confirmation.

Evidence:

- `artifacts.local/evidence/dtr-c0/global-oriented-risk-contract/result.json`,
  SHA-256
  `df7f2bf40efa9dbc8438bb9e47ec1afdbdcd7ddd25765c228239f896275ae0f7`.
- `artifacts.local/evidence/dtr-c0/global-oriented-risk-contract/result.global-scorecard.csv`,
  SHA-256
  `ed40837b97807849181264d060dd9665e73afb780b31939e262f21564571d508`.
- `artifacts.local/evidence/dtr-c0/global-oriented-risk-contract/result.global-scorecard.svg`,
  SHA-256
  `ad5eb88dd6868730557302d270479003d39a2e8325a2d26b84dd8dcd65ac8e82`.

## DTR-C1 metadata-first fresh cohort admission

C1 resolves C0's missing denominators before touching an algorithm. It reads
only the JRDB train native 3-D label and image-timestamp archives. It does not
download or open RGB, LiDAR, bags, detector/tracker output, or predictions.
The entire consumed `packard-poster-session-2019-03-20_1` sequence is excluded,
not only frames 115--257. The other 26 sequences are scanned under the exact
C0 wearer-global OBB CONTACT plus secondary circle-only PROXIMITY contract.

Admission ordering and denominators were fixed before the scan: sequences are
sorted lexicographically, with no event-yield ranking. The roster is the
shortest prefix reaching the preferred `20 bounded CONTACT / 10 unique-first-
responsible / 120 s known non-CONTACT` target; if unavailable, it falls back
to the shortest prefix reaching the minimum `12 / 6 / 60 s` gate. A bounded
event must be preceded and followed by known non-CONTACT, so an interval ending
against `UNKNOWN` is not counted as evaluable.

The preferred gate passes with the first seven fresh sequences:

| Sequence | Bounded CONTACT | Unique responsible | Known non-CONTACT | CONTACT duty |
| --- | ---: | ---: | ---: | ---: |
| bytes-cafe-2019-02-07_0 | 4 | 4 | 94.14 s | 19.44% |
| clark-center-2019-02-28_0 | 1 | 1 | 33.31 s | 9.89% |
| clark-center-2019-02-28_1 | 0 | 0 | 96.92 s | 0.00% |
| clark-center-intersection-2019-02-28_0 | 1 | 1 | 59.74 s | 10.60% |
| cubberly-auditorium-2019-04-22_0 | 7 | 7 | 54.30 s | 37.59% |
| forbes-cafe-2019-01-22_0 | 6 | 6 | 60.63 s | 37.44% |
| gates-159-group-meeting-2019-04-03_0 | 2 | 2 | 10.62 s | 82.01% |
| **Frozen total** | **21** | **21** | **409.66 s** | **26.91%** |

The roster contains 8,368 frames and 578.91 s of wearer timeline: 150.84 s
CONTACT, 19.53 s PROXIMITY, 390.13 s CLEAR, and 18.40 s right-censored
UNKNOWN. The zero-event Clark Center sequence is retained by the ordering rule
and supplies clean negative exposure. All 21 bounded event onsets have exactly
one earliest responsible component, so they are eligible for a later frozen
single-target dropout stress; component identity still does not enter global
alert correctness.

Decision: `DTR_C1_FRESH_GLOBAL_OBB_COHORT_ADMITTED_METADATA_ONLY`. JRDB is not
cohort-insufficient. Freeze the tracked roster and its 21 event definitions.
Raw-sensor acquisition and unchanged R2/R3-C/R7-P/M1-O replay may be
protocolized as a separate second stage, but no arm has run and no algorithm
ranking, recovery, direct-velocity, deployment, product, or safety claim
follows from C1. Forecasting, R8, TeFlow, DeltaFlow, and training remain closed.

Evidence:

- frozen roster: `dtr_c1_fresh_global_obb_roster.json`, SHA-256
  `1ea5067207b957c3c1c7462aed3f5df63231413e5a6da58467b3c36190cf5ae6`;
- `artifacts.local/evidence/dtr-c1/global-obb-cohort-admission/result.json`,
  SHA-256
  `2f6504f642327811892f71f7d2ff5998d15f73f27272f2b24a09880649f99cd8`;
- per-frame truth ledger `result.timeline.jsonl`, SHA-256
  `175429630dcebc8ec1cbb6185a7dae222ba0e7b2709cb341dae6709bbc721166`;
- sequence table `result.sequences.csv`, SHA-256
  `65dba763bdcc01eeaac72c060b63b032fdbf35d60a2ec8c5d396ca209df647dc`;
- bounded-event table `result.events.csv`, SHA-256
  `2180a7eb199b8652e50f853321b970f41b328459918aab432e76c9120819b64d`.

## DTR-C2 confidence-aware scene motion and track-gap bridge

C2 acquired only the seven frozen C1 raw bags and replayed the corrected
wearer-global future-OBB CONTACT contract without route, lifecycle, motion-bound,
training, or threshold changes. R7-P and M1-CT build temporal motion without
evaluator identity; current native boxes remain a privileged scorer-side
spatial attribution ceiling. M1-O remains label-derived. C2 is therefore an
algorithm/mechanics replay, not a deployable sensor result.

M1-CT adds three causal requirements to every admitted raw-LiDAR occupancy-cell
velocity: current and past spatial support, ego-compensated forward-advection
agreement, and velocity consistency with an independent historical sweep. Its
confidence is the minimum of those terms. M1-CTB then opens a second, narrow
channel: unfiltered dense motion can bypass confidence only inside an observable
bounded gap of a previously tracked target. It can bridge occlusion, but cannot
originate a detector-independent alert.

| Arm | CONTACT recall | False segments | False / non-CONTACT min | Event F1 | Median lead | Dropout recovery |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R2 track-only | `20/21` | 29 | 4.25 | **57.14%** | 2.05 s | `0/63` |
| R3-C ceiling | **`21/21`** | 36 | 5.27 | 53.85% | 2.09 s | not scored |
| R7-P naive dense flow | `20/21` | **90** | 13.18 | 30.53% | **2.44 s** | `29/63` |
| M1-O label point-velocity ceiling | `19/21` | 86 | 12.60 | 30.16% | 3.68 s | **`50/63`** |
| M1-CT confidence + consistency | `20/21` | **38** | **5.57** | **50.63%** | 2.08 s | `18/63` |
| **M1-CTB + bounded track-gap bridge** | **`20/21`** | **38** | **5.57** | **50.63%** | **2.08 s** | **`29/63`** |

Relative to R7-P, M1-CT removes 52 false segments (`57.8%`) while preserving
`20/21` CONTACT recall. F1 rises `30.53% -> 50.63%` (+20.10 percentage points)
and median lead remains above two seconds. Confidence alone discards useful
low-confidence occlusion evidence (`18/63` versus R7's `29/63`); the track-gap
bridge restores all recovery available from R7 without reopening those cells as
independent natural-replay alerts. On the earlier consumed R7 stress this same
composition preserves R7's `9/9` by construction: inside a declared track gap,
M1-CTB is exactly R7-P.

Decision:
`DTR_C2_M1_CTB_CONFIDENCE_TRACK_GAP_BRIDGE_FRESH_MECHANICS_SIGNAL`. This is a
strong fresh mechanism result, but not yet a wholesale R2 replacement: R2 still
has fewer natural false segments (29 versus 38) and higher natural F1 (57.14%
versus 50.63%). The next information-bearing increment is a deployable raw-point
direct-velocity source behind the now-supported confidence/bridge interface,
with measured GPU selection; do not rescue C2 with route or confidence-threshold
sweeps. Complex trajectory forecasting and R8 remain closed.

Evidence:

- `artifacts.local/evidence/dtr-c2/fresh-global-obb-replay/result.json`, SHA-256
  `9c19bfcd09c0ee82e485e72d24b27021b862579b29cf40e856ce84e6412d6a8d`;
- scorecard `result.scorecard.csv`, SHA-256
  `17eb81de3fc74e51ffd7f43ec9d337a81a52dfdc6392431f6ac85ff4af85f75f`;
- frozen-bag acquisition receipt `acquisition.json`, SHA-256
  `774f5a7b3dcf4ecfc87e69f3c25a3fb36699dcaddb482a679b785d074dd4ad67`.

## DTR-C3 raw-point direct velocity and multi-resolution evidence routing

C3 changes the motion observation rather than the route matcher. After causal
ego compensation, current and historical raw LiDAR points are reduced to fixed
0.24 m 3-D voxel centroids. Only reciprocal nearest correspondences inside the
unchanged R7 speed range emit direct velocity; raw point counts provide spatial
support. Evaluator identity and future boxes never enter matching. The design is
a deliberately small direct-velocity analogue of the ego-compensated local
correspondence direction used by [ICP-Flow](https://openaccess.thecvf.com/content/CVPR2024/html/Lin_ICP-Flow_LiDAR_Scene_Flow_Estimation_with_ICP_CVPR_2024_paper.html),
with confidence/motion separation motivated by
[SLIM](https://openaccess.thecvf.com/content/ICCV2021/html/Baur_SLIM_Self-Supervised_LiDAR_Scene_Flow_and_Motion_Segmentation_ICCV_2021_paper.html).
It is not a reproduction of either model.

Every sequence benchmarked equivalent SciPy KD-tree and Torch CUDA matching
through the shared execution contract before launch. The actual data selected
CUDA on one sequence and CPU on six; all receipts record measured latency and
the observed device. CPU was retained only when measured faster.

| Arm | CONTACT recall | False segments | Event F1 | Median lead | Dropout recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| R7-P naive dense flow | `20/21` | 90 | 30.53% | 2.44 s | `29/63` |
| M1-CTB cell confidence bridge | `20/21` | **38** | **50.63%** | 2.08 s | `29/63` |
| M1-PD reciprocal raw-point velocity | `20/21` | 85 | 31.75% | 2.83 s | **`52/63`** |
| M1-PDC + independent-history hard gate | `20/21` | 53 | 42.55% | 2.08 s | `36/63` |
| M1-PDCB raw-point gap bridge | `20/21` | 53 | 42.55% | 2.08 s | **`52/63`** |
| **M1-HYBRID multi-resolution router** | **`20/21`** | **38** | **50.63%** | **2.08 s** | **at least `52/63`** |

Raw-point reciprocal motion is much better occlusion evidence than R7 on this
cohort (`52/63` versus `29/63`) but is too broad to originate alerts by itself.
The old independent-history cell gate cuts M1-PD false segments `85 -> 53`, yet
also cuts recovery `52 -> 36` and sometimes fragments one false interval into
several. M1-HYBRID therefore routes evidence by observable state: normal frames
use the lower-false M1-CT path; a bounded gap of an already tracked target may
use M1-PD, with sealed R7 retained as fallback. Its natural score is exactly
M1-CTB, while `52/63` is the confirmed M1-PD lower bound for the union. Retaining
R7 in the gap also preserves the earlier consumed `9/9` recovery by construction.

Decision:
`DTR_C3_M1_HYBRID_RAW_POINT_GAP_BRIDGE_FRESH_MECHANICS_SIGNAL`. Relative to
R7-P, the selected structure confirms the same `20/21` recall, 52 fewer natural
false segments (`-57.8%`), +20.10 F1 points, and at least 23 additional fresh
dropout recoveries. This is the strongest DTR motion mechanism so far, but it
still uses privileged current boxes for scorer-side spatial attribution and does
not beat R2's natural 29 false segments. Do not tune voxel size, correspondence,
confidence, route, or lifecycle on C3. The next layer is detector-independent
occupancy attribution and product-facing sensor integration, not complex
trajectory forecasting.

Evidence:

- `artifacts.local/evidence/dtr-c3/raw-point-direct-velocity-canary/result.json`,
  SHA-256
  `499367d4f059bbe063d7beefd55b3ec6e2ebd3bafce425cc53eb87320c7af5d8`;
- seven per-sequence raw-point manifests and backend receipts under
  `artifacts.local/evidence/dtr-c3/raw-point-direct-velocity-canary/ledgers/`.

## DTR-C4 detector-independent global route risk

C4 removes the remaining scorer-side current-box attribution. Before the C1
roster or native OBB archive is opened, every cell in each sealed truth-blind
motion ledger is queried directly against the unchanged route tube and passed
through the unchanged ONSET/HOLD/ESCALATE/CLEAR lifecycle. The resulting global
alert timeline is written and hash sealed; only then does the evaluator open
future OBB truth. This changes the risk representation, not a route threshold.

The mechanism follows the same architectural separation seen in
[Drive-OccWorld](https://ojs.aaai.org/index.php/AAAI/article/download/33010/35165),
where future occupancy/flow is conditioned on ego action and queried by an
occupancy cost, and in
[PORA](https://arxiv.org/html/2501.16480), where a planned path is evaluated
against spatiotemporal probabilistic occupancy. C4 is deliberately smaller: it
tests current causal direct velocity plus continuous route-entry geometry, not a
learned or multimodal future forecaster.

| Detector-independent arm | CONTACT recall | False segments | False / non-CONTACT min | Event F1 | Median lead |
| --- | ---: | ---: | ---: | ---: | ---: |
| R7-P global naive dense flow | **`21/21`** | 149 | 21.82 | 21.99% | **2.91 s** |
| **M1-CT global confidence motion** | `17/21` | **34** | **4.98** | **47.22%** | 0.46 s |
| M1-PD global raw-point velocity | **`21/21`** | 121 | 17.72 | 25.77% | 2.53 s |
| M1-PDC global confident raw-point velocity | `20/21` | 68 | 9.96 | 36.70% | 1.87 s |

This is the first target/detector-independent global-risk signal in DTR. M1-CT
keeps 17 of 21 bounded CONTACT events with only 34 false segments and 47.22% F1.
That is close to the privileged current-box M1-CT ceiling (`20/21`, 38 false,
50.63% F1) while removing target identity and current native boxes from the
prediction path. The price is material: four events are missed and several
matched intervals begin after physical contact, so this arm is not ready to
replace M1-HYBRID. M1-PDC recovers three of those events but doubles false
segments; opening it globally is not the answer.

Decision:
`DTR_C4_M1_CT_DETECTOR_INDEPENDENT_GLOBAL_RISK_DEVELOPMENT_SIGNAL`. The same
seven C1 sequences were already opened by C2/C3, so this is Development evidence,
not a fresh confirmation. Do not tune route, confidence, or lifecycle on this
cohort. The next representation change is a wearer-global causal occupancy
belief that advects high-confidence motion anchors through short observation
gaps and exposes calibrated occupancy uncertainty to route-risk.

Evidence:

- `artifacts.local/evidence/dtr-c4/detector-independent-global-risk/predictions.json`,
  SHA-256
  `9914ffb202417527dc856c8db1433b832a2da477fe3aa0db47a24e022a38b560`;
- `artifacts.local/evidence/dtr-c4/detector-independent-global-risk/result.json`,
  SHA-256
  `dda143125dc484f22b55ae68dd26247ce1883cfea066d4047bec2c0d941480b9`.

## DTR-C5 cross-estimator consensus ablation

C5 tested one narrow explanation for the C4 misses. M1-PDC could add a global
route-entry cell only when a current M1-CT cell independently agreed in position
and velocity at the already frozen M1 half-confidence boundary. This was not a
threshold sweep: the position and velocity radii were derived directly from
the existing Gaussian confidence scales.

The result was negative: C5 remained at `17/21` recall, increased false segments
`34 -> 39`, and reduced F1 `47.22% -> 44.16%`. It improved median matched lead
`0.46 -> 0.81 s` but recovered none of the four missed events. Same-frame source
agreement therefore does not supply the missing information. This agrees with
[SeFlow](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00143.pdf),
which motivates explicit dynamic/static separation and object/cluster motion
consistency, and with
[ICP-Flow](https://openaccess.thecvf.com/content/CVPR2024/html/Lin_ICP-Flow_LiDAR_Scene_Flow_Estimation_with_ICP_CVPR_2024_paper.html),
which uses locally rigid temporal association rather than isolated point
agreement.

The required backend probe selected SciPy CPU because its measured median was
0.156 ms versus 2.022 ms for Torch CUDA on this tiny batched match; the receipt
records `CPU_FASTER_MEASURED` and the observed RTX 5060 CUDA device. C5 is closed
as `DTR_C5_CROSS_ESTIMATOR_CONSENSUS_DEVELOPMENT_NO_GAIN`; do not iterate its
matching radii on the consumed cohort.

Evidence:

- `artifacts.local/evidence/dtr-c5/cross-estimator-consensus/backend.json`;
- `artifacts.local/evidence/dtr-c5/cross-estimator-consensus/result.json`,
  SHA-256
  `ede3ca21c725d8f935b7cefa4043ed3a9c27c33afafe611b2f03d64c380fdfa2`.

## DTR-C6--C9 wearer route, world occupancy belief, and risk state

C6 corrects a geometry mismatch left by C4. M1-CT velocity is ego-compensated
scene motion in the world, but collision depends on motion relative to the
wearer's short-term route. C6 estimates wearer velocity only from past poses,
transforms confident cells into the world frame, advects them through the
unchanged 0.5 second lifecycle grace, and then evaluates continuous relative
collision geometry. Prediction is hash sealed before the C1 roster or future
OBBs are opened.

| Detector-independent arm | CONTACT recall | False segments | False / non-CONTACT min | Event F1 | Median lead |
| --- | ---: | ---: | ---: | ---: | ---: |
| C4 M1-CT global current motion | `17/21` | 34 | 4.98 | 47.22% | 0.46 s |
| C6 wearer-relative current motion | `17/21` | 51 | 7.47 | 38.20% | 0.80 s |
| C6 wearer-relative world belief | `19/21` | 53 | 7.76 | 40.86% | **1.82 s** |
| C8 fixed-window global-risk bridge | `17/21` | 30 | 4.39 | 50.00% | 0.81 s |
| **C9 self-sustaining global risk belief** | **`18/21`** | **28** | **4.10** | **53.73%** | **1.01 s** |

The C6 result establishes that world occupancy memory contains missing causal
information: it recovers two events and adds 1.36 seconds of median lead over
C4, but allowing any remembered cell to originate an alert is too broad. A
motion-authority router using the already frozen 0.25 m/s floor did not change
the aggregate and was closed without a parameter sweep.

C8/C9 instead split alert authority by state. Current confidence-aware motion
is the only independent ONSET source. World occupancy may only continue an
already active global route-conflict event; it can never originate one from
CLEAR. C8's fixed bridge window reduced false segments but was too short to
recover an event. C9 lets valid belief maintain the active event for as long as
causal occupancy remains, then delegates clearing to the unchanged lifecycle.
This has the same useful asymmetry as
[Occupancy Flow Fields](https://arxiv.org/pdf/2203.03875): flow-traced occupancy
is reachable from current occupancy and can erase unsupported future occupancy,
rather than granting every future cell equal authority. It also keeps the
relative-motion/path-overlap separation emphasized by
[PORA](https://arxiv.org/html/2501.16480).
This recovers one event, reduces false segments `34 -> 28`, raises F1
`47.22% -> 53.73%`, and doubles median lead `0.46 -> 1.01 s` relative to C4.
Its 28 false segments are also below the privileged track-only R2's 29, although
R2 still has higher recall (`20/21`) and F1 (57.14%).

This produces the intended algorithm story:

```text
track-only loses occluded hazards
  -> dense motion restores evidence but creates pseudo-motion
  -> confidence-aware direct velocity suppresses pseudo-motion
  -> global motion originates risk without a detector box
  -> wearer-relative world occupancy supplies causal continuity
  -> route-risk state separates ONSET authority from belief maintenance
```

Decision:
`DTR_C9_SELF_SUSTAINING_GLOBAL_RISK_BELIEF_DEVELOPMENT_SIGNAL`. C9 becomes the
detector-independent natural alert-origin path. M1-HYBRID remains the bounded
detector-gap recovery layer, so the earlier consumed R7 `9/9` recovery is
preserved by composition. The remaining three misses have no M1-PDC evidence
while a C9 state is active; opening raw-point motion as an independent origin
would return to the already measured 68 false segments and is not justified.
Do not tune the belief horizon, route threshold, confidence, or motion floor on
these consumed seven sequences. The next confirmation should use a new cohort;
the next algorithm source upgrade is calibrated occupancy probability or a
semantic route plan, not another same-source gate.

Evidence:

- C6 sealed prediction SHA-256
  `64880b1b130503fc4fd8f6f475fdd934e78df3dc8a15714845f75ea2b504f315`,
  result SHA-256
  `de001057ea0b22cff460a86a25641778aa25ec010e11bd89029e254818c0c086`;
- C8 sealed prediction SHA-256
  `ba32f91984a7ba9faf39f67ae4444d41917336cf7a9e8d53dc7ab401b280ad67`,
  result SHA-256
  `0affd5f151fef5aa5e20dfdb50d18bb7b980d3c47e1349ef4d1b4e50d3a6891a`;
- C9 backend receipt SHA-256
  `668221ca84014961fe6482bae502501939d4cf62afb8a5b23a78c0033571ca9f`,
  sealed prediction SHA-256
  `7d852751f7d14ed01684e78dd2da76a3cf655ea3b28dbd3f138adb9e4ef04046`,
  result SHA-256
  `c9ca1b9fe6e26e19355baac91a9d766e3151962ad5e6d47f3b21aeaef99bdb89`.

## DTR-C10 fixed-C9 fresh confirmation

C10 freezes the smallest still-unscored JRDB subset meeting the unchanged C1
event and non-CONTACT denominators, then applies C9 without changing its
confidence, motion, route, belief, or lifecycle settings. The three raw-bag
workers see timestamps and calibration but not the roster or native future OBB
truth. Their predictions are independently hash sealed, then merged and sealed
again before the score phase first opens truth.

| Fixed C9 cohort | CONTACT recall | False segments | False / non-CONTACT min | Event F1 | Median lead |
| --- | ---: | ---: | ---: | ---: | ---: |
| Seven-sequence Development | `18/21` (85.71%) | 28 | 4.10 | 53.73% | 1.01 s |
| **Three-sequence algorithm-fresh confirmation** | **`17/20` (85.00%)** | **11** | **3.98** | **70.83%** | **1.79 s** |

The fixed mechanism therefore retains recall, slightly lowers false-alert rate,
and improves event F1 and lead on this fresh preferred cohort. The result is
heterogeneous rather than universally solved: the three sequences contribute
`6/6` with 3 false segments, `6/8` with 1 false segment, and `5/6` with 7 false
segments. C9's central state-authority claim also remains observable: remembered
world occupancy supplies 465 belief-only maintenance frames across the cohort,
while 31 belief-risk frames are blocked from originating an alert in CLEAR.

Decision: `DTR_C10_FIXED_C9_ALGORITHM_FRESH_CONFIRMATION_SIGNAL`. The
track-only -> dense recovery -> pseudo-motion -> confidence-aware motion ->
route-risk state story now has an algorithm-fresh confirmation, not merely a
same-cohort Development improvement. This does not calibrate risk probability.
Probability calibration must use a separate future cohort because cell
occupancy scores are not automatically calibrated probabilities, and spatially
correlated cells cannot be combined with an independence product. The admitted
next source upgrade is therefore a separately calibrated route-region occupancy
probability with flow-reachability support, not a C9 threshold sweep.

All GPU-capable launch classes were measured on the shared research runtime.
For these irregular component sets and per-frame collision batches, CPU was
faster on every worker and was selected with `CPU_FASTER_MEASURED`; the receipts
also verify that each CUDA candidate actually ran on the RTX 5060 before CPU was
chosen.

Evidence:

- committed fresh roster SHA-256
  `07ddb799df04178ef7a1ed649f33b21eb1e5fac471f7d4582d11b4c2ba68543e`;
- acquisition SHA-256
  `959cf20c4573c49f6659be80a76c4ee268b410f23eb265f13499286595e63bfc`;
- combined truth-blind prediction SHA-256
  `ce2c5afb22d18f46767feb87cbbb242163a775275ee327764c995fd72302bb10`;
- result SHA-256
  `6ba84a5615aa91156fc9f8fc9005b813b8634232ce5ecb3fefdd41d442239976`.

## DTR-C11 calibrated route-region occupancy

C11 replaces C9's raw route-risk state score with a calibrated route-region
occupancy question: how much confidence-weighted, temporally fresh scene motion
is currently entering the route, and how much remains flow-reachable from the
last admitted state? It voxel-collapses correlated point evidence, normalizes
the intensity by cell area, and fits separate Platt maps for alert onset and
maintenance. It does not multiply spatial cells as if they were independent.
The probability decision is frozen at `0.5`; the unchanged imminent continuous
collision geometry remains the only below-threshold safety bypass.

The coefficients were fitted once on ten consumed sequences, then frozen before
the four raw bags below were acquired. The truth-blind workers sealed both the
fixed C9 arm (`M1_SRB_GLOBAL`) and C11 (`M1_RROQ_GLOBAL`) before scoring opened
the labels.

| Four-sequence fresh cohort | CONTACT recall | False segments | False / non-CONTACT min | Event F1 | Median lead |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed C9 route-risk state | `17/20` (85.00%) | 17 | 5.14 | 62.96% | 2.01 s |
| **C11 calibrated route-region occupancy** | **`17/20` (85.00%)** | **11** | **3.32** | **70.83%** | **1.45 s** |

C11 therefore preserves all 17 recalled events while removing 6 of 17 C9
false segments (`-35.3%`) and adding 7.87 event-F1 points. The cost is 0.56 s
less median lead. Fresh frame calibration is useful but not solved: 5,069 known
frames give Brier `0.2023`, NLL `0.5874`, and equal-count 10-bin ECE `0.1574`.
These are public-replay calibration measurements, not deployment calibration.

Decision: `DTR_C11_ROUTE_REGION_OCCUPANCY_FRESH_SIGNAL`. The evidence supports
the intended story at route-risk level: dense confidence-aware motion can keep
fresh event recall while calibrated route-region authority suppresses
pseudo-motion alerts. Flow reachability is support, not alert authority: a
development-only variant that allowed multi-observation flow traces to originate
alerts kept `34/41` recall but increased false segments from 33 to 38, so it was
rejected before the fresh run. The smallest next falsifier is a new frozen
cohort on which route-conditioned reachability either preserves recall without
erasing useful lead, or fails; no C11 threshold sweep is authorized.

Mechanism basis and evidence boundary:

- [Occupancy Flow Fields](https://arxiv.org/html/2211.04340v2) motivates jointly
  representing future occupancy and flow rather than treating motion and
  occupancy as unrelated outputs; C11 borrows only the reachability idea.
- [Waymo Occupancy Flow Fields](https://waymo.com/research/occupancy-flow-fields-for-motion-forecasting-in-autonomous-driving/)
  reinforces flow-traced occupancy, but its autonomous-driving benchmark is not
  BlindAssist product evidence.
- [Continuous Occupancy Fields](https://arxiv.org/html/2501.16480) motivates
  avoiding grid-resolution-dependent probability mass; C11's cell-area
  normalization is the narrow adopted mechanism.
- [Probabilistic Occupancy Grid Mapping](https://arxiv.org/html/2103.04795)
  supports explicit uncertainty treatment, not a claim that correlated cells
  are independent or that public replay is deployment calibrated.

All GPU-capable launch classes used the shared research launcher and emitted
backend receipts. CUDA execution was verified on the RTX 5060, but the
representative batches selected CPU with `CPU_FASTER_MEASURED`: route collision
was about `0.13 ms` on CPU versus `1.2-1.6 ms` on CUDA, and Platt inference was
about `0.025-0.033 ms` versus `0.35-0.40 ms`. The batches are too small to repay
GPU launch and transfer overhead.

Evidence:

- committed calibrator SHA-256
  `925f9016786122f363ff13e10ab3373398b526b7523dab43c8ce5c8e5d686ec3`;
- committed fresh roster SHA-256
  `88063217b7a4ffd0ee9bcb2c8e71a29c58c4461cf64333d3107406206b26bab9`;
- acquisition SHA-256
  `544e5e2069b35edf9b08b16c2e10d3f4e4d518c52320ca79abf6507abec0bfe5`;
- combined truth-blind prediction SHA-256
  `55fc4831f7429fcefdeb4763c5f58cb24ebba7d84b02dd35e0232c360383eccb`;
- result SHA-256
  `bf38d7bac8cb43bc8b69967fbbbf2e7bf1893709e2d6e950d08140dd3e630629`.

## DTR-C12/C13 route-time onset closures

C11's fresh false-alert reduction cost 0.56 s of median lead relative to its
fixed C9 arm. C12 and C13 therefore kept the complete frozen C11 decision as an
independent baseline and tested whether a new route-time representation could
extend the same alert earlier. Both used the original ten consumed sequences
for fitting and the four now-consumed C11 confirmation sequences for a single
Development decision. The fixed opening gate required no lower recall, no more
false segments, and at least `+0.3 s` median lead. Probability remained `0.5`.

| Consumed four-sequence Development | CONTACT recall | False segments | Event F1 | Median lead | Lead gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen C11 | `17/20` | **11** | **70.83%** | 1.455 s | - |
| C12 conflict-endpoint first-passage innovation | `17/20` | 12 | 69.39% | 1.455 s | 0.000 s |
| C13 peak collision-probability rate | `17/20` | 12 | 69.39% | 1.530 s | +0.075 s |

C12 maps every current M1-CT route-entry cell to the predicted world conflict
endpoint and absolute hit time. Repeated occupancy in the same `0.25 m × 0.5 s`
tubelet is removed from the current route intensity, so only probability mass
newly entering the conflict set can originate the additional channel. It
extended one already-recalled event by about 0.14 s but added one NVIDIA false
segment; pooled median lead did not move.

C13 instead groups current route entries into the existing 0.5 s lifecycle
bins and calibrates the peak time-bin intensity. This tests whether a real mover
forms a temporally concentrated collision-probability rate while pseudo-motion
is diffuse. It preserved recall and extended several events, but its pooled
gain was only 0.075 s and it retained the same added NVIDIA false segment.

Decisions:

- `DTR_C12_CONFLICT_ENDPOINT_FIRST_PASSAGE_DEVELOPMENT_GATE_NOT_MET`;
- `DTR_C13_COLLISION_PROBABILITY_RATE_DEVELOPMENT_GATE_NOT_MET`.

Neither candidate is frozen and neither may enter the remaining algorithm-fresh
JRDB sequences. Do not rescue them with probability, voxel, time-bin, route, or
lifecycle sweeps. The remaining twelve unexposed sequences stay sealed. The
next admissible source upgrade is a stochastic reachability kernel whose spread
comes from point-velocity uncertainty; deterministic point estimates summarized
by another scalar are closed by C12/C13.

The mechanism reserve came from two primary-source themes. Continuous-time
first-exit risk accumulates newly entering probability mass instead of repeatedly
counting occupancy already in collision ([Collision Probabilities for
Continuous-Time Systems](https://arxiv.org/html/2006.01109)). Collision risk is
more naturally represented as a probability rate over time than as a separate
TTC prerequisite ([What Is the Collision Probability?](https://arxiv.org/abs/1711.07060v3)).
For the successor, [SCOPE](https://arxiv.org/html/2407.00144) provides the useful
component hypothesis that ego motion, dynamic occupancy, static structure, and
uncertainty should produce a distribution of future states, while
[Safety-Oriented Pedestrian Occupancy Forecasting](https://arxiv.org/html/2101.02385)
supports retaining dense instance-free occupancy when detector post-processing
drops partially occluded pedestrians. None of those benchmarks is BlindAssist
performance authority.

All three GPU-capable workload classes emitted shared-runtime receipts and
verified real RTX 5060 CUDA execution. CPU was selected with
`CPU_FASTER_MEASURED`: representative route collision was roughly
`0.13-0.25 ms` on CPU versus `0.76-1.02 ms` on CUDA, fitting was roughly
`1.7-2.3 ms` versus `16-18 ms`, and probability inference was roughly
`0.13-0.18 ms` versus `0.33-0.41 ms`.

Evidence:

- C12 result SHA-256
  `bac5f1624e235b7613d338ec31821ff7d6e6f700ba7b99e8eea1c0fbb5a1b3db`;
- C13 result SHA-256
  `ba21410ace87e15e8b687dbbef42bad32274c66c666ffec03d55969ccfbb9add`.

The source split explains why pooling is not enough:

| Source | R2 recall / false | R3-C recall / false | Route authority |
| --- | ---: | ---: | --- |
| THÖR-MAGNI | `10/10` / 42 | `10/10` / 98 | exact ego XY; causal path-tangent yaw proxy |
| JRDB test | `164/175` / 256 | `162/175` / 214 | no synchronized ego pose; straight-relative diagnostic only |
| CODa 16+18+20 | `119/122` / 285 | `121/122` / 325 | source-native pose-authoritative route |

On only the two curve-authoritative sources, THÖR plus CODa, R3-C changes
`129/132` and 327 false segments into `131/132` and 423. Two recovered events
cost 96 extra false segments. Curvature has a real local effect but not a useful
trade-off.

## Event-metric audit

The original frozen gate is retained exactly for comparability, but the replay
now also reports one-to-one ONSET matching, event precision/F1, target-track
exposure-normalized false rate, lead, fragmentation, CLEAR, coverage, and
per-source AUPRC.

| Pooled corrected metric | R2 | R3-C |
| --- | ---: | ---: |
| ONSET event precision | 24.17% | 22.86% |
| ONSET event recall | 80.13% | 80.78% |
| ONSET event F1 | **37.13%** | 35.63% |
| False segments / known-negative target-track minute | **0.461** | 0.504 |
| Fragmented matched-event rate | **10.57%** | 14.11% |
| CLEAR among eligible matched events | **98.46% (`128/130`)** | 98.44% (`126/128`) |

R3-C's frame AUPRC is 0.309 on THÖR, 0.593 on JRDB, and 0.413 on CODa
(source macro 0.439). R2 has no compatible continuous score, so those values
are descriptive rather than an R2-vs-R3 ranking. Evaluator admission coverage
is 9.54% for THÖR, 91.61% for JRDB, and 61.42% for CODa; the different
denominators are exposed rather than silently pooled. False alerts per actual
user wall-clock minute remain `NOT_EVALUABLE` because independent target-track
streams are not merged into a single wearer timeline.

## Natural non-pedestrian availability

A metadata-first scan of CODa sequences 0..20 found two natural, source-native
non-pedestrian positives without downloading RGB or LiDAR:

- sequence 6, `Delivery Truck:2`: one lateral-crossing vehicle event;
- sequence 17, `Scooter:1`: one oncoming micromobility event, right-censored
  after contact.

R2 and R3-C both recall `1/1` in each group. This establishes that the two event
types exist and that the replay path handles them; one event per class is not a
class-generalization claim and does not enter the frozen R3 gate. The scan also
found 42,151 native Bike boxes but zero countable positive Bike events, so Bike
positive recall remains `NOT_EVALUABLE`.

## Static and continuous-collision ceiling

The static CODa ceiling keeps source-native 3-D boxes fixed in the world, uses
current-and-past ego pose for a causal constant-turn route, intersects that
route with oriented footprints, and admits only lower-body or bounded
head-clearance height bands.

Across CODa 16, 18, and 20 it evaluates 182,274 known frames and 12 future
path-contact events: six barrier/boundary, five fixed-structure, and one
temporary-obstacle event.

| Static arm | Event recall | False segments | CLEAR | Median escalation lead |
| --- | ---: | ---: | ---: | ---: |
| P0 current 3 m proximity | `12/12` | 104 | `2/3` | n/a |
| S1 straight route | `11/12` | 22 | `1/1` | n/a |
| S2 straight route + height | `11/12` | 22 | `1/1` | `0.80 s` |
| **S3 sampled curved route + height** | **`12/12`** | **10** | **`3/3`** | **`1.40 s`** |
| S4 continuous curved-route collision | `12/12` | 10 | `3/3` | `1.40 s` |

S3 remains the public-real champion: versus proximity it preserves all events
and removes `94/104` false segments (90.4%). S4 replaces point-only collision
checks with analytic time-of-impact on every 0.1-second curved-route chord. A
four-case controlled canary deliberately places thin crossing, grazing,
turn-entry, and fast transverse contacts between sample points: S3-style point
checks recall `0/4`, while S4 and a 24,000-step-per-chord dense oracle recall
`4/4` with agreeing entry times.

On the public CODa replay, S4 is exactly tied with S3: `12/12`, 10 false
segments, identical CLEAR and lead. Continuous geometry is therefore validated
and retained as an implementation option, but it is not promoted as a new
public-real algorithm win. The stress chords are geometry falsifiers, not a
natural wearer-speed distribution; CODa truth is frame-sampled, so only the
controlled canary has authority over between-frame contacts.

The selected sequences still contain no positive vegetation or head-clearance
path event. Vegetation contributes 33,954 evaluated frames and zero false
alerts, which is exposure without positive recall authority. Positive hanging
branch, thin-branch, drop-off, narrow-gap, and head-clearance performance remain
`NOT_EVALUABLE`.

## Frozen mechanics

R1 forms every causal pairwise velocity allowed by a 1.5-second target history,
uses the component-wise Theil-Sen median as target motion, and analytically
computes first entry into the 3-second, 0.65 m half-width straight route tube.
R2 leaves R1 unchanged and admits R0's least-squares route intersection only
when entry is inside the existing 1.5-second escalation half-horizon.

R3 freezes 0.5 seconds of ego history, a minimum 0.2-second ego span, and
0.1-second route chords. A uses a curved CTRV wearer route and robust target CV;
B uses a straight route and all causal pairwise target-velocity hypotheses; C
uses the curved route, the same distributional hypotheses, and R2's imminent
guard. Distributional entry requires a strict majority (`support > 0.5`); a tie
is false. A/B/C are coupled performance alternatives, not factorial
single-component causal ablations.

Dynamic R2's straight route entry is already analytic continuous time, and R3
computes continuous time-of-impact inside each route chord. DR55 therefore
identified a real point-sampling gap only in static S3. S4 closes that gap by
solving segment entry against the oriented obstacle rectangle Minkowski-expanded
by the 0.65 m route radius. The remaining approximation is the CTRV curve by
0.1-second chords, not endpoint-only collision inside a chord.

Known positive frames drive `ONSET`, sustained positives drive `HOLD`, and the
first entry inside half-horizon emits `ESCALATE` once. A known negative must
persist for 0.5 seconds before `CLEAR`; missing pose, track, or motion history
is `UNKNOWN` and cannot manufacture a clear.

Frozen fingerprints:

- R1: `741b815017297f64cb80f3f9d44282eb7fd16f79f60b04fe4f25ae8a9026f4b8`
- R2: `4142a575911e9d43508e996b0e0cf5062dc5c86d755dfe63d41279caf56302a8`
- R3: `666a29533eff450bf37cfe1c15bc4ec34bea67b27ab1a2279529e78bb588368f`

## Next information-bearing route

The next credible dynamic successor is not another threshold or trajectory
model. It needs a detector-independent residual-occupancy source so a target
that disappears through detector/NMS/tracker failure is not interpreted as
free space. Track covariance, observation availability, time-since-seen, and
bounded gap imputation must remain explicit; an imputed point must not be
reported as an observation. The DR41/DR42/DR45 RGB canary establishes dense
visual presence during induced dropout but not metric route-risk occupancy.
The next source must change that information layer rather than tune the failed
fixed-height projection or the frozen matcher.

Stochastic learned occupancy remains closed by the failed R3 gate. Whole-body
or head-clearance geometry remains the separate static S3/S4 line and requires
new positive route-grounded source truth before an ellipsoid/ESDF model could
make an evidence-backed claim.

### Residual-source admission

The 2026-08-28 source canary first checked RoboSense rather than inferring
accessibility from its paper. Its official validation metadata is usable:
12,034 frames across 702 sequences contain 88,400 3-D boxes, 13,234
sequence-scoped tracks, 15,839 boxes within a 5 m 3-D radius, synchronized pose,
and raw-sensor paths. It is not an immediate raw residual source through the
published distribution, however. The train/validation LiDAR+occupancy payload
is one combined gzip tar split into 23 pieces totalling 239,392,481,862 bytes;
no independently extractable exact-log shard is exposed. The terminal is
`ROBOSENSE_METADATA_ADMITTED_RAW_RESIDUAL_NOT_ADMITTED`.

Changing the source to Argoverse 2 Sensor passed the same accessibility
question. The first lexicographic public validation log was selected before
outcome access. Thirty-two consecutive LiDAR sweeps cover 3.100 seconds and
3,005,181 raw points; all ego poses align at zero timestamp delta. The same
window contains 854 evaluator-only native boxes across vehicle, pedestrian,
bicycle/bicyclist, and wheelchair classes, with 32 unique tracks. Its simple
straight 12 m source-admission tube contains `0` candidate native boxes, so this
window is not an event-evaluation cohort. The emitted 32-row adapter keeps
current LiDAR and pose under `causal_input` and native boxes under
`evaluator_truth`; future frames never enter causal input. This establishes
`AV2_RAW_LOG_SOURCE_ADMITTED` at exact-log granularity, not a DTR improvement.

The next Development step, if opened, is to freeze a multi-log AV2 roster and a
current/past-only residual-occupancy representation while retaining R2 as the
untuned comparator. AV2 remains an automotive retrospective ceiling; it does
not fill wearer, head-clearance, drop-off, product, or safety evidence.

Evidence:

- RoboSense source result:
  `artifacts.local/evidence/dtr-robosense-source-canary/result.json`, SHA-256
  `7d30f41736fd19501ca70f24c4c071f52e84b317a6af023e44cc63b6d08c4da8`.
- AV2 source result:
  `artifacts.local/evidence/dtr-av2-source-canary/result.json`, SHA-256
  `4ddebb8d368b25e29f752d0cc9cd27045d30ef39b455afc6a54a705a84a515d1`.
- Truth-separated AV2 adapter:
  `artifacts.local/evidence/dtr-av2-source-canary/causal-frame-source.jsonl`,
  SHA-256
  `1e70d887d4cd527e5bacf881e42de7d2d62f73de18fdef95c6b15a8e98540d64`.
- R5 RGB dropout result:
  `artifacts.local/evidence/dtr-r5/dropout-canary/result.json`, SHA-256
  `c53981d2ba76a9d696b713169143c44f9dbfc7e04515e5af148c592eb9cc617f`.
- R5 dropout curve:
  `artifacts.local/evidence/dtr-r5/dropout-canary/dropout_curve.png`, SHA-256
  `9b50969acbf78680b61af15d7b2362bf66a59a92f713f305210be64d19745cb2`.
- R6 direct metric result:
  `artifacts.local/evidence/dtr-r6/metric-occupancy-canary/result.json`, SHA-256
  `cced0f312f32059a9894cc177a62d80841bb70366a81db86f5cd900466f9e879`.
- R6 truth-blind metric point ledger:
  `artifacts.local/evidence/dtr-r6/metric-occupancy-canary/result.metric-points.npz`,
  SHA-256
  `404a12631ba60c317d67f7223d20313a2d358a8fbb1d0d4a27f6df38d321ff7b`.
- R6 dropout curve:
  `artifacts.local/evidence/dtr-r6/metric-occupancy-canary/dropout_curve.png`,
  SHA-256
  `dc215850cebf24e4b2bfc07273a47a72bc22f666ea9e8ac36c2947877d9aad9b`.
- R7-P causal occupancy-flow result:
  `artifacts.local/evidence/dtr-r7/occupancy-flow-canary/result.json`, SHA-256
  `019eb7a6c47670c821942fe6a72401899994a9e7bf7115afa9af5eacb8b3b6de`.
- R7-P truth-blind occupancy-flow ledger:
  `artifacts.local/evidence/dtr-r7/occupancy-flow-canary/result.occupancy-flow.npz`,
  SHA-256
  `7ee7302a15393fed44c07b438c1377dea54bc02d6156d07b66f871c68cd6491d`.

## Runtime bridge

The algorithm remains source-independent at the `CausalFrame` / metric-box
boundary. Android USTRF has adapters for metric-depth lower/head occupancy and
for privileged native 3-D boxes. The earlier fixed-height JRDB RGB bridge is
only a deployability hint: on one curated 143-frame window, route intersection
kept `3/3` recall and reduced false alerts from 17 to 9. It is not a
generalization result, and no phone recording is required for this ceiling.

## Literature basis

The dated [DR41-DR60 literature reserve](LITERATURE_RESERVE_2026-08-27.md)
collects the missing-track, stochastic occupancy, continuous collision,
uncertainty, whole-body, and event-evaluation mechanisms used to choose these
falsifiers. It is a mechanism reserve, not transferred BlindAssist evidence.

The robust slope is grounded in [Sen's Theil-Sen estimator](https://doi.org/10.1080/01621459.1968.10480934),
and route-tube entry is the finite-horizon single-command case of
[Velocity Obstacles](https://doi.org/10.1177/027836499801700706). A calibrated
collision probability would require explicit uncertainty and separate
validation, such as [continuous collision probability](https://arxiv.org/abs/2104.01659)
or a [Dynamic Lambda-Field](https://arxiv.org/abs/2103.04795).

## Evidence receipts

- R3 cross-source decision:
  `artifacts.local/evidence/dtr-r3/summary/result.json`, SHA-256
  `5a892b02e0b3ad836965fe1c4d49b0abf6224fddcd452247f874a1ccade28735`.
- R3 source inputs recorded by that summary: THÖR
  `cc9853e252629e2859d0217ef64597cf7ba9981298ecca742b73f2dbc1e62910`, JRDB
  `98df4106ff1cfca8d9e4dd9b408118f5453f274b42291eb03faabd44d6a071f2`, CODa core
  `d602daff2bf22797b1cb9a78fe57be45069b3812f85425006b97f7866510a1c4`, and CODa
  multiclass extension
  `b79c210f50f29f3e9c5a9aaeec0bfd9910e1bfa0d5be4445b852599c99f6e61b`.
- S4 controlled geometry canary:
  `artifacts.local/evidence/dtr-static-continuous/canary/result.json`, SHA-256
  `c51cc4248921895b2586c9092d1564b8dfe337559dfaf709d296d992cbc79c23`.
- S3/S4 CODa replay:
  `artifacts.local/evidence/dtr-static/coda-16-18-20-continuous/result.json`,
  SHA-256
  `4795845150501ef3c3498f417a21e08435ffad59df3d2ff7d437ec8e6b5567a0`.

## Reproduce

From the repository root, after placing the already documented native inputs:

```powershell
python research/active/dtr-r0/thor_magni_native_ceiling.py `
  --manifest-dir <thor-manifest-dir> --include-r3 `
  --output artifacts.local/evidence/dtr-r3/thor-magni/result.json

python research/active/dtr-r0/jrdb_native_ceiling.py `
  --labels-zip <jrdb-test-labels.zip> `
  --timestamps-zip <jrdb-test-timestamps.zip> --include-r3 `
  --output artifacts.local/evidence/dtr-r3/jrdb-test/result.json

python research/active/dtr-r0/coda_native_ceiling.py `
  --sequence-root 20=<coda-20-root> `
  --sequence-root 18=<coda-18-root> `
  --sequence-root 16=<coda-16-root> --include-r3 `
  --output artifacts.local/evidence/dtr-r3/coda-core/result.json

python research/active/dtr-r0/dtr_r3_summary.py `
  --thor artifacts.local/evidence/dtr-r3/thor-magni/result.json `
  --jrdb artifacts.local/evidence/dtr-r3/jrdb-test/result.json `
  --coda-core artifacts.local/evidence/dtr-r3/coda-core/result.json `
  --coda-extension artifacts.local/evidence/dtr-r3/coda-multiclass-extension/result.json `
  --output artifacts.local/evidence/dtr-r3/summary/result.json

python research/active/dtr-r0/continuous_collision_canary.py `
  --output artifacts.local/evidence/dtr-static-continuous/canary/result.json

python research/active/dtr-r0/coda_static_ceiling.py `
  --sequence-root 20=<coda-20-root> `
  --sequence-root 18=<coda-18-root> `
  --sequence-root 16=<coda-16-root> `
  --output artifacts.local/evidence/dtr-static/coda-16-18-20-continuous/result.json

python research/active/dtr-r0/dtr_r5_dropout_canary.py `
  --known-height-result <jrdb-known-height-result.json> `
  --known-height-tracks <jrdb-known-height-sensor-tracks.jsonl> `
  --labels-zip <jrdb-train-labels.zip> `
  --timestamps-zip <jrdb-train-timestamps.zip> `
  --bag <jrdb-sequence.bag> `
  --images-dir <jrdb-stitched-window-dir> `
  --semantic-model <ade20k-semantic-model.pt> `
  --output artifacts.local/evidence/dtr-r5/dropout-canary/result.json `
  --plot artifacts.local/evidence/dtr-r5/dropout-canary/dropout_curve.png

python research/active/dtr-r0/dtr_r6_metric_occupancy_canary.py `
  --known-height-result <jrdb-known-height-result.json> `
  --known-height-tracks <jrdb-known-height-sensor-tracks.jsonl> `
  --labels-zip <jrdb-train-labels.zip> `
  --timestamps-zip <jrdb-train-timestamps.zip> `
  --bag <jrdb-sequence.bag> `
  --dense-ledger <r5-dense-masks.npz> `
  --dense-manifest <r5-dense-masks.json> `
  --calibration-dir <jrdb-calibration-dir> `
  --depth-source <depth-anything-v2-source> `
  --depth-checkpoint <metric-hypersim-vits.pth> `
  --output artifacts.local/evidence/dtr-r6/metric-occupancy-canary/result.json `
  --plot artifacts.local/evidence/dtr-r6/metric-occupancy-canary/dropout_curve.png

python research/active/dtr-r0/dtr_r7_occupancy_flow_canary.py `
  --r6-result artifacts.local/evidence/dtr-r6/metric-occupancy-canary/result.json `
  --known-height-tracks <jrdb-known-height-sensor-tracks.jsonl> `
  --labels-zip <jrdb-train-labels.zip> `
  --timestamps-zip <jrdb-train-timestamps.zip> `
  --bag <jrdb-sequence.bag> `
  --calibration-dir <jrdb-calibration-dir> `
  --output artifacts.local/evidence/dtr-r7/occupancy-flow-canary/result.json

python research/active/dtr-r0/dtr_m0_r7_error_attribution.py `
  --r7-result artifacts.local/evidence/dtr-r7/occupancy-flow-canary/result.json `
  --flow-ledger artifacts.local/evidence/dtr-r7/occupancy-flow-canary/result.occupancy-flow.npz `
  --flow-manifest artifacts.local/evidence/dtr-r7/occupancy-flow-canary/result.occupancy-flow.json `
  --output artifacts.local/evidence/dtr-m0-r7-error-attribution/result.json `
  --table artifacts.local/evidence/dtr-m0-r7-error-attribution/false-segments.csv `
  --timeline artifacts.local/evidence/dtr-m0-r7-error-attribution/timeline.svg

python research/active/dtr-r0/dtr_m1_point_velocity_oracle.py `
  --r7-result artifacts.local/evidence/dtr-r7/occupancy-flow-canary/result.json `
  --m0-result artifacts.local/evidence/dtr-m0-r7-error-attribution/result.json `
  --known-height-tracks artifacts.local/evidence/dtr-r0/jrdb-known-height-bridge-v1/result.sensor-tracks.jsonl `
  --labels-zip artifacts.local/datasets/dtr-r0-jrdb-rgb-bridge-v1/train_labels.zip `
  --timestamps-zip artifacts.local/datasets/dtr-r0-jrdb-rgb-bridge-v1/train_timestamps.zip `
  --bag artifacts.local/datasets/dtr-r0-jrdb-rgb-bridge-v1/packard-poster-session-2019-03-20_1.bag `
  --calibration-dir artifacts.local/datasets/ustrf-canonical-observation-source-authority-data-pack-r0/jrdb_toolkit/calibration `
  --output artifacts.local/evidence/dtr-m1/point-velocity-oracle/result.json

python research/active/dtr-r0/dtr_m2_extent_gap_audit.py `
  --m1-result artifacts.local/evidence/dtr-m1/point-velocity-oracle/result.json `
  --output artifacts.local/evidence/dtr-m2/extent-gap-audit/result.json

python research/active/dtr-r0/dtr_m3_realized_future_contract_decomposition.py `
  --m2-result artifacts.local/evidence/dtr-m2/extent-gap-audit/result.json `
  --output artifacts.local/evidence/dtr-m3/realized-future-contract-decomposition/result.json

python research/active/dtr-r0/dtr_c0_global_oriented_risk_contract.py `
  --m3-result artifacts.local/evidence/dtr-m3/realized-future-contract-decomposition/result.json `
  --output artifacts.local/evidence/dtr-c0/global-oriented-risk-contract/result.json

python research/active/dtr-r0/dtr_c1_global_obb_cohort_admission.py `
  --labels artifacts.local/datasets/dtr-r0-jrdb-rgb-bridge-v1/train_labels.zip `
  --timestamps artifacts.local/datasets/dtr-r0-jrdb-rgb-bridge-v1/train_timestamps.zip `
  --output artifacts.local/evidence/dtr-c1/global-obb-cohort-admission/result.json `
  --roster research/active/dtr-r0/dtr_c1_fresh_global_obb_roster.json
```

## DTR-C14 confidence-covariant stochastic route conflict

C14 tested the next representation change on the already consumed C11
Development cohort; it did not change a route threshold.  It replayed the
truth-blind R7-to-M1 temporal match and recovered each admitted cell's measured
forward-advection residual, velocity disagreement, history span, and support.
Those observations define block-diagonal position/velocity covariance.  A
fixed third-degree cubature rule then advances eight equally weighted 4-D
state points through the unchanged continuous route-collision geometry.

The design follows three external mechanisms rather than importing a product
claim: [forward stochastic reachability defines probability-weighted occupied
sets](https://ar5iv.labs.arxiv.org/html/1803.07180),
[RigidFlow++ uses spatial and forward/backward consistency to reject unreliable
scene-flow correspondences](https://ar5iv.labs.arxiv.org/html/2310.11284), and
[DifFlow3D treats per-point scene-flow uncertainty as a reliability
quantity](https://arxiv.org/html/2311.17456v4).  Dynamic-BKI additionally
supports propagating scene occupancy with flow rather than treating motion as
an object-track-only attribute: <https://ar5iv.labs.arxiv.org/html/2108.03180>.

The fixed Development comparison was:

| arm | recall | false segments | event F1 | median lead |
| --- | ---: | ---: | ---: | ---: |
| frozen C11 `M1_RROQ_GLOBAL` | `17/20` | 11 | 70.83% | 1.455 s |
| C14 `M1_SRC_GLOBAL` | `18/20` | 16 | 66.67% | 1.504 s |

C14 recovered one additional bounded CONTACT event, but added five false
segments and gained only `0.049 s` median lead.  It therefore failed the frozen
gate (recall not lower, false segments not higher, median lead gain at least
`0.3 s`) and no algorithm-fresh sequence was opened.  Result SHA-256:
`7616044a39a90df846073762d8ba1662985176ab7f3066094889f29cb112fc42`.

This closes symmetric covariance spreading of every admitted cell as a direct
ONSET replacement.  A successor must add directional or multimodal evidence
that distinguishes plausible mover futures from uncertainty mass; it must not
rescue C14 by scanning the probability threshold, covariance scale, cubature
weights, or route geometry.

All C14 point matching, cubature, collision, Platt fitting, and probability
inference used the shared research launcher.  Each representative CPU/GPU
short test observed a real CUDA tensor on the RTX 5060; CPU was selected with
the recorded reason `CPU_FASTER_MEASURED` for these small workloads.

## DTR-C15--C18 confidence-aware scene-motion closures

C15--C18 kept the C11 route geometry, probability decision at `0.5`,
maintenance model, and `0.5 s` lifecycle fixed.  They changed only the motion
representation on the already consumed four-sequence C11 Development cohort:

- C15 fit equal-prior static-versus-rigid velocity hypotheses inside each
  frame-local occupancy component.  Mean mover posteriors were `0.979--0.990`,
  showing that the component partition was not a useful mover identity.
- C16 retained the current and causally matched historical signed velocities
  as two empirical modes.  This removed C14's invented symmetric directions
  and recovered `0.272 s` lead, but either mode could still create risk mass.
- C17 required both empirical modes to agree on discounted route entry.  It
  strongly reduced false alerts but delayed useful early evidence.
- C18 retained C16's mean collision mass and replaced its two-frame confidence
  with a three-frame causal chain: the minimum of both frozen pair confidences
  and frozen-scale velocity-delta consistency.  A missing second-level match is
  `UNKNOWN` and contributes zero onset mass.

The fixed Development comparison was:

| arm | recall | false segments | event F1 | median lead | lead gain vs C11 |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen C11 `M1_RROQ_GLOBAL` | `17/20` | 11 | 70.83% | 1.455 s | -- |
| C15 `M1_CVM_GLOBAL` | `17/20` | 12 | 69.39% | 1.455 s | 0.000 s |
| C16 `M1_EVM_GLOBAL` | `17/20` | 13 | 68.00% | 1.726 s | +0.272 s |
| C17 `M1_TRC_GLOBAL` | `17/20` | 8 | 75.56% | 1.062 s | -0.392 s |
| C18 `M1_TFMC_GLOBAL` | `17/20` | 9 | 73.91% | 1.079 s | -0.376 s |

All four retained C11 recall.  C16 established that signed historical motion
contains earlier route-entry information; C17/C18 established that temporal
consistency can suppress pseudo-motion below C11's false-alert count.  None
simultaneously met the frozen gate of recall not lower, false segments not
higher, and at least `0.3 s` median lead gain, so no algorithm-fresh sequence
was opened.  Result SHA-256 values are:

- C15: `e2950f3e230323a966355108ad96f495e5d7b852caf93daf94b823a633bc6e98`;
- C16: `c9df4ffd5aa98983d6ece7971c8173db9dd170da39740acc2ad247bc9b95da39`;
- C17: `13ec91eead39222b01616d1145d5e9e2d32b155aa735d97d1f19418bfd901727`;
- C18: `89c561b7d2fcd601e7e1a77ba4c5061028de87f9c9bdc4039b165e2cf66a48c7`.

C19 therefore froze C16's early route-conflict mass and C18's three-frame
confidence as separate channels and learned one training-only joint
calibration.  This tested whether early motion and trustworthy motion were
complementary without changing the downstream decision contract.

This route follows dynamic occupancy work that preserves a velocity
distribution separately from occupancy confidence
([Nuss et al.](https://ar5iv.labs.arxiv.org/html/1605.02406)) and multi-frame
scene-flow work that uses preceding-frame or delta cues for temporal
consistency ([M-FUSE](https://openaccess.thecvf.com/content/WACV2023/papers/Mehl_M-FUSE_Multi-Frame_Fusion_for_Scene_Flow_Estimation_WACV_2023_paper.pdf),
[DeltaFlow](https://proceedings.neurips.cc/paper_files/paper/2025/file/80613b043d43ffaae9824ee9d4b291e5-Paper-Conference.pdf)).
These papers motivate components only; they do not establish BlindAssist
performance.

All C15--C18 point matching, tensor transforms, collision geometry, calibration,
and inference ran through the shared research launcher.  Every representative
CPU/GPU receipt observed a real CUDA tensor on the RTX 5060.  CPU was selected
with `CPU_FASTER_MEASURED` because these batches took microseconds to a few
milliseconds on NumPy and CUDA launch/transfer overhead dominated.

## DTR-C19--C21 coherent pseudo-motion closures

C19--C21 kept the same consumed four-sequence Development cohort and the C11
route, probability, maintenance, and lifecycle contract fixed.  They tested
three mechanisms without opening the remaining algorithm-fresh sequences:

- C19 fit one fixed L2 logistic calibration over the standardized C16 early
  score and C18 three-frame confidence.  Both learned coefficients were
  positive (`0.445112`, `0.172793`), so confidence did not learn a veto; the
  fusion instead averaged away early evidence.
- C20 multiplied pair confidence by a threshold-free Gaussian neighborhood
  vote over position and velocity agreement.  Nearly every cell received local
  support (for example `44,509/44,562` on `clark-center-2019-02-28_0`), and its
  event metrics were identical to C16.  The false motion is locally coherent,
  not a sparse outlier population.
- C21 subtracted the coordinate-wise median scene velocity from both signed
  empirical modes.  It recovered one additional event but produced 18 false
  segments, showing that a global median is not a valid static-background
  estimator in crowded or co-moving scenes.

| arm | recall | false segments | event F1 | median lead | lead gain vs C11 |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen C11 `M1_RROQ_GLOBAL` | `17/20` | 11 | 70.83% | 1.455 s | -- |
| C19 joint calibration | `17/20` | 12 | 69.39% | 1.268 s | -0.187 s |
| C20 local motion vote | `17/20` | 13 | 68.00% | 1.726 s | +0.272 s |
| C21 scene-bias residual | `18/20` | 18 | 64.29% | 1.396 s | -0.059 s |

None met the frozen gate, so the fresh cohort remains sealed.  Result SHA-256
values are:

- C19: `fe49132e642b6925bb802bbceb7ed2f05378627d5b15151b33ccc4fdbd752c5a`;
- C20: `897d56ad5f8de62dd3b743f349835fbcaf646104863013c7bde12e84a6ca6858`;
- C21: `ea29f38ec1f1bde2713c1e2f042eabbf775ceddc5d89c6ab615cec0bfaf2488c`.

This closes downstream fusion, local rigidity voting, and single-vector scene
bias subtraction on the current LiDAR pseudo-flow.  It authorized C22 to
compare short RGB point tracks with ego-induced rigid image trajectories, treat
their residual as independent-motion confidence, and lift that cue through
existing calibration into LiDAR cells before route risk.  This
follows the independent-motion residual used by
[SLIM](https://openaccess.thecvf.com/content/ICCV2021/papers/Baur_SLIM_Self-Supervised_LiDAR_Scene_Flow_and_Motion_Segmentation_ICCV_2021_paper.pdf)
and the local rigidity evidence in
[VoteFlow](https://openaccess.thecvf.com/content/CVPR2025/papers/Lin_VoteFlow_Enforcing_Local_Rigidity_in_Self-Supervised_Scene_Flow_CVPR_2025_paper.pdf),
but changes the information source because C20 shows that local rigidity alone
cannot separate BlindAssist's coherent pseudo-motion.  These papers motivate
the component only; they do not establish BlindAssist performance.

All C19--C21 work used the shared research launcher after the DTR doctor passed.
Representative receipts verified real RTX 5060 CUDA execution.  C19 probability
inference selected GPU because its measured time was equal/slightly faster
(`0.4580 ms` versus `0.4586 ms` CPU).  The other small kernels selected CPU with
`CPU_FASTER_MEASURED`; measured launch/transfer overhead made GPU roughly
`5--40x` slower, rather than silently falling back from a declared GPU run.

## DTR-C22 ego-rigid visual-motion confidence

C22 changed the information source on the already consumed R7 143-frame,
three-event, nine-dropout canary.  Each R7 cell proposed its prior world
position from signed velocity.  Five fixed vertical anchors were projected
into the cylindrical RGB panorama; sparse forward/backward PyrLK tracks were
compared with both the ego-rigid static reprojection and the R7-proposed moving
reprojection.  Agreement, forward/backward consistency, and valid-track
coverage formed one component confidence before the unchanged R7 route test.
Missing or failed tracks contributed low confidence, not a static/safe label.

The source change produced the desired false-alert effect but failed the
dropout requirement:

| arm | critical recall | target false segments | event F1 | induced dropout recovery |
| --- | ---: | ---: | ---: | ---: |
| R7 occupancy flow | `3/3` | 20 | 22.22% | `9/9` |
| C22 visual-validated flow | `3/3` | 12 | 22.22% | `0/9` |

C22 removed exactly the eight R7-added false segments, reducing attributed
flow cells from `24,141` to `3,771`.  However, only 9 target-attributed flow-risk
frames remained, event matching covered `2/3`, and none overlapped the fixed
dropout windows.  Therefore accept
`DTR_C22_EGO_RIGID_VISUAL_MOTION_DEVELOPMENT_GATE_NOT_MET`: the independent
visual residual is a strong pseudo-motion suppressor, but this coarse stitched
cell projection is too sparse and time-misaligned to retain detector-independent
recovery.  Result SHA-256 is
`16ada5327f089a0c740f6b639e43b914074c0f1fdef7acbc8fb80bd91ec38ad8`.

Do not rescue C22 with a confidence threshold or temporal grace sweep.  C23
must improve the observation correspondence itself: track raw LiDAR-supported
points independently in the five undistorted perspective cameras, compensate
LiDAR-to-image time with full ego SE(3), subtract exact per-point rigid
reprojection, and only then aggregate residual confidence to the existing BEV
cell.  The four consumed C11 bags already contain all five RGB streams, both
LiDARs, TF, timestamps, and official intrinsics/extrinsics; no new download or
depth model is required.

C22 ran through the shared launcher after the DTR doctor passed.  The launcher
verified the RTX 5060 CUDA runtime, while the OpenCV 4.10 wheel reported no CUDA
device and no SparsePyrLK CUDA provider.  The representative 512-track probe
therefore selected `opencv-cpu-sparse-pyrlk` in `0.0510 s` with the explicit
receipt reason `GPU_BACKEND_UNAVAILABLE`; this was a provider limitation, not
a silent CUDA fallback.

## DTR-C23--C24 confidence-aware point scene motion

C23 first tested the object-level hypothesis on the same consumed R7 Packard
canary.  It associated R7 frame-local components, fit one rigid SE(2) transform
per candidate, and combined rigidity residual, inlier support, size agreement,
association margin, R7/rigid velocity agreement, and up to five causal velocity
observations into a motion confidence.  It preserved critical recall `3/3` and
all `9/9` induced-dropout recoveries, improved event F1 from `22.22%` to
`26.09%`, and reduced target false segments from `20` to `16`, but missed the
fixed `<=14` target.  Result SHA-256 is
`56a281c805e51702194f993abc15dd81588d5f2e5bf93e37c003f63bbb456d36`.

The failure was structural, not a confidence-threshold problem.  The seven C23
factors had event-versus-nuisance AUC only `0.474--0.517`; raising confidence
would preferentially remove dropout evidence.  Segment attribution showed that
C23 removed four R7-added false segments and added none, while the four
remaining R7 additions came from one component translation being broadcast to
all cells.  One component triggered two targets, and another target inherited
motion from only one edge cell.  Replacing that translation with the rigid
centroid velocity also left 16 false segments and lost one of three events.

C24 therefore changed the representation to point-local motion.  It reopened
the raw upper/lower LiDAR causally, applied ego compensation and 3-D voxel
reciprocal-nearest correspondence (`M1-PD`), then required an independent prior
flow observation to advect to the current cell with consistent velocity
(`M1-PDC`).  Route geometry, speed range, event lifecycle, and all scoring
thresholds stayed unchanged.

| arm | critical recall | target false segments | event F1 | median first lead | induced dropout recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| R7 component translation | `3/3` | 20 | 22.22% | 2.740 s | `9/9` |
| M1-PD cell-local direct velocity | `3/3` | 13 | 30.00% | 2.463 s | `9/9` |
| **C24 M1-PDC three-frame consistency** | **`3/3`** | **12** | **31.58%** | **2.381 s** | **`9/9`** |

C24 meets the requested M1 Development gate: it keeps R7's `9/9` recovery and
removes all eight R7-added target false segments, returning to the R2 false
count while increasing event F1 by `9.36` points.  Attributed cells fall from
`24,141` to `1,684`; the explicit trade-off is median first-alert lead
`-0.359 s`, while median escalation lead remains identical at `1.438 s`.
Result SHA-256 is
`2cd71c4d8934a38956c9d5d2c2d10f2c0f36cf7eed4e9924889574cc230c838e`.
The algorithm-fresh cohort remains sealed; this authorizes one fresh
confirmation of the fixed C24 representation, not more Packard tuning.

The mechanism follows the static/dynamic and correct-association decomposition
in [SeFlow](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/143_ECCV_2024_paper.php),
the associate-then-fit rigid-object order in
[ICP-Flow](https://openaccess.thecvf.com/content/CVPR2024/html/Lin_ICP-Flow_LiDAR_Scene_Flow_Estimation_with_ICP_CVPR_2024_paper.html),
the local rigidity principle in
[VoteFlow](https://openaccess.thecvf.com/content/CVPR2025/html/Lin_VoteFlow_Enforcing_Local_Rigidity_in_Self-Supervised_Scene_Flow_CVPR_2025_paper.html),
and explicit multi-frame context in [Flow4D](https://arxiv.org/html/2407.07995v1).
These sources motivated the representation only; they are not evidence of
BlindAssist performance.

Both runs used the shared research launcher after the DTR doctor verified the
RTX 5060 CUDA runtime.  Representative C23 component ICP selected NumPy CPU at
`0.2535 ms` versus Torch CUDA `3.6737 ms`.  C24's `4x2048` raw-point match
selected SciPy cKDTree CPU at `4.5333 ms` versus observed RTX 5060 CUDA cdist
`7.8822 ms`.  Both receipts state `CPU_FASTER_MEASURED`; CPU was chosen because
it was measured faster for these irregular batches, not because CUDA silently
failed or was bypassed.

## DTR-C25 algorithm-fresh point-motion confirmation

C25 froze C24 without changing its point matching, confidence, route geometry,
motion bounds, lifecycle, or score threshold.  Before any bag or result was
opened, it selected the deterministic minimum five-sequence remainder reaching
the existing C1 admission floor: 3,358 frames, 12 bounded global-OBB CONTACT
events, 12 unique first-responsible objects, and 130.98 s known non-CONTACT
wearer time.  Five truth-blind workers sealed all route-conflict timelines
before the roster and future OBB labels were opened.  The sealed prediction
SHA-256 is
`4817321126ca291528ab48c7ff4bf23059b0a064a1617058c26b07d2bef39bd2`.

| arm | CONTACT recall | false segments | event F1 | median first lead | induced dropout recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| R7 component translation | `11/12` | 52 | 29.33% | 4.200 s | `30/36` |
| M1-PD cell-local direct velocity | `11/12` | 29 | 42.31% | 3.067 s | `33/36` |
| M1-PDC hard three-frame consistency | **`12/12`** | **21** | **53.33%** | 1.624 s | `25/36` |

The fixed C25 gate is not met.  M1-PDC improves natural recall, cuts R7 false
segments by 59.6%, and raises event F1 by 24.00 points, but its induced-dropout
recovery is lower than R7 (`25/36 < 30/36`).  Lead was reported rather than
gated and exposes a second material cost: median first alert is 2.576 s later
than R7.  The dropout loss is localized: M1-PDC recovers only `1/6` Gates and
`0/6` Packard-0 trials versus R7's `6/6` and `3/6`; it matches R7 on the other
three sequences.  M1-PD itself recovers `33/36` while still cutting false
segments by 44.2%, so the fresh result supports point-local direct velocity and
rejects using independent-history consistency as a hard evidence veto.

Accept
`DTR_C25_POINT_FLOW_ALGORITHM_FRESH_GATE_NOT_MET_HARD_TEMPORAL_VETO`.
The five sequences are now consumed: do not tune confidence, correspondence,
route, lifecycle, or thresholds against them.  The next representation change
is a route-conditioned residual future-occupancy flow over M1-PD tokens, with
temporal confidence used as a soft weight/uncertainty signal rather than as an
alert-deleting gate.  First run one realized-future-occupancy headroom falsifier
through the frozen global-OBB route/lifecycle contract; train no forecasting
head unless that oracle can materially recover lead without damaging recall or
false segments.  Result SHA-256 is
`c7045f39b774093a4ed4fd4a5c4494ea698474a1a2ab7d0e0bf50bf1fdc778a5`.

## DTR-C26 support-conditioned future-occupancy headroom

C26 ran the one consumed-cohort falsifier required before any forecasting
training.  It did not use global CONTACT truth as a free alert.  It began only
from cells already present in the sealed M1-PD ledger, associated a cell only
when exactly one current native OBB lay within the frozen 0.08485 m cell margin,
and required the same identity to have complete realized OBB support through
the full 3 s horizon.  Only those cells replaced constant velocity with their
first realized future OBB route entry.  Unsupported, ambiguous, and
right-censored cells retained M1-PD.  The combined raw signal then passed
through the unchanged urgent boundary and `RiskEventLifecycle`.

| arm | CONTACT recall | false segments | event F1 | median first lead |
| --- | ---: | ---: | ---: | ---: |
| R7 reference | `11/12` | 52 | 29.33% | **4.200 s** |
| M1-PD reference | `11/12` | 29 | 42.31% | 3.067 s |
| M1-PDC reference | **`12/12`** | **21** | **53.33%** | 1.624 s |
| supported realized-future oracle | `11/12` | 25 | 45.83% | 2.261 s |

The oracle fails all three componentwise-envelope checks.  It does not recover
the one M1-PD miss, removes seven of the original 29 false segments but produces
a net reduction of only four after lifecycle segmentation, and loses 0.806 s
median lead.  All 12 responsible objects have some unique support inside their CONTACT
window, but the missed SVL-1 event remains missed; support arriving somewhere
inside an event is not equivalent to causal support at an origin that can
anticipate it.  Across the 29 false-segment windows, unsupported/ambiguous and
right-censored evidence dominates the cell-frame ledger, so a future head
restricted to current M1-PD support cannot reliably remove the remaining risk.

Accept `DTR_C26_SUPPORTED_FUTURE_OCCUPANCY_HEADROOM_NOT_MET`.  Do not train or
sweep a residual future-occupancy model on the current M1-PD representation.
The next information change must first create an occlusion-persistent,
identity-free point support field: retain point-local velocity (no component
broadcast), propagate evidence with soft confidence/age rather than a hard
three-frame veto, and expose missingness as `UNKNOWN`.  Only after that support
representation shows recall/false/lead headroom may route-conditioned future
occupancy reopen.  Result SHA-256 is
`d555907d815aeca7abd349492ec7ec94f3b69cc8d7a040d99c5f1cbdd66da152`.

## DTR-C27 positive-support point memory

C27 tested the smallest identity-free persistence mechanism on the consumed
C25 ledgers.  M1-PDC cells originated world-coordinate lineages; reciprocal
M1-PD cells could refresh position and velocity with soft confidence, and
unmatched lineages were advected for the already exercised maximum `0.8 s`
dropout duration.  Evaluator identity and source component IDs were unavailable
to association.  The frozen M1-PDC lifecycle was mechanically unioned with the
extension lifecycle so extension `UNKNOWN` could not erase a baseline alert.

| arm | CONTACT recall | false segments | event F1 | median first lead | induced dropout recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| M1-PDC reference | `12/12` | **21** | **53.33%** | 1.624 s | `25/36` |
| C27 persistent point support | **`12/12`** | 26 | 48.00% | **2.967 s** | `27/36` |
| R7 recovery reference | `11/12` | 52 | 29.33% | 4.200 s | **`30/36`** |

The fixed gate is not met.  C27 preserves every PDC event's first-alert lead
and gains two dropout recoveries over PDC, but it adds five false segments and
remains three recoveries below R7.  The failure is structural: sealed dynamic
ledgers contain positive support, confidence, and age, but not current
`KNOWN_FREE`, occluder, or out-of-FoV evidence.  A missing cell therefore cannot
be distinguished as departed, occluded, or unsensed.

Accept `DTR_C27_PERSISTENT_POINT_SUPPORT_DEVELOPMENT_GATE_NOT_MET`.  Do not tune
age, half-life, reciprocal radius, confidence, route, or lifecycle on C25.  The
next information source is the four-state LiDAR visibility sidecar frozen in
[C28 visibility-conditioned point memory](C28_VISIBILITY_CONDITIONED_POINT_MEMORY_2026-08-28.md):
`HIT / KNOWN_FREE / OCCLUDED / UNSENSED`.  Only occlusion may retain a positive;
known-free clears it and unsensed remains `UNKNOWN`.  Result SHA-256 is
`2ea585c20029f66d64c2435faa38f997615607058734c419ea6d13f3ba78c505`.

## DTR-C28 visibility-conditioned point memory

C28 added the missing causal ray state rather than changing a route threshold.
Each remembered point retained its raw endpoint height voxels, and current
upper/lower LiDAR rays classified those 3-D supports as `HIT`, `KNOWN_FREE`,
`OCCLUDED`, or `UNSENSED`.  `KNOWN_FREE` cleared the lineage, `UNSENSED` stayed
unknown, and only `OCCLUDED` could emit bounded old-velocity persistence.  A
current occupancy `HIT` without motion-consistent PD evidence did not inherit
the old velocity.

| arm | CONTACT recall | false segments | event F1 | median first lead | induced dropout recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| M1-PDC reference | `12/12` | **21** | **53.33%** | 1.624 s | `25/36` |
| C28 visibility-conditioned memory | **`12/12`** | 46 | 34.29% | **2.532 s** | `26/36` |
| R7 recovery reference | `11/12` | 52 | 29.33% | 4.200 s | **`30/36`** |

The route is evaluable: `66,732 / 68,240` absent-lineage observations received
a causal ray state.  Nevertheless, visibility did not validate motion identity.
An `OCCLUDED` cell explains why a support is hidden but does not prove its old
velocity remains correct; repeated motion hits can also refresh static
pseudo-motion.  C28 therefore preserves `12/12` and earlier lead but adds 25
false segments for only one dropout recovery.

Accept
`DTR_C28_VISIBILITY_CONDITIONED_POINT_MEMORY_DEVELOPMENT_GATE_NOT_MET`.  Do not
tune C28 on the consumed C25 cohort.  Retain the four-state sidecar as an input,
not as alert authority.  The successor must search the structure that grants
dense motion authority only under observable uncertainty and current temporal
motion consistency.  Result SHA-256 is
`ea3a485cb0f94dfe7eda320e57e1107052f50e5446082ed0646355fc2695bd1f`.

## DTR-C30 confidence + temporal-consistency authority

C30 expanded the truth-blind authority trace with all current M1-PD reciprocal
residual cells, then used a frozen 16-iteration SkyDiscover AdaEvolve search to
discover point-wise motion authority.  The retained rule requires supported
direct flow, displacement/velocity temporal consistency, and compatible local
velocity peers or an observed lineage.  Occupancy visibility alone never grants
a motion vector.

| arm | CONTACT recall | false segments | event F1 | median first lead | induced dropout recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| M1-PDC baseline | `12/12` | 21 | 53.33% | 1.624 s | `25/36` |
| C30 retained consensus authority | **`12/12`** | **20** | **54.55%** | **1.734 s** | `29/36` |
| C30 best recovery Pareto arm | `12/12` | 29 | 45.28% | 2.869 s | **`33/36`** |

The retained candidate preserves every event with no later alert, removes one
false segment, improves F1 and lead, and recovers four additional dropout
samples.  It nevertheless misses the frozen `>=30/36` gate by one sample.
Accept
`DTR_C30_CONFIDENCE_TEMPORAL_CONSENSUS_DEVELOPMENT_GATE_NOT_MET`; retain this
candidate as the representation baseline, but do not tune its constants on
C25.  The next falsifier is a causal short-window component state that can add
the one missing recovery without adding a false segment.

## DTR-C31 temporal velocity-component authority

C31 corrected the remaining representation defect: C30's raw residual
`dp_m/dv_mps` fields were zero-filled, so local consensus was not genuine
cross-frame consistency.  All supported reciprocal raw rows now vote into local
velocity components.  A component becomes hard authority only after its
observed center is better explained by `c + v * dt` than by staying at `c`.
During a short observed occlusion, its last real support footprint is transported
with decay; `KNOWN_FREE` revokes it and ordinary `HIT/UNSENSED` never refresh it.

| arm | CONTACT recall | false segments | event F1 | median first lead | induced dropout recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| M1-PDC | `12/12` | 21 | 53.33% | 1.624 s | `25/36` |
| C30 local consensus | `12/12` | **20** | **54.55%** | 1.734 s | `29/36` |
| broad C31 soft-to-hard component | `12/12` | 28 | 46.15% | **2.732 s** | **`33/36`** |
| **C31 signed transport authority** | **`12/12`** | **21** | **53.33%** | **2.667 s** | **`30/36`** |

Accept `DTR_C31_TEMPORAL_COMPONENT_AUTHORITY_DEVELOPMENT_GATE_MET`.  Relative
to PDC it adds five dropout recoveries and `1.043 s` median lead without adding
false segments or losing/delaying an event.  Relative to broad component birth,
the static-versus-transported residual removes seven false segments.  This
consumed-cohort gate authorizes source-disjoint confirmation of the frozen C31
mechanism, not further C25 tuning.  Result SHA-256 is
`1787d88a13c5dcc689dc28ce8a4f46c2d7ae6b0c3114ffab2e05d6c5acfe1e8d`.

## DTR-C31 source-disjoint confirmation

The frozen mechanism did not transfer. A truth-blind source preflight retained
six of the seven remaining algorithm-unexposed JRDB sequences; Gates-to-Clark
was `NOT_EVALUABLE` because frame 0 preceded every causal native pose. The
final cohort contains 4,811 frames, six bounded CONTACT events, 18 induced
dropout trials, and 278.08 seconds of known non-CONTACT exposure. All C30/C31
predictions and dropout ledgers were sealed before labels were opened.

| arm | CONTACT recall | false segments | Event F1 | median first lead | dropout recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| M1-PDC | `4/6` | **25** | **22.86%** | **2.291 s** | `5/18` |
| C30 local consensus | `4/6` | 27 | 21.62% | 2.291 s | `5/18` |
| C31 signed transport | `4/6` | 35 | 17.78% | 2.291 s | `6/18` |

Accept `DTR_C31_SOURCE_DISJOINT_CONFIRMATION_GATE_NOT_MET`. C31 gained only
one dropout recovery versus the frozen required `+2`, recovered neither missed
event, added no lead, and added ten false segments. Do not tune C31 and do not
open C32 probabilistic body-route occupancy on this support. The false
inflation is concentrated in four CONTACT-bearing sequences, so a successor
must change component information/authority rather than spread the same
accepted components with uncertainty. The full failure localization and
receipts are in `C31_FRESH_CONFIRMATION_2026-08-29.md`.

## DTR-X0 motion-source attribution

X0 opened only the already consumed confirmation truth and attributed the two
missed CONTACT events, all 25 PDC false segments, and the ten C31 false ranges
that do not overlap a PDC false range. It did not alter or rescore predictions.

The two misses separate cleanly. Huang-2 has 32 responsible-OBB raw cells over
23 frames, including correct motion in six frames and three frames before the
existing 1.5 s urgent boundary, but no correct cell enters the frozen route
tube: `ROUTE_GEOMETRY_MISS`. Huang-lane has zero responsible-OBB raw cells in
the full `-3..0 s` window: `NO_MOTION_SUPPORT`.

| sealed diagnostic set | bad flow | static pseudo-motion | real mover, noncritical | total |
| --- | ---: | ---: | ---: | ---: |
| M1-PDC false segments | 14 | 10 | 1 | 25 |
| C31 non-overlapping incremental false segments | 2 | 8 | 0 | 10 |

Accept `DTR_X0_MOTION_SOURCE_ATTRIBUTION_COMPLETE`. Source errors account for
`34/35` false units, so learned motion authority is not the next experiment.
Freeze the risk scorer and compare current raw direct flow with exactly one
stronger scene-flow source. Retain continuous collision/body-route geometry as
a bounded Huang-2 canary, not as a substitute for the missing Huang-lane source
or false-flow repair. C31 tuning, C32, forecasting, and model training remain
closed. Full definitions, rows, hashes, and claim limits are in
`X0_MOTION_SOURCE_ATTRIBUTION_2026-08-29.md`.

## DTR-X1--X4 lagged scene-flow source terminal

X1c and X2 opened one full source-only replay after the two-scan-lag voxel
source recovered Huang-lane motion headroom and suppressed `25/34` frozen
source-error representatives.  On the full six-sequence opened Development
cohort, X3 recovered CONTACT `4/6 -> 6/6`, improved median lead
`2.291 -> 3.816 s`, and raised dropout recovery `5/18 -> 8/18`.  It also raised
false segments `25 -> 94` and reduced Event F1 `22.86% -> 11.32%`.

Accept `DTR_X3_FULL_LAG_FLOXEL_GATE_NOT_MET`.  The recall, lead, and dropout
checks passed, but all three selectivity checks failed.  Close X3 without
tuning; this consumed Development result does not authorize source-disjoint
confirmation.

X4 then replaced autograd flow with a deterministic, single-threaded float64
rigid-cluster vote on only the opened positive/error slices.  Three independent
cold roots had identical canonical arrays and effects.  Every run suppressed
`31/34` source-error units; across 19 evaluated positive frames, 17 had
associated cells but none had correct motion or route entry.  X4 also
missed the one-scan-period compute gate (`0.3990--0.4502 s` p95 versus
`0.06961 s`).  Accept
`DTR_X4_DETERMINISTIC_CLUSTER_VOTE_REPEATABILITY_GATE_NOT_MET`; the negative is
deterministic, X4 is closed, and no full X4 replay or parameter sweep is
authorized.  Full gates, hashes, and claim limits are in
`X1_X3_LAG_FLOXEL_SOURCE_2026-08-29.md`.

Read-only full-replay attribution then assigned `81/94` X3 false segments to
source failure: 66 static pseudo-motion, 14 bad-flow magnitude, and one
direction reversal.  Of 73 X3-only additions, `65/73` (89.04%) have the same
source causes.  Accept `DTR_X3_FULL_REPLAY_FAILURE_ATTRIBUTION_COMPLETE` and
require the next source to be static-aware and direction-consistent; do not
rescue X3 through threshold, seed, backbone, route, lifecycle, or scorer work.

X5 tested reciprocal cycle agreement across overlapping causal five-scan
windows using only the sealed X3 cells.  It suppressed `32/34` source-error
units but retained one correct positive frame and zero correct route-entry
frames.  Accept `DTR_X5_OVERLAP_CYCLE_SOURCE_FALSIFIER_GATE_NOT_MET`: close
same-source consistency filtering and require a static-world anchor or an
independently observable dynamic signal.

## DTR-X6--X7 causal static-world anchor

X6 introduced an independent causal raw-LiDAR world-occupancy anchor while
retaining X3 candidates, velocity, route geometry, and scorer. On the same
60-frame positive/error roster it preserved three correct positive frames and
two correct route-entry frames, suppressed `26/34` source-error units, and met
the one-scan-period compute bound. Accept
`DTR_X6_STATIC_WORLD_PERSISTENCE_FALSIFIER_GATE_MET`; the pass authorized one
frozen full replay only.

X7 completed all 4,811 timeline frames across the six opened Development
sequences. Relative to X3, it reduced false segments `94 -> 72` and improved
Event F1 `11.32% -> 14.29%`, while preserving `6/6` CONTACT recall, `3.816 s`
median lead, and `8/18` dropout recovery. Relative to PDC it still has 47 more
false segments and lower F1. Accept
`DTR_X7_FULL_STATIC_WORLD_ANCHOR_GATE_NOT_MET`: the anchor contributes real
selectivity but is insufficient as a standalone source. Do not tune the opened
map radius or age; require independently observable scene-motion evidence.
Full mechanics, count amendment, hashes, and claim limits are in
`X6_X10_STATIC_VISUAL_LEARNED_AUTHORITY_2026-08-29.md`.

## DTR-X10 cross-fitted learned motion authority

X10 applied one fixed standardized L2 logistic head to the sealed X9 cells.
Six folds trained only on the other five sequences' native-OBB motion-validity
labels; each held-out probability ledger was sealed before its labels were
available to scoring. The inference features contain current geometry, motion,
support, X7/X9 retention, and local velocity dispersion, but no route, event,
future, or label input.

At the frozen `0.5` threshold, X10 retained 108,403/238,726 cells, reduced false
segments `64 -> 47`, improved Event F1 `15.79% -> 20.34%`, preserved `6/6`
CONTACT recall and `8/18` dropout recovery, and retained `3.526 s` aggregate
median lead. Accept
`DTR_X10_CROSS_FITTED_MOTION_AUTHORITY_GATE_NOT_MET`: the cross-sequence effect
is real, but false segments remain above PDC and F1 below 35%. Do not try a
second classifier, feature reweighting, or threshold on the consumed folds;
require a new observable motion signal. Full hashes and boundaries are in
`X6_X10_STATIC_VISUAL_LEARNED_AUTHORITY_2026-08-29.md`.

## DTR-X11--X16 RGB-authorized continuation

X11--X13 tested raw-camera static veto, CIWT track agreement, and stitched-RGB
positive dynamic birth. All three frame-local routes closed: X11 was weaker
than stitched X8, X12 deleted every positive, and X13 retained correct motion
but no correct route-entry frame. X14 supplied the missing causal structure:
only an independently RGB-authorized birth may move at its unchanged X7
velocity for the frozen R1 `0.50 s` clear-grace interval. Its 60-frame canary
kept 13 correct frames and four correct route frames, suppressed `30/34`
source-error units, and met the compute gate.

X15 replayed that frozen mechanism over all 4,811 frames. Relative to PDC it
improved CONTACT recall `4/6 -> 5/6`, false segments `25 -> 18`, Event F1
`22.86% -> 34.48%`, and median lead `2.291 -> 3.061 s`; dropout recovery fell
`5/18 -> 2/18`. Accept
`DTR_X15_FULL_RGB_AUTHORIZED_CONTINUATION_GATE_NOT_MET`. The false/F1 misses
are narrow, but the dropout regression is material and no duration or visual
confidence sweep is authorized.

X16 composed the already frozen X10 sequence-held-out authority ahead of the
same RGB birth and continuation. It reduced false segments to 15 and raised
median lead to `3.630 s`, but CONTACT recall fell to `4/6`, Event F1 to 32%,
and dropout stayed `2/18`. Accept
`DTR_X16_CROSSFIT_RGB_AUTHORIZED_CONTINUATION_GATE_NOT_MET`; the composition is
over-selective and closes without rescue. X15 remains the current algorithm
reference. Eleven of its 18 false ranges are in memorial-court, so the next
information change must attribute that scene's persistent non-contact motion
and add gap-persistent identity/occupancy rather than another cell filter.
Mechanics, hashes, and claim limits are in
`X11_X15_RGB_AUTHORIZED_CONTINUATION_2026-08-29.md`.

Read-only X15 attribution sharpens that successor. Every one of the 18 false
ranges contains RGB authorization outside the route followed by transport into
the route; 14 ranges have no same-frame risky birth. In memorial, seven of 11
ranges are transport-only and four are mixed. Lifecycle HOLD accounts for 147
of 684 active false frames, so it lengthens exposure but does not create the
repeated transport cohorts. The next single falsifier should require an
ego-compensated, temporally persistent RGB movable-instance mask/track before
birth, then leave X15 continuation and downstream logic frozen.

## DTR-X17--X21 track-carried component ancestry

X17 confirmed that YOLO11 instance-track persistence is a strong continuity
source (`5/6` CONTACT, `14/18` dropout, `4.922 s` lead), but mask-wide X7
admission produced 33 false segments. X18's dense X15 seeding was
score-equivalent. X19 restricted seeding to raw X13 births yet retained
mask-wide admission; 30 of its 31 false segments overlapped X17. This closes
seed and tracker tuning: the failure was authority amplification inside a live
instance mask.

X20 keyed continuation to exact `(class, track, X7 component)` ancestry. It
reached `5/6` CONTACT, 16 false segments, 37.04% Event F1, and `3.785 s` lead,
but only `2/18` dropout recovery because current-X7 component support still
broke inside the induced gaps.

X21 removes that final dependency without reopening mask-wide birth. A raw X13
birth inside a track is the only state origin; later frames transport only the
stored component row, and retain it only while its anchor remains inside the
same current live track mask. No new current-X7 cell is absorbed. On the sealed
4,811-frame Development replay X21 reached **`5/6` CONTACT, 11 false segments,
45.45% Event F1, `3.061 s` median lead, and `8/18` dropout recovery**. All six
frozen checks passed. Accept
`DTR_X21_TRACK_CARRIED_COMPONENT_ANCESTRY_GATE_MET` for Development and move the
unchanged arm to one source-disjoint confirmation; do not tune the consumed
cohort. Mechanics, hashes, and claim limits are in
`X17_X21_TRACK_CARRIED_COMPONENT_ANCESTRY_2026-08-29.md`.

## DTR-X8--X9 independent RGB static evidence

X8 compared synchronized stitched-RGB tracks against ego-rigid static and
X7-moving reprojections. It vetoed only when valid tracks favored the static
hypothesis; missing or invalid visual evidence retained the X7 candidate. On
the 60-frame roster it preserved three correct positive frames and two correct
route-entry frames, suppressed `27/34` source-error units, and met the
frame-local compute bound. Accept
`DTR_X8_RGB_STATIC_VETO_FALSIFIER_GATE_MET`; one frozen full replay followed.

X9 matched all 4,811 requested RGB frames and evaluated 4,803 consecutive
pairs. It vetoed 62,757 of 301,483 X7 cells and reduced false segments
`72 -> 64`, with Event F1 `14.29% -> 15.79%`; `6/6` CONTACT recall, `3.816 s`
aggregate median lead, and `8/18` dropout recovery were unchanged. Accept
`DTR_X9_FULL_RGB_STATIC_VETO_GATE_NOT_MET`: the independent source has additive
effect but remains far above PDC's 25 false segments and below the frozen 35%
F1 floor. Close this fixed visual veto without threshold or camera-fusion
tuning. Mechanics, hashes, and claim limits are in
`X6_X9_STATIC_WORLD_RGB_AUTHORITY_2026-08-29.md`.

## Claim ceiling

These are retrospective public-real privileged algorithm ceilings. THÖR uses
controlled-lab global tracks; JRDB is a large robot-relative/interpolated
diagnostic without synchronized ego pose; CODa uses source-native boxes,
identities, pose, timestamps, and calibration. Future path and future contact
are evaluator-only. The R5 RGB result is a three-event curated induced-dropout
stress canary with evaluator-only identity binding, not source-disjoint or
natural-distribution evidence.

No RGB/LiDAR detector, detector disappearance robustness, calibrated collision
probability, natural wearer motion, Android runtime, BLV benefit, product
reliability, or safety performance is established. Dynamic positive authority
is predominantly pedestrian plus one scooter and one delivery-truck event;
static positive authority is the 12 observed barrier/fixed/temporary events.
`UNKNOWN` and `NOT_EVALUABLE` are never counted as safe.

R6 further shows that the current curated dropout cohort cannot isolate a
static direct-metric source: all three events lack an admissible residual
closing-velocity input under the frozen matcher. It establishes neither a
negative metric-depth result nor a spatiotemporal occupancy capability.

R7-P is a privileged classical raw-LiDAR motion ceiling on the same consumed
Development window. Its `9/9` dropout recovery coexists with `20` false
segments and an 86.0% global route-risk frame rate. It establishes a causal
motion-information signal, not detector-independent dynamic occupancy quality,
source-disjoint generalization, or permission to train an RGB student.

DTR-M0 is scorer-side post-outcome attribution on that consumed result. It
localizes R7's flow-caused errors but adds no performance, generalization,
product, or safety evidence. Its component-discontinuity flags are not stable
identity or split/merge truth.

DTR-X0 is likewise scorer-side post-outcome attribution, now on the consumed
C31 source-disjoint confirmation. Its native identity/trajectory labels choose
the next source experiment but add no performance or generalization evidence.
Unmatched risk cells are consistent with static pseudo-motion, not proof that
every unmatched cell belongs to a static surface.

DTR-M1-O is a privileged, label-dependent oracle on the same consumed JRDB
window. Its `6/9` dropout recovery and 17 false segments close estimator work
under the frozen hard point/cell downstream; they do not establish AV2-native
scene-flow performance, source-disjoint generalization, route-forecasting
quality, Android behavior, product benefit, or safety.

DTR-M2-D is a privileged read-only geometric diagnosis on that same consumed
window. Its three `POINT_MISS_FOOTPRINT_MISS` dropout trials and five
`FOOTPRINT_HIT_TRUTH_NEGATIVE` false segments close only the current-OBB plus
constant-M1-velocity swept-footprint hypothesis. They are not a general
negative on time-varying occupancy, route-conditioned forecasting, or body
collision geometry, and they authorize no fresh M2-O or product claim.

DTR-M3-D opens realized future labels only for read-only truth-contract
decomposition. Its decisive event is positive under the evaluator circle but
negative under every realized future OBB. It therefore closes fresh M3-O until
the intended contact semantics are chosen and rescored on fresh evidence. The
result is not learned forecasting, causal future-information, product, or
safety evidence.

DTR-C0 adopts global realized-OBB CONTACT plus simultaneous circle-only
PROXIMITY, but the consumed window is CONTACT-saturated and has no non-contact
denominator. Its descriptive overlap scores and `6/6` M1/R7 target-component
diagnostic are not performance evidence. Fresh global-OBB confirmation is
required before estimator work; no Android, product, user, or safety claim
follows.

DTR-C1 is metadata-only source/cohort admission. Its 21 bounded events and
409.66 s non-CONTACT exposure prove that the frozen JRDB roster has evaluation
denominators, not that any algorithm detects them. Event identities are frozen
only for later dropout intervention and attribution; global correctness remains
wearer-level. No raw-sensor replay, detector robustness, direct-motion
estimation, product behavior, user benefit, or safety performance is established.
