# DTR: route-conditioned obstacle-risk events

Status: `DTR_R3_GATE_NOT_MET / R2_DYNAMIC_RETAINED /
S4_CONTINUOUS_GEOMETRY_VALIDATED_NO_PUBLIC_GAIN /
R5_RGB_DROPOUT_CANARY_GATE_NOT_MET /
R6_DIRECT_METRIC_SINGLE_FACTOR_NOT_EVALUABLE_STATIC_OCCUPANCY_MATCHER_UNREACHABLE /
R7_P_CAUSAL_OCCUPANCY_FLOW_DEVELOPMENT_GATE_NOT_MET_NO_R8 /
DTR_M1_O_POINT_VELOCITY_ORACLE_CEILING_NOT_MET_CLOSE_SCENE_FLOW_ROUTE /
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
```

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

DTR-M1-O is a privileged, label-dependent oracle on the same consumed JRDB
window. Its `6/9` dropout recovery and 17 false segments close estimator work
under the frozen hard point/cell downstream; they do not establish AV2-native
scene-flow performance, source-disjoint generalization, route-forecasting
quality, Android behavior, product benefit, or safety.
