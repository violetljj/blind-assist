# Current decisions: L10-R0 active; Dynamic Travel Risk R2 established

Status: `L10_R0_ACTIVE / SC14_CAUSAL_MICRO_MOTION_ACTION_BELIEF_MECHANICS_SIGNAL /
SC14_CORE_CAUSAL_ACTION_HANDOFF_GUARD_IMPLEMENTED /
L10_CORE_SEEK_GUIDE_REACQUIRE_CONTROLLER_IMPLEMENTED /
SC15_SC16_SC17_ACTIVE_VIEW_PROXY_ROUTES_CLOSED /
SC18_CAUSAL_ACTION_OUTCOME_REPAIR_MECHANICS_SIGNAL /
SC19_DEFICIT_CONDITIONED_ACTION_UTILITY_MECHANICS_SIGNAL /
SC20_FACTORIZED_ENDPOINT_FALSE_READY_SUPPRESSION_GATE_NOT_MET /
SC21_FUNCTIONAL_SET_INTEGRITY_FRESH_SOURCE_NOT_EVALUABLE /
SC22_INTEGRITY_ENDPOINT_COMPOSITION_MECHANICS_SIGNAL /
SC24_SEMANTIC_TOPOLOGY_INTEGRITY_CONSUMED_MECHANICS_SIGNAL /
SC28_REDUNDANCY_GATED_ACTION_FAMILY_CONSUMED_MECHANICS_SIGNAL /
SC29_REDUNDANCY_GATED_ACTION_FAMILY_FRESH_GATE_NOT_MET /
SC31_CONFLICT_ADMITTED_REDUNDANCY_GATED_FRESH_DEVELOPMENT_SIGNAL /
SC32_BOUNDARY_ORDINAL_SOURCE_NOT_EVALUABLE /
SC34_ORIENTATION_QUOTIENT_ORDINAL_FRESH_GATE_NOT_MET` and
`DTR_R2_PUBLIC_REAL_PRIVILEGED_CEILINGS_ESTABLISHED /
DTR_C2_M1_CTB_CONFIDENCE_TRACK_GAP_BRIDGE_FRESH_MECHANICS_SIGNAL /
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

The mechanics representation is now implemented in `core:assist`: a causal
paired-point Horn fit produces `UNKNOWN / SET_VALUED / LOCKED`, and a fail-closed
admitter binds it to provider, goal, session, parent entity, exact frame, clock,
and freshness. The product handoff reducer now rejects `HANDOFF_READY` unless a
separate endpoint join supplies current `READY` position, visibility,
grounding, orientation, and reachability plus a `LOCKED` causal action model.
Explicit user confirmation remains the only completion transition. This is a
runtime contract implementation, not evidence that a live RGB-D source or any
of those endpoint conditions currently succeeds.

The earlier SC1W/SC2 semantic-carrier and opportunity policy is now also in
`core:assist`. Fresh semantic evidence alone can acquire, guide, or reacquire;
continuity-only evidence is limited to two observation requests, and LOST
requires two later fresh hits. Non-exhaustive proposal failure maps to
`UNKNOWN + SWEEP/SCAN`, not `NONE/STOP`. Any identity loss or missing current
endpoint evidence revokes `HANDOFF_READY` back to `Approach`. This runtime port
inherits the existing Development representation result (`124 -> 0` STOP,
`24 -> 0` target-present false-NONE, 100% non-navigation action coverage,
`30/30` end-to-end episodes) but adds no executed-view causality, live-camera,
metric-arrival, product, user-benefit, or safety evidence.

Three bounded successors now close the cheap-observability branch. On consumed
ArTVideo Development, SC15 found 17 passive `PAN/SCAN` transitions; aligned
motion had lower semantic gain than opposed motion (`0.6667` versus `0.8000`)
and higher wrong-gate creation (16.67% versus 0%). On a protocol-frozen fresh
DSText V2 indoor video, SC16 processed 2,824 recognition observations and 2,677
transitions; Pareto improvement in text height, sharpness, and centering yielded
only `+0.0093` aligned-minus-opposed semantic gain and lower gate crossing
(1.66% versus 4.28%). SC17 then reused that consumed Development result for one
fixed three-view exact-consensus rule; precision moved only 90.97% to 91.88%,
wrong outputs fell 24.31%, and five of 49 tracks still formed wrong consensus.

Accept the three frozen gate negatives. Do not sweep temporal horizon, quality
weights, centering/scale thresholds, edit distance, vote count, or OCR gates on
the opened sources. The next active-observation algorithm must use a new source
with an actually issued action plus before/after observation receipt, allowing
causal outcome feedback and policy repair, or add genuinely new identity
information. None of SC15--SC17 changes the existing SC2 action-interface
mechanics result or establishes live active-view, arrival, product, user, or
safety benefit.

SC18 now lands the causal successor in `core:assist`. Each observation action
issues a current goal/session/entity/frame/clock-bound receipt; only a later
admitted comparable observation can classify the result as improved, no gain,
or contradicted. The caller must acknowledge that the matching instruction was
executed; a merely issued prompt has no outcome authority. Missing, stale,
mismatched, unauthorized, or unexecuted evidence remains `UNKNOWN`. A
comparable no-gain result prevents an immediate repeat and selects a different
observation action.

On one protocol-frozen 250-episode controlled-world comparison, repeated
same-action choices after no gain fell from `182/206` to `0/215`; task success
was unchanged at 88.5%, target-absent false completion remained 0/50, and
reacquisition moved from 82.35% to 80.75%, within the frozen tolerance. Accept
`SC18_CAUSAL_ACTION_OUTCOME_REPAIR_MECHANICS_SIGNAL`, with a strict synthetic-
mechanics ceiling. The zero-repeat primary result is structurally expected and
does not prove that the chosen opposite pan is optimal. The next source must
collect real issued-action receipts so action utility can be learned per
evidence deficit; do not tune the consumed seed or return to passive quality
proxies.

SC19 replaces long-term fixed repair with bounded online utility learning.
Utility is keyed only by evidence deficit and camera-relative bearing; updates
require execution-confirmed comparable receipts, duplicate receipt IDs are
ignored, and `UNKNOWN` never becomes success or failure. Candidate sets remain
hard safety boundaries: notably, association ambiguity cannot explore blind
approach.

In a protocol-frozen authored seven-context world, with 1,260 trials and 215
unknown outcomes, authoritative improvement rose from `419/1,045` (40.10%) to
`658/1,045` (62.97%). Expected cumulative regret fell 302.40 -> 20.38
(-93.26%), final-window optimal selection rose 42.86% -> 99.52%, and ambiguous-
context approach fell 100% -> 0. Accept
`SC19_DEFICIT_CONDITIONED_ACTION_UTILITY_MECHANICS_SIGNAL` under a strict
synthetic contextual-policy ceiling. The probabilities are authored, so the
metric size cannot be transferred to natural video or users. The next evidence
source is live execution-confirmed action receipts under unchanged safety
candidate sets; do not tune SC19's consumed contexts, probabilities, seed, or
exploration coefficient.

Device/demo integration is intentionally paused. SC20 instead opens the
factorized endpoint layer on the already-consumed SceneFun3D 420683 trajectory.
Across six tasks, 901 real posed RGB-D frames, and 5,406 task-frames, the
evaluator contained 185 observation-ready task-frames. A centered-large-parent
proxy emitted 429 ready frames, of which 300 were false (30.07% precision,
69.73% recall, F1 0.420). The task-functional observer joins horizontal
stand-off, depth-consistent visibility, camera orientation, and the existing
SC11 grounding support; it emitted 121 frames, of which 120 were true and one
was false (99.17% precision, 64.86% recall, F1 0.784).

The frozen Development gate is nevertheless not met because true-ready task
coverage fell from `6/6` to `5/6`. The missing under-bed drawer task had 35
evaluator-ready frames, but no predicted frame jointly satisfied position and
visibility; at eligible stand-off, at most one of its three selected functional
points was depth-visible. Read this as strong false-arrival suppression plus a
localized upstream candidate-integrity/visibility deficit, not endpoint
completion. Do not tune standoff, visibility, orientation, or parent-proxy
thresholds on this consumed scene. Reachability remains `UNKNOWN`, so SC20
cannot emit `HANDOFF_READY`; the next eligible endpoint source must add
candidate integrity on separately versioned evidence or independent free-space
and human-reachability authority.

SC21 changes candidate representation rather than repairing SC20 thresholds.
Inside a task-relational selected set, candidates are connected in parent-
normalized 3-D coordinates; only a unique largest component of at least two
members may quarantine isolated candidates. Ties and singleton sets remain
`SET_VALUED + REQUEST_INTEGRITY_VIEW`. The radius (`0.4` normalized parent
extent) and dominance rule were frozen before opening the two-scene
`421002 + 420673` cohort. That source is `NOT_EVALUABLE`: only `3/20` tasks had
an authorized exact parent binding and there were zero integrity opportunities.
This is a source-denominator result, not an algorithm negative.

A read-only diagnostic then applied the unchanged rule to consumed SC11. It
quarantined the under-bed drawer's isolated candidate at 1.445 m while retaining
the two candidates 5.33 cm and 5.80 cm from functional truth. Legal commits rose
`4/6 -> 5/6`, wrong parts fell `2 -> 1`, and mean target-set recall stayed
83.33%. This is a localized consumed-source mechanism effect only.

SC22 composes that fixed integrity rule with unchanged SC20 endpoint factors and
adds no parameter. On the same consumed real RGB-D trajectory, observation-
ready task coverage rose `5/6 -> 6/6`; true-ready recall rose 64.86% -> 83.24%
and F1 0.784 -> 0.906. Precision stayed above 99% (99.17% -> 99.35%) and false
ready frames stayed `1 -> 1`. The missing under-bed task recovered 34 of its 35
evaluator-ready frames. Accept
`SC22_INTEGRITY_ENDPOINT_COMPOSITION_MECHANICS_SIGNAL` under a strict consumed-
trajectory ceiling: it shows that candidate-set integrity and factorized
arrival are complementary, not that transfer or product arrival is confirmed.

Do not tune SC20 or SC21 on 420683. Fresh confirmation needs an RGB-D proposal
cohort with enough authorized parent-bound tasks and actual integrity
opportunities. Reachability remains `UNKNOWN`; SC22 cannot emit
`HANDOFF_READY`, and explicit user confirmation remains the only completion
authority.

SC23--SC29 then tested whether SC21 transfers beyond the localized under-bed
case. An outcome-blind source admission scanned the official SceneFun3D
one-video roster in order and selected scenes only by description count and
target-region multiplicity. On the first three fresh scenes, SC21 had 37
parent-bound tasks and two integrity opportunities, but legal commits changed
`12 -> 13`, wrong parts `50 -> 49`, and mean target-set recall fell
93.24% -> 90.54%. Accept
`SC23_FUNCTIONAL_SET_INTEGRITY_FRESH_GATE_NOT_MET`: pure spatial connectivity
cannot distinguish two functions that share one parent.

The observed conflict was exact: two socket regions (`plug_in`) formed the
largest component while a remote-control region (`key_press`) was spatially
isolated. SC24 therefore adds a task-action semantic witness before unchanged
SC21 topology. On consumed SC23, legal commits improved `13 -> 16`, wrong parts
`49 -> 41`, and recall recovered 90.54% -> 93.24%. This is
`SC24_SEMANTIC_TOPOLOGY_INTEGRITY_CONSUMED_MECHANICS_SIGNAL`, not fresh
confirmation; candidate function labels are privileged SceneFun3D proposals.

The next fresh SC25 source had 31 evaluable tasks but only two exact semantic
admissions, below the frozen minimum four. It is `NOT_EVALUABLE`, despite a
directional `9 -> 10` legal-commit, `61 -> 58` wrong-part, and 72.58% -> 75.81%
recall change. SC26 widened semantics only across action families while keeping
all candidates within a family set-valued. On consumed SC25 it admitted 31/31
tasks, improved legal commits `9 -> 11`, reduced wrong parts `61 -> 54`, and
raised recall 72.58% -> 79.03%.

Fresh SC27 showed why broad semantic deletion must remain selective: legal
commits improved `5 -> 7` and wrong parts `32 -> 27`, but recall fell 83.33% ->
79.17% because an `Open the storage drawer` target was labeled `rotate` while a
single `hook_turn` candidate satisfied the broad open/close family. SC28 adds a
no-regret redundancy condition: exact task-action cues may filter directly,
but a broad family needs at least two compatible candidates or it preserves the
full set. On consumed SC27 this restored recall to 83.33% and still reduced
wrong parts `32 -> 30`, yielding
`SC28_REDUNDANCY_GATED_ACTION_FAMILY_CONSUMED_MECHANICS_SIGNAL`.

The independently admitted SC29 cohort was evaluable (29 tasks, 27 semantic
admissions) but contained zero candidate-changing cross-family conflicts; both
arms were identical at 12 legal commits, 82.76% recall, and 21 wrong parts.
Therefore `SC29_REDUNDANCY_GATED_ACTION_FAMILY_FRESH_GATE_NOT_MET` is a source-
opportunity terminal, not a regression. Do not tune SC23--SC29 or keep selecting
cohorts by multi-target count. The next legal source admission must count
provider-public, same-parent cross-action-family conflict opportunities without
using target membership or algorithm outcome; otherwise no fresh integrity
claim is evaluable. Reachability remains `UNKNOWN` and handoff remains
forbidden.

SC30 fixes the source denominator instead of changing SC28. It ignores task
target membership and selector output, groups provider functional proposals by
parent OBB and frozen action family, and admits only scenes containing a parent
with at least two action families plus at least two proposals in one family.
The first bounded 20-scene roster found only one eligible scene and was
`NOT_EVALUABLE`. V2 changed only the roster ceiling (`20 -> 80`), preserved all
thresholds and ordering, and admitted the first three eligible scenes after 32
candidates: `421267`, `422356`, and `422377`. Each contained exactly one
provider-public redundancy-eligible conflict parent; no target IDs or algorithm
outcomes entered selection.

On this source-disjoint cohort, SC31 had 16 parent-bound tasks, 10 semantic
admissions, and two metric-changing tasks. Relative to SC21 topology, the
unchanged SC28 successor improved legal commits `3 -> 4`, reduced wrong parts
`24 -> 20` (-16.67%), preserved mean target-set recall at 62.5%, and produced
zero cross-parent violations. Accept
`SC31_CONFLICT_ADMITTED_REDUNDANCY_GATED_FRESH_DEVELOPMENT_SIGNAL` as the first
fresh proposal-conditional evidence that source-aligned action semantics can
remove same-parent functional interference without aggregate recall loss.

The result is not task-wise no-regret. `Unplug the printer` improved from an
all-wrong set to one legal target, while `second outlet of the power strip`
lost its target after repeated same-action candidates were reduced by topology;
the gains offset at aggregate recall. Do not tune SC31, ordinal words, topology,
or the admitted scenes. The next legal representation is a fresh target-
conditioned ordinal axis for repeated same-action controls (for example second,
third, fourth outlet), with action-family filtering unchanged. Candidate labels
are still privileged, reachability remains `UNKNOWN`, and `HANDOFF_READY`
remains forbidden.

## 2026-08-28 ordinal grounding decision

SC32 tested a parent-boundary-anchored absolute ordinal axis and admitted no
scene in its frozen 80-candidate roster. The failure is source/representation
specific, not an ordinal-algorithm negative: the diagnostic scene's three
near-uniform outlet proposals were authorized to the bed OBB, so the parent
boundary could not honestly define the power strip's first slot. Do not relax
the opened boundary, pitch, or residual thresholds.

SC33 replaced that unavailable polarity source with an orientation quotient.
When public task text supplies a complete rank inventory and provider-public
geometry supplies an unoriented odd-length lattice, only the reversal-invariant
middle rank may uniquely commit; endpoint ranks retain both hypotheses and ask
for an orientation observation. Source admission selected `421658 / 42445769`
after 16 candidates without reading target IDs. The fresh SC34 run recovered
`second outlet` from an empty SC31 selection to the unique correct proposal,
raising legal commits `0/3 -> 1/3` and mean target-set recall `66.67% -> 100%`
with zero task-wise regressions. Wrong parts stayed `4`, so the frozen composite
gate requiring wrong-part reduction was not met. Record this as a fresh narrow
false-negative-recovery effect inside a formal gate failure; do not tune or
reopen the cohort. The next information source must authorize axis polarity
(for example a live view/gravity relation) to resolve first/last, while RGB
proposal quality, reachability, arrival, and `HANDOFF_READY` remain unknown.

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

That representation-level question has now received a read-only M2-D audit.
It reused the exact M1 ledger and changed no prediction or gate, comparing the
continuous zero-radius cell paths against current native oriented footprints
translated by the same robust M1 velocity over 0--3 s. The three failed
dropout trials are all `POINT_MISS_FOOTPRINT_MISS`, with the nearest footprint
still 0.0374 m outside the frozen 0.65 m route body. The five M1 new/modified
false segments are all `FOOTPRINT_HIT_TRUTH_NEGATIVE`.

The terminal is `DTR_M2_D_EXTENT_GAP_NOT_SUPPORTED_NO_FRESH_M2_O`.
Do not launch a fresh swept-footprint M2-O from the current-extent hypothesis.
The earlier positive center-plus-body-extent future uses realized future
native boxes; it is not equivalent to translating the current OBB at constant
M1 velocity. The remaining gap may require time-varying future occupancy,
different route/contact semantics, or both, but M2-D does not adjudicate those
alternatives and does not authorize route-conditioned forecasting, R8, or a
different scene-flow estimator.

M3-D has now performed the required realized-future contract decomposition
without changing any prediction or event. For all three repeated
`pedestrian:35` dropout rows, the evaluator's realized-center circular radius
hits while the realized future native OBB misses. At first contact frame 185,
the evaluator circle is 0.0209 m inside threshold and the true oriented box is
0.0374 m outside the 0.65 m route body; the realized-center versus M1-CV center
residual is only 0.0103 m. Thus the decisive flip is circle-versus-OBB truth
semantics, not demonstrated causal future-dynamics headroom.

The false rows separate into three target-owned realized-OBB misses--genuine
constant-transport false positives--and two other-component attribution
errors. In particular, `pedestrian:9` is triggered by `pedestrian:34`, whose
own realized future is positive under both circle and OBB contracts.

The terminal is
`DTR_M3_D_EVALUATOR_CIRCLE_OBB_SEMANTICS_MISMATCH_NO_FRESH_M3_O`. Keep M3-O,
learned/residual future occupancy, route-conditioned forecasting, R8, and
scene-flow estimator competition closed. The next admissible action is to
choose and freeze the event meaning. If the intended claim is oriented-body
collision, revise the evaluator and rescore a fresh cohort before testing any
dynamics model. If circularized proximity is intentional, describe it as
such; M3-D establishes a semantics difference, not that one definition is
universally correct.

The task contract is now reset by C0: primary truth is the wearer-level union
of realized future OBB CONTACT, legacy circle-only proximity is a simultaneous
secondary label, and target identity cannot make an otherwise correct global
route-risk alert false. Frozen R2, R3-C, R7-P, and M1-O predictions were
replayed without training or tuning.

This consumed window is not a valid global scorecard. Overlapping p33, p34,
and p36 OBB-contact horizons cover `143/143` frames, producing one left- and
right-censored always-CONTACT interval, zero bounded events, and 0.0 minutes of
known non-CONTACT wearer time. Consequently global CONTACT recall/F1 and false
segments per wearer minute are `NOT_EVALUABLE`. The descriptive 1.0 overlap
F1 and zero unmatched segments for every arm are saturation artifacts, not
performance.

After removing the three p35 circle-only repetitions, the dropout contract has
six OBB-CONTACT rows. R7-P and M1-O contain stressed-target raw contributions
in `6/6`; R2 and R3-C have a global raw alert in `5/6` only because other
targets contribute, while the stressed target contributes in `0/6`. Do not
promote this consumed `6/6` diagnostic to a 100% recovery claim.

The terminal is
`DTR_C0_GLOBAL_ORIENTED_RISK_CONTRACT_NOT_EVALUABLE_ALWAYS_CONTACT_WINDOW`.
Retain the global OBB CONTACT plus secondary PROXIMITY contract, but freeze a
fresh cohort with bounded contact events and known non-contact wearer time
before comparing algorithms or opening a deployable direct-motion estimator.
Forecasting, R8, and scene-flow estimator competition remain closed.

C1 has now frozen that fresh truth roster without opening an algorithm. The
train label/timestamp archives contain 27 common sequences; the entire consumed
`packard-poster-session-2019-03-20_1` sequence is excluded. The other 26 were
scanned metadata-only in lexicographic order. The shortest prefix reaching the
predeclared preferred `20 bounded / 10 unique responsible / 120 s non-CONTACT`
target contains seven sequences and provides `21 / 21 / 409.66 s` across
8,368 frames. CONTACT occupies 150.84 s of 560.51 s known truth, for a 26.91%
duty cycle; one selected sequence contains no CONTACT and contributes 96.92 s
of negative exposure.

Accept `DTR_C1_FRESH_GLOBAL_OBB_COHORT_ADMITTED_METADATA_ONLY`. This establishes
that JRDB can supply event recall/F1 and false-segments-per-non-CONTACT-minute
denominators under the corrected global OBB contract. It does not establish
R2, R3-C, R7-P, or M1-O performance. Freeze the tracked sequence/event roster;
if the second stage is opened, acquire raw data only for that roster and replay
the unchanged four arms without training or tuning. Dropout uses only the 21
unique-first-responsible events and remains a mechanism secondary. Forecasting,
R8, TeFlow, DeltaFlow, and deployable estimator work remain closed until the
fresh global replay is adjudicated.

C2 has now adjudicated that replay without changing route geometry, lifecycle,
motion bounds, or thresholds. R7-P recalled `20/21` bounded CONTACT events but
created 90 false segments (13.18 per known non-CONTACT minute), for 30.53% F1.
M1-CT requires raw-LiDAR motion to carry spatial support, ego-compensated
forward-advection agreement, and velocity consistency with an independent
historical sweep. It preserved `20/21`, cut false segments to 38 (`-57.8%`),
raised F1 to 50.63%, and retained 2.08 s median lead.

That confidence gate alone reduced induced-gap recovery from R7's `29/63` to
`18/63`. M1-CTB therefore keeps a second channel closed by default and admits
raw dense motion only during an observable bounded gap of a previously tracked
target. Natural-replay scores remain exactly M1-CT, while recovery returns to
`29/63`. Accept
`DTR_C2_M1_CTB_CONFIDENCE_TRACK_GAP_BRIDGE_FRESH_MECHANICS_SIGNAL` as fresh
evidence that confidence-aware scene motion can suppress pseudo-motion without
sacrificing the occlusion recovery already available from dense flow.

This does not yet authorize wholesale R2 replacement: R2 has the same `20/21`
recall with 29 false segments and 57.14% F1 on natural replay. Current native
boxes are also a privileged spatial-attribution ceiling. The next DTR increment
may replace the occupancy-cell motion source with deployable raw-point direct
velocity behind the now-supported confidence/track-gap interface, using measured
GPU backend selection. Do not tune C2 confidence, route, lifecycle, or motion
thresholds; do not open complex trajectory forecasting or R8 from this result.

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
