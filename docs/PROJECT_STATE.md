# Project state

Updated: 2026-08-30

BlindAssist is a runnable Android showcase research prototype. The default
research policy is effect-first: demonstrate a genuine, visible effect in a
controlled setup, report the setup and metric honestly, and add only checks
that protect interpretation.

## Current operating surface

- Current ten-meter route: [L10-R0 Goal-Lock Copilot](../research/active/l10-r0/README.md)
- Current obstacle/risk route: [Dynamic Travel Risk R2](../research/active/dtr-r0/README.md)
- Current question and stop condition: [CURRENT_DECISION.md](CURRENT_DECISION.md)
- Workstation entrypoint: `tools/ba.ps1`
- CARLA DTR asset bridge: [CARLA integration playbook](CARLA_PLAYBOOK.md)
- Android entrypoint: `scripts/run_android_gradle.ps1`
- Closed experiment lookup: [history-index.md](history-index.md)

Current route snapshot:

- JRDB/public DTR retains R2 and freezes X21 as a same-source Development gate
  pass pending a genuinely source-disjoint confirmation.
- CARLA DTR retains X24 as a same-source C2 Development pass; X26 and X30 did
  not meet their gates, C8--C10 are source-level `NOT_EVALUABLE` terminals with
  no X31 metrics, and the frozen CARLA C11 source is pending its first capture
  (`NOT_RUN`).
- L10 PanoLab active entrance-ray recovery passed `4/4`; this does not establish
  a pixel portal. Generic Panoramax pixel-portal mining is closed.
- Hypersim established a synthetic posed-portal mechanism ceiling. SceneNN real
  RGB-D confirmation did not meet the gate because reference-side visibility
  was not authoritative; a fresh visibility-qualified real source is required.

## Current evidence

GRAIL owner orientation is a completed negative-result chain rather than the
daily mainline. Its latest terminal remains
`STOP_G1_ACTIVE_MULTIVIEW_APPEARANCE / DEVELOPMENT_GATE_NOT_MET /
NO_FINAL_TEST`: fixed three-view appearance increased the PRESERVE tendency,
collapsed Doorway FLIP to `0/24` in both seeds, and left owner-group macro
balanced accuracy near chance. No view selector, G0 fusion, or further
pose/model sweep is authorized from that consumed result.

Dynamic Travel Risk R2 now combines robust finite-horizon occupancy consensus
with one fixed imminent route-intersection guard and stable
`ONSET / HOLD / ESCALATE / CLEAR` events. On the 19-session, route-authoritative
THÖR-MAGNI ceiling it recalls `10/10` events with 42 false-alert segments,
strictly improving R0's `9/10` and 55. On 27 JRDB test sequences it recalls
`164/175` with 256 false alerts versus R0's `161/175` and 260. CODa adds a hard
pose-authoritative development/holdout check: R2 recalls `119/122` pedestrian
events with 285 false alerts versus R0's `122/122` and 286, retaining the small
recall cost instead of tuning it away.

CARLA 0.9.16 is now connected through a BlindAssist-owned, external-process
asset contract. The bridge admits the frozen V16 privileged-source canary and
the corrected V17 valid negative by exact status, authority, result path, and
SHA-256, then materializes an ignored project-side context for experiments.
This makes CARLA available as a synthetic DTR causal lab without adding it to
Android or treating it as source-disjoint X21 confirmation.

DTR-CARLA-C0 turned that bridge into a runnable algorithm benchmark: six
causal twin families, 12 episodes, four separately replayed sensor modalities,
and a truth-blind RGB -> RGB-D -> CARLA-flow-teacher -> privileged-current-state
comparison through unchanged R2. All arms recalled `7/7` events, but O0 RGB was
best (`3` false segments, 82.35% event F1); O1/O2T/O3 produced `6/8/7` false
segments. The route-turn pair failed in all arms and O2T also failed the
static/dynamic background pair. Therefore no observation-increment effect is
claimed. The planned-route successor has since run through X24; consumed C0
still cannot authorize threshold tuning or promotion of CARLA flow to a
deployment input.

The static CODa ceiling adds causal curved-route and bounded vertical
occupancy for walls/barriers, fixed structures, and temporary obstacles. It
recalls all `12/12` observed path-contact events while reducing three-metre
proximity false alerts from 104 to 10 (`90.4%`) and clearing `4/4` eligible
events. The selected data contains no positive vegetation/head-clearance event;
GOOSE lacks the synchronized route/ground-clearance truth needed to turn tree
crown semantics into honest human head-collision truth, so that partition is
`NOT_EVALUABLE`. These remain privileged public-data algorithm ceilings, not
RGB/LiDAR detector, Android runtime, natural-distribution, user-benefit, or
safety evidence.

The detector-dropout residual route now separates static geometry from temporal
motion. R5 person masks and R6 calibrated RGB/raw-LiDAR metric occupancy each
recovered `0/9` induced windows; R6 is `NOT_EVALUABLE` because its static
matcher has no admissible closing signal on the nearly stationary wearer
window. R7-P then sealed a truth-blind, latest-past-pose raw-LiDAR BEV
occupancy-flow ledger without evaluator identity in temporal association. It
recovered `9/9`, proving a narrow spatiotemporal information effect, but raised
original false segments `12 -> 20` while event F1 remained 22.22%; global flow
route-risk was active in `123/143` frames. The terminal is therefore
`R7_P_CAUSAL_OCCUPANCY_FLOW_DEVELOPMENT_GATE_NOT_MET_NO_R8`: do not tune the
consumed cohort and do not open an RGB student from this result.
Read-only DTR-M0 attribution then separated the 20 false segments into 11
unchanged R2 inheritance, eight flow-new, and one flow-extended. All nine
flow-caused/modified entries were unsupported by scorer-side target linear
velocity: five were `STATIC_PSEUDO_MOTION`, while four moving-target cases had
large flow/target velocity disagreement and remain
`ATTRIBUTION_OR_FRAGMENTATION` without a proven split/merge identity. The next
eligible experiment is a fresh, frozen point-wise scene-flow or direct-velocity
source ceiling; route/lifecycle tuning and R8 remain closed.
That M1-O ceiling is now complete on the same JRDB window using causal
native-box piecewise-rigid point velocity over current raw LiDAR. It suppressed
`8/9` M0 diagnostic false entries but retained only `6/9` dropout recoveries,
produced five new motion-source-induced false segments, and reduced event F1 to
16.00%. The oncoming miss repeats across all three dropout durations and shows
that correct selective velocity is insufficient under the frozen hard
point/cell-to-tube aggregation. The terminal is
`DTR_M1_O_POINT_VELOCITY_ORACLE_CEILING_NOT_MET_CLOSE_SCENE_FLOW_ROUTE`:
TeFlow/DeltaFlow, R8, route forecasting, and consumed-cohort aggregation tuning
remain closed.
Read-only M2-D then tested whether the missing representation was only target
extent: the exact M1-attributed cells were compared with the same components'
current native OBBs translated continuously at the same robust M1 velocity.
All three repeated `pedestrian:35` misses were
`POINT_MISS_FOOTPRINT_MISS`; the closest footprint remained 0.0374 m outside
the 0.65 m route body. All five M1-induced/modified false segments were
`FOOTPRINT_HIT_TRUTH_NEGATIVE`. The terminal is therefore
`DTR_M2_D_EXTENT_GAP_NOT_SUPPORTED_NO_FRESH_M2_O`: do not launch fresh M2-O
from this extent-only hypothesis. The discrepancy between realized future
native extent and constant-velocity current-footprint geometry remains
localized but unresolved; it does not reopen forecasting or R8.
M3-D has now decomposed that discrepancy with realized future labels. All
three repeated `pedestrian:35` rows are
`EVAL_CIRCLE_HIT_REALIZED_OBB_MISS`: at first evaluator contact, the circle is
0.0209 m inside threshold while the realized OBB is 0.0374 m outside, with
only 0.0103 m realized-center versus M1-CV residual. Of the five M1 false
segments, three target-owned rows are realized-OBB forecast false positives
and two are other-component attribution errors. The terminal is
`DTR_M3_D_EVALUATOR_CIRCLE_OBB_SEMANTICS_MISMATCH_NO_FRESH_M3_O`.
Fresh M3-O, residual future occupancy, route forecasting, R8, and scene-flow
estimator work remain closed. The next decision is semantic, not a model run:
freeze whether the event means circularized proximity or oriented-body
contact; an OBB-contact claim requires evaluator revision and fresh rescoring.
C0 now selects oriented-body CONTACT as primary truth and retains circle-only
PROXIMITY as a simultaneous secondary label. Correctness is wearer-global: all
future native OBBs are unioned, while component identity is diagnostic only.
However, the consumed window is structurally saturated: overlapping p33, p34,
and p36 contacts make all `143/143` frames CONTACT, with zero bounded events
and 0.0 known non-CONTACT wearer minutes. Primary global recall/F1 and
false-segments/minute are therefore `NOT_EVALUABLE`; the descriptive 1.0
overlap scores are not algorithm success.
The legacy nine dropout rows become six OBB-CONTACT rows after excluding the
three p35 circle-only repetitions. R7-P and M1-O contribute the stressed target
in `6/6`; R2/R3-C global alerts occur in `5/6` but the target contributes in
`0/6`. This is consumed mechanism diagnosis only. The terminal is
`DTR_C0_GLOBAL_ORIENTED_RISK_CONTRACT_NOT_EVALUABLE_ALWAYS_CONTACT_WINDOW`.
A fresh global-OBB cohort must contain bounded CONTACT and known non-CONTACT
wearer time before any deployable motion-estimator work; forecasting and R8
remain closed.
C1 has now completed that metadata-only admission without reading RGB, LiDAR,
bags, detector output, or predictions. After excluding the entire consumed
Packard sequence, a lexicographic shortest-prefix rule freezes seven fresh
JRDB train sequences with 21 bounded CONTACT events, 21 unique-first-
responsible events, and 409.66 s known non-CONTACT wearer time across 8,368
frames. The cohort CONTACT duty cycle is 26.91%, including a retained zero-
CONTACT sequence with 96.92 s negative exposure. The decision is
`DTR_C1_FRESH_GLOBAL_OBB_COHORT_ADMITTED_METADATA_ONLY`: JRDB supplies valid
future replay denominators, but C1 is not an algorithm result. Raw-sensor
acquisition plus unchanged R2/R3-C/R7-P/M1-O replay is a separate next stage;
forecasting, R8, TeFlow/DeltaFlow, estimator competition, and training remain
closed.
That C2 replay is now complete. Across seven fresh sequences and 21 bounded
CONTACT events, R7-P recalled `20/21` with 90 false segments and 30.53% F1.
Confidence plus identity-free temporal consistency retained `20/21`, reduced
false segments to 38 (`-57.8%`), raised F1 to 50.63%, and retained 2.08 s median
lead. Confidence alone recovered `18/63` induced track gaps versus R7's `29/63`.
M1-CTB therefore permits raw dense motion only inside an observable bounded gap
of a previously tracked target; it restores `29/63` while keeping natural scores
identical to M1-CT. This is
`DTR_C2_M1_CTB_CONFIDENCE_TRACK_GAP_BRIDGE_FRESH_MECHANICS_SIGNAL`, not a full
R2 replacement: R2 still has 29 natural false segments and 57.14% F1. The next
source upgrade is deployable raw-point direct velocity behind the supported
confidence/bridge interface, not route-threshold tuning or trajectory
forecasting.
C3 has now executed that raw-point upgrade across the same seven frozen
sequences. Ego-compensated reciprocal 3-D voxel direct velocity retains `20/21`
CONTACT recall and recovers `52/63` induced track gaps, versus R7's `29/63`, but
its 85 natural false segments are too broad for independent alert creation. An
independent-history gate reduces false segments to 53 but recovery to `36/63`.
The selected M1-HYBRID therefore uses M1-CT normally and admits M1-PD plus R7
fallback only inside an observable prior-track gap. It keeps C2's 38 natural
false segments, 50.63% F1, and 2.08 s median lead while confirming at least
`52/63` gap recoveries. Decision:
`DTR_C3_M1_HYBRID_RAW_POINT_GAP_BRIDGE_FRESH_MECHANICS_SIGNAL`. Current-box
spatial attribution remains privileged; detector-independent attribution and
runtime integration are next, not threshold tuning or trajectory forecasting.

The detector-independent continuation has since established C11 as the current
fresh baseline: `17/20` CONTACT recall, 11 false segments, 70.83% event F1, and
1.455 s median lead. C14 recovered point-wise temporal position/velocity
residual covariance from the truth-blind R7-to-M1 ledger chain and propagated a
fixed eight-point cubature distribution through continuous route collision.
It raised recall to `18/20`, but also raised false segments to 16 and gained only
0.049 s median lead. The fixed Development gate failed, no remaining fresh
sequence was opened, and symmetric covariance spreading is closed rather than
rescued with threshold or covariance-scale tuning.

C15--C18 replaced that symmetric uncertainty with component hypotheses,
signed current/history velocity modes, route-entry consensus, and frozen-scale
three-frame confidence. Every arm retained C11's `17/20` recall. C16 improved
lead to `1.726 s` but raised false segments to 13; C17 reduced false segments to
8 and reached `75.56%` F1 but delayed median lead to `1.062 s`; C18 reached 9
false segments and `73.91%` F1 at `1.079 s` lead. No arm met the joint frozen
gate, so no algorithm-fresh sequence was opened.

C19--C21 then closed three attempts to extract a reliable mover decision from
the same LiDAR pseudo-flow. Fixed two-channel calibration reached `17/20`, 12
false segments, `69.39%` F1, and `1.268 s` lead; local position/velocity voting
was nearly universally supported and reproduced C16; global median scene-bias
subtraction reached `18/20` but raised false segments to 18 and reduced F1 to
`64.29%`. The remaining fresh cohort stayed sealed. The structural result is
that the false motion is locally coherent and cannot be removed by another
downstream fusion or one global background vector.

C22 changed the source on the consumed R7 canary: ego-rigid RGB point-track
residual confidence cut target-attributed false segments `20 -> 12`, exactly
removing the eight R7-added segments, while frozen-base critical recall remained
`3/3`. The visually admitted flow nevertheless recovered `0/9` induced dropout
windows and matched only `2/3` events, so the gate failed and no fresh cohort
opened. The cue is a strong pseudo-motion suppressor, but coarse stitched
cell-level projection is too sparse/time-misaligned for detector-independent
recovery. C23 must improve observation correspondence rather than tune C22:
track raw LiDAR-supported points in all five undistorted perspective cameras,
apply full ego SE(3) LiDAR-to-image compensation, subtract exact rigid
reprojection, and aggregate only afterward to BEV cells. Required C11 bag RGB,
dual-LiDAR, TF, timestamps, and calibration are already local. Threshold,
duration, route, lifecycle, score-fusion, local-vote, and global-bias tuning
remain closed.

C23--C24 then changed correspondence and motion representation rather than
tuning the route.  Object-rigid temporal confidence reduced R7 false segments
`20 -> 16` but exposed component-to-cell velocity broadcast as the remaining
structural error.  C24 replaced it with causal ego-compensated reciprocal 3-D
point motion plus independent three-frame consistency.  On the consumed
Packard Development canary it retained `3/3` CONTACT recall and `9/9` dropout
recovery while returning false segments `20 -> 12` and raising event F1
22.22% -> 31.58%.  This opened one fixed algorithm-fresh confirmation, not a
parameter sweep.

C25 has now adjudicated that confirmation on five previously unopened JRDB
sequences: 3,358 frames, 12 bounded global-OBB CONTACT events, and 130.98 s
known non-CONTACT time.  R7 reached `11/12`, 52 false segments, 29.33% F1,
4.200 s median lead, and `30/36` induced-dropout recovery.  Point-local M1-PD
kept `11/12`, cut false segments to 29, raised F1 to 42.31%, and recovered
`33/36`.  Hard three-frame M1-PDC reached `12/12`, 21 false segments, and 53.33%
F1, but delayed median lead to 1.624 s and recovered only `25/36` dropout
trials.  Therefore
`DTR_C25_POINT_FLOW_ALGORITHM_FRESH_GATE_NOT_MET_HARD_TEMPORAL_VETO`: the
point-wise source generalizes, but temporal consistency cannot remain a hard
evidence-deletion gate.  This cohort is consumed.  The next source-level
question is route-conditioned residual future occupancy over M1-PD with soft
confidence/uncertainty, preceded by one realized-future occupancy headroom
falsifier under the unchanged global-OBB route and lifecycle contract.

C26 has now run that falsifier without training.  Perfect realized future OBB
motion was permitted only for uniquely current-box-supported cells from the
sealed M1-PD ledger; ambiguous, unsupported, or right-censored cells kept their
original constant-velocity entry, and the combined signal used the unchanged
lifecycle.  The privileged oracle remained `11/12`, reduced false segments only
`29 -> 25`, and lowered median lead `3.067 -> 2.261 s`.  It therefore reached none of
the C25 componentwise best anchors (`12/12`, 21 false, 4.200 s).  Decision:
`DTR_C26_SUPPORTED_FUTURE_OCCUPANCY_HEADROOM_NOT_MET`.  A forecasting model on
the current M1-PD support is closed; the next representation must first provide
occlusion-persistent point support with soft confidence/age and explicit
`UNKNOWN`, without returning to component-wide velocity broadcast.

C27 has now falsified positive-support memory alone.  PDC-seeded,
PD-refreshed identity-free lineages retained `12/12` CONTACT recall, preserved
every PDC event's first-alert lead, and improved median lead to 2.967 s, but
produced 26 false segments versus PDC's 21 and recovered `27/36` induced gaps
versus R7's `30/36`.  Decision:
`DTR_C27_PERSISTENT_POINT_SUPPORT_DEVELOPMENT_GATE_NOT_MET`.  The missing input
is not another age or confidence setting: it is the causal reason a support is
absent.  C28 therefore adds raw-LiDAR `HIT / KNOWN_FREE / OCCLUDED / UNSENSED`
ray state so only occlusion may persist, known-free can clear departed ghosts,
and unsensed space remains `UNKNOWN`.

C28--C31 then produced a consumed-cohort Development win, but the single
authorized C31 source-disjoint confirmation did not transfer. One of seven
remaining JRDB sequences was structurally `NOT_EVALUABLE` before prediction
because its first frame had no current-or-past native pose. On the final six
sequences (4,811 frames, six CONTACT events, 18 dropout trials, and 278.08 s
known non-CONTACT), M1-PDC scored `4/6`, 25 false segments, 22.86% F1, 2.291 s
median lead, and `5/18`; frozen C31 scored `4/6`, 35 false, 17.78% F1, the same
lead, and `6/18`. Decision:
`DTR_C31_SOURCE_DISJOINT_CONFIRMATION_GATE_NOT_MET`. Do not tune C31 or open
C32 probabilistic body-route occupancy on the same component support; a
successor must first change component information/authority.

X0 has now selected that successor using read-only scorer-side attribution on
the already opened six-sequence truth. Huang-2 has correct early raw motion but
no frozen point/route entry (`ROUTE_GEOMETRY_MISS`); Huang-lane has no
responsible-object raw support in the `-3..0 s` window
(`NO_MOTION_SUPPORT`). PDC's 25 false segments are 14 `BAD_FLOW`, ten
`STATIC_PSEUDO_MOTION`, and one real noncritical mover. C31's ten
non-overlapping additions are two `BAD_FLOW` and eight
`STATIC_PSEUDO_MOTION`. Source errors therefore account for `34/35` false
units. Decision: `DTR_X0_MOTION_SOURCE_ATTRIBUTION_COMPLETE`; next compare the
current direct flow with exactly one stronger scene-flow source under the
unchanged scorer. Learned authority remains closed, while continuous geometry
is only a bounded Huang-2 canary.

X1--X3 then tested one independent Floxels-inspired five-scan source. The
two-scan-lag causal adapter retained localized positive headroom (three correct
frames, two correct route-entry frames), and the frozen 35-unit error slice
suppressed `25/34` source-error units. The sealed full six-sequence Development
replay did not transfer that local effect: lag-Floxel improved CONTACT recall
from `4/6` to `6/6`, median lead from `2.291 s` to `3.816 s`, and dropout
recovery from `5/18` to `8/18`, but false segments rose from 25 to 94 and F1
fell from 22.86% to 11.32%. Decision:
`DTR_X3_FULL_LAG_FLOXEL_GATE_NOT_MET`. The result includes one truth-blind,
fail-closed empty-support amendment at Huang-lane frame 4, bound by the sealed
evidence chain; it remains opened Development, not confirmation, deployment,
real-time, or safety evidence. Do not tune the source, route, thresholds, or
lifecycle against these outcomes.

X4 separately removed CUDA autograd and early stopping with a deterministic
CPU float64 rigid-cluster vote. Three cold runs produced identical canonical
array hashes and effect signatures, and suppressed `31/34` error units, but
recovered `0/19` positive frames and had `0/19` correct route-entry frames.
Its p95 source compute was `0.399--0.450 s` versus a `0.0696 s` median scan
period. Decision:
`DTR_X4_DETERMINISTIC_CLUSTER_VOTE_REPEATABILITY_GATE_NOT_MET`; close this
representation without a parameter sweep. The next DTR action is read-only
full-replay failure attribution on X3, not another scorer rescue.

That attribution is now complete. Of X3's 94 false segments, 66 are
`STATIC_PSEUDO_MOTION`, 14 are bad-flow magnitude, and one is a direction
reversal: source failures are `81/94` (86.17%). X3 resolved nine PDC false
segments but added 73 new ones; `65/73` incremental errors (89.04%) are still
static pseudo-motion or bad flow. Decision:
`DTR_X3_FULL_REPLAY_FAILURE_ATTRIBUTION_COMPLETE`. The next admissible source
must be `STATIC_AWARE_DIRECTION_CONSISTENT_SCENE_FLOW`; threshold, seed,
backbone, tracker, route, lifecycle, and scorer sweeps remain closed.

X5 tested whether reciprocal cycle agreement between causal overlapping
five-scan windows could supply that authority from the sealed X3 cells. On its
60-frame opened falsifier it suppressed `32/34` source-error units, but retained
only one correct positive frame and zero correct route-entry frames. Decision:
`DTR_X5_OVERLAP_CYCLE_SOURCE_FALSIFIER_GATE_NOT_MET`. Close both rigid-cluster
and same-source consistency-filter routes; the next source must add a static
world anchor or independently observable dynamic evidence, not filter the same
unsemantic geometric flow again.

X6 added the first such independent source: causal raw-LiDAR occupancy in world
coordinates under native ego pose. Its single 60-frame falsifier retained three
correct positive frames and two correct route-entry frames, suppressed `26/34`
source-error units, and met the one-scan-period compute check. Decision:
`DTR_X6_STATIC_WORLD_PERSISTENCE_FALSIFIER_GATE_MET`; this opened one frozen
full replay only.

X7 completed all 4,811 cohort timeline frames (4,787 source-supported plus 24
fail-closed causal warm-up frames). The anchor removed 340,734 of 642,217 X3
candidate cells and cut false segments `94 -> 72`, while preserving X3's `6/6`
CONTACT recall, `3.816 s` median lead, and `8/18` dropout recovery. Event F1
rose `11.32% -> 14.29%` but remained below PDC's `22.86%`; the false-segment,
F1, and below-PDC selectivity checks all failed. Decision:
`DTR_X7_FULL_STATIC_WORLD_ANCHOR_GATE_NOT_MET`. Retain the world anchor as a
useful component, close it as a standalone source, and require an independent
scene-motion observation next rather than a map-radius or scorer sweep.

X8 supplied synchronized stitched-RGB track residual as that independent
observation. Its one 60-frame falsifier retained three correct positive frames
and two correct route-entry frames, suppressed `27/34` source-error units, and
met the frame-local compute bound. Decision:
`DTR_X8_RGB_STATIC_VETO_FALSIFIER_GATE_MET`; this opened one frozen full replay.

X9 processed the complete X7 timeline and vetoed 62,757 of 301,483 cells. It
preserved `6/6` CONTACT recall, `3.816 s` aggregate median lead, and `8/18`
dropout recovery, while reducing false segments `72 -> 64` and raising F1
`14.29% -> 15.79%`. The false-segment maximum, F1 minimum, and below-PDC checks
still failed. Decision: `DTR_X9_FULL_RGB_STATIC_VETO_GATE_NOT_MET`. The two
independent static observations have additive but insufficient selectivity;
do not tune their constants on this opened cohort.

X10 added one fixed sequence-held-out learned motion-authority head on the
sealed X9 cells. Every fold trained on the other five sequences only and sealed
held-out probabilities before scoring. With a fixed L2 logistic model and
`0.5` threshold it retained 108,403/238,726 cells, cut false segments
`64 -> 47`, raised Event F1 `15.79% -> 20.34%`, preserved `6/6` CONTACT recall
and `8/18` dropout recovery, and kept `3.526 s` aggregate median lead. Decision:
`DTR_X10_CROSS_FITTED_MOTION_AUTHORITY_GATE_NOT_MET`. This is a real
cross-sequence learned-authority gain but remains above PDC's 25 false segments
and below the frozen 35% F1 floor. Close same-feature model/threshold changes;
the next source must add a new observable motion signal.

X11--X15 added that observable through calibrated raw and stitched RGB motion.
Raw multiview and CIWT positive authority closed; frame-local stitched RGB
dynamic birth also failed because correct motion did not survive to route
entry. Reusing the frozen R1 `0.50 s` clear grace as causal motion continuation
changed the result: the X14 canary met all checks, and the X15 six-sequence
replay reached `5/6` CONTACT, 18 false segments, 34.48% Event F1, `3.061 s`
median lead, and `2/18` dropout recovery. Accept
`DTR_X15_FULL_RGB_AUTHORIZED_CONTINUATION_GATE_NOT_MET`: this is the strongest
current Development mechanism, but it misses the false/F1 gates narrowly and
regresses dropout materially. Eleven of its 18 false ranges are concentrated
in memorial-court.

X16 then composed the frozen X10 held-out authority before the unchanged RGB
birth and continuation. It crossed the selectivity target with 15 false
segments and `3.630 s` lead, but recall fell to `4/6`, Event F1 to 32%, and
dropout remained `2/18`. Accept
`DTR_X16_CROSSFIT_RGB_AUTHORIZED_CONTINUATION_GATE_NOT_MET` and close this
over-strict composition without a model, threshold, or duration sweep. Retain
X15 as the active algorithm reference; the next representation must explain
memorial's persistent non-contact motion and restore detector-gap continuity,
not add another deletion filter.

Read-only residual replay confirms the failure layer. All 18 X15 false ranges
contain births authorized outside the route and later transported into it;
14/18 contain no same-frame risky birth at all. In memorial, seven of 11 ranges
are transport-only and four are mixed. Of 684 active false frames, 537 have raw
risk and 147 are lifecycle HOLD tails. The next observable therefore needs
ego-compensated movable-instance identity/masks across frames before birth;
shortening HOLD, merging ranges, or repeating cell-level motion closure would
not remove the originating transport cohorts.

X17--X21 then isolated persistence from admission. X17/X18 learned tracks gave
`14/18` dropout recovery but 33 false segments; X19's cleaner raw-X13 seed still
gave 31 because the live mask could absorb unrelated X7 components. X20 keyed
authority to exact `(class, track, X7 component)` ancestry and reached `5/6`
CONTACT, 16 false segments, 37.04% Event F1, and `3.785 s` lead, but retained
only `2/18` dropout recovery. X21 carries only the already authorized component
row through the same live track mask, with no current-X7 absorption. Its sealed
six-sequence replay reached **`5/6` CONTACT, 11 false segments, 45.45% Event
F1, `3.061 s` lead, and `8/18` dropout recovery**; all frozen checks passed.
Record `DTR_X21_TRACK_CARRIED_COMPONENT_ANCESTRY_GATE_MET` as Development
promotion and move the unchanged arm to source-disjoint confirmation. This is
not product, real-device, safety, or cross-source evidence.

L10 is active in parallel and does not depend on GRAIL owner orientation. SC1W
separates fresh semantic identity, DINO/motion continuity, and a RapidOCR CTC
word carrier for current-camera steering. SC2 adds opportunity-correct active
search: an incomplete OCR miss is `UNKNOWN + SEARCH`, with explicit
`SWEEP/SCAN/PAN/APPROACH/SIDESTEP/HOLD`, rather than false semantic absence or
terminal STOP. SC4 adds source-isolated, belief-latched OCR routing: CRAFT may
rescue cold-start acquisition, but after RapidOCR has acquired once it cannot
override LOST or navigate.

On video1+video10 Development, the broader SC3 word-scope source reached 88.84%
identity recall, 96.18% target support, 91.19% correct-direction coverage,
99.18% navigation precision, five wrong frames, and 30/30 end-to-end success.
On consumed video14, the final SC4 routing diagnostic rescued the two
primary-never-acquired goals: end-to-end rose 18/24 -> 24/24, identity recall
84.51% -> 95.77%, target support 88.73% -> 100%, with 100% navigation precision
and zero wrong identities. Fresh video16 independently showed the primary path
is strong on locally unique text: 96.15% recall, 100% support, 21/21 end-to-end,
100% navigation precision, and zero wrong identities. A broader per-frame SC3
fallback failed its video16 gate after one target-gap false identity; SC4 closes
that override route.

Fresh video17 then passed SC4's non-regression/safe-neutral gate but exposed the
next base-controller limit: repeated `Dairy/Milk` package instances reduced the
unchanged exact-track score to 48.23% identity recall, 61.11% end-to-end, and
53.26% navigation precision. SC5 now represents up to four source-latched
physical hypotheses with independent local/context DINO, OCR-context, motion,
and evidence-age state; appearance can propagate but cannot create identity or
navigate. On the consumed video17 diagnostic, a legal `SET_VALUED` reading kept
100% navigation precision and raised natural-frame navigation coverage from
73.45% to 81.42%. The exact-instance `53.26%` score is not a legal baseline for
that result: every video17 goal maps to multiple physical IDs, while the public
input supplies no reference, initial box, context anchor, or functional role.
Exact-instance and functional disambiguation are therefore `NOT_EVALUABLE`, not
a failed matcher.

The first 2x context fingerprint also did not beat the existing local crop
signal (`0.9297` versus `0.9408` pairwise AUC), so it is retained as state but
not promoted as the next information source. SC6 adds exactly one legal source:
`PublicInstanceBinding(goal text + public anchor frame + public anchor box/crop)`.
The public anchor may create exact-instance authority; OCR only admits
candidates; DINO/motion may associate, propagate, and reference-reacquire but
cannot create identity without the binding. Evaluator-native physical IDs stay
private.

On the fresh source-disjoint video18 cohort (two frozen bindings, six gap
episodes), stateless reference matching reached 86.49% exact-instance precision
and 57.14% coverage, with 15 wrong-instance frames and six wrong physical-ID
switches. The reference-bound temporal belief reached **100% precision, 91.67%
coverage (+34.53 pp), zero wrong-instance frames, zero wrong switches, 6/6 gap
reacquisition, 6/6 end-to-end, and zero authority violations**. This passes all
frozen SC6 gates. The consumed video17 diagnostic was reported but not used for
model or threshold selection. Frozen UNIQUE behavior remained identical on all
651 video16 and 546 video14 parity decisions. This is narrow public-reference
replay evidence; executed active observation, functional grounding, metric
arrival, live product benefit, user benefit, and safety remain unproven.

The parallel Named-POI functional-entrance branch now has a retained non-OCR
dual-family portal proposer, but PB1 did not close target-to-portal binding. A
shared candidate head with a permutation-invariant set summary was trained on a
20-building, building-disjoint public-image cohort. On six PB1-unseen
confirmation buildings it ranked truth at Top-3 `6/6` and rejected `30/30`
wrong-target pairings, but collapsed to `NONE` on all 36 episodes, reducing
correct `COMMIT / SET_VALUED` truth coverage from the selected support-ray
baseline's `4/6` to `0/6`. The corrected decision is
`L10_PB1_FRESH_BUILDING_GATE_NOT_MET_ALL_NONE_STOP_EMBEDDING_FUSION`; the
mechanical original pass was rejected as an evaluator-integrity defect. The
consumed cohort is closed to embedding, weight, threshold, backbone, and fusion
tuning. L10-AV0 remains blocked until a changed target-identity information
source or representation produces a correctly admitted portal on a new
building-disjoint cohort.

PB2-A has now changed that source rather than retuning PB1. A source-only audit
froze 12 PB1-disjoint public buildings with one reference plus facade,
entrance, side, and partial queries, split six Development / six confirmation.
Development selected fixed whole-image CLIP+DINO as the generic baseline and
SALAD as the specialized VPR challenger. On 24 confirmation queries SALAD
changed Recall@1 `9/24 -> 8/24`, Recall@3 `18/24 -> 20/24`, correct positive
acceptance `12/24 -> 8/24`, and wrong-building confirmation `23/120 -> 3/120`.
MixVPR accepted `16/24` positives but also `48/120` wrong buildings. Record
`L10_PB2A_SPECIALIZED_VPR_IDENTITY_GATE_NOT_MET_STOP_SINGLE_FRAME_APPEARANCE_ONLY`.
Do not sweep backbone, threshold, normalization, reference weighting, or fusion
on this consumed cohort. The next legal successor must add information such as
logo/OCR, map or POI metadata, coarse GPS, or genuinely ordered multi-view
evidence. PB2-B and L10-AV0 remain blocked.

The cross-paired `x/120` values are source-label negative proxies, not physical
target-absence authority: dense same-city images were not exhaustively labeled
for every co-visible gallery building. Physical target-absent rejection is
therefore `NOT_EVALUABLE`; the negative decision is unchanged because Recall@1
and target-present positive acceptance independently fail the gate.

PB3 changed to a public POI alias pack plus high-precision scene-text evidence
with an asymmetric join: a unique text proof may authorize its entity and veto
other requests; missing text remains `UNKNOWN` and falls back to the frozen
appearance decision. The new PB1/PB2-A-disjoint cohort contains four
Development and four test entities with one identity-bearing plus one context
query each, frozen by human source audit before any PB3 model call.

On test, Development-selected DINOv2 accepted all `8/8` positives but also
`23/24` source-label-negative wrong requests. Metadata-backed text produced
`5/5` correct proofs, and the asymmetric join kept positive acceptance at
`8/8` while reducing wrong accepts to `9/24` (60.87% relative). The frozen
coverage clause nevertheless failed: only `2/4` identity-bearing queries were
proved, below `3/4`. Record
`L10_PB3_METADATA_BACKED_TEXT_IDENTITY_BRANCH_GATE_NOT_MET`. Fixed target-blind
tiling added no proof. Retain the precision/veto mechanism only; do not promote
it to portal binding or active view, and do not tune aliases, OCR, thresholds,
or crops on the consumed source. The next legal increment is a fresh logo or
Chinese identity source, or a genuinely ordered execution-receipt-backed
`APPROACH_TEXT / SWEEP_SIGN` observation. PB2-B and general L10-AV0 remain
blocked.

PB4 changed the representation rather than reopening PB3. SCIL decomposes
official bilingual metadata into English-word, traditional/simplified-folded
Han, and public-mark carriers; distinctive atoms are weighted against the
frozen candidate pack, strong carrier disagreement becomes `UNKNOWN`, and a
proof authorizes only its own entity. A source-only audit rejected invalid
sign/directory/exhibit/wayfinding candidates before model access and froze four
Development plus four test entities, each with one identity-bearing and one
context view.

On fresh test, the Development-selected CLIP appearance arm reached Recall@1
`5/8`, positive acceptance `2/8`, identity-bearing acceptance `0/4`, and
source-label-negative wrong acceptance `1/24`. English canonical and flat
bilingual whole-alias controls produced `0/8` and `1/8` correct proofs. SCIL
produced **`6/6` correct proofs, zero wrong proofs, `4/4` identity-bearing
coverage**, and five Han/public-mark-participating proofs. The asymmetric join
raised positive acceptance to **`6/8`** and reduced the wrong-request proxy to
**`0/24`**. All six frozen gates passed:
`L10_PB4_SCRIPT_CONTRASTIVE_IDENTITY_LATTICE_GATE_MET`.

Retain SCIL as conditional positive place-identity authority. The next seam is
not another OCR/alias/threshold sweep: join only a current SCIL-proved frame to
the already frozen functional portal-set proposer. Text-`UNKNOWN` remains
blocked and requests `APPROACH_TEXT / SWEEP_SIGN`. No portal ownership,
active-view, navigation, arrival, user, product, or safety result is implied.

PB5 then joined the sealed SCIL proof to the unchanged dual-family portal
proposer through a typed entity authority token and a truth-blind structural
opportunity predicate. The audit froze all six proof-positive frames before
portal execution: three in-scope glass-door positives, two no-portal controls,
and one monumental open-entrance OOD challenge. Human labels never controlled
authorization.

Raw Top-1/Top-3 portal truth retention was `2/3` / `2/3`; the structurally
authorized set retained truth on only **`1/3`** positives and falsely authorized
**`1/2`** no-portal controls. The OOD entrance (`0/1` leakage), 18 wrong-request
pairs, two SCIL-`UNKNOWN` rows, and all `COMMIT`/guidance states remained
blocked. Record
`L10_PB5_SCRIPT_PROVED_AUTHORITY_CARRYING_PORTAL_LATTICE_GATE_NOT_MET`.

SCIL remains conditional identity authority, but geometry-only portal
authorization is closed on these consumed rows. Do not sweep its thresholds or
boxes. Change the portal-opportunity information source or representation and
confirm it on a source-disjoint cohort containing no-portal and open-aperture
controls before claiming portal ownership.

PB6 through PB10 have now executed that information-source ladder without
retuning PB5. Synthetic door semantics passed Development and failed fresh
(`0/4` positive truth Top-3); ADE20K door components did the same (`3/4` in
Development, `0/4` fresh). A typed Qwen2-VL grid retained truth on only `1/4`
and typed all four controls as doors, while a specialized door ontology retained
no candidate on any positive. Those sources are closed on their consumed rows.

The strongest retained signal is complementary rather than authoritative. A
fixed row-wise relative-depth aperture construction retained portal truth in
Top-3 on `4/4`; official Trans4Trans class-5 glass-door masks aligned to a
truth-member aperture on the same `4/4`. The join still fired on `1/2`
no-portal and `1/2` open-mouth controls (balanced accuracy `0.750`). A
threshold-free glass cut removed all control errors but collapsed positive
coverage to `1/4` (balanced accuracy `0.625`). Record
`L10_PB10_GLASS_DOOR_PLANE_AND_TOPOLOGICAL_CUT_DEVELOPMENT_GATE_NOT_MET`.

PB11 then froze metric relief behind a calibrated aperture-rim plane on eight
fresh SUN RGB-D capture sequences: four door planes, two NONE controls, and two
doorless openings with visible source-depth clear spans of `1.5381 m` and
`1.7735 m`. Source sensor depth made all `8/8` rows evaluable, but the minimum
door score was `0.7166` while the maximum control score was `0.9831`; strict
margin was `-0.2665` and ROC AUC `0.625`. The planar cabinet controls scored
like opaque doors, while the glass double door was the weakest positive.
Record `L10_PB11_METRIC_PORTAL_CLOSURE_P0_PRIVILEGED_GATE_NOT_MET`. This closes
metric rim-plane closure as door authority; per protocol, DepthART was not run
and no scorer or cohort parameter was tuned after output. The next admissible
representation is RGB door-part topology (leaf/frame/handle or hinge), with
depth used only after semantic topology exists.

PB12 tested that successor on another eight fresh SUN RGB-D capture sequences
with a source-disjoint detector trained on `door`, `handle`, `cabinet door`, and
`refrigerator door`. The frozen smallest-enclosing-parent rule rejected both
handled-furniture controls and both large doorless openings, but authorized
`0/4` real architectural doors: positive recall `0.0`, control false-positive
rate `0.0`, and balanced accuracy `0.500`. The model did expose real structure
(door parents on `2/4`, door handles on `2/4`, and coherent cabinet-door/handle
groups), but parent and child never formed an authorized real-door pair. Record
`L10_PB12_DOOR_PART_TOPOLOGY_DEVELOPMENT_GATE_NOT_MET`. Do not tune its boxes,
parent priority, confidence, crop, image size, or checkpoint on these rows. The
next admissible representation is a distinct pixel-level door-part source that
localizes leaf, frame, handle, and furniture-door competitors.

PB13 attempted that pixel representation with a pinned Florence-2-large-ft
snapshot, six pre-frozen referring-segmentation expressions, and a new eight-row
SUN RGB-D cohort spanning four source buckets. Its first formal launch stopped
before model load because the evaluator guard incorrectly compared the
protocol-frozen `USE_TF=0` with `1`; one unchanged mechanical replay was legal
because no cohort image or output had been consumed. On the replay, the first
frame's `operation_part` output contained a component that violated the frozen
minimum-three-vertex polygon contract. The process stopped after three model
calls, before any frame or aggregate metric was adjudicated. Record
`L10_PB13_FLORENCE_PIXEL_PART_TOPOLOGY_DEVELOPMENT_NOT_EVALUABLE_OUTPUT_CONTRACT_FAILURE`.
This is neither a positive nor negative quality result. The cohort is consumed:
do not repair the parser, prompts, beams, or topology on these pixels. PB14 must
use a new proposal/segmentation information source and a fresh cohort.

PB14 supplied that new source with a frozen YOLOE-26n-seg open-vocabulary mask
model, one full-frame parent lane, and one exact-parent-box child-rescale lane on
eight further fresh SUN RGB-D sequences. The execution was mechanically complete
in `9.60 s` with 17 fixed calls and `104,960,512` peak allocated CUDA bytes, but
the information source produced zero `architectural_leaf`, `operation_part`, or
`hinge` instances. It instead emitted two `closet_door` and six
`doorless_opening` parents on the positives; the nine parent crops still produced
no child instance. Consequently all four positives and all four controls remained
unauthorized: recall `0.0`, control false-positive rate `0.0`, and balanced
accuracy `0.500`. Record
`L10_PB14_YOLOE_MULTISCALE_PART_TOPOLOGY_DEVELOPMENT_GATE_NOT_MET`. This localizes
the failure before topology, at open-vocabulary semantic grounding. Do not tune
YOLOE prompts, thresholds, image size, parent ordering, crops, or mask assignment
on the consumed cohort. PB15 changes the information source to two-scale
GroundingDINO boxes plus SAM2.1 box-conditioned masks on another fresh cohort.

PB15 ran that frozen source on eight more source-disjoint SUN RGB-D captures.
After a pre-model, pre-RGB receipt-digest typo was corrected and all dependent
hashes were rebound, the valid run completed `195` GroundingDINO and `37` SAM2.1
calls in `90.90 s`, with `2,096,144,896` peak allocated CUDA bytes. It authorized
three of four architectural doors, one of two handled-furniture controls, and
zero of two large doorless openings: recall `0.75`, control false-positive rate
`0.25`, and balanced accuracy `0.750`. Record
`L10_PB15_GROUNDED_SAM_MULTISCALE_PART_TOPOLOGY_DEVELOPMENT_GATE_NOT_MET`.
The shared failure is upstream of topology: mutually exclusive part prompts
frequently produced the same nearly full-parent box, and SAM2.1 therefore made
nearly identical whole-object masks. Small parent-mask area differences then
created both the architectural false negative and furniture false positive, and
also contaminate the three apparent true positives. Do not rescue the consumed
cohort with GroundingDINO prompt, threshold, scale, cap, crop, SAM, priority, or
assignment changes. PB16 changes to native SAM 3.1 text-conditioned concept
instance masks on fresh source-disjoint pixels.

PB16 loaded the SAM 3.1 multiplex detector through a custom strict image-only
adapter assembled from Meta's multiplex recipe: all `1,166` detector keys
loaded with zero missing or unexpected keys, while no tracker or video predictor
was instantiated. This is not an official supported SAM 3.1 image API. The
public SAM 3 image builder had already failed a synthetic smoke because its
four-scale dual neck does not match the checkpoint's three-scale detector. A
separate first formal launch stopped before model load, RGB decoding, or output
when WDDM exposed `8,150` rather than the frozen `8,151` MiB; only the same-GPU
minimum-memory contract was corrected to `8,000` MiB.

The valid frozen run completed `88` model calls in `34.94 s`, peaking at
`4,685,969,920` allocated CUDA bytes. It produced `0/4` true positives, `0/4`
control false positives, and balanced accuracy `0.500`. Record
`L10_PB16_SAM3_NATIVE_PART_TOPOLOGY_DEVELOPMENT_GATE_NOT_MET`. The parent
concept `architectural door` returned zero native instances in all eight frames;
operation-part masks still appeared on three positives, furniture concepts were
active, and both control families remained clean. The route therefore failed
before topology at architectural-parent semantics. The cohort is consumed: do
not rescue it with concept wording, confidence, cap, assignment, adapter, or
topology changes. Any successor must change both supported weight/API and parent
semantic representation on fresh source-disjoint pixels.

PB17 then used the original `facebook/sam3` Safetensors weights through the
official Transformers `Sam3Model`/`Sam3Processor` image API and changed the
observable to direct nonempty native masks for the simple noun `door`, with no
mandatory operation part or parent-child assignment. A synthetic mechanical
smoke strictly loaded all weights and peaked at `3,928,569,856` allocated CUDA
bytes. Its eight formal capture sequences had zero overlap with all 48 PB11--PB16
sequences.

The one-shot run stopped on frame 1 after one image encode and one text call:
the mask tensor kept native spatial dimensions instead of the frozen source
size. Processor-only diagnosis verified the requested target was exactly the
source `[530,730]`. In the frozen Transformers implementation every positive
retained count is resized, while a zero count preserves native dimensions; the
error therefore mechanically establishes no `door` instance on the first fresh
positive at `0.5`. The required `4/4` positive gate is false. Record
`L10_PB17_OFFICIAL_SAM3_DOOR_STATE_DEVELOPMENT_GATE_NOT_MET`; aggregate balanced
accuracy remains unevaluable and is not reported. The cohort is consumed and
cannot be rerun with an empty-output-contract, prompt, threshold, processor, or
postprocessing repair. A successor must add a distinct observable source on
fresh pixels.

PB18 then returned to eight entity/file-disjoint real Named-POI images and
tested a Script-Proved Ingress-Connected Portal Graph. A correct localized SCIL
carrier could reach a portal proposal only through the same ADE20K host
component and walkable-ingress pixels. The source produced seven correct
identity proofs, zero wrong proofs, and one `UNKNOWN`; target portal Top-1/Top-3
was `1/6` / `4/6`. Both no-portal controls were rejected, but one tenant portal
was also authorized and neither tenant control achieved an exact target-hit plus
tenant-reject pair. Opportunity-balanced Top-3 accuracy was `0.7083`.

The terminal is
`L10_PB18_SCRIPT_PROVED_INGRESS_CONNECTED_PORTAL_GRAPH_GATE_NOT_MET`. Close the
opened graph to SCIL, semantic-class, connectivity, threshold, rank, proposer,
and fusion rescue. The admissible successor changes the information source to
an entity-owned OSM entrance node plus provider pose and explicit line-of-sight
occlusion on a new cohort; it cannot inherit arrival, access, traversability,
guidance, `HANDOFF_READY`, product, user-benefit, or safety authority from PB18.

PB19 implemented the external entity-owned entrance plus explicit occlusion
evaluator, but the Mapillary source stopped before any projection. The initial
primary direct stratum was marker-blind `NOT_EVALUABLE` on `4/4`. A source-only
repair screened 858 targets, 74 mechanically valid direct combinations, and 32
full panoramas. The first adjudicator retained four direct rows; an independent
second adjudicator rejected Markthal Rotterdam because its hall exposed
multiple walkable entrances, leaving only three rows with overlapping portal
intervals. Record
`L10_PB19_MAPILLARY_SOURCE_NOT_EVALUABLE_UNIQUE_PORTAL_TRUTH_3_OF_4`.
The 4/4 denominator is unchanged, no output marker was shown, and no bearing,
relative compass angle, predicted x, projection hit, or formal evaluator call
was made. Panoramax is the next admissible provider only after source-blind
orientation/provenance admission; this is not an algorithm negative.

Panoramax then opened a strict orientation allowlist: full uncropped 2:1
equirectangular pixels, matching sensor dimensions, owning-item true-north EXIF
heading agreeing with `view:azimuth`, and explicit zero/non-conflicting pose.
General provider projection remains closed outside that gate. A first 24-episode
PanoLab source still failed pixel admission at `0/12` across the tenant and
multi-entrance strata before any algorithm/oracle call, confirming that facade
pixels alone do not name the intended OSM entrance node.

A separate fresh four-episode active-ray cohort did establish causal action
value. Four distinct target ways and sequences used official reciprocal
adjacent images, with two non-target-building occlusions and two target-self
occlusions. The v2 replay achieved strict orientation `8/8`, correct initial
visibility roles `4/4`, reciprocal action receipts `4/4`, zero occluded-state
false authorizations, post-action ray authorization `4/4`, and active recovery
`4/4` (100%). Mean camera displacement was `9.394 m`; mean authority-count
change was `+1.0`. The earlier v1 result is retained as a source-serialization
defect: its sanitized items omitted frozen `prev`/`next` links, while the core
four recovery outcomes were already unchanged.

The terminal is
`L10_PANOLAB_ACTIVE_ENTRANCE_RAY_RECOVERY_DEVELOPMENT_GATE_MET`. This advances
the real active-observation mechanism: `SIDESTEP_TO_ENTRANCE_FACE` can change an
exact entity-owned entrance ray from geometry-blocked to geometry-authorized.
It establishes no pixel portal hit or ownership, public/legal access,
walkability, collision-free motion, arrival, `HANDOFF_READY`, product benefit,
user benefit, or safety. The next pixel-portal route needs independent entrance
identity evidence, not another matcher or threshold sweep.

That successor was tested through three genuinely different information
sources. Exact entrance-node text reached `1/8` multiview proofs over eight
nodes and 22 Panoramax views, with zero wrong proofs; whole-viewport
cross-sequence DINO reached only `1/4` unique correct portal bindings and made
`9/12` wrong-reference ray bindings. Both cohorts are consumed Development
evidence and are closed to OCR, matcher, threshold, viewport, portal-proposer,
or fusion rescue.

A third cohort changed the source to separately annotated exact portal patches
on three new target ways. All six images passed strict orientation and all three
pairs were cross-sequence, but independent source audit admitted only `1/3`
reference portals and `0/3` query portals. Joint pixel-source admission was
`0/3`, query truth was not created, and matcher calls were zero. Record
`L10_PANOLAB_EXACT_PORTAL_PATCH_SOURCE_NOT_EVALUABLE_REFERENCE_1_OF_3_QUERY_0_OF_3_NO_MATCHER_CALL`.
This was an asset/source ceiling, not an EfficientLoFTR negative.

The close-range source expansion has now consumed the five remaining eligible
ways from the 111-row local metadata pool. It froze ten strict-orientation
images into five cross-collection pairs at `7.290-22.319 m` from the exact OSM
entrance node (mean `13.997 m`). Reference-only source admission was `0/5`,
query-only admission was `1/5`, and joint admission was `0/5`; no matcher call
ran. Record
`L10_PANOLAB_CLOSE_PORTAL_SOURCE_NOT_EVALUABLE_REFERENCE_0_OF_5_QUERY_1_OF_5_JOINT_0_OF_5_NO_MATCHER_CALL`.
The same local rule now has zero unconsumed candidates. Do not resample it or
weaken unique-portal truth. A Panoramax continuation must add a genuinely new
independent collection for a near strict-DIRECT seed or discover new federated
coverage per collection, with metadata selection frozen before pixels.

That federated continuation has now run once and is closed. Five new-city
catalog searches plus the exact Panoramax web-viewer 5.2.0 effective-zero pose
contract scanned 215 frozen targets. Strict projection supplied 25
cross-collection targets, viewer-equivalent projection supplied 32, and global
geometry retained 18 DIRECT ways before freezing five episodes, with zero
search or geometry errors. Record
`L10_PANOLAB_FEDERATED_VIEWER_EQUIVALENT_PORTAL_METADATA_GATE_MET`. The disjoint
pixel audit then admitted `0/5` reference portals and `2/5` query portals, so
joint admission was `0/5` and matcher calls remained zero. Record
`L10_PANOLAB_FEDERATED_PORTAL_SOURCE_NOT_EVALUABLE_REFERENCE_0_OF_5_QUERY_2_OF_5_JOINT_0_OF_5_NO_MATCHER_CALL`.
Do not continue city mining or widen the point ray on this consumed cohort. The
next fresh source must freeze an independently measured entrance extent or
portal interval before pixels; Panoramax can remain a pixel provider but is not
the missing entrance-identity credential.

That independent-extent successor is now evaluated. A SAVeNoW/TUM LoD3
preflight found 61 unique door polygons and 74 Panoramax items across four
collections, but zero near, directionally usable door pairs. Minimum
camera-to-door distance was `68.866 m`, minimum directionally possible distance
was `175.052 m`, and pixel requests stayed at zero. Record
`L10_LOD3_PANORAMAX_DOOR_REACHABILITY_GATE_NOT_MET_ZERO_NEAR_VISIBLE_DOOR_PAIRS`.
The geographic pairing is closed; the LoD3 model's published `1-3 cm` relative
accuracy is not global Door-to-Panoramax registration.

The width-first route then froze an OSM-mapped entrance width and host-wall
tangent for three building-disjoint targets. Of 359 returned images, 194 were
perspective-projectable and 40 remained in field of view under the complete
declared camera-horizontal-accuracy circle; six images from six distinct
collections were frozen. Record
`L10_WIDTH_FIRST_PERSPECTIVE_PORTAL_METADATA_GATE_MET_3_OF_3_BUILDINGS_6_DISTINCT_COLLECTIONS`.
Reference-only and query-only portal admission were both `0/3`, so joint
admission was `0/3` and no matcher ran. Record
`L10_WIDTH_FIRST_PERSPECTIVE_PORTAL_SOURCE_NOT_EVALUABLE_REFERENCE_0_OF_3_QUERY_0_OF_3_JOINT_0_OF_3_NO_MATCHER_CALL`.
Mapped width improved metadata reachability but did not overcome `4-5 m`
camera-position uncertainty and grazing/off-facade views. Do not narrow the
declared error, reselect the consumed cohort, or tune the matcher. Generic
Panoramax mining is closed; the next fresh source must supply sub-metre
camera-to-portal registration or a directly posed portal mask/mesh, with
Panoramax retained only as the pixel carrier. This is not access,
traversability, arrival, `HANDOFF_READY`, product, user-benefit, or safety
evidence.

The directly posed successor has now met a frozen synthetic Development gate.
Hypersim scene `ai_034_001` supplied three trajectories of 100
semantic-instance masks and nine door instance IDs. Before RGB materialization,
the selector froze the first three door-disjoint instances satisfying fixed
mask-size/margin, cross-trajectory, and query-distractor rules. The single
replay lifted the reference contour from the provider world-position image and
projected it with the official scene projection matrix and held-out trajectory
pose; query RGB and query masks were evaluator-only.

The result was exact door Top-1 `3/3`, wrong-door commits `0/3`, centroid inside
target `3/3`, episode IoU `0.7063 / 0.4944 / 0.7528`, median IoU `0.7063`, and
mean centroid error `20.04 px` over `0.940-3.843 m` camera baselines. Record
`L10_HYPERSIM_POSED_PORTAL_TRANSFER_DEVELOPMENT_GATE_MET`. This proves the
posed-reference mechanism only on a privileged synthetic indoor ceiling. It
does not prove a doorway aperture, real/outdoor transfer, named-entrance
ownership, access, traversability, approach, arrival, `HANDOFF_READY`, product,
user-benefit, or safety. Real posed-door confirmation (prefer ScanNet++ when
access is available) is next; outdoor confirmation still requires same-domain
sub-metre camera-to-LoD3-door registration. Panoramax-only relative SfM is
parked because it cannot create that absolute portal anchor.

The first real posed RGB-D confirmation is now a frozen negative. SceneNN v1
found zero fully contained target-door frames across `9,000` poses. Its
pre-RGB-D v2 successor admitted clipped visible surfaces in three fresh scenes
and ran exactly one six-frame replay. Exact door selection was `2/3`, wrong
commits `1/3`, centroid inside `1/3`, median IoU `0.2647`, and nominal
metric-world centroid error `0.985` median. Record
`L10_SCENENN_REAL_RGBD_PARTIAL_METRIC_PORTAL_TRANSFER_CONFIRMATION_GATE_NOT_MET`.
SN01 showed why: mesh projection alone did not encode occlusion, so the
reference depth plane belonged to foreground content despite a low residual.
Do not tune the consumed frames, plane fit, or gates. The next real source must
add source-side visibility authority through a provider 2D instance mask or
z-buffered labelled mesh/depth consistency on a fresh cohort; generic Panoramax
and Panoramax-only SfM remain closed.

SC7 freezes a zero-OCR, provider-neutral reference-bound door-instance route on
Ego4D EgoTracks, but real execution is `NOT_EVALUABLE_CREDENTIALS_PENDING`
because the approved AWS profile is absent. That is a source-access condition,
not an algorithm negative.

SC8 advances the next functional-grounding layer without waiting for SC7. On one
real SceneFun3D Development scene, evaluator-provided functional masks supply
unlabeled proposals while ARKit boxes freeze the parent instance. Of ten public
task descriptions, six have a single parent binding and four remain
`NOT_EVALUABLE_PARENT_BINDING`. Nearest-part selection from the parent center
made legal commits on `4/6`, averaged 50% target-set recall, and selected two
wrong parts. The instance-bound task-relational selector reached **6/6 legal
commits, 100% target-set recall, zero wrong parts, and zero cross-parent identity
violations**. The parent center was outside the requested functional region in
all `6/6` tasks. Decision:
`SC8_TASK_RELATIONAL_FUNCTIONAL_BINDING_DEVELOPMENT_SIGNAL`.

This is proposal-conditional selection evidence. It does not establish RGB
functional-part proposal generation, reachability, orientation, approach pose,
arrival, completion, product benefit, user benefit, or safety.

SC9--SC11 then exercised real proposal generation behind the frozen SC8 seam on
the same opened scene. Generic-handle RGB with parent-surface lifting (SC9), RGB
ray triangulation (SC9T), and generic-handle RGB plus native depth (SC10) all
missed the source gate: proposal recall was at most 50%, and legal task commits
were at most `2/6`. Those routes are preserved negatives and are not eligible
for threshold, clustering, seed, or fusion rescue on this cohort.

SC11 changed the information source to task-grounded Grounding DINO regions,
then used native ARKit depth, the authorized parent box, and multi-view
consensus. Relative to its single-view arm, proposal recall improved from `8/10`
to `10/10`, legal task commits from `3/6` to `4/6`, mean target-set recall from
50% to 83.33%, and wrong parts from eight to two; proposal precision was 84.62%.
The provider was sealed before evaluator functional truth was loaded, and the
run executed on CUDA. Decision:
`SC11_TASK_GROUNDED_RGBD_MULTIVIEW_FUNCTIONAL_PROPOSAL_DEVELOPMENT_SIGNAL`.

This is one-scene, privileged-parent, posed-RGB-D Development evidence. Four
descriptions remain `NOT_EVALUABLE_PARENT_BINDING`, and two of six evaluable
tasks still fail because a wrong cluster survives inside the correct parent.
It is not open-vocabulary, phone-camera, reachability, orientation, arrival,
completion, product, user-benefit, or safety evidence.

SC12 and SC13 opened the next action-geometry layer on two source-disjoint
SceneFun3D scenes. Task-signed parent geometry regressed (`7/9 -> 6/9` signed
translation-direction hits), while a decoupled local contact-surface normal
retained `4/6` hits and improved mean error only 30.39 -> 28.18 degrees. These
static routes are terminal negatives on their consumed scenes; axis, PCA,
threshold, seed, and fusion sweeps are forbidden.

SC14 replaces static inference with causal paired functional-point motion. On
scene 421013, motion-type correctness improved `7/8 -> 8/8`, translation
direction hits `4/7 -> 7/7`, and mean direction error 38.89 -> 0.91 degrees.
The one rotational action passed its axis gate and had 2.22 cm pivot-line error.
Decision: `SC14_CAUSAL_MICRO_MOTION_ACTION_BELIEF_MECHANICS_SIGNAL`.

The positive is only a simulated paired-point mechanics ceiling generated from
SceneFun3D motion truth. Correspondences, perturbation, and safety are
privileged; no real user action or natural RGB-D motion was observed. It does
not establish RGB tracking, safe probing, reachability, body orientation,
arrival, `HANDOFF_READY`, user completion, product benefit, or safety.

The action representation has nevertheless landed in `core:assist`. Runtime
paired 3-D correspondences now produce a fail-closed
`UNKNOWN / SET_VALUED / LOCKED` action belief with source, identity, frame,
clock, and freshness admission. `HANDOFF_READY` additionally requires current
position, visibility, grounding, orientation, and reachability to be `READY`;
completion still requires explicit user confirmation. This closes a software
bypass but creates no new live-source, arrival, product, or safety evidence.

The front half of the loop is now a core runtime contract as well. The ported
SC1W/SC2 controller separates fresh semantic identity, bounded appearance
continuity, and carrier bearing; emits explicit seek/observe/guide actions;
requires two fresh hits after LOST; and revokes stale handoff readiness. Its
source Development artifact records `124 -> 0` terminal STOP frames,
`24 -> 0` target-present false-NONE frames, 100% action coverage on
non-navigation frames, and `30/30` end-to-end replay episodes. Those are replay
decision metrics, not evidence that a physical camera/user action improves the
next observation.

SC15--SC17 tested whether that missing observation effect could be recovered
cheaply. Passive direction alignment on ArTVideo failed (`0.6667` aligned mean
semantic gain versus `0.8000` opposed; 16.67% versus 0% wrong-gate creation).
A fresh protocol-frozen DSText V2 indoor run then produced 2,824 OCR
observations and 2,677 natural transitions, but a Pareto height/sharpness/
centering improvement yielded only `+0.0093` semantic-gain delta and worse gate
crossing. Fixed three-view exact consensus raised precision only +0.91 pp and
still formed wrong consensus on 5/49 tracks. These are terminal negatives for
the opened proxy representations, not evidence against active observation
itself. The next source must record issued action -> before/after result so the
controller can repair policy from causal outcome.

SC18 implements that receipt seam in `core:assist`: observation actions are
bound to goal/session/entity/frame/clock and the pre-action semantic deficit;
only later comparable admitted evidence may report improvement, no gain, or
contradiction, and only after a matching execution acknowledgement; an issued-
only prompt, missing observation, or invalid evidence remains `UNKNOWN`. A
no-gain receipt changes the next repeated action. In a frozen 250-episode controlled
world, repeated no-gain actions fell `182/206 -> 0/215`, task success stayed
88.5%, absent false completion stayed 0/50, and reacquisition changed 82.35% ->
80.75%. This is a synthetic mechanism signal, not live executed-view evidence.
The next active increment is deficit-conditioned action utility from real
issued-action receipts, not a sweep of the fixed repair rule or old proxies.

That algorithmic increment is now implemented as SC19. A reusable contextual
UCB policy learns separate action utility for each evidence-deficit/bearing
context, updates only from execution-confirmed comparable outcomes, ignores
`UNKNOWN`, deduplicates receipts, and keeps unsafe actions outside the candidate
set. On a frozen authored seven-context mechanism world, improvement increased
40.10% -> 62.97%, expected regret fell 93.26%, final-window optimal selection
reached 99.52%, and ambiguous-context approach fell 100% -> 0. This validates
the online learning mechanics only. Real promotion still requires live phone
actions producing the same confirmed receipts without relaxing safety sets.

With device/demo integration paused, SC20 moves to the endpoint algorithm
instead. On the consumed SceneFun3D 420683 posed RGB-D trajectory, replacing a
centered-large-parent proxy with a factorized task-functional endpoint observer
reduced false ready frames `300 -> 1`, increased precision `30.07% -> 99.17%`,
and increased F1 `0.420 -> 0.784`. It retained true-ready coverage on only five
of six tasks versus six of six for the proxy, so the frozen gate remains not
met. Reachability is absent and explicitly `UNKNOWN`; no `HANDOFF_READY`, real
arrival, product, user-benefit, or safety claim follows.

SC21 adds a parent-normalized connected-component integrity layer before the
endpoint join. Its frozen two-scene fresh source was `NOT_EVALUABLE` (`3/20`
parent-bound tasks, zero integrity opportunities), but an unchanged read-only
SC11 diagnostic removed the 1.445 m isolated under-bed candidate, improving
legal functional commits `4/6 -> 5/6` and wrong parts `2 -> 1` without recall
loss. SC22 then composed the fixed SC21 rule with SC20. On the consumed real
RGB-D trajectory it restored endpoint task coverage `5/6 -> 6/6`, improved
recall 64.86% -> 83.24% and F1 0.784 -> 0.906, while retaining one false-ready
frame and above-99% precision. This is a composition-mechanics signal, not
fresh transfer; reachability remains `UNKNOWN` and handoff remains forbidden.

SC23--SC29 tested transfer of candidate integrity on three successive
source-disjoint, outcome-blind SceneFun3D cohorts. Pure topology failed fresh
SC23 because same-parent functions can be spatially separated. Adding an exact
task-action semantic witness produced a consumed mechanism gain (`13 -> 16`
legal commits, `49 -> 41` wrong parts, 90.54% -> 93.24% recall), but fresh SC25
had only two semantic opportunities and was `NOT_EVALUABLE`. Hierarchical
action families increased coverage but fresh SC27 traded `32 -> 27` wrong parts
for 83.33% -> 79.17% recall. A redundancy gate removed that regression on the
consumed cohort while retaining `32 -> 30` wrong-part reduction. Fresh SC29 had
29 evaluable tasks but zero candidate-changing cross-family conflicts, so its
identical arms do not confirm the algorithm. The next source must admit by
same-parent cross-action-family conflict opportunity, not generic multi-target
count; all current semantics remain privileged proposal ceilings, reachability
is `UNKNOWN`, and handoff is forbidden.

SC30 replaced generic multi-target source admission with provider-public
conflict admission: functional proposals are grouped by parent and frozen
action family without reading task target IDs or selector outputs. A 20-scene
roster found only one eligible scene; an unchanged-threshold V2 expanded the
roster ceiling and admitted three scenes after 32 candidates. SC31 then produced
the first fresh proposal-conditional semantic-integrity signal: on 16 evaluable
tasks, legal commits improved `3 -> 4`, wrong parts fell `24 -> 20`, mean recall
stayed 62.5%, and cross-parent violations remained zero. The aggregate result
contains one improved printer-unplug task and one regressed repeated-outlet
task, so task-wise no-regret is not established. The next representation must
model target-conditioned ordinal structure inside repeated same-action controls;
candidate semantics remain privileged and reachability remains `UNKNOWN`.

SC32--SC34 now isolate that ordinal gap. A frozen parent-boundary axis found no
eligible source in 80 candidates because functional controls can be bound to a
larger support OBB rather than their physical carrier. The successor therefore
keeps both PCA polarities and commits only a reversal-invariant rank. On the
fresh `421658` three-outlet cohort, the middle-rank rule changed `second outlet`
from empty/zero-recall to a unique correct commit: legal commits `0 -> 1`, mean
recall `66.67% -> 100%`, zero task-wise regressions. Wrong parts remained `4`,
so SC34 is formally `GATE_NOT_MET` under its frozen composite gate despite the
fresh false-negative-recovery effect. Endpoint ranks remain set-valued pending
an authorized orientation observation; no RGB, reachability, arrival, handoff,
product, or safety claim follows.

SC35--SC39 next sought real-view polarity while keeping demo/device integration
paused. Fresh `422155` exposed low-resolution camera poses but no admissible
parent, contextual, or self-carrier ordinal lattice, so its source outcomes are
`NOT_EVALUABLE`. On target-exposed `422200`, a stable temporal camera order
proved reverse-side and swapped the second/third outlets in SC38, reducing mean
recall `100% -> 33.33%` with two task-wise regressions. SC39 added frame-aligned
depth visibility and public ordinal-inventory offset `[2,3,4]`. All 22 in-frame
views were depth-consistent for all candidates (best maximum residual 7.48 mm),
but only four satisfied the frozen horizontal span and zero met the five-pose
temporal consensus gate. The terminal is
`SC39_NOT_EVALUABLE_NO_DEPTH_VISIBLE_ACTIVE_VIEW`, not a negative. Reuse the
visibility/rank representations only on a separately admitted sufficiently
observed source; do not tune this cohort or claim RGB, reachability, arrival,
handoff, product, or safety effects.

SC40 kept that algorithm fixed and exhausted the 128-scene unopened suffix.
There were zero public directional ordinal `plug_in` tasks, so no point-cloud,
depth, backend, truth, or evaluator stage opened. The result is
`SC40_NOT_EVALUABLE_NO_FRESH_DEPTH_VISIBLE_ACTIVE_VIEW`: the current official
SceneFun3D roster is closed as a fresh confirmation source for this contract,
while the representation remains eligible on a genuinely new synchronized
repeated-control source.

SC41 moved to the fresh official SWITCH terminal-verification source and tested
whether goal-conditioned incomplete-state contrast plus current-video
similarity improved desired-state CLIP. It did not: the baseline was `9/16`, the
successor `3/16`, with seven regressions. The same-scene term dominated score
variation but its argmax was correct on `0/16`, establishing that raw appearance
persistence is anti-causal when task completion should visibly change state.
The terminal is `SC41_SWITCH_CAUSAL_TERMINAL_VERIFICATION_GATE_NOT_MET`. This
opened cohort is closed to tuning; the next terminal-state route must encode an
explicit localized before/after effect and use a new unexposed cohort. No demo,
device, navigation, reachability, action-execution, arrival, product, or safety
claim is added.

SC42 introduced an explicit Goal-Localized Effect Axis on 64 fresh real OSCaR
`open/close/on/off` clips. It moved static desired-state selection from `25/64`
to `27/64`, but missed its gate. The source audit also found that OSCaR Frame 3
is only a temporal endpoint and may precede a visible requested effect, so this
is not completion truth and the remaining roster is not an authorized arrival
confirmation source.

SC43 moved to 95 fresh SWITCH immediate-final-state tasks and tested a local
Qwen2-VL-2B `do(action) - do(no-action)` logit contrast. It moved `19/95` to
`24/95` but predicted A on all 95 rows, exposing label-prior collapse rather
than visual causal reasoning. The terminal is
`SC43_CAUSAL_INTERVENTION_LOGIT_CONTRAST_GATE_NOT_MET`; its shared-ID modality
variants and prompt/token tweaks are closed. The next endpoint route must
visibly ground actuator, effect carrier, before state, after state, and conflict
before deterministic completion reduction. No demo or product integration is
authorized by SC42/SC43.

SC44 admitted the public ungated RoboPulse hard subset as a genuinely new signed
progress source. A source-stratified hash froze 90 rows across nine manipulation
sources, each with start/end task anchors and BEFORE/AFTER front plus wrist views.
The new deterministic reducer consumed five explicit factor margins, but all five
Qwen2-VL factors were positive on all `90/90` rows. Baseline balanced accuracy was
`49.90%`; the reducer was `50.00%` and predicted progress on every row. The
terminal is `SC44_GOAL_ANCHORED_FACTOR_REDUCER_GATE_NOT_MET`. Prompt-exposed
factors are now closed; the legal successor needs a learned differential factor
representation, calibrated `UNKNOWN`, disjoint supervision, and a new frozen
evaluation cohort. No demo or product integration is authorized by SC44.

SC45 delivered that learned representation on a fully disjoint role split after
excluding every SC44 ID. Six RoboPulse sources trained a 32-dimensional
multi-view Progress Factor Tensor, `droid_oxe` calibrated a split-conformal
`UNKNOWN`, and 380 fresh `human_pika`/`libero_data` rows remained evaluation-only
until provider seal. The untrained goal-axis baseline was `61.58%` balanced
accuracy. SC45 reached `65.96%` selective balanced accuracy at `93.95%` known
coverage, a `+3.47` point gain over the baseline on the same known rows. This is
the first content-sensitive endpoint gain after SC42-SC44, but it missed the
frozen `+5` point gate and is recorded as
`SC45_LEARNED_PROGRESS_FACTOR_TENSOR_GATE_NOT_MET`. The residual is source-
specific (`human_pika` `59.78%`, `libero_data` `71.91%`); the next representation
must localize hands and the physical effect carrier rather than retune this
global tensor. No demo or product integration is authorized by SC45.

SC46 made that representation change on the fresh Guardian UR5-Fail source-
defined `400/30/140` train/validation/test execution splits. GroundingDINO
localized task entities plus the robot gripper in three start/end viewpoints;
CLIP encoded global, task-region, and joint interaction-region state changes;
a frozen 80-dimensional factor tensor and split-conformal reducer retained 33
`UNKNOWN`s. The global baseline was `49.85%` balanced accuracy. SC46 reached
`59.94%` selective balanced accuracy at `76.43%` coverage, a real `+10.15`
point gain over the baseline on the same 107 known rows, but missed the frozen
`70%` absolute gate and `60%` per-class recall floor (failure recall `57.14%`).
Record `SC46_GUARDIAN_LOCAL_EFFECT_CARRIER_TENSOR_GATE_NOT_MET`. Localization
was present on all 140 test rows; the residual is semantic state discrimination,
especially `translation_object`, not box availability. Do not retune the opened
split. The next legal successor needs an explicit object-state/change carrier
or contact/release representation on another fresh cohort. No demo or product
integration is authorized by SC46.

SC47 tested whether training-only robot phase signals could supply the missing
state semantics without becoming runtime inputs. On the fresh DROID-OOD
`200/50/100` train/calibration/evaluation roles, CLIP video features distilled
dense pseudo-progress (`R2=0.62`), gripper closedness (`0.73`), and vertical
displacement (`0.75`) on calibration, while speed and gripper-opening rate were
not learnable. The learned visual baseline reached `61.61%` balanced accuracy.
The conformal successor reached `63.22%` selective balanced accuracy at `61%`
coverage, a `+9.11` point gain over the baseline on the same 61 known rows, but
missed coverage, absolute-accuracy, and per-class gates; success recall was only
`32%`. Record `SC47_DROID_OOD_PRIVILEGED_PHASE_DISTILLATION_GATE_NOT_MET`.
Process phase is visually recoverable, but it is not terminal object-state
evidence. Do not tune the consumed validation split. The next legal successor
needs direct, object-bound final-state supervision on a new cohort. No demo or
product integration is authorized by SC47.

The named-destination front half now has a zero-OCR public-reference falsifier.
Across six source-audited Hong Kong entities and 11 frozen evaluation images,
CLIP name-only and global CLIP+DINO references each retrieved `6/11` correct
entities. Adding mutual DINO patch matches plus affine consistency kept `6/11`
but reduced wrong-goal confirmations from `3/55` to `0/55` and raised balanced
accuracy from `65.45%` to `68.18%`. The predeclared top-1 gate failed, so record
`NAMED_POI_FACADE_FINGERPRINT_DO_NOT_TUNE_LOCALIZE_REFERENCE_COVERAGE_GAP`.
Local geometry is a useful non-OCR veto but not yet a sufficient finder. The
next source change is a richer facade/entrance/sign/context reference bank, not
weight or threshold tuning on this opened cohort. No entrance, navigation,
metric arrival, product, user-benefit, or safety claim follows.

That source change has now materialized 50 prior-file-disjoint Commons facets
and a second 29-image fresh source. Filename-derived entrance truth was rejected
as `NOT_EVALUABLE` after all three alleged false-ready frames visibly contained
stairs, escalators, or passage structures. On 18 fresh human-labelled views,
scene identity retrieved `12/18`, confirmed seven, and produced zero wrong-goal
confirmations. Identity-only readiness had `2/8` true ready and five false
ready; adding the frozen CLIP/GroundingDINO entrance graph suppressed every
ready decision, including all eight positives. Although the entrance graph
score retained `0.725` AUC, the scene-level identity-AND-entrance join failed.
Record `NAMED_POI_SCENE_LEVEL_IDENTITY_AND_ENTRANCE_GATE_NOT_MET`; do not tune
the opened sources.

A target-local binding reducer now has fresh real-image evidence. On 20 third-
batch public images, GroundingDINO proposed a truth-overlapping candidate for
all `4/4` human-boxed entrances. The generic highest-score rule nevertheless
made zero correct and 20 false commits. Proposal-context CLIP+DINO binding cut
false commits to five but also made zero correct unique commits; it is retained
only as a fail-closed filter. Record
`NAMED_POI_TARGET_LOCAL_CROP_BINDING_SAFETY_FILTER_ONLY_NO_CORRECT_COMMIT`.
The bottleneck is no longer proposal availability but transporting POI identity
to the correct proposal. The next source/representation is a reciprocal patch-
match target-support field joined spatially to entrance proposals, not OCR-only
recognition or tuning this consumed cohort. No entrance, navigation, metric
arrival, product, user-benefit, or safety claim follows.

That support branch has now reached its single-frame ceiling. A coarse 9x9
field failed; native 16x16 downward support rays reduced false commits on a
25-image fresh batch from 24 to nine while preserving one correct commit. A
source audit then refused tenant entrances as target truth and expanded Commons
pagination from 50 to 500 files. On the resulting 32-image V7 batch, a
multi-facet reference bank plus unchanged rays preserved `2/6` correct unique
commits, reduced false commits `30 -> 7`, raised precision
`6.25% -> 22.22%`, and retained every oracle-available true entrance (`4/6`)
in `COMMIT / SET_VALUED`.

The frozen terminal is
`NAMED_POI_MULTIFACET_SUPPORT_RAY_PROPOSAL_CEILING_REACHED_SINGLE_FRAME_UNIQUENESS_NOT_MET`:
retain the ray as a candidate-preserving veto, not a finished locator. The
active entrance belief now converts ambiguity into `CENTER_AND_APPROACH`,
requires two consecutive bound views before commit, and enters last-bearing
reacquisition after loss. Temporal real-input evidence remains pending; no
public access, guidance, arrival, product, user-benefit, or safety claim follows.

## Demonstration track

Semantic Anchor to Marker Pose remains the live-device showcase closure. Its
device evidence and DTR-R0 research evidence must be reported separately. It is
not the current work priority and receives no new integration in SC20.

## Boundaries

- Historical routes remain closed unless a new versioned experiment changes
  the task representation or introduces a genuinely new information source.
- Curated Development evidence may support the showcase, but its scope must be
  stated.
- Live-camera behavior requires a ready device and a device run; a build or JVM
  test is not a substitute.
- `UNKNOWN` and `NOT_EVALUABLE` are not negative evidence or proof of safety.
- Protected final claims follow [formal governance](formal/RESEARCH_GOVERNANCE.md).
