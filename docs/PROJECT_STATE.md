# Project state

Updated: 2026-08-28

BlindAssist is a runnable Android showcase research prototype. The default
research policy is effect-first: demonstrate a genuine, visible effect in a
controlled setup, report the setup and metric honestly, and add only checks
that protect interpretation.

## Current operating surface

- Current ten-meter route: [L10-R0 Goal-Lock Copilot](../research/active/l10-r0/README.md)
- Current obstacle/risk route: [Dynamic Travel Risk R2](../research/active/dtr-r0/README.md)
- Current question and stop condition: [CURRENT_DECISION.md](CURRENT_DECISION.md)
- Workstation entrypoint: `tools/ba.ps1`
- Android entrypoint: `scripts/run_android_gradle.ps1`
- Closed experiment lookup: [history-index.md](history-index.md)

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

## Demonstration track

Semantic Anchor to Marker Pose remains the live-device showcase closure. Its
device evidence and DTR-R0 research evidence must be reported separately.

## Boundaries

- Historical routes remain closed unless a new versioned experiment changes
  the task representation or introduces a genuinely new information source.
- Curated Development evidence may support the showcase, but its scope must be
  stated.
- Live-camera behavior requires a ready device and a device run; a build or JVM
  test is not a substitute.
- `UNKNOWN` and `NOT_EVALUABLE` are not negative evidence or proof of safety.
- Protected final claims follow [formal governance](formal/RESEARCH_GOVERNANCE.md).
