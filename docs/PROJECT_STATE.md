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
