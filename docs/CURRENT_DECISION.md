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
DTR_C3_M1_HYBRID_RAW_POINT_GAP_BRIDGE_FRESH_MECHANICS_SIGNAL /
DTR_C11_ROUTE_REGION_OCCUPANCY_FRESH_SIGNAL /
DTR_C14_STOCHASTIC_ROUTE_CONFLICT_DEVELOPMENT_GATE_NOT_MET /
DTR_C15_COMPONENT_VELOCITY_MIXTURE_DEVELOPMENT_GATE_NOT_MET /
DTR_C16_EMPIRICAL_VELOCITY_MODES_DEVELOPMENT_GATE_NOT_MET /
DTR_C17_TEMPORAL_ROUTE_CONSENSUS_DEVELOPMENT_GATE_NOT_MET /
DTR_C18_THREE_FRAME_MOTION_CONFIDENCE_DEVELOPMENT_GATE_NOT_MET /
DTR_C19_JOINT_MOTION_CONFIDENCE_DEVELOPMENT_GATE_NOT_MET /
DTR_C20_LOCAL_MOTION_VOTING_DEVELOPMENT_GATE_NOT_MET /
DTR_C21_SCENE_BIAS_RESIDUAL_MOTION_DEVELOPMENT_GATE_NOT_MET /
DTR_C22_EGO_RIGID_VISUAL_MOTION_DEVELOPMENT_GATE_NOT_MET /
DTR_C27_PERSISTENT_POINT_SUPPORT_DEVELOPMENT_GATE_NOT_MET /
DTR_X3_FULL_LAG_FLOXEL_GATE_NOT_MET /
DTR_X3_FULL_REPLAY_FAILURE_ATTRIBUTION_COMPLETE /
DTR_X4_DETERMINISTIC_CLUSTER_VOTE_REPEATABILITY_GATE_NOT_MET /
DTR_X5_OVERLAP_CYCLE_SOURCE_FALSIFIER_GATE_NOT_MET /
DTR_X6_STATIC_WORLD_PERSISTENCE_FALSIFIER_GATE_MET /
DTR_X7_FULL_STATIC_WORLD_ANCHOR_GATE_NOT_MET /
DTR_X8_RGB_STATIC_VETO_FALSIFIER_GATE_MET /
DTR_X9_FULL_RGB_STATIC_VETO_GATE_NOT_MET /
DTR_X10_CROSS_FITTED_MOTION_AUTHORITY_GATE_NOT_MET /
DTR_X11_RAW_MULTIVIEW_STATIC_VETO_FALSIFIER_GATE_NOT_MET /
DTR_X12_CIWT_TRACK_MOTION_AGREEMENT_FALSIFIER_GATE_NOT_MET /
DTR_X13_STITCHED_DYNAMIC_BIRTH_AUTHORITY_FALSIFIER_GATE_NOT_MET /
DTR_X14_RGB_AUTHORIZED_MOTION_CONTINUATION_FALSIFIER_GATE_MET /
DTR_X15_FULL_RGB_AUTHORIZED_CONTINUATION_GATE_NOT_MET /
DTR_X16_CROSSFIT_RGB_AUTHORIZED_CONTINUATION_GATE_NOT_MET /
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

SC35--SC39 attempted to supply that polarity authority from real camera poses
without touching the demo path. Fresh scene `422155` was source-inadequate: the
low-resolution trajectory was available, but its functional controls had no
admissible parent/context/self-carrier ordinal lattice. Those source outcomes
are `NOT_EVALUABLE`, not algorithm negatives.

On target-exposed `422200`, SC37's `5/5` temporal image order was stable yet
reverse-side. SC38 consequently swapped the second and third outlets and
regressed two of three tasks (`100% -> 33.33%` mean recall), proving that camera
x-order without visibility cannot authorize linguistic left/right. SC39 then
added frame-aligned measured depth plus the provider-public ordinal inventory
`[2,3,4]`: all 22 real-image in-frame views agreed with candidate depth, with a
best maximum residual of 7.48 mm, but only four views had the frozen horizontal
span and none reached the frozen five-pose temporal support. Record
`SC39_NOT_EVALUABLE_NO_DEPTH_VISIBLE_ACTIVE_VIEW`; do not lower or sweep its
thresholds, and do not run an evaluator on an unadmitted source. The next legal
source must supply a depth-visible repeated-control lattice with sufficient
natural observation support. Visibility and hidden-prefix rank mapping may be
reused unchanged; fresh transfer, RGB, reachability, arrival, and handoff remain
unestablished.

SC40 then scanned the entire previously unopened official suffix without
changing the parsed SC39 algorithm object. Across all 128 remaining one-video
scenes, zero had a public directional ordinal `plug_in` task, so point-cloud,
depth, backend, target, and evaluator stages never opened. Record
`SC40_NOT_EVALUABLE_NO_FRESH_DEPTH_VISIBLE_ACTIVE_VIEW`. The SceneFun3D
one-video roster is exhausted for fresh confirmation of this exact contract;
do not widen its parser or substitute labels after exposure. Further ordinal
work requires a new information source with repeated controls, directional rank
language, synchronized depth, and sufficient visible-view support.

## 2026-08-28 terminal verification decision

SC41 opened a genuinely new source and froze all scoring before truth access:
the official SWITCH Basic v1 open `verification_state/video2img` task at commit
`510a96b59c8688a2122d725d142c5b720962cc47`. All 16 real-video tasks were
evaluable. Desired-outcome CLIP selected `9/16` correct, while the proposed
goal-conditioned incomplete-state contrast plus maximum current-video
similarity selected `3/16`, a `-37.5` point change with seven regressions.

Record `SC41_SWITCH_CAUSAL_TERMINAL_VERIFICATION_GATE_NOT_MET`. The diagnostic
same-scene term had the largest median candidate span (`0.1425`) but its argmax
was correct on `0/16`: a successful outcome should often look different from
the pre-completion video. Current-scene appearance persistence is therefore not
terminal-state evidence for this task. Do not tune weights, prompts, scales,
sampling, model, or membership after exposure. A legal successor must add an
explicit localized state-transition/effect representation and evaluate it on a
new unexposed cohort; the `9/16` baseline is descriptive, not a positive gate.
No navigation, approach, reachability, executed action, user arrival, handoff,
product, or safety claim follows.

## 2026-08-28 explicit state-effect decisions

SC42 tested a new Goal-Localized Effect Axis on 64 hash-frozen real OSCaR
`open/close/on/off` clips. Object-selected CLIP patches were projected onto an
explicit `after - before` language direction. The static desired-state baseline
was `25/64`; the successor was `27/64`, with 13 rescues, 11 regressions, and a
descriptive `close` change from `7/20` to `11/20`. Record
`SC42_GOAL_LOCALIZED_EFFECT_AXIS_GATE_NOT_MET`. More importantly, OSCaR Frame 3
is temporal endpoint truth, not functional completion truth; official captions
can state that a requested effect is still not visible. Do not tune this cohort
or score the unopened remainder against the same inadequate arrival authority.
The result is narrow evidence that a directed local state axis carries some
signal, not a completion result.

SC43 used the genuinely effect-authoritative SWITCH `final_state/img2img` task:
95 current-state/action/four-outcome rows, with truth hidden until provider
seal. Plain Qwen2-VL-2B selected `19/95`; a frozen `do(action) - do(no-action)`
log-probability contrast selected `24/95`, a `+5.26` point change with 16
regressions. Record `SC43_CAUSAL_INTERVENTION_LOGIT_CONTRAST_GATE_NOT_MET`.
The successor chose A on all 95 rows, proving label-prior collapse rather than
content-sensitive causal reasoning. Do not change prompts, label tokens,
resolution, scaling, examples, or sampling on this opened task, and do not count
its shared-ID video/text variants as fresh confirmation.

The next legal terminal-state algorithm must expose `actuator`, `effect
carrier`, `before`, `after`, and `conflict` as separately grounded variables and
use a deterministic reducer with `UNKNOWN`; hidden reasoning compressed directly
into a choice token is closed. It requires new visual supervision and a genuinely
new outcome-authoritative cohort. No online search, identity, localization,
navigation, physical execution, arrival, handoff, product, or safety claim
follows.

SC44 supplied that new outcome authority from the public ungated RoboPulse hard
subset. Ninety rows were frozen by source-stratified ID hash across nine real and
simulated manipulation sources; each row includes task-start/task-end references,
BEFORE/AFTER front and wrist views, and signed progress truth. The Goal-Anchored
Factor Reducer exposed effect carrier, actuator/contact, spatial orientation,
conflict integrity, and handoff distance as separate Qwen2-VL-2B label margins,
then required positive conflict integrity and at least three positive factors.
The direct baseline scored `43/90` with `49.90%` balanced accuracy. The reducer
also scored `43/90` with `50.00%` balanced accuracy and one regression, so record
`SC44_GOAL_ANCHORED_FACTOR_REDUCER_GATE_NOT_MET`.

The factor audit is the decision: every one of the five margins was positive on
all `90/90` rows, and the successor therefore predicted progress on `90/90`.
Explicit factor names do not make frozen VLM logits into grounded state
variables. Do not tune prompts, tokens, thresholds, weights, or this cohort.
The next legal successor must learn a content-sensitive differential factor
representation from disjoint supervision and expose calibrated `UNKNOWN` before
the same deterministic reducer; it must use a new frozen evaluation cohort. No
demo or product integration follows.

SC45 implemented that representation change without reopening SC44. It excluded
all 90 consumed IDs, used six source domains for training, one source for split-
conformal calibration, and held out `human_pika` plus `libero_data` as 380 fresh
evaluation rows. A 32-dimensional Progress Factor Tensor separates effect
carrier, wrist actuator/contact, front-view spatial orientation, cross-view
conflict, and completed-reference handoff distance. A fixed linear learner emits
probabilities; only the frozen conformal set reducer may emit progress,
regression, or `UNKNOWN`.

The untrained visual goal-axis baseline reached `234/380` (`61.58%` balanced
accuracy). SC45 resolved `357/380` (`93.95%` coverage) at `65.96%` selective
balanced accuracy, versus `62.49%` for the baseline on those same known rows: a
real `+3.47` point cross-source gain, but below the frozen `+5` point gate.
Record `SC45_LEARNED_PROGRESS_FACTOR_TENSOR_GATE_NOT_MET`. The source split is
diagnostic: selective accuracy was `59.78%` on `human_pika` and `71.91%` on
`libero_data`. The strongest learned groups were effect carrier and cross-view
conflict; spatial orientation was weakest. Do not tune the opened split,
regularization, conformal alpha, or factor weights. A legal successor must add a
new object/hand-local effect-carrier representation and use another frozen
cohort; global CLIP factor tensors remain reusable as a baseline, not endpoint
authority.

SC46 executed that successor on Guardian UR5-Fail without reusing SC45 rows.
The source-native roles are 400 training, 30 calibration, and 140 evaluation
executions. A task-conditioned GroundingDINO carrier exposes up to two task
entities and the robot gripper in all three start/end views. CLIP supplies global,
task-region, and combined interaction-region goal-axis deltas; box confidence,
area, overlap, count, and gripper-task distance complete the 80-dimensional
tensor. A fixed learner supplies probabilities and the localization-plus-
conformal reducer alone may emit success, failure, or `UNKNOWN`.

The global baseline reached `49.85%` balanced accuracy. SC46 resolved `107/140`
(`76.43%` coverage) at `59.94%` selective balanced accuracy, versus `49.79%`
for the baseline on those same rows: a material `+10.15` point gain. It still
missed the frozen `70%` absolute gate and the `60%` per-class floor because
failure recall was `57.14%`; record
`SC46_GUARDIAN_LOCAL_EFFECT_CARRIER_TENSOR_GATE_NOT_MET`. All 140 evaluation
rows met the localization eligibility rule, so the remaining error is not
missing boxes. The learned mass is dominated by task/gripper detection
confidence and counts, while task and interaction goal-axis deltas remain
weaker; `translation_object` failures are only `7/15` correct. Do not tune this
opened split, detector threshold, prompt, conformal alpha, or classifier. A
legal successor needs explicit object state/change or contact/release evidence
on a new frozen cohort; local effect carriers remain a promising mechanism, not
endpoint authority.

SC47 then changed the supervision source rather than retuning SC46. The public,
ungated DROID-OOD source supplied 250 source-train and 100 source-validation
real robot trajectories. A task-stratified hash assigned 200 rows to training
and 50 to conformal calibration; all 100 source-validation rows stayed outcome-
invisible until provider seal. Training-only dense pseudo-progress, gripper
closedness, vertical displacement, Cartesian speed, and gripper-opening rate
supervised a five-channel phase teacher. Evaluation provider inputs were only
task text and twelve three-view video samples.

The teacher transferred dense progress (`R2=0.620`), gripper closedness
(`0.725`), and vertical displacement (`0.746`) to held-out calibration video,
but failed on Cartesian speed (`-0.190`) and opening rate (`-0.207`). The learned
visual baseline reached `61.61%` balanced accuracy on all 100 evaluation rows.
The phase-distilled successor resolved `61/100` at `63.22%`, versus `54.11%`
for the baseline on those same known rows: `+9.11` points, but below the frozen
`70%` accuracy and coverage gates. Its failure recall was `94.44%`, while
success recall collapsed to `32%`; only ten evaluation rows received a
singleton success set. Record
`SC47_DROID_OOD_PRIVILEGED_PHASE_DISTILLATION_GATE_NOT_MET`.

The decision is sharper than generic domain shift: visible approach/contact/
lift phase is learnable, but process completion is not proof that the correct
object reached the requested final state. Do not tune the consumed split,
sampling, teacher targets, classifier, or conformal alpha. A legal successor
must bind an explicit final object-state description to the target entity on a
new cohort; high-frequency derivative teachers are closed.

The user-facing priority is now the named-destination front half rather than a
demo. A first zero-OCR public-reference run froze six Hong Kong entities, 12
reference images, six calibration views, 11 evaluation views, and all 55 wrong
goal pairs. CLIP name-only and global CLIP+DINO references each retrieved
`6/11`; adding mutual DINO patch matches with affine consistency still retrieved
`6/11`, but reduced wrong-goal confirmations `3 -> 0` and improved balanced
accuracy `65.45% -> 68.18%`. Accept the false-confirmation reduction as a
mechanism observation, but record
`NAMED_POI_FACADE_FINGERPRINT_DO_NOT_TUNE_LOCALIZE_REFERENCE_COVERAGE_GAP`
because the frozen top-1 gate failed.

Do not tune weights, patch geometry, calibration thresholds, or roles on this
opened cohort. Change the information source: build a larger public target
knowledge pack with explicit facade, real entrance, logo/sign, architectural
context, and on-site wayfinding facets. OCR remains one independent
high-precision branch; it cannot be the only identity authority, and no branch
alone can emit arrival. A fresh successor must improve correct-entity retrieval
before the route spends work on entrance binding and continuous guidance.

The multi-facet source change is now complete. Fifty prior-file-disjoint
Commons views supplied entrance/facade/wayfinding facets, followed by a second
29-image fresh source after filename-derived truth proved invalid. On 18
human-labelled evaluation views, the unchanged entity score retrieved `12/18`,
confirmed seven, and made zero wrong-goal confirmations. Identity-only readiness
gave `2/8` true ready with five false ready; the scene-level CLIP plus
GroundingDINO entrance graph emitted no ready decisions at all. Its raw entrance
ranking still reached `0.725` AUC, so the terminal is
`NAMED_POI_SCENE_LEVEL_IDENTITY_AND_ENTRANCE_GATE_NOT_MET`, not evidence that
entrance cues are absent.

The identity seam was then moved inside each entrance candidate. On a third
prior-file-disjoint 20-image source, GroundingDINO covered all `4/4` human-boxed
actionable entrances at IoU >= 0.30. Committing its strongest generic proposal
made 20 false commits and zero correct commits. Candidate-local CLIP+DINO target
binding reduced false commits `20 -> 5` but still made zero correct unique
entrance commits; one of four positives survived only in a `SET_VALUED` set.
Retain this reducer only as a `75%` false-commit filter, not as a locator. Record
`NAMED_POI_TARGET_LOCAL_CROP_BINDING_SAFETY_FILTER_ONLY_NO_CORRECT_COMMIT`.

Do not tune V3 context scale, thresholds, boxes, prompt, or weights. The next
representation must transport entity evidence spatially: project reciprocal
reference-to-query patch support into the image and require an entrance proposal
to overlap or connect to that target-support field. Only a correct unique edge
may feed tracking, bearing guidance, and arrival logic. OCR remains an optional
high-precision branch, not the main or sole authority.

The spatial successor is now consumed through V7. Coarse 9x9 support failed,
while native 16x16 downward target-support rays became a strong veto. On the
final 32-image source with six target-building entrances and 26 strong
negatives, changing only to a multi-facet public reference bank preserved two
correct commits, reduced false commits `30 -> 7`, raised precision
`6.25% -> 22.22%`, and retained truth in `COMMIT / SET_VALUED` on `4/6`
positives. That equals the proposal oracle's `4/6` availability ceiling.

Record
`NAMED_POI_MULTIFACET_SUPPORT_RAY_PROPOSAL_CEILING_REACHED_SINGLE_FRAME_UNIQUENESS_NOT_MET`.
Do not tune V4/V5/V7 grids, ray width, thresholds, boxes, or roles. The static
locator gate did not pass because correct unique commits remained `2/6`.
Retain the ray as a false-commit/candidate-preservation mechanism only.

The next active seam is implemented: `SET_VALUED` requests a centered closer
view; a sole surviving candidate may commit only if normalized target scale does
not decrease; loss enters last-bearing reacquire. This prevents persistence
alone from promoting an action-inconsistent distractor.

The first real ordered portal source is now admitted without inference. Commons
contained no videos across six named-POI categories, but the previously unopened
IFC Man Cheung Street series frames 06--07 show the same exterior glass door
bank from a wide upper-interior view and a closer escalator-aligned view. Record
`ADMIT_REVERSE_SIDE_SAME_PORTAL_ACTIVE_VIEW_PROXY`. This is an exit-side proxy,
not an outside entrance route or commanded-action proof.

The first real non-OCR two-frame prefix retained portal-set truth in `0/2`
frames. The original temporal belief falsely changed `SET_VALUED -> COMMIT` on
an almost-full-image/foreground-escalator trajectory. The new approach-scale
gate changes the frozen replay to `SET_VALUED -> SET_VALUED`, eliminating that
observed false commit but not recovering the door bank. Record
`ACTIVE_VIEW_SCALE_GATE_PREVENTS_OBSERVED_FALSE_COMMIT_PROPOSAL_SOURCE_STILL_MISSING`.
Do not tune the consumed prompt/ray/policy; change the portal proposal
representation.

The proposal representation has now changed rather than tuning the consumed
prompt or thresholds. A non-OCR dual-family functional portal-set proposer
groups repeated door posts into a multi-leaf entrance and pairs vertical
handles to infer an aperture. Lattice-only failed its first frozen six-building
gate at top-three `3/6`; the handle-pair successor was developed only after that
cohort was consumed. On a separate six-building frozen confirmation cohort, the
unchanged successor retained portal truth at top-one `5/6` and top-three `6/6`;
`50/60` stored top-ten candidates and `14/18` top-three candidates lay in the
human-frozen functional portal regions. Record
`RETAIN_DUAL_FAMILY_FUNCTIONAL_PORTAL_SET_PROPOSER`.

This crosses the current-image glass-entrance proposal bottleneck, not the full
last-ten-metre task. Target-entity identity must be confirmed separately before
these top-three proposals enter the scale-gated active belief. Public access,
traversability, ordered/commanded-view causality, tracking, reacquisition,
guidance, arrival, user benefit, and safety remain unproved. SkyDiscover policy
search remains blocked until a genuine ordered or commanded real episode is
available; the obsolete oracle benchmark family is excluded from the route.

L10-PB1 has now tested that missing target-to-portal seam rather than tuning the
proposer. Twenty building identities were split without overlap into 10 train,
four Development, and six confirmation entities. The dual-family proposer,
CLIP, and DINOv2 were frozen; only a shared candidate MLP plus a symmetric set
summary and explicit `NONE` head were trained. The six confirmation buildings
were unseen by PB1 training/selection, although their pixels had previously
served the separate proposer-only confirmation.

Development selected the unchanged native support ray as the strongest
baseline. On confirmation it retained truth at Top-1 `4/6`, Top-3 `5/6`, made
two wrong portal commits, covered truth in `COMMIT / SET_VALUED` on `4/6`, and
wrongly admitted `27/30` wrong-target pairings. The learned head ranked truth at
Top-1 `4/6` and Top-3 `6/6` and rejected all `30/30` wrong-target pairings, but
it also emitted `NONE` on every one of the 36 target/query episodes: correct
`COMMIT / SET_VALUED` coverage fell `4/6 -> 0/6` and false `NONE` was `6/6` on
proposer-available positives.

The original mechanical promotion expression therefore had an evaluator bug:
zero commits could satisfy “wrong portal commit reduced” while Top-3 was counted
from latent rankings. The post-result integrity adjudication changes no outputs,
labels, or denominators and refuses that degenerate pass. Record
`L10_PB1_FRESH_BUILDING_GATE_NOT_MET_ALL_NONE_STOP_EMBEDDING_FUSION`. Do not tune
weights, thresholds, embeddings, backbones, or fusion on this consumed cohort.
A legal successor must change the public target-identity information source or
representation and use a new building-disjoint confirmation cohort. L10-AV0
remains blocked because there is still no correctly admitted portal to act on.

L10-PB2-A changed the identity representation to dedicated VPR rather than
reopening PB1. Twelve PB1-disjoint buildings were source-audited before any
model call and split without overlap into six Development and six confirmation
entities; every entity has one reference and four facade / entrance / side /
partial queries. Portal proposals, portal crops, PB1 decisions, tracking, and
active observation were not invoked. Development alone selected fixed
whole-image CLIP+DINO as baseline and SALAD as challenger.

On 24 confirmation queries the selected baseline versus SALAD was Recall@1
`9/24` versus `8/24`, Recall@3 `18/24` versus `20/24`, correct requested-place
acceptance `12/24` versus `8/24`, and wrong-building confirmation `23/120`
versus `3/120`. SALAD therefore provided stronger rejection and slightly better
latent Top-3 ranking, but no positive identity gain. MixVPR reached `16/24`
positive acceptance only with `48/120` wrong-building confirmations. All arms
used Development-frozen thresholds; test did not select an arm or threshold.

Post-result integrity review limits every cross-paired `x/120` value to a
source-label negative proxy. The audit did not exhaustively mark every gallery
building that may be co-visible in a dense same-city image, so physical
target-absent rejection is `NOT_EVALUABLE`. No model output, positive label,
threshold, or denominator changed. The decision is unaffected because Recall@1
and target-present positive acceptance already fail the identity gate.

Record
`L10_PB2A_SPECIALIZED_VPR_IDENTITY_GATE_NOT_MET_STOP_SINGLE_FRAME_APPEARANCE_ONLY`.
This closes CLIP, DINOv2, fixed CLIP+DINO, SALAD, MixVPR, threshold,
normalization, reference-weighting, and fusion sweeps on this consumed cohort.
Do not open PB2-B or L10-AV0. Change the information source to logo/OCR, map or
POI metadata, coarse GPS, or genuinely ordered multi-view evidence. This is a
curated public-image confirmation, not a universal VPR negative or any access,
navigation, arrival, user-benefit, product, or safety result.

L10-PB3 then changed both the information source and decision authority. A
public POI alias pack plus frozen RapidOCR may produce a unique positive entity
proof; such a proof accepts only that requested entity and vetoes other
requests. If text is absent or ambiguous, the result remains `UNKNOWN` and the
frozen appearance decision is preserved rather than converted to `NONE`.
Eight entities absent from PB1 and PB2-A were source-audited before any PB3
model call and split four Development / four test, with one deliberately
identity-bearing and one context query per entity.

Development selected DINOv2 as the appearance baseline and froze lexical score
and uniqueness-margin thresholds. On eight fresh test queries, the appearance
baseline reached Recall@1 `6/8` and positive acceptance `8/8`, but accepted
`23/24` source-label-negative wrong requests. Metadata-backed text emitted five
proofs, all correct, and the asymmetric join preserved positive acceptance at
`8/8` while reducing wrong-request accepts to `9/24` (`60.87%` relative
reduction). However, it correctly proved only `2/4` source-labeled
identity-bearing views versus the frozen `3/4` requirement. Record
`L10_PB3_METADATA_BACKED_TEXT_IDENTITY_BRANCH_GATE_NOT_MET`.

This is a strong precision/veto mechanism signal, not a promoted general
identity branch. Canonical names and the public alias pack were identical at
`5/5` precision and `5/8` coverage on test. A permitted target-blind full-frame
plus four-tile diagnostic added zero proofs, so digital crop/scale tuning is
closed too. Do not tune aliases, lexical thresholds, OCR models, or crops on
this cohort. The next legal source must add a fresh logo or Chinese identity
representation, or a genuinely ordered executed `APPROACH_TEXT / SWEEP_SIGN`
observation with before/after receipt. PB2-B and general L10-AV0 remain blocked;
wrong-request counts remain source-label proxies rather than exhaustive
physical target-absence authority.

L10-PB4 has now crossed the missing identity-coverage gate with a genuinely
new representation. The Script-Contrastive Identity Lattice (SCIL) separates
official public metadata into English-word, traditional/simplified-folded Han
uni/bi-gram, and public-mark carriers. Candidate-pack distinctiveness weights
the atoms, agreeing carriers receive only a small bonus, and independently
strong carriers that name different entities veto the frame to `UNKNOWN`.
Only a unique SCIL proof authorizes its entity; no proof preserves the frozen
appearance decision.

A source-only pixel audit rejected placard, directory, exhibit, wayfinding,
neighbor-sign, and previously human-opened candidates before any PB4 model
call. It froze eight PB1/PB2-A/PB3-formal-cohort-disjoint entities into four
Development and four test entities, each with one identity-bearing and one
context query. Development selected CLIP for appearance and independently
froze all text score/margin thresholds. Test remained sealed.

On eight test queries, appearance reached Recall@1 `5/8`, accepted only `2/8`
correct requests and `0/4` identity-bearing requests, and accepted `1/24`
source-label-negative wrong requests. English canonical matching proved `0/8`;
flat bilingual whole-alias matching proved `1/8`. SCIL emitted **`6/6` correct
proofs with zero wrong proofs**, including **`4/4` identity-bearing views** and
five proofs with Han or public-mark participation. Its asymmetric join raised
positive acceptance from `2/8` to **`6/8`** and reduced the wrong-request proxy
from `1/24` to **`0/24`** (`100%` relative, `4.17 pp` absolute). All six frozen
clauses passed. Record
`L10_PB4_SCRIPT_CONTRASTIVE_IDENTITY_LATTICE_GATE_MET`.

SCIL is now the conditional proof-positive named-place authority; PB3 remains
closed and its cohort must not be tuned. This does not yet bind an entrance.
The next legal experiment may join only a current SCIL-proved frame to the
separately frozen functional portal-set proposer. `UNKNOWN` frames remain
blocked and request `APPROACH_TEXT / SWEEP_SIGN`. The identity-bearing stratum
is deliberate opportunity evidence and wrong-request rows remain source-label
proxies; no natural prevalence, open-world identity, access, active-view,
navigation, arrival, product, user-benefit, or safety claim follows.

L10-PB5 has now tested that exact seam and stopped it at the correct layer. The
Script-Proved Authority-Carrying Portal Lattice (SP-ACPL) retained every sealed
PB4 test frame with a correct SCIL proof. A source-only portal audit froze three
in-scope glass-door positives, two no-portal negative controls, and one visible
monumental entrance outside the paired-handle representation before any portal
proposer call on these frames. Human labels were evaluator truth only: the
unchanged proposer ran on all six proof-positive frames.

A truth-blind opportunity predicate, frozen from the earlier six
source-disjoint positive portal frames, allowed only sufficiently strong,
scaled, lower-image paired-handle proposals to carry the SCIL entity token. Raw
portal truth retention was `2/3` at Top-1 and Top-3, but authorized truth
retention was only **`1/3`** and one of two no-portal controls falsely received
portal authority. The OOD aperture stayed blocked (`0/1` leakage), as did all
18 source-label wrong-request pairs, both SCIL-`UNKNOWN` rows, and every
`COMMIT` or directional state. The failed positive-coverage and negative-control
clauses record
`L10_PB5_SCRIPT_PROVED_AUTHORITY_CARRYING_PORTAL_LATTICE_GATE_NOT_MET`.

Retain SCIL as conditional identity authority, but do not attach portal
ownership from this geometry-only opportunity gate. Score, box, family, SCIL,
and geometry threshold tuning is closed on the consumed PB5 rows. The next
legal experiment must change the portal-opportunity information source or
representation, preferably a frozen semantic door/opening layer tested on a
source-disjoint cohort with no-portal and open-aperture controls. No access,
traversability, active-view, tracking, guidance, arrival, product,
user-benefit, or safety result follows.

PB6--PB9 then changed that source four times and closed each failed
representation instead of tuning it. PB6 synthetic semantics reached `4/6`
positive truth Top-3 with `0/16` false controls in Development, then failed a
new cohort at `0/4` with `1/2` no-portal and `1/2` open-mouth leakage. PB7
ADE20K components reached `3/4`, `0/2`, and `0/2` in Development, then retained
no positive on fresh pixels. PB8's typed VLM selected `DOOR` on all eight
frames and retained truth in only `1/4`; PB9's author-final specialized
detector retained no door candidate on any positive. Do not reopen their
prompts, classes, confidence, boxes, crops, model sizes, or thresholds.

PB10 changed modality. Frozen row-relative far-depth components retained a
truth member in Top-3 on **`4/4`** positives, and official Trans4Trans class-5
argmax masks aligned to a truth-member component on **`4/4`**. But one of two
no-portal controls and one of two large open-mouth controls also fired, for
balanced accuracy `0.750`. A separately frozen, threshold-free bottom-to-far
glass cut achieved `0/2` and `0/2` control leakage but only `1/4` positive
coverage, balanced accuracy `0.625`. Record
`L10_PB10_GLASS_DOOR_PLANE_AND_TOPOLOGICAL_CUT_DEVELOPMENT_GATE_NOT_MET`.
RDAT remains only a coarse proposal diagnostic and Trans4Trans only a semantic
alignment source.

PB11 Metric Portal Closure is now closed. Its cohort was frozen before output
across eight distinct SUN RGB-D captures with true K, official depth, sealed
aperture quadrilaterals, and role counts `4 + 2 + 2`. All `8/8` rows evaluated,
but `min(door)=0.7166` was below `max(control)=0.9831`; strict margin was
`-0.2665` and ROC AUC `0.625`. The strongest no-portal result was a planar wall
storage panel, while the weakest positive was a glass double door. Record
`L10_PB11_METRIC_PORTAL_CLOSURE_P0_PRIVILEGED_GATE_NOT_MET`.

This is a representation failure, not a metric-depth checkpoint failure:
rim-plane closure identifies planar occlusion but cannot say that the plane is
a door. Do not run DepthART C1 or reopen the cohort, quadrilateral, rim,
interior, relief, quantile, threshold, or score on these rows. The next decision
must add RGB door-part topology (leaf-frame-handle or hinge relationships) and
may use depth only as a subsequent geometry consistency check.

PB12 Door Part-Parent Topology has now tested and closed one such box-level
source. A DoorDetect-trained four-class YOLO11n checkpoint was frozen before
output on eight new SUN RGB-D captures: four handled architectural doors, two
handled furniture-door controls, and two large doorless openings. The exact
rule assigned each handle to the smallest enclosing `door`, `cabinet door`, or
`refrigerator door` parent and authorized only a handle assigned to `door`.
It rejected all `4/4` controls but also authorized `0/4` positives, giving
positive recall `0.0`, control false-positive rate `0.0`, and balanced accuracy
`0.500`. Record `L10_PB12_DOOR_PART_TOPOLOGY_DEVELOPMENT_GATE_NOT_MET`.

The useful signal is narrower: architectural door parents were present on
`2/4`, visible architectural-door handles on `2/4`, the closet produced
coherent cabinet-door/handle assignments, and both large openings stayed
silent. The failure is that parent and child did not co-occur reliably, plus one
real door had overlapping door/cabinet parents that assigned the handle to the
cabinet class. Do not repair this by changing thresholds, crops, box overlap,
parent priority, image size, checkpoint, or fusion on the consumed cohort. The
next decision must change to a distinct pixel-level door-part information source
that explicitly localizes leaf, frame, handle, and furniture-door competitors.

PB13 changed to text-conditioned Florence-2 pixel polygons and froze six
expressions plus an eight-frame, four-bucket cohort before any output. A first
pre-output environment-guard bug consumed no pixels and allowed one mechanical
replay. That replay stopped on frame 1 after the `operation_part` result exposed
a component with fewer than three valid vertices under the preregistered polygon
contract. Zero frames and zero aggregate metrics were adjudicated. Record
`L10_PB13_FLORENCE_PIXEL_PART_TOPOLOGY_DEVELOPMENT_NOT_EVALUABLE_OUTPUT_CONTRACT_FAILURE`;
do not call it a miss or a safe rejection.

The PB13 cohort is consumed and cannot be rerun with a relaxed parser or revised
prompt. The next decision must change both information source/representation and
cohort; a native open-vocabulary mask source or a separately grounded box-to-mask
pipeline is admissible, but Florence prompt/parser rescue on these pixels is not.

PB14 tested the native-mask branch with pinned YOLOE-26n-seg and MobileCLIP2-B.
The frozen representation ran all eight full frames plus nine exact parent-box
crops (`17` calls total) in `9.60 s`, with peak allocated CUDA memory
`104,960,512` bytes. It rejected all four controls, but authorized none of the
four visible-part architectural doors. Full-frame output contained zero
`architectural_leaf`, zero `operation_part`, and zero `hinge` instances; two
positives were instead split into `closet_door` plus `doorless_opening`, and the
other two had no parent. Parent rescaling recovered no handle or hinge. Metrics
are positive recall `0.0`, control false-positive rate `0.0`, balanced accuracy
`0.500`; record
`L10_PB14_YOLOE_MULTISCALE_PART_TOPOLOGY_DEVELOPMENT_GATE_NOT_MET`.

This is a source-semantic failure before the frozen pixel topology, not evidence
for revising the matcher. The PB14 rows are consumed: do not compare alternate
prompt wording, confidence, NMS, image size, crop expansion, YOLOE size, parent
priority, or fusion on them. The next decision takes the separately grounded
box-to-mask branch: fixed two-scale GroundingDINO parent/part grounding followed
by SAM2.1 box masks and the same competitor-conservative pixel assignment, on a
new PB15 cohort.

PB15 completed that branch on eight further source-disjoint SUN RGB-D captures.
The first launch stopped before model load or cohort RGB decoding because the
frozen GroundingDINO tree-receipt digest was missing one hexadecimal character;
the receipt digest and dependent protocol/cohort bindings were corrected before
any output existed. The valid frozen run completed `195` GroundingDINO calls and
`37` SAM2.1 calls in `90.90 s`, with `2,096,144,896` peak allocated CUDA bytes.
It authorized `3/4` architectural doors, leaked on `1/2` handled-furniture
controls, rejected both large doorless openings, and reached balanced accuracy
`0.750`. Record
`L10_PB15_GROUNDED_SAM_MULTISCALE_PART_TOPOLOGY_DEVELOPMENT_GATE_NOT_MET`.

The attractive positives are not clean part evidence. GroundingDINO repeatedly
mapped mutually exclusive child prompts such as handle, push bar, panic bar, and
hinge to nearly the entire parent crop; SAM2.1 faithfully converted those boxes
to nearly identical whole-object masks. In the false negative, a slightly
smaller cabinet mask absorbed the pseudo-parts despite an architectural parent
being present. In the furniture false positive, a slightly smaller
architectural mask absorbed whole-cabinet pseudo-parts. The same collapse is
present in the three nominal true positives. PB15 is therefore closed as a
semantic/localization-source failure, not a matcher opportunity: do not tune its
prompts, thresholds, caps, crop geometry, SAM settings, parent priority, or
assignment rule on the consumed pixels. PB16 must replace the grounded-box chain
with a native text-conditioned concept-instance mask source on a fresh cohort;
the current admissible source is SAM 3.1.

PB16 froze SAM 3.1 multiplex weights, a strict image-only detector adapter,
five parent concepts, five operation-part concepts, and eight further
source-disjoint SUN RGB-D captures. The public SAM 3 image builder was rejected
during pre-formal smoke because this SAM 3.1 checkpoint exposes a three-scale
multiplex detector rather than the builder's four-scale dual neck. PB16 instead
assembled the detector components from Meta's multiplex recipe and strictly
loaded all `1,166` detector keys with zero missing or unexpected keys; this is a
custom image-only adapter, not an official supported SAM 3.1 image API. A first
formal launch then stopped before model load, RGB decoding, or output because
WDDM reported `8,150` rather than the frozen `8,151` MiB. Only that mechanical
contract was corrected to the same GPU with an `8,000` MiB minimum.

The valid run completed `8` image encodes plus `80` text calls in `34.94 s`,
with `4,685,969,920` peak allocated CUDA bytes. It authorized `0/4`
architectural doors and `0/4` controls: positive recall `0`, true-negative rate
`1`, and balanced accuracy `0.500`. Record
`L10_PB16_SAM3_NATIVE_PART_TOPOLOGY_DEVELOPMENT_GATE_NOT_MET`. The decisive
observable is global: the frozen `architectural door` parent concept returned
zero instances on all eight frames. Child masks still appeared on three of four
positives, while furniture parent concepts and both control classes behaved
cleanly, so topology never received an architectural parent to authorize. PB16
is closed as a parent semantic/source failure. Do not alter its concept text,
confidence, caps, mask assignment, adapter, or topology on the consumed pixels.
A successor must change the supported weight/API representation and parent
semantic representation together, then use a fresh source-disjoint cohort.

PB17 made both changes: it materialized the original `facebook/sam3`
Safetensors checkpoint, used the official Transformers image API with reusable
vision embeddings, and replaced parent-plus-child topology with direct native
mask presence for the simple noun `door`. A synthetic-only smoke strictly loaded
all weights with zero missing, unexpected, mismatched, or error keys; its peak
CUDA allocation was `3,928,569,856` bytes. The formal cohort froze eight new
capture sequences with zero overlap against the 48 PB11--PB16 sequences.

The formal run stopped on the first positive after one image encode and one
`door` call because the postprocessed mask tensor retained its native spatial
shape rather than the frozen source-image shape. Input-only diagnosis confirmed
the processor supplied the correct `[530,730]` target. The frozen Transformers
implementation resizes every nonempty retained mask and leaves native shape only
when the retained count is zero; therefore the first fresh positive had no
`door` instance at confidence `0.5`. That single miss makes the required `4/4`
positive conjunction impossible. Record
`L10_PB17_OFFICIAL_SAM3_DOOR_STATE_DEVELOPMENT_GATE_NOT_MET`. No full confusion
matrix or balanced accuracy is claimed. Do not repair the empty-mask assertion,
prompt, threshold, processor, or postprocessing and rerun the consumed cohort.
The next route must add genuinely different observable information on fresh
pixels rather than try another open-vocabulary door phrasing.

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

C3 has now replaced the motion observation with causal ego-compensated raw-point
direct velocity. Three-dimensional 0.24 m voxel centroids emit velocity only
under reciprocal nearest correspondence and the unchanged R7 speed bounds.
Backend choice was measured per sequence before matching: one sequence selected
Torch CUDA and six selected SciPy KD-tree because it was faster on their actual
point counts; every receipt records the observed device.

M1-PD preserves `20/21` natural CONTACT recall and raises induced-gap recovery
from R7's `29/63` to `52/63`, but 85 natural false segments make it unsuitable
as an independent alert source. Applying the existing independent-history hard
gate reduces false segments to 53 and F1 rises to 42.55%, but recovery falls to
`36/63`. This localizes the tradeoff to evidence routing rather than a missing
confidence threshold.

Accept `DTR_C3_M1_HYBRID_RAW_POINT_GAP_BRIDGE_FRESH_MECHANICS_SIGNAL`.
M1-HYBRID uses M1-CT on normal frames and opens reciprocal M1-PD plus sealed R7
fallback only inside an observable bounded gap of a previously tracked target.
Its natural replay therefore remains `20/21`, 38 false segments, 50.63% F1, and
2.08 s median lead, while M1-PD alone confirms a lower bound of `52/63` recovery
for the gap union. Relative to R7 this is 52 fewer natural false segments,
+20.10 F1 points, and at least 23 more fresh dropout recoveries. Keeping R7 as a
gap fallback also preserves the earlier consumed `9/9` mechanism by
construction.

This remains public LiDAR replay with privileged current native boxes for
scorer-side spatial attribution. Do not tune C3 voxel size, nearest-neighbor,
confidence, route, motion, or lifecycle parameters. The next source change is
detector-independent occupancy attribution and runtime integration; complex
trajectory forecasting and R8 remain closed.

The later detector-independent line now has an algorithm-fresh C11 baseline:
calibrated route-region occupancy retained `17/20` bounded CONTACT recall while
reducing the fixed C9 baseline's 17 false segments to 11, raising event F1 from
62.96% to 70.83%. C14 then recovered per-cell temporal residual covariance and
propagated eight fixed sigma points through the unchanged collision geometry.
On the consumed C11 Development cohort it improved recall `17/20 -> 18/20`, but
false segments rose `11 -> 16` and median lead improved only `0.049 s`.
Therefore accept `DTR_C14_STOCHASTIC_ROUTE_CONFLICT_DEVELOPMENT_GATE_NOT_MET`:
do not open the remaining fresh cohort and do not tune probability, covariance,
cubature, or route parameters. A successor needs directional or multimodal
future evidence, not symmetric uncertainty spreading.

C15--C18 then tested confidence-aware directional motion without changing the
route, probability threshold, maintenance model, or lifecycle. C15's
frame-local component mixture labelled nearly every component moving and did
not improve C11. C16's signed current/history modes retained `17/20` recall and
raised median lead to `1.726 s`, but false segments rose to 13. C17's
route-entry consensus reduced false segments to 8 and raised F1 to `75.56%`,
while lead fell to `1.062 s`. C18's frozen-scale three-frame chain retained
`17/20`, 9 false segments, and `73.91%` F1, but lead remained `1.079 s`.
Therefore accept the four Development closures and keep the remaining cohort
sealed. They establish two distinct signals—early signed motion and temporally
trustworthy motion—but no single scalar yet preserves both.

C19's fixed training-only joint calibration then retained `17/20` recall but
produced 12 false segments, `69.39%` F1, and `1.268 s` median lead; both learned
coefficients were positive, so confidence did not provide a learned veto. C20's
threshold-free local position/velocity vote was almost universally supported
and reproduced C16 (`17/20`, 13 false, `1.726 s` lead), proving the pseudo-motion
is locally coherent rather than isolated. C21's coordinate-wise median scene
bias subtraction reached `18/20` but also 18 false segments and only `64.29%`
F1, so one global background motion is invalid in crowded/co-moving scenes.
Accept all three Development closures and keep the remaining cohort sealed.

C22 then changed the information source on the consumed R7 canary. Ego-rigid
RGB point-track residual confidence reduced target-attributed false segments
`20 -> 12` while the frozen base still retained `3/3` critical recall. But the
visually admitted flow recovered `0/9` induced dropout windows, matched only
`2/3` events, and left event F1 unchanged at `22.22%`. Accept
`DTR_C22_EGO_RIGID_VISUAL_MOTION_DEVELOPMENT_GATE_NOT_MET`. It establishes that
independent visual residual can remove the eight added pseudo-motion segments,
but coarse stitched cell projection does not provide sufficiently dense,
time-aligned target support. Keep all fresh cohorts sealed. C23 must change
correspondence granularity: five undistorted perspective cameras, raw
LiDAR-supported point tracks, full ego SE(3) LiDAR-to-image compensation, then
BEV aggregation. Do not tune C22 confidence, grace, route, or lifecycle.

C23--C24 instead established the required representation change directly in
raw LiDAR.  C23's object-rigid confidence retained `3/3` critical events and
`9/9` induced dropout recoveries but left 16 false segments because one
component translation was still broadcast to all occupied cells.  C24 replaced
that broadcast with ego-compensated reciprocal point correspondence and a
causal independent-history consistency observation.  On the consumed Packard
Development canary, M1-PD reached `3/3`, 13 false segments, 30.00% F1, and
`9/9`; M1-PDC reached `3/3`, 12 false segments, 31.58% F1, and `9/9`.  The fixed
Development gate passed and authorized exactly one algorithm-fresh confirmation.

C25 froze that representation and its gate before selecting or acquiring the
deterministic five-sequence remainder.  Truth-blind workers sealed predictions
for 3,358 frames before future OBB truth was opened.  On 12 bounded CONTACT
events, R7 reached `11/12` recall, 52 false segments, 29.33% F1, 4.200 s median
lead, and `30/36` induced-dropout recovery.  M1-PD reached `11/12`, 29 false,
42.31% F1, 3.067 s, and `33/36`.  M1-PDC reached `12/12`, 21 false, 53.33% F1,
1.624 s, and `25/36`.

Accept
`DTR_C25_POINT_FLOW_ALGORITHM_FRESH_GATE_NOT_MET_HARD_TEMPORAL_VETO`.
M1-PDC passes the recall, false-segment, and F1 checks but fails the frozen
dropout non-regression check; its lead loss is also material although not gated.
The failure is concentrated in Gates (`1/6` versus R7 `6/6`) and Packard-0
(`0/6` versus `3/6`), while the ungated point-local M1-PD recovers `33/36` and
still removes 44.2% of R7 false segments.  The justified conclusion is that
point-local direct velocity generalizes, while independent-history consistency
must not delete evidence as a hard gate.  Do not tune this consumed cohort.

The next admissible information upgrade is route-conditioned residual future
occupancy over the causal M1-PD point tokens, using confidence and temporal
consistency as soft weights and calibrated uncertainty rather than an alert
source or hard veto.  Before training, run one realized-future occupancy oracle
through the unchanged global-OBB route/lifecycle scorer.  Continue only if it
shows material lead headroom without reducing recall or increasing false
segments; otherwise close the forecasting branch without a model sweep.

C26 has now executed that single falsifier.  It did not convert global CONTACT
truth directly into an alert.  Future OBB motion could replace M1-PD constant
velocity only where a sealed point cell associated to exactly one current
native OBB under the existing 0.08485 m margin and that identity remained fully
observed for the complete 3 s horizon.  Unsupported, ambiguous, and
right-censored cells retained M1-PD, and the combined raw evidence passed through
the unchanged R2 urgent boundary and lifecycle.

The support-conditioned oracle scored `11/12` CONTACT recall, 25 false segments,
45.83% F1, and 2.261 s median lead.  It removed seven of M1-PD's 29 original false
segments but gained zero net event recall and lost 0.806 s median lead.  It therefore
failed every componentwise C25 envelope check: PDC's `12/12` recall, PDC's 21
false segments, and R7's 4.200 s lead.  Accept
`DTR_C26_SUPPORTED_FUTURE_OCCUPANCY_HEADROOM_NOT_MET` and do not train or sweep
a residual future-occupancy model on current M1-PD tokens.

The next admissible source change is an occlusion-persistent identity-free point
support field.  Preserve cell-local direct velocity, carry each token forward
with soft confidence and evidence age, expose missingness as `UNKNOWN`, and
never broadcast one component translation to all cells.  This must demonstrate
joint recall/false/lead headroom before route-conditioned learned future
occupancy can reopen.

C27 has now run that bounded positive-support canary on the consumed C25
cohort.  M1-PDC alone remains `12/12`, 21 false segments, 53.33% F1, 1.624 s
median lead, and `25/36` induced-gap recovery.  PDC-seeded, reciprocal
M1-PD-refreshed point lineages retain `12/12`, preserve every PDC event's
first-alert lead, improve median lead to 2.967 s, and recover `27/36`, but false
segments rise to 26 and recovery remains below R7's `30/36`.

Accept `DTR_C27_PERSISTENT_POINT_SUPPORT_DEVELOPMENT_GATE_NOT_MET`.  Do not tune
the `0.8 s` age, confidence decay, reciprocal radius, route, or lifecycle.  The
five sealed dynamic ledgers contain no current visibility or known-empty
authority, so positive support cannot distinguish departed, occluded, and
unsensed states.  C28 changes that source with a conservative raw-LiDAR ray
ledger: `HIT` confirms presence, `KNOWN_FREE` clears departure, `OCCLUDED`
alone permits persistence, and `UNSENSED` remains `UNKNOWN`.  It must beat the
same `12/12`, 21-false, `30/36`, per-event no-later envelope without a parameter
sweep before learned future occupancy can reopen.

C28--C31 completed that representation path on the consumed C25 cohort, and
C31 met its Development gate there (`12/12`, 21 false segments, 53.33% F1,
2.667 s median lead, `30/36` dropout recovery). Its one authorized
source-disjoint confirmation has now failed. A raw-source preflight retained
six algorithm-unexposed JRDB sequences (4,811 frames, six CONTACT events, 18
dropout trials, 278.08 s known non-CONTACT); Gates-to-Clark was structurally
`NOT_EVALUABLE` because frame 0 had no causal native pose. After predictions
were sealed, M1-PDC scored `4/6`, 25 false, 22.86% F1, 2.291 s, and `5/18`.
C31 scored `4/6`, 35 false, 17.78% F1, the same 2.291 s, and `6/18`.

Accept `DTR_C31_SOURCE_DISJOINT_CONFIRMATION_GATE_NOT_MET`. The mechanism
recovered neither missed event, added no lead, gained only one dropout recovery
versus the frozen `+2` requirement, and added ten false segments. Do not tune
C31 and do not open C32 probabilistic body-route occupancy on the same support.
X0 has now opened only this consumed truth to choose that successor. Huang-2's
miss contains correct raw motion in six frames, including three early frames,
but none enters the frozen route tube (`ROUTE_GEOMETRY_MISS`). Huang-lane has
zero responsible-OBB raw cells in the entire `-3..0 s` window
(`NO_MOTION_SUPPORT`). Of 25 PDC false segments, 14 are `BAD_FLOW`, ten are
`STATIC_PSEUDO_MOTION`, and one is a real noncritical mover. The ten
non-overlapping C31 additions are two `BAD_FLOW` and eight
`STATIC_PSEUDO_MOTION`.

Accept `DTR_X0_MOTION_SOURCE_ATTRIBUTION_COMPLETE`. Because source errors are
`34/35` false units and one miss has no raw support, learned motion authority is
not yet authorized. The next dynamic-risk falsifier replaces only current raw
direct flow with exactly one stronger scene-flow source while freezing route,
horizon, lifecycle, cohort, and evaluator. A separate continuous-geometry
canary is admissible only for the localized Huang-2 miss. C31, C32,
probabilistic occupancy, and model training remain closed.

X1--X3 replaced only the motion source with one independent
Floxels-inspired, symmetric five-scan field transported through a measured
two-scan lag. The localized gates passed: X1c found three correct positive
frames and two correct route-entry frames, while X2 suppressed `25/34` frozen
source-error units. The sealed full six-sequence replay then scored `6/6`
CONTACT recall, 94 false segments, 11.32% F1, 3.816 s median lead, and `8/18`
dropout recovery, versus PDC's `4/6`, 25, 22.86%, 2.291 s, and `5/18`.

Accept `DTR_X3_FULL_LAG_FLOXEL_GATE_NOT_MET`. Higher recall, lead, and dropout
recovery do not compensate for 69 added false segments and lower F1. The run is
opened Development only; one Huang-lane empty-support frame was emitted
fail-closed by a truth-blind amendment and explicitly bound into the sealed
evidence chain. The CUDA-autograd source also showed ledger-level cold-rerun
drift, although its X1c/X2 decisions repeated. Do not tune this consumed source,
route, threshold, horizon, or lifecycle.

X4 tested the narrow deterministic alternative: CPU float64 rigid-cluster
multi-scan votes, no autograd, random seed, optimizer, or early stopping. Three
cold roots had identical canonical arrays and effects and suppressed `31/34`
error units, but produced zero correct positive or route-entry frames out of 19;
p95 source compute was `0.399--0.450 s` against a `0.0696 s` scan period.
Accept `DTR_X4_DETERMINISTIC_CLUSTER_VOTE_REPEATABILITY_GATE_NOT_MET` and close
that representation without a parameter sweep. The only admissible next step
is read-only X3 full-replay failure attribution that identifies the missing
source information; no new full replay opens until a bounded source-only
falsifier preserves the X3 positives and removes its false-motion mechanism.

The read-only X3 full-replay attribution is now complete. Of 94 X3 false
segments, 66 are `STATIC_PSEUDO_MOTION`, 14 are bad-flow magnitude, and one is
a direction reversal, so source failures account for `81/94` (86.17%). X3
resolved nine PDC false segments but created 73 new ones; static pseudo-motion
or bad flow accounts for `65/73` (89.04%) of those additions. The remaining
primary causes are five real noncritical movers, four wrong bindings, three
fragmentation segments, and one route-geometry miss; 11 segments also carry a
comparative temporal-fragment flag.

Accept `DTR_X3_FULL_REPLAY_FAILURE_ATTRIBUTION_COMPLETE`. The next admissible
information source is `STATIC_AWARE_DIRECTION_CONSISTENT_SCENE_FLOW`: it must
preserve X3's recovered positive motion while explicitly distinguishing static
surfaces and rejecting direction-inconsistent transport. This is a source
authority change, not permission to sweep threshold, seed, backbone, tracker,
route, horizon, lifecycle, fusion, or scorer.

X5 then applied one source-only causal check to the sealed X3 cells: local flow
had to agree reciprocally with the preceding overlapping five-scan window. On
60 opened frames it suppressed `32/34` frozen source-error units, but recovered
only one correct positive frame and zero correct route-entry frames. Accept
`DTR_X5_OVERLAP_CYCLE_SOURCE_FALSIFIER_GATE_NOT_MET` and close same-source
cycle filtering without a threshold sweep. The surviving direction is narrower:
introduce a static-world anchor or independently observable dynamic evidence;
do not derive another authority from the same unsemantic geometric flow.

X6 supplied that independent static-world observation using raw LiDAR and
native ego poses. On the same single 60-frame falsifier it preserved three
correct positive frames and two correct route-entry frames, suppressed `26/34`
source-error units, and measured `0.06435 s` source compute p95 against the
`0.06961 s` scan period. Accept
`DTR_X6_STATIC_WORLD_PERSISTENCE_FALSIFIER_GATE_MET`; this authorized one frozen
full replay, not tuning.

X7 completed that six-sequence replay. The anchor removed `340,734/642,217`
X3 candidate cells and reduced false segments `94 -> 72` while preserving
`6/6` CONTACT recall, `3.816 s` median lead, and `8/18` dropout recovery.
Event F1 improved only `11.32% -> 14.29%`, remained below PDC's 22.86%, and all
three selectivity checks still failed. Accept
`DTR_X7_FULL_STATIC_WORLD_ANCHOR_GATE_NOT_MET`: the causal anchor has genuine
source-level effect but is not a selective standalone risk source. Do not tune
its map age or radius. The next admissible source must add independently
observable scene-motion evidence; details and sealed hashes are in
`X6_X10_STATIC_VISUAL_LEARNED_AUTHORITY_2026-08-29.md`.

X8 added that observation as synchronized stitched-RGB image tracks compared
against ego-rigid static and X7-moving reprojections. Missing visual evidence
retains the candidate. Its single 60-frame falsifier preserved three correct
positive frames and two route-entry frames, suppressed `27/34` source-error
units, and ran the frame-local projection/LK/veto path at `0.03184 s` p95.
Accept `DTR_X8_RGB_STATIC_VETO_FALSIFIER_GATE_MET`; it opened one frozen full
replay only.

X9 completed that full replay. The visual veto removed 62,757 of 301,483 X7
cells and cut false segments `72 -> 64` while preserving `6/6` recall,
`3.816 s` aggregate median lead, and `8/18` dropout recovery. Event F1 improved
only `14.29% -> 15.79%`; false segments remain above PDC's 25 and the frozen
maximum of 16. Accept `DTR_X9_FULL_RGB_STATIC_VETO_GATE_NOT_MET` and close this
fixed RGB veto without a threshold or camera-fusion rescue. Static-world and
RGB evidence both have measurable additive effect, but neither supplies enough
authority for selective event risk.

X10 then tested one fixed sequence-held-out learned authority on the sealed X9
cells. Six L2-logistic folds used native-OBB labels from the other five
sequences only, with exact label-member access receipts and a frozen `0.5`
threshold. It retained 108,403 of 238,726 cells, reduced false segments
`64 -> 47`, and raised Event F1 `15.79% -> 20.34%`, while preserving `6/6`
CONTACT recall and `8/18` dropout recovery; aggregate median lead moved from
`3.816 s` to `3.526 s`. Accept
`DTR_X10_CROSS_FITTED_MOTION_AUTHORITY_GATE_NOT_MET`: the learned authority has
cross-sequence effect but still fails all three selectivity checks. Do not
rescue it by changing model, feature set, regularization, or threshold on these
opened folds; the next successor must add a new observable motion signal.

X11--X15 supplied that signal through calibrated RGB motion. Raw multiview and
CIWT authority closed, and frame-local stitched RGB birth was too short-lived.
The X14 bounded-continuation canary passed; its X15 full replay reached `5/6`
CONTACT, 18 false segments, 34.48% Event F1, `3.061 s` median lead, and `2/18`
dropout recovery. Accept
`DTR_X15_FULL_RGB_AUTHORIZED_CONTINUATION_GATE_NOT_MET`: it is the strongest
current Development reference, but it narrowly misses false/F1 and materially
regresses dropout. X16 composed X10 ahead of the same RGB birth/continuation;
it reached 15 false segments and `3.630 s` lead but fell to `4/6` recall, 32%
F1, and the same `2/18` dropout. Accept
`DTR_X16_CROSSFIT_RGB_AUTHORIZED_CONTINUATION_GATE_NOT_MET` and close the
over-selective composition without rescue. Eleven of X15's 18 false ranges
remain concentrated in memorial-court; the next source change must model that
persistent non-contact motion and gap continuity, not add another cell filter.

X17--X21 resolved that tradeoff with learned instance persistence plus bounded
component ancestry. Mask-wide X17/X18/X19 bridges restored dropout but produced
31--33 false segments. X20 restricted continuation to an exact
`(class, track, X7 component)` lineage and reached `5/6` CONTACT, 16 false
segments, 37.04% Event F1, and `3.785 s` lead, but only `2/18` dropout. X21
keeps raw X13 birth as the only entry and transports only the stored component
row while its anchor remains inside the same live track mask; it never absorbs
a new current-X7 cell. The sealed Development replay reached **`5/6` CONTACT,
11 false segments, 45.45% Event F1, `3.061 s` median lead, and `8/18` dropout
recovery**. Accept `DTR_X21_TRACK_CARRIED_COMPONENT_ANCESTRY_GATE_MET` for
Development. Freeze the arm and move to source-disjoint confirmation; do not
tune the consumed six sequences.

## What stops here

- Do not record the superseded 24 canary or 120 staged local RGB clips.
- Do not widen the public cohorts or tune tracker, support, tube, horizon,
  urgency, lifecycle, or guard thresholds against the opened outcomes.
- Do not create separate per-class test matrices to make the numbers look more
  complete.
- Do not treat `UNKNOWN` or `NOT_EVALUABLE` as safe.

The next algorithm action is one frozen, source-disjoint X21 confirmation.
When deployment evidence is wanted, privileged evaluation boxes/tracks must
still be replaced behind the frozen metric-frame adapters. Phone recording and
live-device testing are not the current blocker. Full methods, receipts, and
claim limits are in [the DTR route README](../research/active/dtr-r0/README.md).
