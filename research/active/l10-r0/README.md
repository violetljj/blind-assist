# L10-R0 Goal-Lock Copilot

Status: `ACTIVE / L10-SC14 CAUSAL MICRO-MOTION ACTION BELIEF /
CORE SC1W-SC2 SEEK-GUIDE-REACQUIRE CONTROLLER`

L10-R0 is the ten-meter copilot line. It does not depend on GRAIL owner
orientation. The first targets are deliberately legible and demonstrable:
room signs, exits, named entrances, elevator buttons, and service desks.

The algorithm is a goal-conditioned evidence controller rather than another
single-frame recognizer. It combines:

- long-term goal evidence from text, appearance, and sign structure;
- short-term center/motion memory for stable LEFT / RIGHT / FORWARD guidance;
- asymmetric acquire/retain belief and explicit LOST -> REACQUIRE behavior;
- three-frame near/completion evidence before declaring TASK COMPLETE.

The first matched benchmark compares a per-frame grounding baseline, a sticky
local tracker, and L10-R0 in the same seeded closed-loop target, distractor,
occlusion, reappearance, approach, and completion scenarios.

```powershell
python research/active/l10-r0/benchmark.py `
  --output artifacts.local/evidence/l10-r0/controlled-v1.json
```

The result is a controller/mechanics Development result. It is not real-camera,
natural-distribution, Android, navigation-safety, or user-benefit evidence.

## Proposal-conditional real-RGB replay

`artvideo_replay.py` accepts the public ArTVideo frame/JSON layout. It uses the
first annotated target crop as the visual goal, hides that target proposal for
four frames, and compares static-template, sticky-local, and L10 dual-memory
association using real crop pixels. Ground-truth boxes supply proposals and
evaluator identity; transcriptions never enter matching.

```powershell
python research/active/l10-r0/artvideo_replay.py `
  --dataset artifacts.local/datasets/artvideo-l10-r0 `
  --output artifacts.local/evidence/l10-r0/artvideo-proposal-replay-v1.json
```

On the available two-video canary (10 tracks, 30 gap episodes), L10-R0 reached
90.8% target-frame accuracy, 86.7% reacquisition, and 6.9% wrong selections.
The static template was stronger on accuracy/reacquisition (95.7%/96.7%) but
made 17.6% wrong selections. This is evidence for conservative ambiguity
rejection only; it does not evaluate detection, OCR, distance, guidance,
completion, or live camera control.

## Proposal-free OCR replay

`artvideo_ocr_replay.py` starts from full RGB frames rather than GT proposals.
RapidOCR supplies every text box and transcription; ArTVideo transcription only
defines the requested goal, and GT track boxes are evaluator-only. A cached run
can be replayed without rerunning OCR:

```powershell
$env:PYTHONPATH='artifacts.local/runtime/semantic-anchor-v1/site-packages'
& 'artifacts.local/runtime/artvideo-l10-ocr-probe/.venv/Scripts/python.exe' `
  research/active/l10-r0/artvideo_ocr_replay.py `
  --dataset artifacts.local/datasets/artvideo-l10-r0 `
  --models artifacts.local/runtime/semantic-anchor-v1/models `
  --cache artifacts.local/evidence/l10-r0/artvideo-proposal-free-text-v0/ocr-cache.json `
  --output artifacts.local/evidence/l10-r0/artvideo-proposal-free-text-v0/result.json
```

Across 83 frames, CPU OCR took about 30--32 seconds (2.59 fps; median 0.262 s,
p95 0.583 s), and returned text on 81/83 frames. For the ten eligible tracks,
OCR proposal coverage was 94.22% (261/277) and lexical goal coverage was 93.50%
(259/277). The same 30 four-frame-gap episodes produced:

| Controller | Target-frame accuracy | Wrong selections | Gap reacquire |
|---|---:|---:|---:|
| per-frame text | 78.85% | 115 | 90.0% |
| sticky text | **80.76%** | 102 | 90.0% |
| L10 candidate-bound text | 75.62% | **97** | 90.0% |

This establishes a useful raw-RGB OCR source, not an L10 win: candidate-bound
text reduced five wrong frames versus sticky but lost 35 correct frames and
added one median reacquisition frame. A single preregistered source change added
the existing gray/edge/HSV long-short crop evidence. It reached zero wrong
selections but only 57.27% accuracy and 76.67% reacquisition, so that source is
closed as `CLOSE_SOURCE_NO_FURTHER_MATCHER_TUNING`; it must not be rescued by a
threshold sweep.

## L10-SC0: semantic admission, visual continuity

`artvideo_semantic_visual_replay.py` implements the first task-level source
successor. It enforces an authority split rather than blending every cue into
one identity score:

- OCR lexical evidence is the only source allowed to acquire or reacquire a
  goal identity;
- a frozen DINOv2-S crop embedding maintains short/long appearance continuity
  after semantic admission;
- current-camera motion is a small continuity prior, not owner-canonical
  orientation;
- a failed visual/semantic margin enters LOST instead of silently switching to
  another proposal.

The local DINOv2-S source runs at about 110 crops/s on the available GPU. On
clean singleton OCR-to-GT associations, pooled-patch retrieval reached 100%
top-1; per-video same-vs-different track AUC was 0.993 on video1 and 0.931 on
video10. GT was attached only after crops had been embedded, so these are
continuity-source diagnostics rather than identity/verifier claims.

On the unchanged two-video, ten-track, 30-episode proposal-free replay, the
frozen L10-SC0 controller crossed all three B1 gates:

| Controller | Target-frame accuracy | Wrong selections | Gap reacquire |
|---|---:|---:|---:|
| sticky text B1 | 80.76% | 102 | 90.0% |
| L10-SC0 | **84.43%** | **5** | **100.0%** |

The tradeoff is deliberate abstention: misses rose from 48 to 104. The tracked
runner reproduces the frozen result from the cached OCR boxes and embeddings:

```powershell
$env:PYTHONPATH='artifacts.local/runtime/semantic-anchor-v1/site-packages'
& 'E:/codex-tools/bin/blindassist-python.cmd' -B `
  research/active/l10-r0/artvideo_semantic_visual_replay.py `
  --dataset artifacts.local/datasets/artvideo-l10-r0 `
  --ocr-cache artifacts.local/evidence/l10-r0/artvideo-proposal-free-text-v0/ocr-cache.json `
  --embedding-cache artifacts.local/evidence/l10-r2/dino-crop-source-v0/embeddings.npz `
  --embedding-index artifacts.local/evidence/l10-r2/dino-crop-source-v0/embedding-index.json `
  --videos video1 video10 `
  --output artifacts.local/evidence/l10-r2/semantic-visual-memory-v0/tracked-repro.json
```

A previously unseen ArTVideo video12 clip was then opened once with the same
code, weights, and gates. Absolute transfer was strong across 11 tracks and 33
gap episodes: 97.24% accuracy, zero wrong target-present selections, and 96.97%
reacquisition. It did **not** pass the relative holdout gate: this clip's sticky
OCR baseline already reached 99.55% accuracy and 100% reacquisition; L10-SC0
only reduced its wrong selections from 50 to 28. Therefore the correct result
is `SOURCE_GATE_NOT_MET` for cross-video relative uplift, not a generalized
algorithm win.

One generic top-two expected-information-gain camera policy was also rejected:
it improved shifted synthetic task success by 1--7 points but raised wrong-lock
frames to 18--22%. Active observation remains in scope only as a
deficit-specific request (for example, unreadable decisive token or ambiguous
token-to-door association); it cannot acquire identity or override semantic
`NONE / UNKNOWN`.

## L10-SC1W: identity, continuity, and steering are different authorities

The video12 replay also exposed a structural mistake hidden by aggregate
identity accuracy: every one of SC0's 28 wrong selections occurred inside an
evaluator-injected target-absent gap, and only 18/28 of those selected OCR lines
had the target's coarse bearing. Across 1,091 correct identity frames, 814
carriers were merged OCR lines, but only 400/814 merged-line centers agreed with
the target bearing. A stable identity therefore does not make the OCR line
center a trustworthy steering point.

`artvideo_dual_state_replay.py` implements SC1W as a three-authority controller:

- fresh lexical evidence owns `TARGET / NONE / UNKNOWN / UNCERTAIN` identity;
- DINOv2 appearance and camera-relative motion own only short visual
  `BOUND / COASTING / PROVISIONAL / LOST` continuity;
- RapidOCR's recognition-timestep alignment supplies a goal-related word
  carrier for current-camera LEFT / FORWARD / RIGHT steering. This is a
  CTC-aligned sub-box, not a separately detected physical object box.

Only fresh semantic `TARGET` may emit `NAVIGATE`. Weaker target-related evidence
may emit `OBSERVE` for at most two frames, but cannot acquire identity, navigate,
reacquire, or complete by appearance alone. LOST still requires two fresh
semantic hits. The association weights and gates are unchanged from SC0; the
new information source changes steering geometry, not the matcher.

On the unchanged video1+video10 Development replay (10 tracks, 30 gaps, 681
target frames), SC1W preserved SC0's 575 correct identity frames, five wrong
frames, and 30/30 reacquisitions while changing the control surface:

| Metric | SC0 line carrier | SC1W CTC word carrier | Delta |
|---|---:|---:|---:|
| identity bearing accuracy | 85.04% | **95.13%** | **+10.09 pp** |
| direction-ready coverage | 78.27% | **86.78%** | **+8.51 pp** |
| gap `OBSERVE` bearing accuracy | 70.59% | **94.12%** | **+23.53 pp** |
| navigation precision | 99.14% | **99.14%** | 0.00 pp |
| target-support coverage | 84.43% selection | **91.78% identity-or-observe** | **+7.35 pp** |
| identity reacquisition | 100.0% | **100.0%** | 0.00 pp |

All 653 steering outputs used an actual RapidOCR word result; no merged-line
fallback was needed on these two clips. Appearance-only identity violations and
continuity-only navigation violations were both zero. Two fixed alternatives
were closed without tuning: character-proportional line splitting fell to
76.70% bearing accuracy, and morphology-derived carriers fell to 78.09%.

A once-opened video13 predecessor holdout remains a useful negative result, not
confirmation: its line-carrier SC1 reached 99.74% navigation precision and only
three wrong identity frames, but just 70.0% reacquisition and 40.23% direction
accuracy. It exposed the line-level geometry/source bottleneck that SC1W changes;
video13 is consumed and must not be rerun as fresh confirmation.

SC1W was then frozen before any OCR, embedding, annotation parsing, or outcome
access on the source-disjoint official ArTVideo video14 clip. Its one permitted
run covered eight tracks, 24 gap episodes, and 426 target frames and passed all
seven preregistered gates:

| Frozen video14 metric | SC0 line carrier | SC1W | Delta |
|---|---:|---:|---:|
| identity target recall | 84.51% | **84.51%** | 0.00 pp |
| navigation precision | 100.0% | **100.0%** | 0.00 pp |
| wrong identity frames | 0 | **0** | 0 |
| identity reacquisition | 75.0% | **75.0%** | 0.00 pp |
| target-support coverage | 84.51% | **88.73%** | **+4.22 pp** |
| identity bearing accuracy | 50.00% | **76.67%** | **+26.67 pp** |
| correct-direction coverage | 42.25% | **68.08%** | **+25.83 pp** |

The frozen controller used 234 CTC word carriers, 43 single-token boxes, and
101 explicit merged-line fallbacks. This is a clean transfer of the authority
split and a large coarse-direction gain on a new clip, while the unchanged
75.0% reacquisition and only 61.9% CTC-word coverage identify the next source
work. The frozen protocol SHA-256 is `DADBF91F...F0293`; the single result is
`23384E49...C8FE7`. No result-driven retry or tuning is permitted on video14.

The cached Development result can be reproduced without OCR or embedding
recomputation:

```powershell
$env:PYTHONPATH='artifacts.local/runtime/semantic-anchor-v1/site-packages'
& 'E:/codex-tools/bin/blindassist-python.cmd' -B `
  research/active/l10-r0/artvideo_dual_state_replay.py `
  --dataset artifacts.local/datasets/artvideo-l10-r0 `
  --ocr-cache artifacts.local/evidence/l10-r4/rapidocr-ctc-word-carrier-v0/ocr-cache.json `
  --embedding-cache artifacts.local/evidence/l10-r4/rapidocr-ctc-word-carrier-v0/embeddings.npz `
  --embedding-index artifacts.local/evidence/l10-r4/rapidocr-ctc-word-carrier-v0/embedding-index.json `
  --models artifacts.local/runtime/semantic-anchor-v1/models `
  --model artifacts.local/models/p1_a2_dinov2_small_ed25f3a `
  --videos video1 video10 `
  --output artifacts.local/evidence/l10-r4/rapidocr-ctc-word-carrier-v0/repro.json
```

This is real-RGB, proposal-free OCR replay with evaluator-injected target gaps.
It is evidence for a semantic/continuity/steering representation and
current-camera coarse direction, not live active-view causality, metric arrival,
open-world identity, product, user-benefit, or safety.

## L10-SC2--SC4: acquire the goal without letting a fallback hijack belief

The video14 aggregate `18/24` result originally looked like a reacquisition
failure. A failure-layer audit showed that all six misses belonged to two goals
that RapidOCR never acquired before the injected gap. Among episodes with a
real pre-gap lock, LOST -> REACQUIRE was already `18/18`. This changed the next
problem from tracker tuning to acquisition-source coverage.

`artvideo_opportunity_active_search.py` also corrects an invalid control
inference: a non-exhaustive OCR candidate set cannot prove that the goal is
absent. It maps evidence deficits to explicit `SWEEP / SCAN / PAN / APPROACH /
SIDESTEP / HOLD` actions and changes source `STOP` into `UNKNOWN + SEARCH`, while
leaving identity authority unchanged. On video1+video10 it removed all 124
source STOP frames and all 24 target-present false-NONE decisions; every
non-navigation frame received an action. These are action decisions, not proof
that executing the motion causes a better next view.

This SC1W/SC2 authority split now runs in `core:assist` through
`GoalCopilotController.kt`. The controller admits only goal/session/entity/
current-frame-bound evidence: fresh semantics may acquire, navigate, and
reacquire; continuity-only evidence may request at most two better views;
missing or non-exhaustive proposals become `UNKNOWN + SWEEP/SCAN`, never
terminal absence. LOST requires two later fresh semantic hits before guidance
resumes. It also owns the product handoff reducer: missing identity or endpoint
evidence revokes an earlier `HANDOFF_READY`, preventing stale confirmation.

The focused JVM sequence check covers search, guide, bounded coast, two-hit
reacquisition, endpoint admission, readiness revocation, and explicit user
completion. This ports the already measured Development representation effect
(`124 -> 0` STOP frames, `24 -> 0` target-present false-NONE frames, 100%
non-navigation action coverage, `30/30` end-to-end episodes); it is not a new
camera-action experiment and still does not establish causal readability gain,
metric arrival, product benefit, or safety.

The semantic-source successor then evolved through three bounded versions:

- SC2 isolates RapidOCR and CRAFT memory. CRAFT may provide current-frame text
  identity and bearing but cannot write the RapidOCR/DINO motion anchor.
- SC3 assigns fuzzy lexical authority only to a one-token CRAFT word box. A
  multi-token box such as `Makan Dulu` cannot claim the physical identity of the
  requested token `Dulu`; it must match the complete goal span.
- SC4 adds a belief latch: CRAFT is a cold-start acquisition specialist only.
  Once RapidOCR has ever acquired the target, CRAFT cannot override a later
  LOST/gap frame. This preserves a strong primary path instead of forcing every
  detector to vote on every frame.

The evidence is deliberately mixed rather than flattened into one headline:

| Evidence | Primary | Successor | Decision |
|---|---:|---:|---|
| video1+video10 Development, SC3 | 84.43% identity recall / 91.78% support / 86.78% direction-ready | **88.84% / 96.18% / 91.19%**, 99.18% navigation precision, five wrong frames, 30/30 end-to-end | effect |
| consumed video14 diagnostic, SC4 | 84.51% recall / 88.73% support / 75.0% end-to-end | **95.77% / 100% / 100%**, 100% navigation precision, zero wrong frames | cold-start rescue effect |
| once-opened video15, SC2 exact scope | 0% recall / 0% end-to-end | 7.20% recall / 33.33% end-to-end, 19 correct and zero wrong identity frames | relative effect, **absolute capability inadequate** |
| once-opened video16, SC3 | primary itself: **96.15% recall, 100% support, 21/21 end-to-end, 100% navigation precision, zero wrong frames** | one fallback inside a target gap added one wrong frame and reduced precision to 99.81% | SC3 gate not met |
| once-opened video17, SC4 | 48.23% recall / 61.11% end-to-end / 53.26% navigation precision | identical; CRAFT was blocked 16 times after primary lock | safe-neutral routing; base physical-instance belief inadequate |

The video17 failure is not hidden as generic domain shift. Three long tracks are
repeated product-label instances: `Dairy` has another semantic candidate in
56/65 frames, and the two `Milk` tracks do so in 42/46 and 15/15 frames. The
current single-hypothesis controller chooses one semantic equivalent as a
physical instance and navigates, which is exactly the behavior the first L10
product should avoid. The next algorithmic step is an explicit multi-hypothesis
goal belief that emits active disambiguation until a functional target/entrance
association is available; it is not a return to GRAIL owner-canonical
orientation.

The supported first-demo claim is therefore narrower but real: on text-rich,
locally unique targets, source-disjoint video16 establishes a strong
search/lock/reacquire/direction controller, while video14 establishes that a
belief-latched second OCR source can rescue a genuine primary acquisition blind
spot without adding wrong identities. Repeated identical physical instances,
executed view-change causality, functional aperture/entrance association,
metric arrival, and TASK COMPLETE remain open.

## L10-SC5: task-conditioned multi-hypothesis goal belief

SC5 replaces the implicit `goal -> one carrier` assumption with at most four
source-latched physical hypotheses. Fresh RapidOCR or word-scope-legal
cold-start CRAFT evidence may create a hypothesis. Each hypothesis owns its own
local DINO, 2x-context DINO, neighboring-OCR fingerprint, motion, and evidence
age. DINO/motion may propagate a candidate but never creates identity, selects
a target, navigates, arrives, or completes. The frozen SC4 controller remains
the complete `UNIQUE` path; `SET_VALUED` may choose any legal current instance,
while `AMBIGUOUS` retains the set and requests observation without navigating.

The consumed video17 diagnostic changes the interpretation of the proposed
gate. All six public text goals map to multiple annotated physical IDs. With no
public reference image, initial box, context anchor, or functional role, the
old exact-track navigation precision of 53.26% is not a legal baseline for an
any-instance task. Under the legal `SET_VALUED` reading, frozen SC4 already has
100% navigation precision with 73.45% natural-frame coverage. SC5 preserves
100% precision and reaches 81.42% coverage (+7.97 pp). For exact-instance use,
SC5 keeps `AMBIGUOUS`, covers a legal instance on 83.63% of evaluable frames,
emits zero navigation decisions, and records zero identity-authority
violations.

The proposed low-cost context fingerprint is not yet the missing source. On
video17, pairwise physical-continuity AUC was 0.9297 for the 2x-context DINO
embedding versus 0.9408 for the existing local crop. The context crop lowered
the negative-pair median similarity (0.5268 versus 0.5665), but that is not a
net discrimination gain. UNIQUE-path parity was exact on video16 (651/651) and
video14 (546/546), preserving their frozen results rather than reopening them.

Decision: `SC5_SET_VALUED_COVERAGE_EFFECT_EXACT_INSTANCE_BLOCKED`. Admit a
public exact-instance binding (reference crop, initial box, or context anchor)
or evaluator-authoritative functional/aperture association before testing
exact-instance or functional disambiguation. video17 cannot prove executed
active-view causality, functional grounding, arrival, or task completion.

Implementation: `artvideo_multi_hypothesis_belief.py`. Evidence:
`artifacts.local/evidence/l10-r8/sc5-video17-diagnostic-v0/result.json`, SHA-256
recorded with the terminal delivery.

## L10-SC6: public reference-bound exact-instance belief

SC6 changes the information source, not the matcher. A
`PublicInstanceBinding` freezes goal text, a public anchor frame, and a public
anchor box/crop before evaluation. Its opaque public identity is mapped to the
native physical track only inside evaluator-private truth. The anchor may create
exact-instance identity authority; OCR remains candidate qualification;
DINO/motion may associate, propagate, and reference-reacquire an authorized
hypothesis but cannot create identity on appearance alone. SC4 routing, SC5
`K=4`, TTL, association gates, controller, local/context DINO, and UNIQUE paths
remain frozen.

The three frozen arms are: `SC5-TEXT`, which correctly remains `AMBIGUOUS`
without a reference; `RB0-STATELESS`, which independently applies the frozen
local-DINO reference score on each eligible frame; and `SC6-RB-BELIEF`, which
binds once and retains/reacquires the authorized hypothesis through SC5 temporal
state. The consumed video17 diagnostic failed the coverage-gain gate and was not
used for model or threshold selection.

The formal source-disjoint run uses ArTVideo video18, selected from an
annotation-only screen before RGB/model access: two public bindings and three
frozen four-frame proposal-gap episodes per binding. An initial assistant-chosen
minimum of three bindings admitted no video; before any pixel decode, OCR,
embedding, or outcome, protocol v1 changed only that roster minimum to two.
No algorithm, threshold, crop, weight, backbone, metric, or effect gate changed.

| arm | exact precision | coverage | wrong frames | wrong switches | gap reacquisition | end-to-end |
|---|---:|---:|---:|---:|---:|---:|
| SC5-TEXT | 0% | 0% | 0 | 0 | 0/6 | 0/6 |
| RB0-STATELESS | 86.49% | 57.14% | 15 | 6 | 6/6 | 5/6 |
| **SC6-RB-BELIEF** | **100%** | **91.67%** | **0** | **0** | **6/6** | **6/6** |

SC6 gains 34.53 pp coverage over RB0 while reducing, rather than increasing,
wrong commits. Identity-authority violations are zero. UNIQUE parity is exact
on video16 (651/651 decisions) and video14 (546/546 decisions). Decision:
`SC6_REFERENCE_BOUND_BELIEF_EFFECT`. The legal reference source plus frozen
temporal belief is sufficient on this small cohort, so the stop condition for
opening PDM/NearID/Doppelgangers/FlowVerify evidence is not met.

Implementations: `artvideo_public_instance_binding.py` and
`artvideo_reference_bound_belief.py`. Formal evidence:
`artifacts.local/evidence/l10-r9/sc6-fresh-reference-bound-v0/formal/result.json`,
SHA-256 `835632016e77e3fd4d6540ff85ae0c3d36d0115746daf93cfbf3b51375ecb9ff`.
The claim is limited to public-reference-bound ArTVideo replay with private
evaluator IDs and injected proposal gaps. It is not executed active motion,
functional/locational arrival, completion, product, user-benefit, or safety
evidence.

## L10-SC7: zero-OCR goal-locked door-instance route

SC7 first removes text as identity authority instead of turning L10 into a
stronger OCR system. The frozen G0 question is whether one provider-neutral
reference-bound belief can acquire, retain, and reacquire a specified building
door instance in real egocentric video. Ego4D EgoTracks supplies the public
visual template, dense long-term boxes, natural exits/occlusions, and re-entry
needed for evaluator truth; `object_title` and target boxes are forbidden after
source-order cohort admission.

The comparison is frozen as stateless reference matching versus the same
reference evidence plus target-blind temporal geometry. The gate requires at
least +10 pp exact-instance coverage and +10 pp correct-direction coverage,
at least 95% precision, zero wrong/absent commits, and 80% gap recovery within
three visible frames. OCR is absent from the goal and every candidate vector.

The signed Ego4D agreement is complete, the official CLI runtime is installed,
and the source-audit and provider-neutral belief seams have focused tests. The
machine does not yet have the approved AWS profile, so annotation admission and
real-video evaluation remain `NOT_EVALUABLE_CREDENTIALS_PENDING`; this is not
an algorithm negative. `prepare_egotracks_sc7.ps1` will download only the
EgoTracks source and freeze train/val cohorts once credentials are provisioned.

Protocol: `L10_SC7_GENERAL_GOAL_LOCKED_PROTOCOL.md`. Implementations:
`general_goal_locked_belief.py`, `egotracks_sc7_source_audit.py`, and
`prepare_egotracks_sc7.ps1`.

## L10-SC8: instance-bound task-relational functional parts

SC8 advances a different claim layer while SC7 remains credential-blocked. It
does not ask whether L10 can keep following the right parent instance; it asks
whether an already-authorized parent can be converted into the task-specific
handle or interaction part without using the parent box center as a handoff.

`functional_part_binding.py` keeps the parent binding opaque and immutable. It
uses only the public task description and unlabeled functional-part proposal
centroids, returning `UNIQUE`, `SET_VALUED`, `AMBIGUOUS`, or `NOT_EVALUABLE`.
Ordinal relations such as `top`, `second`, and `bottom` select a horizontal part
row; appearance, affordance labels, and evaluator target IDs are not inputs.
Multiple functionally legal handles remain a set for downstream geometry rather
than being collapsed into an arbitrary completion point.

The first SceneFun3D Development canary freezes one real scene, ARKit parent
boxes, evaluator-only description-to-part truth, and GT functional masks as the
proposal source. Six of ten task descriptions have a unique parent binding and
are evaluable; four lack an ARKit parent box and remain `NOT_EVALUABLE`.

| Arm | legal task commit | mean target-set recall | wrong parts |
|---|---:|---:|---:|
| nearest part to parent center | 4/6 (66.67%) | 50.0% | 2 |
| **SC8 task-relational binding** | **6/6 (100%)** | **100%** | **0** |

The parent center itself lies more than the fixed 2 cm functional-contact
tolerance from the requested part in all 6/6 tasks. SC8 crosses the frozen
+25 pp legal-commit gate with zero cross-parent identity violations, yielding
`SC8_TASK_RELATIONAL_FUNCTIONAL_BINDING_DEVELOPMENT_SIGNAL`.

This is proposal-conditional part-selection evidence, not an RGB functional-
part detector. It does not establish reachability, orientation, approach pose,
arrival, completion, product benefit, user benefit, or safety. The next source
increment is a task-conditioned multi-view RGB/RGB-D part proposer behind this
frozen binding interface; parent-box or box-scale completion must not return.

Implementation: `scenefun3d_functional_handoff_ceiling.py`. Evidence:
`artifacts.local/evidence/l10-r10/sc8-scenefun3d-functional-binding-v0/result.json`,
SHA-256 `db9068a4d9a01ed6af73ce81853f98ec6210c7e5b80526fdcd68cc2b758c6d7f`.

## L10-SC9--SC11: make the functional proposals observable

SC9 starts where SC8 deliberately stopped: the frozen task-relational selector
no longer receives evaluator masks as proposals. The provider sees posed RGB or
RGB-D observations, the authorized ARKit parent box, and—only in SC11—the public
task concept. Functional annotations and description-to-part IDs are loaded
only after the provider artifact is sealed. All four variants ran on the same
opened SceneFun3D Development scene, so the sequence is a mechanism/source
audit, not a parameter search or an independent confirmation cohort.

The three generic-handle routes did not cross the frozen proposal gate:

| Source | proposal recall | proposal precision | legal task commit | mean task recall | wrong parts |
|---|---:|---:|---:|---:|---:|
| SC9 RGB + parent surface | 50.0% | 55.56% | 2/6 | 33.33% | 6 |
| SC9T RGB + ray triangulation | 40.0% | 44.44% | 1/6 | 16.67% | 8 |
| SC10 generic RGB + native depth | 50.0% | 50.0% | 1/6 | 16.67% | 6 |

Ray triangulation changed the 3D representation but regressed, while native
depth removed the ray-depth ambiguity without removing generic detector false
handles. These consumed negatives are retained; detector thresholds, DBSCAN,
seeds, fusion weights, and the opened cohort must not be swept to rescue them.

SC11 changes the missing information source. A task-grounded Grounding DINO
provider proposes `drawer handle`/`cabinet knob` regions, native ARKit depth
back-projects them, parent geometry rejects cross-instance observations, and
the frozen multi-view consensus supplies functional candidates to SC8. The
proposal provider remains evaluator-truth blind.

| Arm | proposal recall | proposal precision | legal task commit | mean task recall | wrong parts |
|---|---:|---:|---:|---:|---:|
| single-view grounded RGB-D | 8/10 (80.0%) | 94.44% | 3/6 (50.0%) | 50.0% | 8 |
| **multi-view grounded RGB-D** | **10/10 (100%)** | **84.62%** | **4/6 (66.67%)** | **83.33%** | **2** |

This crosses the frozen source gate: multi-view aggregation increases task
legal commits and target-set recall while reducing wrong parts, yielding
`SC11_TASK_GROUNDED_RGBD_MULTIVIEW_FUNCTIONAL_PROPOSAL_DEVELOPMENT_SIGNAL`.
Execution used an NVIDIA GeForce RTX 5060 Laptop GPU (`cuda:0`), 248 sampled
frames, and 3.12 GB recorded peak CUDA memory. Two of six evaluable tasks still
fail because a wrong functional cluster survives inside the correct parent;
four other descriptions remain `NOT_EVALUABLE_PARENT_BINDING`.

SC11 is narrow, privileged-parent, posed-RGB-D Development evidence. It does
not establish open-vocabulary or phone-camera coverage, exact-instance
acquisition, reachability, orientation, approach pose, arrival, completion,
product benefit, user benefit, or safety. A successor must add a genuinely new
candidate-integrity or action-geometry source on separately versioned evidence;
it must not tune this opened result.

Implementations: `scenefun3d_multiview_functional_proposer.py`,
`scenefun3d_depth_functional_proposer.py`, and
`scenefun3d_grounded_depth_functional_proposer.py`. Evidence:

- SC9: `artifacts.local/evidence/l10-r11/sc9-multiview-functional-proposer-v0/result.json`,
  SHA-256 `ebb57ab5acb8e746f142ae4c87bd431607eaeac3e03da0cd8bceade535b578ee`.
- SC9T: `artifacts.local/evidence/l10-r11/sc9t-multiview-ray-functional-proposer-v1/result.json`,
  SHA-256 `298df0805d383ee0229b1a68ec37523e7d1a8895da837a4dba0587feaa269099`.
- SC10: `artifacts.local/evidence/l10-r12/sc10-native-depth-functional-proposer-v0/result.json`,
  SHA-256 `3bcc5fac308eeedde023c1f26329d46cd2452bc9af6a2e2fd5e74a8fd6676ea6`.
- SC11 provider: `artifacts.local/evidence/l10-r13/sc11-task-grounded-rgbd-functional-proposer-v0/provider.json`,
  SHA-256 `bc8e7f509cf2ebc1538127fe3dcdf59663292145c7c042c8ed81303f118a0d0b`.
- SC11 result: `artifacts.local/evidence/l10-r13/sc11-task-grounded-rgbd-functional-proposer-v0/result.json`,
  SHA-256 `ec188e8396159b3cc364bd759be7328eb1d458e6082ca219dd98e99af1e9c380`.

## L10-SC12--SC14: do not infer action kinematics from static appearance

SC12 advances from functional grounding toward orientation and handoff while
keeping completion authority separate. It predicts two axes: where the camera
should approach and face the authorized functional contact, and how that
contact should translate or rotate. Motion annotations are loaded only after
the action-pose provider is sealed.

The first static representation failed on a source-disjoint SceneFun3D scene
(`421013 / 42444703`). Parent-surface geometry already gave `7/9` signed
translation-direction hits, while task-signed geometry fell to `6/9`; mean
angular error regressed from 20.40 to 30.39 degrees. SC13 therefore decoupled
the parent-facing approach axis from a local 3D contact-surface action axis and
kept door/window kinematics set-valued. On another fresh scene
(`421010 / 42444709`), direction hits remained `4/6` and mean error improved
only from 30.39 to 28.18 degrees. Decisions:

- `SC12_TASK_CONDITIONED_DUAL_AXIS_ACTION_POSE_GATE_NOT_MET`;
- `SC13_DECOUPLED_APPROACH_AND_LOCAL_ACTION_FRAME_GATE_NOT_MET`.

These are consumed static-geometry negatives. Task-verb remapping, local PCA
radius, OBB face selection, thresholds, seeds, and geometry fusion must not be
swept to rescue either opened scene. The failure is structural: static geometry
can suggest an axis but does not reveal whether a door slides or hinges, nor
whether a control surface actually moves along that axis.

SC14 changes the information source instead of the matcher. A causal action
belief consumes paired before/after points on the already-authorized functional
region. It compares translation-only and rigid-motion explanations, recovers a
signed translation direction or a rotation axis and pivot line, and locks the
action model only after motion evidence. The canary freezes a 2 cm translation
or 5 degree rotation, 1.5 mm measurement noise, and at most 128 evaluator-paired
points. A rank-deficient pivot-line solve is gauge-fixed by choosing the point
on the axis closest to the coordinate origin; this is a numerical invariant,
not a scientific parameter.

On `421013`, eight actions had enough paired functional points:

| Arm | motion type | translation direction hit | mean translation error | rotation axis / pivot |
|---|---:|---:|---:|---:|
| static task/parent geometry | 7/8 | 4/7 | 38.89 deg | not action-observed |
| **SC14 causal micro-motion** | **8/8** | **7/7** | **0.91 deg** | **1/1 axis; 2.22 cm pivot-line error** |

This crosses the frozen gate and yields
`SC14_CAUSAL_MICRO_MOTION_ACTION_BELIEF_MECHANICS_SIGNAL`. The independent
`421010` diagnostic produced only four sufficiently dense paired regions and is
`SC14_NOT_EVALUABLE_INSUFFICIENT_CAUSAL_PROBES`; its causal translation arm was
still `4/4` versus the static `2/4`, but it is not promoted to confirmation.
The `421005` source audit had only two parent-bound tasks and is likewise
`NOT_EVALUABLE`, not negative evidence.

SC14 is a simulated paired-point mechanics ceiling. SceneFun3D motion truth
generates the perturbation, point correspondences are evaluator-authoritative,
and no real user action was requested. It does not establish RGB tracking,
safe probing, collision-free reachability, body orientation, arrival,
`HANDOFF_READY`, user completion, product benefit, or safety. The next legal
source is passive natural before/after RGB-D motion or an explicitly authorized
benign micro-interaction protocol; static action-axis guessing is closed.

### Runtime landing: causal action belief cannot bypass endpoint readiness

The SC14 representation now has a pure-Kotlin runtime contract in
`core/assist/.../goal/CausalActionGeometry.kt`. It fits paired parent-frame 3-D
points with a Horn rigid transform, keeps low-motion or high-residual evidence
`SET_VALUED`, and emits a signed translation axis or rotation axis plus a
gauge-fixed pivot line only as `LOCKED`. Admission is fail-closed on provider
identity, goal/session/parent binding, exact current frame, comparable clock,
availability, and expiry.

`GoalHandoffEvent.HandoffReady` now requires the result of an endpoint join.
That join accepts only a current `LOCKED` action belief together with `READY`
position, visibility, grounding, orientation, and reachability. A blocked join
is rejected by the reducer, while `CompletedByUser` still requires the existing
explicit button or voice confirmation. This is implementation landing, not a
new empirical result: no live paired-RGB-D provider has been admitted, the
SceneFun3D canary cannot enter the product path, and no arrival, product, user,
or safety claim changes.

Implementations: `scenefun3d_action_ready_pose.py`,
`scenefun3d_decoupled_action_pose.py`, and
`scenefun3d_causal_action_probe.py`; runtime bridge:
`core/assist/src/main/java/com/linnan/blindassist/goal/CausalActionGeometry.kt`.
Evidence:

- SC12: `artifacts.local/evidence/l10-r14/sc12-task-conditioned-dual-axis-action-pose-v0/result.json`,
  SHA-256 `6373cb5595a43a824649b852b5771a3c096e1957ae0910f87be7671ef46156ce`.
- SC13: `artifacts.local/evidence/l10-r15/sc13-decoupled-approach-local-action-frame-v0/result.json`,
  SHA-256 `f828369e8dfab033befc21e5b14bcac03718ae1177fd1381b17d1b1c21df8a07`.
- SC14 observations: `artifacts.local/evidence/l10-r16/sc14-causal-micro-motion-421013-v0/observations.json`,
  SHA-256 `9c7a42ac44b32c75be22bbb14ad9b410d40dc6e6237a13e634c5260f1d31d946`.
- SC14 provider: `artifacts.local/evidence/l10-r16/sc14-causal-micro-motion-421013-v0/provider.json`,
  SHA-256 `2208d2ee765273d4fb9050c2e88471a065257e3dd505760d4dd8dffb8e787e9f`.
- SC14 result: `artifacts.local/evidence/l10-r16/sc14-causal-micro-motion-421013-v0/result.json`,
  SHA-256 `bb11fa3b87ea350ad793d3c62daf81e183b6806f08c82b2c1df1535677d1ca01`.

## Public text-source admission

The 2026-08-28 source canary changed the information source rather than the
matcher. DSText V2's official Zenodo record exposes 18 CC-BY-4.0 video archives
totalling 448,078,468 bytes, and a 9.6 MB sample contained two readable MP4s.
After RRC registration, the official V2 training annotation archive was also
retrieved: 90 XML ground-truth files, 21,393,054 compressed bytes, SHA-256
`7f0e72642530390d96f8983e666fba236d3d57c1e1853c5a3fe95c972d2a03f4`.
The source result is now `DSTEXT_V2_ANNOTATION_SOURCE_ADMITTED`: official media
and temporal annotation authority are available, so a frozen track-gap audit
may proceed. This admission alone is not a track-gap, presence, or semantic-
reacquisition result.

HierText supplies an independent geometry authority without pretending to fill
that temporal gap. Across all 1,724 validation images, the fixed cohort contains
20,400 legible printed multi-word lines and 85,380 legible words. The complete
line center and the target word center disagree on L10's frozen coarse direction
for 20,866 words (`24.439%`); mean absolute normalized horizontal displacement is
`0.0751`, with p90 `0.1891`. This admits word polygons as evaluator geometry for
the merged-line carrier problem. It does not establish temporal identity,
reacquisition, active-view utility, arrival, or handoff.

Evidence:

- DSText source result:
  `artifacts.local/evidence/l10-dstext-v2-source-canary/result.json`, SHA-256
  `01a27d353bc7517f1a85e85f0c32840892e76b461aa0312308deff2e05bb411d`.
- HierText geometry result:
  `artifacts.local/evidence/l10-hiertext-word-carrier-canary/result.json`,
  SHA-256
  `7faf93015fc4b46afa17c02678b024239529a68afc9f8a110c790eacda5ce728`.
- Truth-only adapter sample:
  `artifacts.local/evidence/l10-hiertext-word-carrier-canary/adapter.jsonl`,
  SHA-256
  `3bc752813610e5123c2d7b8c825efb7133781ed8f1e8b84251b146a321fe3668`.

## Android text-goal canary

`apps/demos/semantic-anchor-demo-app` is now the runnable L10-R0 surface. Its
CameraX analyzer feeds ML Kit line/block text boxes into a pure Kotlin
candidate-bound controller. The controller exposes the full demonstration
loop:

`SEARCH -> TARGET_FOUND -> LOCKED -> LEFT/RIGHT/FORWARD -> LOST -> SCAN -> REACQUIRED -> NEAR -> TASK_COMPLETE`

Belief belongs to one box trajectory, combines short predicted motion with a
slower spatial prototype, rejects a distant same-text handoff while locked, and
requires two fresh hits after LOST. NEAR/TASK COMPLETE requires three centered
large-box frames. That last signal is explicitly a visual-scale proxy, not
metric distance, navigation safety, or user-confirmed task completion. The
current bundled recognizer is Latin-script, so the first canary targets are
English/numeric room signs, exits, entrances, elevator labels, and service-desk
signs.

The focused JVM check is:

```powershell
pwsh -NoProfile -File scripts/run_android_gradle.ps1 `
  :semantic-anchor-demo-app:testDebugUnitTest `
  --tests com.linnan.blindassist.semanticanchor.SemanticAnchorSessionTest
```

Mechanism references: [GoMatching long/short matching](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2d66a70c770de7835678f1c1e65fe5e1-Abstract-Conference.html),
[QueryNLT joint language/visual tracking](https://openaccess.thecvf.com/content/CVPR2024/html/Shao_Context-Aware_Integration_of_Language_and_Visual_References_for_Natural_Language_Tracking_CVPR_2024_paper.html),
and [AECNav evidence consolidation](https://arxiv.org/abs/2608.10817).
