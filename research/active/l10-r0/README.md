# L10-R0 Goal-Lock Copilot

Status: `ACTIVE`

- **Core controller:** SC1W--SC2 seek/guide/reacquire and the SC14 causal
  action-belief handoff guard remain implemented; the later SceneFun3D ordinal
  source line terminates at `SC40_NOT_EVALUABLE_NO_FRESH_DEPTH_VISIBLE_ACTIVE_VIEW`.
- **Active observation:** PanoLab passed
  `L10_PANOLAB_ACTIVE_ENTRANCE_RAY_RECOVERY_DEVELOPMENT_GATE_MET` (`4/4`
  reciprocal `SIDESTEP_TO_ENTRANCE_FACE` recoveries). This authorizes an exact
  entrance ray geometrically, not a pixel portal.
- **SEVN backend:** after the frozen V2 stack failed to confirm on a fresh
  panel, a portal-private PP-OCRv6 medium witness changed the observation edge
  rather than its scoring thresholds. On 40 further fresh PAN episodes with
  zero overlap against all 205 prior addresses and 220 prior frames, exact
  visible-number OCR improved `22/37 -> 29/37` and binding improved
  `16/0/24 -> 21/0/19` correct/wrong/UNKNOWN. All five new proposals were
  correct, both ambiguous witness sets abstained, and precision stayed `100%`.
  This passes the frozen same-source Development gate; it is not cross-provider
  or cross-city confirmation.
- **Pixel portal:** generic Panoramax mining is closed after the width-first
  source admitted `0/3` reference, `0/3` query, and `0/3` joint portals. It may
  carry imagery only with independent sub-metre portal registration.
- **Posed transfer:** Hypersim met the synthetic Development gate; SceneNN is
  terminal at
  `L10_SCENENN_VISIBLE_METRIC_PORTAL_TRANSFER_DEVELOPMENT_GATE_NOT_MET`.
  Strict-triangle z-buffering repaired reference-plane contamination, but the
  edge-clipped credential still lacked complete portal extent. 3RScan E0 then
  met `L10_3RSCAN_REGISTERED_ENDPOINT_EXTENT_DEVELOPMENT_GATE_MET`: complete
  registered extent raised median planar IoU `0.2727 -> 0.7688` and reduced
  median world-centroid error `0.3760 m -> 0.1428 m` with `0/3` wrong doors. The
  first learned RGB successor improved one fresh rescan `0.0706 -> 0.3312` IoU,
  but its coordinate homography extrapolated outside the door and did not meet
  the canary.
- **3RScan exact-target binding:** coherent-cycle transport confirmed on three
  new physical targets and rejected `4/4` cross-scene target-absent pairs, but
  one same-scene sibling door remained a stable false binding under bilateral
  masks, an official Doppelgangers classifier, two references, and a displaced
  second query. A universal complementary corroboration fixed that consumed
  panel (`3/3` positives, `0/2` false commits) but rejected both newly frozen
  FC31/FC08 positives because its specialist global/active supports did not
  transfer; the new exact-target-absent FC30 negative was correctly rejected.
  A zero-model selective cascade now commits strong local bindings directly and
  requests complementary corroboration only for ambiguity. Across both consumed
  panels it reached `5/5` positives, `0/3` false commits, precision `1.0`, and
  requested the extra branch for only `1/8` rows. Its post-observation `0.25`
  direct-exit point is Development only. On a frozen never-consumed SC34 target
  from a new scan family, cycles were strong (`0.443/0.981`) and both new
  cross-scene negatives had zero cycles, but the extent gate rejected a stable
  wrong-structure binding at IoU `0.065`. Robust projective transport raised IoU
  only to `0.082`; geometry fitting is frozen as the wrong repair layer.

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
blindassist-python.cmd -B `
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
blindassist-python.cmd -B `
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

### SC15--SC17: cheap active-view proxies do not predict identity gain

SC15 first asked whether the frozen SC2 action already aligned with useful
natural view changes on the consumed video1+video10 Development stream. Action
output was sealed before evaluator target geometry and target-associated OCR
outcomes were read. There were 17 evaluable `PAN/SCAN` opportunities: 12
geometrically aligned and five opposed. Aligned transitions had lower mean
semantic gain than opposed transitions (`0.6667` versus `0.8000`) and created
wrong semantic gates in 16.67% versus 0%. Decision:
`SC15_PASSIVE_ACTION_CONDITIONED_OBSERVATION_GAIN_GATE_NOT_MET`.

SC16 then changed both source and representation. Before any payload or OCR
outcome was opened, a protocol selected the only `Street_View_Indoor` entry in
the already-admitted DSText V2 archive and defined a three-coordinate Pareto
view quality vector: rectified text height, crop sharpness, and horizontal
centering. The official track quadrilateral was evaluator-authoritative and
PP-OCRv6 recognition-only supplied the independent semantic result. Across 216
natural frames, 49 eligible tracks, 2,824 observations, and 2,677 transitions,
422 were Pareto-aligned and 444 Pareto-opposed. Aligned minus opposed mean
semantic gain was only `+0.0093` against the frozen `+0.10` gate; improvement
rate delta was `+0.0322` against `+0.15`, and semantic gate crossing was lower
(`1.66%` versus `4.28%`). Decision:
`SC16_PARETO_OBSERVABILITY_SEMANTIC_GAIN_GATE_NOT_MET`.

One final Development-only mechanism check reused the now-consumed SC16 OCR
observations without opening another video. SC17 required an exact normalized
majority across views at offsets `0/3/6`. Precision increased only 90.97% to
91.88% (`+0.91 pp`), wrong outputs fell 24.31%, correct coverage remained
85.05%, and five of 49 tracks still formed at least one wrong consensus. It did
not earn a fresh test:
`SC17_TRIVIEW_SEMANTIC_CONSENSUS_DEVELOPMENT_GATE_NOT_MET`.

These terminals close centering/last-bearing as an observability proxy, a
height-sharpness-centering Pareto rule, and fixed three-view OCR majority on
their opened evidence. Do not sweep horizon, feature weights, quality
thresholds, edit distance, vote count, or semantic gates. The next eligible
algorithm needs a new source that records an actually issued observation action
and its before/after result, then learns or repairs policy from causal outcome;
alternatively it must introduce genuinely new identity information. Passive
video association remains useful only as a source audit.

Implementations: `artvideo_passive_action_gain.py`,
`dstext_pareto_observability.py`, and
`dstext_triview_semantic_consensus.py`; frozen SC16 protocol:
`dstext_observability_protocol_v0.json`. Evidence:

- SC15 result: `artifacts.local/evidence/l10-r17/sc15-passive-action-conditioned-observation-gain-v0/result.json`,
  SHA-256 `e1d170dda9fedf5b2edb38d7c107dd84431bad8f40fa9373c5131fd646ff5455`.
- SC16 protocol: `research/active/l10-r0/dstext_observability_protocol_v0.json`,
  SHA-256 `cc7bea02855763cd3907094c977a9aa87957fa88314d7312447d7bf4fb7299bd`.
- SC16 result: `artifacts.local/evidence/l10-r18/sc16-dstext-pareto-observability-v0/result.json`,
  SHA-256 `9b97f1c59945d46e891f1ee584d1d7372cc47432ba8f64e75049e1cc56cf9f14`.
- SC17 result: `artifacts.local/evidence/l10-r19/sc17-triview-semantic-consensus-development-v0/result.json`,
  SHA-256 `7807d5f419ef6d81690a3896fcf4ac63f26045dd671cd88973e94b182c148e4e`.

### SC18: causal action-outcome receipt and one-step policy repair

SC18 implements the successor required by the three proxy negatives. Every
runtime observation-seeking instruction now emits a goal/session/entity/frame/
clock-bound receipt containing the pre-action semantic state and evidence
deficit. Only a later, admitted, comparable observation may close it as
`IMPROVED`, `NO_GAIN`, or `CONTRADICTED`; absent, stale, mismatched, or
non-admitted evidence closes as `UNKNOWN`. A matching execution acknowledgement
is mandatory; issuing a prompt alone cannot claim an action effect. A
comparable no-gain outcome stops
the controller from repeating the same instruction and selects a distinct
repair action. Guidance actions do not manufacture observation receipts.

A protocol-frozen controlled-world check compared the unchanged GoalLock
controller with this one-step repair on 250 episodes (200 target-present, 50
target-absent). Repeated same-action decisions after an observable no-gain
transition fell from `182/206` (88.35%) to `0/215`; target-present task success
was unchanged at 88.5%, and target-absent false completion remained 0/50.
Reacquisition decreased slightly from 82.35% to 80.75%, inside the frozen
five-point non-inferiority margin. All preregistered gates passed, yielding
`SC18_CAUSAL_ACTION_OUTCOME_REPAIR_MECHANICS_SIGNAL`.

The primary reduction is partly structural—the candidate is designed not to
repeat the just-failed action. Its useful evidence is the absence of a task-
success or false-completion regression in this controlled world, not a claim
that the opposite pan is optimal. This remains synthetic search mechanics: no
phone/user action was executed, simulator state is not natural observation,
and no live readability, identity, arrival, product, user-benefit, or safety
claim follows. The next algorithmic step is deficit-conditioned action utility
learned from real issued-action receipts, not another static image-quality
proxy or a repair-rule sweep on this consumed seed.

Implementations: `GoalObservationActionOutcome.kt`, the receipt/repair path in
`GoalCopilotController.kt`, `action_outcome_benchmark.py`; frozen protocol:
`action_outcome_protocol_v0.json`. Evidence:

- SC18 protocol: `research/active/l10-r0/action_outcome_protocol_v0.json`,
  SHA-256 `103287b676fc81a3a5c27d74d2fcf3cab143e4b24f384caf075659d4da27b680`.
- SC18 result: `artifacts.local/research/l10-r0/sc18/action_outcome_result.json`,
  SHA-256 `92eb8554559e0012aad4db4f8cdbcc901558c25ecfa5f2bb7b04847b4b12413b`.

### SC19: deficit-conditioned online action utility

SC19 removes the fixed repair graph as the long-term policy. A bounded
contextual UCB learner now keys utility by observation deficit and camera-
relative bearing, updates only from execution-confirmed `IMPROVED / NO_GAIN /
CONTRADICTED` receipts, deduplicates receipt IDs, and ignores `UNKNOWN`
entirely. Its candidate sets are safety boundaries rather than learned
preferences: for example, `ASSOCIATION_AMBIGUOUS` cannot explore
`APPROACH_FOR_IDENTITY`. The policy object can be shared across goal sessions,
while its public snapshot contains only context/action counts rather than goal
identity.

The frozen controlled mechanism world contains seven heterogeneous deficit
contexts, 180 trials per context, a fixed 15% unknown-outcome process, and the
same `0.35` exploration coefficient as runtime. Compared with SC18's fixed
one-step repair:

| Metric | fixed repair | contextual utility |
|---|---:|---:|
| authoritative improvements | 419/1,045 (40.10%) | 658/1,045 (62.97%) |
| expected cumulative regret | 302.40 | 20.38 |
| final-window optimal-action rate | 42.86% | 99.52% |
| ambiguous-context unsafe approach | 100% | 0% |

Improvement rose `+22.87 pp` and expected regret fell 93.26%, crossing every
preregistered gate and yielding
`SC19_DEFICIT_CONDITIONED_ACTION_UTILITY_MECHANICS_SIGNAL`. This is a stronger
algorithmic mechanism than fixed opposite-pan repair, but the effect sizes come
from authored context probabilities. They do not establish natural action
utility, live-phone compliance, causal readability, exact identity, arrival,
product benefit, user benefit, or safety. Real promotion requires logging the
same execution-confirmed receipt schema on live actions and checking whether
the learned ranking transfers without changing the frozen candidate safety
sets.

Implementation: `GoalActionUtilityPolicy.kt`; controlled evaluator:
`action_utility_benchmark.py`; frozen protocol:
`action_utility_protocol_v0.json`. Evidence:

- SC19 protocol: `research/active/l10-r0/action_utility_protocol_v0.json`,
  SHA-256 `7d5cef68311c9047bbb431ae9a8549fbae11569f2ea51a4ffbda73fd6eaf03ce`.
- SC19 result: `artifacts.local/research/l10-r0/sc19/action_utility_result.json`,
  SHA-256 `499aa8411eacc9711000eec4cde6ad94897ec18272362f918139edc3e264de62`.

### SC20: factorized functional endpoint observer

SC20 pauses device/demo integration and attacks a different missing algorithm
layer: rejecting false arrival. It reuses the consumed SceneFun3D 420683 ARKit
trajectory and SC11's frozen task-functional candidate sets, then compares the
old centered-large-parent proxy with an explicit join of horizontal stand-off,
depth-consistent functional-point visibility, camera orientation, and upstream
grounding support. Reachability has no authority in this source and is fixed to
`UNKNOWN`; therefore neither arm may emit `HANDOFF_READY`.

Across six tasks, 901 real posed RGB-D frames, 5,406 task-frames, and 185
evaluator observation-ready frames:

| Metric | centered-large parent proxy | factorized functional endpoint |
|---|---:|---:|
| ready frames | 429 | 121 |
| true / false ready | 129 / 300 | **120 / 1** |
| precision | 30.07% | **99.17%** |
| recall | **69.73%** | 64.86% |
| F1 | 0.420 | **0.784** |
| tasks with at least one true-ready frame | **6/6** | 5/6 |

The result is formally
`SC20_FACTORIZED_ENDPOINT_GATE_NOT_MET`: the frozen gate required no loss of
task coverage. The missing under-bed drawer task had 35 evaluator-ready frames,
but no factorized output jointly satisfied stand-off and visibility; at valid
stand-off, no more than one of its three selected SC11 points was depth-visible.
This localizes the next problem to candidate integrity/visibility rather than
justifying threshold repair. The 300 -> 1 false-ready reduction is a strong
mechanism effect, but it does not override the failed coverage gate.

Implementation: `scenefun3d_factorized_endpoint_observer.py`. Evidence:

- SC20 provider:
  `artifacts.local/evidence/l10-r19/sc20-factorized-endpoint-420683-v0/provider.json`,
  SHA-256 `fe1ab068991e1f70881c33c96eec76477975ca2cb0021ae0ce87380b7498ec5c`.
- SC20 result:
  `artifacts.local/evidence/l10-r19/sc20-factorized-endpoint-420683-v0/result.json`,
  SHA-256 `95a5a36b6cb0074689b0b98c14d9e3d4e2ec38c0e8ee6bfc7bee203aaf2f3188`.

This is a Development diagnostic on an already-consumed path with privileged
parent bindings, not a fresh confirmation or controller-executed trajectory.
Do not sweep endpoint thresholds on 420683. The next endpoint successor must
change the information source: separately versioned functional candidate
integrity, or independent free-space and human-reachability evidence. Explicit
user confirmation remains the only completion authority.

### SC21--SC22: candidate-set integrity closes the localized endpoint gap

SC20's only uncovered task contained a structurally suspicious selected set:
two candidates lay 5--6 cm from the under-bed drawer truth, while a third was
1.445 m away. SC21 introduces one representation change before endpoint
reasoning. It builds connected components in parent-normalized 3-D coordinates
inside the already selected task-relational set. A candidate may be quarantined
only when there is one unique largest component with at least two members;
ties, all-singleton sets, and singleton selections stay `SET_VALUED` and request
another integrity view. Exact-parent identity authority is unchanged.

The component radius (`0.4` normalized parent extent), minimum component size
(`2`), source roster, and gates were frozen before opening the source-disjoint
`421002 + 420673` cohort. That formal source cannot adjudicate SC21:

- 20 task descriptions;
- only 3 parent-bound evaluable tasks and 17 `NOT_EVALUABLE_PARENT_BINDING`;
- zero integrity opportunities;
- decision `SC21_NOT_EVALUABLE_INSUFFICIENT_PARENT_BOUND_TASKS`.

This rejects the cohort as an integrity benchmark; it is not a negative result
for the algorithm. One permitted read-only diagnostic then applied exactly the
same frozen rule to the consumed SC11 RGB-D candidates. It produced one
integrity opportunity, quarantined the 1.445 m under-bed outlier, and retained
the two 5.33/5.80 cm candidates:

| Consumed SC11 diagnostic | task-relational set | + SC21 integrity |
|---|---:|---:|
| legal functional commits | 4/6 | **5/6** |
| mean target-set recall | 83.33% | **83.33%** |
| wrong parts | 2 | **1** |

SC22 then composes that fixed representation with the unchanged SC20
stand-off, depth-visibility, and orientation factors. It has no new parameter:

| Consumed 420683 endpoint metric | SC20 | SC21 + SC20 |
|---|---:|---:|
| ready frames | 121 | 155 |
| true / false ready | 120 / 1 | **154 / 1** |
| precision | 99.17% | **99.35%** |
| recall | 64.86% | **83.24%** |
| F1 | 0.784 | **0.906** |
| tasks with true-ready evidence | 5/6 | **6/6** |

The under-bed task alone moved from `0/35` to `34/35` true-ready frames. This
crosses every frozen SC22 composition gate and yields
`SC22_INTEGRITY_ENDPOINT_COMPOSITION_MECHANICS_SIGNAL`. It establishes a strong
compositional mechanism: factorized arrival works better when selected
functional sets first reject spatially disconnected hypotheses. It is still a
read-only result on an already-consumed real RGB-D path, not fresh transfer or
controller-executed motion.

Implementations and protocols:

- `scenefun3d_functional_set_integrity.py` and
  `functional_set_integrity_protocol_v0.json`, protocol SHA-256
  `c882f9dee4f3a2617578459886a7cef869ca6db7968c0993d240d95bb0e06986`;
- `sc11_functional_set_integrity_diagnostic.py`, consumed diagnostic SHA-256
  `ffa0832c6d8b92d9c1b4a09ee52c4aab074992ce1fa97c09a08405f124565d57`;
- SC21 fresh provider/result SHA-256
  `d7a417b9f8e8a5642c3903c824fbe72fdaa90eb7fc95d499d822708111d29501` /
  `cd397293816ca5b3e53bfd13a90252e9a2491fb4b3334313fc86dad45504cde4`;
- `scenefun3d_integrity_endpoint_observer.py` and
  `integrity_endpoint_protocol_v0.json`, protocol SHA-256
  `785d47525296e52a0466f2a9edcec60727a82a4fbe5ae9546c210fac2e5767e9`;
- SC22 provider/result SHA-256
  `a19844bc12be56966f3d0063dda60cb56a67329aed361b1f71866dfc5b7d67c9` /
  `bbe7bc15f52db13b9215d51f3d68ebd29ee40f03dcb2bee262d1f1c3d232812e`.

Do not tune either component on 420683. Fresh confirmation requires a real
RGB-D proposal cohort with enough authorized parent-bound tasks and actual
integrity opportunities. Reachability is still `UNKNOWN`, so SC22 cannot emit
`HANDOFF_READY`; explicit user confirmation remains the only completion
authority.

### SC23--SC29: semantic witnesses prevent topology from deleting another function

SC23 used a frozen, outcome-blind source admission over the official SceneFun3D
one-video roster. It selected the first three scenes meeting description and
multi-target-count requirements, without running the selector. The fresh result
showed that SC21's spatial coherence is insufficient by itself:

| Fresh SC23, 37 evaluable tasks | task-relational set | + topology |
|---|---:|---:|
| legal commits | 12 | **13** |
| mean target-set recall | **93.24%** | 90.54% |
| wrong parts | 50 | **49** |

Both integrity opportunities shared one parent. For `Plug the device in the
socket behind the TV`, the isolated candidate was a wrong `key_press` region;
for `Turn on the TV using the remote control`, that same isolated candidate was
the only target. Geometry had no authority to distinguish the tasks. SC24 adds
a task-action semantic witness before unchanged topology. It never receives
target membership, but its functional action labels are a privileged
SceneFun3D proposal ceiling:

| Consumed SC23 diagnostic | SC21 topology | + exact action semantics |
|---|---:|---:|
| legal commits | 13 | **16** |
| mean target-set recall | 90.54% | **93.24%** |
| wrong parts | 49 | **41** |

Fresh SC25 had 31 evaluable tasks but only two exact semantic admissions, below
the frozen minimum four, so it is `NOT_EVALUABLE`. SC26 then introduces
hierarchical action families: exact cues remain precise, while broad
`open/close` and activation families exclude only other families and retain all
within-family candidates. On consumed SC25 this raised legal commits `9 -> 11`,
recall 72.58% -> 79.03%, and reduced wrong parts `61 -> 54`.

Fresh SC27 exposed a no-regret failure. The family layer reduced wrong parts
`32 -> 27` and raised legal commits `5 -> 7`, but recall fell 83.33% -> 79.17%:
an `Open the storage drawer of the fireplace` target had action label `rotate`,
while one `hook_turn` candidate satisfied the broad open family. SC28 therefore
requires redundant family support. Exact cues may filter with one compatible
candidate; a broad family needs at least two compatible candidates or the full
set is preserved. On consumed SC27 it restored recall to 83.33% and retained a
`32 -> 30` wrong-part reduction.

Fresh SC29 was fully evaluable but did not exercise the mechanism: 29 tasks, 27
semantic admissions, zero changed tasks, and identical arms (12 legal commits,
82.76% recall, 21 wrong parts). Its gate is correctly not met. Multi-target
count is not the right source denominator for semantic integrity. The next
source admission must count same-parent cross-action-family conflicts using
provider-public candidate labels and geometry, while remaining blind to target
membership and algorithm outcome.

Key evidence SHA-256:

- SC23 fresh provider/result:
  `17f636394b9691ccf1238e792b1fca28b4230b92cdc560713ccb22cb4a052174` /
  `bd1291efe48781056ade2d205c2f14fd854b9119a78ff77718a1aef249657994`;
- SC24 consumed semantic provider/result:
  `2e2e74011342390afc99ca65075177ac49b82f1a995080203b10c316c5d8dc1b` /
  `2eef62fe41c98987b84d3fa08cfe01974f9cacbcf7550d42320756747e1dc91d`;
- SC25 fresh result:
  `7f7051cc6a149f9d77821cef31b5334d3947d765fef60629d3363bb08f195c93`;
- SC26 consumed action-family result:
  `e774357d1f3d363e0215d59d243e18cf196ecf9526a20dffe9fe32b2d87b301b`;
- SC27 fresh result:
  `386dbd9bac97719a25e4b009aa74f3c73c8a949848c1e6a369b9887e9196301b`;
- SC28 consumed redundancy-gated result:
  `87b459a3cc35896fd476b6216b161ae91cd2c57e10d7daf7b48aea9a69c6e07c`;
- SC29 fresh provider/result:
  `dd8c5bfb7c097e4729d71e29a352db841e73dfa740ef682cde9f84848602b443` /
  `3fc9ca5aeceba09a719c08c316711f9b5f24e768b42dc94c67ac6eceef9bec7b`.

Implementations: `scenefun3d_integrity_source_admission.py` and
`scenefun3d_semantic_topology_integrity.py`. These results do not establish RGB
semantic proposal quality, exact-instance acquisition, reachability, arrival,
`HANDOFF_READY`, product benefit, or safety. Do not tune the opened cohorts or
semantic rules; change the source opportunity representation next.

### SC30--SC31: conflict-aligned source admission earns the first fresh signal

SC29 showed that semantic admission count is not enough: a scene may contain
many action-cued tasks while no authorized parent contains cross-action-family
interference. SC30 therefore moves the opportunity definition entirely into
provider-public proposal space. It groups functional proposals by parent OBB
and three frozen action families (`CONNECT`, `ACTIVATE_CONTROL`, and
`OPEN_CLOSE`). A scene is eligible only when one parent contains at least two
families and at least one family has two proposals, matching SC28's redundancy
gate. Description target IDs, target membership, task selector output, and
evaluator scores are never read.

The bounded V1 roster found only one eligible scene in 20 candidates. V2
changed only the roster ceiling from 20 to 80; ordering, mappings, thresholds,
and required three-scene count stayed fixed. It stopped after 32 candidates and
admitted `421267 / 42444733`, `422356 / 42446579`, and
`422377 / 42447329`, each with one redundancy-eligible conflict parent.

The unchanged SC28 algorithm then crossed every frozen SC31 aggregate gate:

| Fresh conflict-admitted SC31, 16 evaluable tasks | SC21 topology | + redundancy-gated action semantics |
|---|---:|---:|
| legal commits | 3/16 (18.75%) | **4/16 (25.0%)** |
| mean target-set recall | **62.5%** | **62.5%** |
| wrong parts | 24 | **20** |
| cross-parent violations | 0 | **0** |

There were 10 semantic admissions and two metric-changing tasks. `Unplug the
printer` improved from zero target recall and two wrong parts to one legal
target. Conversely, `Plug the device in the second outlet` lost its target
while removing two wrong parts: all candidates shared the correct action label,
but the current topology lacked the requested outlet's ordinal axis. SC31 is
therefore a fresh aggregate Development signal, not task-wise no-regret or
general functional grounding.

Evidence SHA-256:

- SC30 V1 result:
  `03ce206143c976e8ac5558f9a4b54736db1c2831dac88468d5c7a93bd93ad6e0`;
- SC30 V2 protocol/result:
  `13e1f9026006b9505fc830e247e6f9084a98cd5997ceece5c82f9d39ec709c54` /
  `c45b4915fa3ece3fe1bd376f1746cd42e0b1e8897c5971f7e1d3f0e510550ab0`;
- SC31 protocol/provider/result:
  `ca4f5316c064dab32600ee7d39cdbddb96620f85052a24ae1ad24f6d93552355` /
  `5606754b9db206bb26ec21b6beb013e77945a9a896bc87eaa0a13a135306b4a7` /
  `7d594da76e7e812c9176993a819cd61144525133341fb00e23c5ef450810474e`.

Implementations: `scenefun3d_conflict_source_admission.py` and the unchanged
`scenefun3d_semantic_topology_integrity.py`. Do not tune this cohort. The next
eligible successor must add a frozen task-conditioned ordinal representation
for repeated same-action controls on separately admitted evidence. SC31 still
uses privileged functional labels and does not establish RGB proposal quality,
reachability, arrival, `HANDOFF_READY`, product benefit, or safety.

### SC32--SC34: polarity-safe ordinal grounding recovers the middle control

SC32 froze a parent-normalized, boundary-anchored absolute slot axis. Its
80-candidate source roster admitted zero scenes. A sealed diagnostic showed why:
three outlet proposals were regular and collinear, but their authorized parent
was the enclosing bed OBB, not the power strip. The support-object boundary is
therefore not an honest source of outlet numbering; the opened thresholds and
cohort remain closed.

SC33 changes the representation instead of the matcher. For a complete public
rank inventory over an odd-length provider-public lattice, it retains both PCA
axis polarities. Requested rank `k` maps to both positions `k` and `N+1-k`;
only when those hypotheses coincide may the selector uniquely commit. Thus the
middle of three candidates is grounded without inventing left/right authority,
while first and third remain set-valued and request an orientation view.

The outcome-blind admission stopped after 16 candidates on
`421658 / 42445769`. Fresh SC34 results on its three outlet tasks were:

| Metric | SC31 baseline | + orientation quotient |
|---|---:|---:|
| legal commits | 0/3 | **1/3** |
| mean target-set recall | 66.67% | **100%** |
| wrong parts | 4 | 4 |
| task-wise regressions | 0 | **0** |

`second outlet` changed from an empty selection to the unique correct proposal.
The formal decision is still
`SC34_ORIENTATION_QUOTIENT_ORDINAL_FRESH_GATE_NOT_MET` because the frozen gate
also required wrong-part reduction; this route repaired a false negative rather
than removing a wrong candidate. Treat it as a narrow fresh mechanism effect
inside a composite-gate failure. Do not reopen the scene or thresholds. The next
legal source must add view/gravity polarity authority to distinguish endpoint
ranks. Privileged labels and exact parent binding remain ceilings; RGB proposal,
reachability, arrival, `HANDOFF_READY`, product, and safety claims remain
forbidden.

Execution receipts used the shared `research-l10-r0` environment and
`tools/research_backend.py`. On the real `421658` point-cloud matching probe,
NumPy CPU was measured faster than Torch CUDA (38.37 ms versus 189.67 ms median),
so CPU execution is an explicit `CPU_FASTER_MEASURED` selection rather than a
device or provider fallback.

### SC35--SC39: real-view polarity needs visibility and enough observation support

SC35 changed the information source from an unoriented lattice to real
ARKitScenes camera trajectories. The official high-resolution pose asset was
HTTP 403, while the admitted low-resolution trajectory exposed 1,870 poses on
the fresh `422155 / 42445680` scene. That scene had 18 functional annotations
but only four parent-bound annotations, no active-view parent, no 3DOD door
anchor, and no self-carrier local action lattice. All four frozen source
variants are therefore `INCOMPLETE` or `NOT_EVALUABLE`, not active-view
algorithm negatives.

The target-exposed `422200 / 42445100` scene was then used only as a consumed
mechanism diagnostic. It contains one provider-public three-outlet metric
lattice and public directional ordinal inventory `[2, 3, 4]`. SC37 found a
stable `5/5` camera-frame order, but SC38 showed that stable projection alone is
not semantic polarity: the selected reverse-side view swapped the second and
third outlets. Mean recall fell `100% -> 33.33%`, wrong parts fell `6 -> 4`, and
two of three tasks regressed, yielding
`SC38_TEMPORAL_CONSENSUS_ACTIVE_VIEW_CONSUMED_GATE_NOT_MET`. Do not flip this
opened order from target IDs or tune its trajectory/view thresholds.

SC39 adds two authorities rather than changing those thresholds. Frame-aligned
measured depth must agree with every candidate, and the public ordinal inventory
maps absolute ranks even when the first physical slot is not annotated. Across
592 poses, 592 aligned depth/intrinsic frames, 168 geometry-valid views, and 22
true-image in-frame views, all 22 in-frame views were depth-consistent for every
candidate; the best maximum candidate residual was 7.48 mm. Only four views met
the frozen horizontal-span requirement, however, so none reached the unchanged
five-pose temporal-consensus minimum. The formal source decision is
`SC39_NOT_EVALUABLE_NO_DEPTH_VISIBLE_ACTIVE_VIEW`: the visibility and hidden-slot
representations are reusable, but this source cannot adjudicate their ordinal
effect and no SC40 evaluation is authorized.

SC39 protocol/result/backend hashes are
`6214c4d34ea6697e9b66cd2a9b13d1ad7b4962f17cdeffb77f6681e6e88aca2d`,
`f7c77420c4594b268838757d22bd103df67e5fe0618f8abda65db2b7bc0efcc9`,
and `4b62ad95879fd871db97d286870ac69f1a62934f565299da0eb9369b29840989`.
The measured point-center launch selected NumPy CPU because its 214.44 ms median
beat Torch CUDA's 355.65 ms; depth download, extraction, decoding, and scalar
visibility scoring stayed CPU by contract. No fresh transfer, RGB proposal,
online view reachability, tracking, arrival, `HANDOFF_READY`, product, user, or
safety claim follows. A successor needs a separately admitted source with a
depth-visible ordinal lattice and enough naturally supported views; it must not
lower SC39's span, alignment, visibility, or temporal thresholds.

SC40 exhausted the independently unopened remainder before changing route. It
kept the SC39 `frozen_algorithm` field-for-field identical as a parsed JSON
object, resumed strictly after visit `435724`, and scanned all 128 remaining
official one-video scenes. Zero scenes contained even one provider-public
directional ordinal `plug_in` task, so no point-cloud builder, depth download,
backend selection, target binding, or evaluator was launched. The decision is
`SC40_NOT_EVALUABLE_NO_FRESH_DEPTH_VISIBLE_ACTIVE_VIEW`; protocol/result hashes
are `84690388a48a7f8017d9f6efe31b18aa41799a116c0938dc9b76b7bb81e3b714`
and `95a2238d458297f915937070e5e02453fed3530e43d10a1c5982296281224d83`.

This closes the official SceneFun3D one-video roster as a fresh confirmation
source for this exact directional-outlet contract. It does not close the
visibility-aware ordinal representation. A successor must change the
information source to a dataset or collected stream that actually contains
repeated controls, public directional rank language, synchronized depth, and
enough observable views; parser widening, action-label substitution, and
threshold relaxation on this opened roster are forbidden.

### SC41: current-scene similarity is not terminal-state evidence

SC41 changed information source rather than reopening the consumed ordinal
cohort. A broad public-source audit routed terminal verification to the official
[SWITCH](https://github.com/BAAI-Agents/SWITCH) 30% open subset instead of the
gated 1.27 TB VL-LN release or BusyBox's public LeRobot rows, whose published
feature schema does not expose the optional instrumented interface-state
telemetry. The frozen source is
`BAAI-Agents/SWITCH-Basic-v1-open@510a96b59c8688a2122d725d142c5b720962cc47`,
task `verification_state/video2img`: 16 real current-state/task videos, each
paired with four candidate outcome images and a natural-language goal.

Before truth was opened, SC41 sealed a provider for two fixed zero-shot CLIP
selectors. The baseline chose the candidate most similar to the desired visible
outcome. The successor added a goal-conditioned incomplete-state contrast and
maximum similarity to eight uniformly sampled frames from the task video. The
ground-truth field was stripped before JSON parsing and remained unavailable
through download, backend selection, embedding, scoring, prediction, and
provider sealing.

| selector | correct | accuracy |
| --- | ---: | ---: |
| desired-state text baseline | 9/16 | 56.25% |
| counterfactual + same-scene successor | 3/16 | 18.75% |

The absolute gain is `-37.5` points, with seven baseline-correct tasks regressed,
so the frozen gate decisively fails as
`SC41_SWITCH_CAUSAL_TERMINAL_VERIFICATION_GATE_NOT_MET`. A read-only diagnostic
explains the failure without reopening the cohort: the median four-candidate
score span was `0.0400` for desired text, `0.0164` for the counterfactual margin,
and `0.1425` for same-scene similarity; the same-scene argmax was correct on
`0/16` tasks. The correct terminal image often contains the requested visible
state change, so rewarding resemblance to the pre-completion video is an
anti-causal signal, not continuity evidence.

Do not tune prompts, weights, scales, sampling, model, or task membership on
this opened cohort. The reusable result is the need for an explicit transition
or effect variable--for example, a localized control state before/after the
action--rather than raw appearance persistence. Desired-state CLIP's `9/16` is
descriptive signal, not a predeclared positive. The test establishes no online
search, physical approach, reachability, executed action, user arrival,
`HANDOFF_READY`, product, or safety effect.

The frozen protocol, backend launch, provider, and result SHA-256 values are
`c8deb74c4f3fcb260280f0f04589fb5efa8a7bb0c089f8587933074fc0232329`,
`eb1e4d6d54b0041c1b4aa4206dd19e38b6855abedc57461bb5e80da0405bea03`,
`2274126fa7a8bc5877965f40c5f031b04438cf844ced1f7875a20256fbe70dd8`,
and `b54f8f837385306361ecbffb420dd086cf2a68d970d5abd565871498564ddbba`.
The representative launch selected `clip-torch-cpu` because its 234.55 ms
median beat Torch CUDA's 342.22 ms on the verified RTX 5060 Laptop GPU; model
identity and actual device/provider matched the frozen receipt.

### SC42: a directed local effect is signal, but OSCaR Frame 3 is not completion truth

SC42 moved to the official public
[OSCaR dataset](https://huggingface.co/datasets/ali-vosoughi/oscar-dataset),
whose [NAACL paper](https://aclanthology.org/2024.findings-naacl.226/) reports
real egocentric object-state-change clips. Before opening any held-out frame
captions or temporal keys, it selected 64 hash-frozen `open`, `close`, `turn
on`, and `turn off` clips from the 500-clip human-verified EPIC benchmark split.
The Goal-Localized Effect Axis selected five CLIP patches by object similarity
and projected them onto the normalized `after-state - before-state` language
direction. The static desired-state baseline chose `25/64` (`39.06%`) terminal
frames; the successor chose `27/64` (`42.19%`), a `+3.125` point change with 13
rescues and 11 regressions. `close` improved descriptively from `7/20` to
`11/20`, but the frozen overall gate failed as
`SC42_GOAL_LOCALIZED_EFFECT_AXIS_GATE_NOT_MET`.

Post-outcome source inspection identified a more important authority boundary.
OSCaR's held-out `Frame_3` is the temporally last annotated frame, not a promise
that the requested effect is visible or complete. An independently opened
Development example explicitly describes a toggled light whose expected
illumination has not appeared by Frame 3. SC42 therefore measures terminal-frame
retrieval under a curated action subset, not functional completion. Its small
uplift is reusable representation evidence only; neither its failure nor its
score is evidence that BlindAssist can or cannot confirm arrival. Do not tune
the opened 64 clips or consume the remaining roster against the same inadequate
completion truth.

The protocol, backend, provider, and result hashes are
`1f0b2dc91cd67072d1247dce411130b6daee54e9996160a9783030306be3835b`,
`d7bcf98c634f68afec5f61048eb7ee755458ed246e9e763c571e12072f46ccac`,
`c82754a34bf6922b18e9a4cefb37ed8e290963f71111b7a949d720a9b251a619`,
and `141fb43331182ce693bde07c3463a050cb15ad74d87431bc1868aeddd0183ea0`.
The representative launch selected CPU because its 145.14 ms median beat CUDA's
367.66 ms with exact device identity and no provider fallback.

### SC43: hidden causal prompting collapses before it becomes a state model

SC43 then opened the independently unused official SWITCH
`final_state/img2img` task, whose 95 rows ask which of four visual states occurs
first after an action from a shown current state. This gives the immediate
effect authority missing from OSCaR Frame 3. A fixed Qwen2-VL-2B baseline used
plain four-choice next-token logits. The successor used a Causal Intervention
Logit Contrast: log probability under a structured `do(action)` prompt minus
the matching `do(no-action)` log probability. GT was removed before JSON parsing
and remained unavailable until the provider was sealed.

| selector | correct | accuracy |
| --- | ---: | ---: |
| plain multi-image VLM | 19/95 | 20.00% |
| intervention logit contrast | 24/95 | 25.26% |

The successor gained `+5.26` points but caused 16 regressions and failed the
frozen gate as `SC43_CAUSAL_INTERVENTION_LOGIT_CONTRAST_GATE_NOT_MET`. The
read-only output audit is decisive: the successor predicted A on `95/95` rows;
the causal arm predicted A on 94 and D on one, while the no-action arm split
only between C (43) and D (52). The apparent gain is therefore label-prior
arithmetic, not content-sensitive causal state reasoning. Do not repair it by
changing label tokens, prompt wording, resolution, contrast scaling, examples,
or self-consistency on this opened task, and do not reopen its paired video/text
modalities as if shared IDs were fresh scientific cohorts.

The reusable successor requirement is now explicit: expose and supervise
`actuator`, `effect carrier`, `before state`, `after state`, and `conflict` as
separate visual variables, then let a deterministic reducer decide immediate
effect or `UNKNOWN`. A letter logit may consume those variables only after their
visual grounding is independently observable; prompting a small VLM to keep
them hidden is closed. The protocol, backend, provider, and result hashes are
`f404838b2c652fc0772144cfc925fe9ffedbe85b5253d901f219ed21020d07c8`,
`ecced33725cc6afa58a6c19db4ef5a3517d7e1df78484777b18ff1b653d22da2`,
`afb32e730469ac515fc4f860a323d77f06c69cd6e489529a795cc86eff704102`,
and `cab9d77a240508ba152eaadb6ea4b7ca76470a890a7006c60fa68d838e4926ee`.
The real three-prompt workload selected the verified RTX 5060 GPU at 3.96 s
versus 71.67 s on CPU, with exact CUDA execution and no fallback.

### SC44: named factors still collapse unless the representation learns them

SC44 opened the public ungated
[RoboPulse hard subset](https://huggingface.co/datasets/yuheng2000/RoboPulse),
whose 1,800 rows provide a task-start reference, completed-task reference,
BEFORE and AFTER front/left-wrist/right-wrist views, and a signed physical
progress label. A source-stratified SHA-256 selection froze ten rows from each
of nine manipulation sources (`90` total) without consulting conversations or
Hop labels. This is fresh progress/regression authority rather than OSCaR's
temporal endpoint or SWITCH's four-choice label roster.

The Goal-Anchored Factor Reducer queried five separately named physical margins:
effect carrier, actuator/contact, spatial orientation, conflict integrity, and
handoff distance. A deterministic reducer returned progress only when conflict
integrity was positive and at least three of five factors were positive.

| selector | correct | balanced accuracy |
| --- | ---: | ---: |
| direct binary Qwen2-VL-2B | 43/90 | 49.90% |
| goal-anchored factor reducer | 43/90 | 50.00% |

The result is `SC44_GOAL_ANCHORED_FACTOR_REDUCER_GATE_NOT_MET`. Only one
baseline-correct row regressed, but this apparent stability has no value: every
factor margin was positive for all `90/90` rows, so the reducer predicted
progress on all 90. Margin ranges never crossed zero (effect carrier
`0.72..2.70`, actuator/contact `0.69..2.20`, spatial/orientation `0.81..2.98`,
conflict/integrity `0.64..1.62`, handoff distance `0.55..2.08`). The direct
baseline was nearly as collapsed at 88 progress predictions and two regression
predictions.

This closes prompt-only factor exposure, not factorized state modeling. Do not
change prompt wording, label tokens, factor thresholds, reducer weights, or the
opened cohort. A legal successor must learn content-sensitive before/after
factor differences from disjoint supervision, expose calibrated `UNKNOWN`, and
freeze a new evaluation cohort before outcome access. The protocol, backend,
provider, and result hashes are
`30672f0cdbb963cffe0a79385295df16e1e0f4a2a67c9d3cd97e27f22b187980`,
`d8bf4fb4caac66cd8e780688c82b9b9fceb4387c2717f0120604e91e2fa955e7`,
`82cf70a7b9f05ff0dd04e7b55afd93478561096179be6b9e30accfd47845b493`,
and `9be2c7cf87da89d3730b0ac6412de59e7ec4e39eb1878be045e867d75b92c6b7`.
The representative launch selected the verified RTX 5060 GPU at 8.73 s versus
53.40 s on CPU, with exact CUDA execution and no fallback. No target identity,
navigation, physical execution, user arrival, explicit confirmation, product,
or safety claim follows.

### SC45: a learned progress tensor breaks the constant-label collapse, narrowly

SC45 changes both representation and cohort rather than repairing SC44. It
excludes the 10 hash-selected SC44 rows from every RoboPulse source, trains on
six whole source domains (`1,140` rows), calibrates only on `droid_oxe` (`190`
rows), and leaves `human_pika` plus `libero_data` (`380` rows) outcome-invisible
until provider seal. The class counts were balanced without resampling:
training `572/568`, calibration `95/95`, and evaluation `190/190`
progress/regression.

The 32-dimensional Progress Factor Tensor is constructed from normalized CLIP
image and task-text embeddings. It keeps five named groups: front-view effect
carrier direction, left/right wrist actuator/contact direction, absolute
front-view spatial anchors, cross-view conflict/agreement, and distance to the
completed reference. A fixed standardized logistic learner supplies class
probabilities. A split-conformal set at frozen `alpha=0.25` is the only final
reducer: singleton sets become progress/regression and non-singletons remain
`UNKNOWN`.

| selector | scope | balanced accuracy | coverage |
| --- | --- | ---: | ---: |
| untrained visual goal axis | all 380 | 61.58% | 100% |
| learned factor tensor | 357 known rows | 65.96% | 93.95% |
| baseline on the same 357 known rows | 357 known rows | 62.49% | 93.95% |

The successor gains `+3.47` points on identical known rows and retains 23
explicit `UNKNOWN`s, so it breaks SC44's `90/90` constant-progress collapse.
However, the preregistered gate required at least `+5` points; record
`SC45_LEARNED_PROGRESS_FACTOR_TENSOR_GATE_NOT_MET`. Correct-known-per-total is
`235/380` (`61.84%`), so abstention was not used to hide the denominator.

The outcome localizes the remaining representation deficit. `libero_data`
reaches `71.91%` selective accuracy at `93.68%` coverage, while the human-hand
`human_pika` source reaches only `59.78%` at `94.21%`. Learned coefficient L2
mass is largest for effect carrier (`1.23`) and cross-view conflict (`0.89`),
then actuator/contact (`0.60`), handoff distance (`0.56`), and spatial
orientation (`0.22`). The next legal successor therefore needs object/hand-
localized effect-carrier deltas on a new cohort, not another C/threshold/alpha
or source-role sweep.

The protocol, backend, provider, and result hashes are
`6fc16c63bc97f7d94d682473ccd1733ae2f5303e28ecf0bb78f0c1de02bf5c71`,
`bf1761343b0f96d21bf0621f047838d64a66f2493f23957d1fd322402a8971a9`,
`bbb4acde7be06e4b9ab28e07e6415bd1f607a0640564c1c4ad1dd96b37d7ce85`,
and `f1b70d49fd249dd172ae9433a918c7aa759213742c30f23492dcc59c54682b70`.
The representative 32-image batch selected the verified RTX 5060 GPU at
`0.423 s` versus CPU `0.605 s`, with exact CUDA execution and no fallback.
This is endpoint-mechanism evidence only; it does not establish online identity,
search, localization, navigation, reachability, physical execution, user
arrival, explicit completion confirmation, product benefit, or safety.

### SC46: local effect carriers produce a large relative gain but remain sub-gate

SC46 follows the SC45 failure localization instead of tuning its consumed split.
Exa reviewed 95 result slots across three source workstreams (progress/reward,
failure/recovery, and hand-object/state-change localization). Guardian UR5-Fail
was selected because its public Apache-2.0 train, validation, and test datasets
provide a small immediately runnable real UR5 cohort with three RealSense views,
start/end images, task/subtask text, and execution-success labels. The frozen
source-native roles contain `400/30/140` rows and balanced train/calibration
truth (`200/200`, `15/15`) without resampling.

The Local Effect Carrier Tensor uses GroundingDINO to materialize up to two
task-conditioned entity boxes plus a robot-gripper box in each of three
start/end viewpoints. CLIP then encodes the full frame, the task-entity union,
and the joint task-entity/gripper interaction crop against a before/successful-
after text axis. Detection confidence/count, region area/IoU, and gripper-task
distance join those state deltas. The three per-view vectors plus cross-view
mean and standard deviation form a fixed 80-dimensional tensor. A standardized
logistic learner supplies probabilities; only frozen localization eligibility
and a split-conformal singleton set may emit success/failure, otherwise
`UNKNOWN`.

| selector | scope | balanced accuracy | coverage |
| --- | --- | ---: | ---: |
| global CLIP goal axis | all 140 | 49.85% | 100% |
| local effect carrier tensor | 107 known rows | 59.94% | 76.43% |
| baseline on the same 107 known rows | 107 known rows | 49.79% | 76.43% |

The `+10.15` point same-known gain is much larger than SC45's `+3.47` points,
so local carriers are a real representation advance. The frozen gate still
fails: absolute selective balanced accuracy is below `70%`, and failure recall
is `57.14%` versus the required `60%` (`success` recall `62.75%`). Record
`SC46_GUARDIAN_LOCAL_EFFECT_CARRIER_TENSOR_GATE_NOT_MET`. All `140/140`
evaluation rows satisfied localization eligibility; 33 `UNKNOWN`s came only
from non-singleton conformal sets. The residual is semantic: `translation_object`
failures were only `7/15` correct, and coefficient mass is led by task/gripper
confidence and counts rather than the intended task/interaction state deltas.
The next legal successor therefore needs explicit object-state/change or
contact/release evidence on a new cohort, not prompt, detector-threshold,
classifier, alpha, or split tuning.

The protocol, implementation, grounder receipt, encoder receipt, sealed
provider, and result hashes are
`faebbbc1f82650bd0a3181712ef6c4161baadc611e7d38b3b49c995228510414`,
`ee2056ece4b19a2b3e6f2e663bd2b6229c2e58367f27f3d2b865675dafeb0548`,
`e35abbf3ed712359b4ef9fdd571526aeca17ce5a0d88bc9a2edb84197337b428`,
`48be5a8870a73a1455d4c1b2135f2b5355a405f4edd90e9ca4e2a0233a327beb`,
`9b17671df00ef399581462cec1259d1b357500d172d8aae9be27b5fc386886c1`,
and `ca407cefb26cc5c68c7ed53c7d0c9c556aadb5ca8a423d48092f62cc551e0330`.
Both representative model probes selected verified RTX 5060 CUDA with no
fallback: GroundingDINO `0.976 s` versus CPU `2.707 s`, and CLIP `0.073 s`
versus CPU `0.265 s`. Source pages:
[train](https://huggingface.co/datasets/paulpacaud/ur5fail_train_dataset),
[validation](https://huggingface.co/datasets/paulpacaud/ur5fail_val_dataset), and
[test](https://huggingface.co/datasets/paulpacaud/ur5fail_test_dataset). This
is endpoint-mechanism evidence only; it does not establish target identity,
active search, metric localization, navigation, reachability, physical
execution, user arrival, explicit confirmation, product benefit, or safety.

### SC47: privileged phase distillation learns process, not terminal state

SC47 follows SC46's semantic-state failure instead of tuning the consumed
Guardian test split. Exa reviewed 90 result slots across three workstreams:
object-state change, grasp/contact/release, and small immediately runnable
public datasets. DROID-OOD was selected because its needed subset is only
384,056,897 bytes yet contains 350 real three-view trajectories, task text,
recorded gripper/Cartesian state, dense pseudo-progress, and source success
labels. The selected 703-file inventory is pinned at revision
`0fd7c566e1812b90b63dca1c295818b781de8a7c` with aggregate SHA-256
`8312e18413c7961a5652791d80a3eb884421abf0262c11c6ff353b70c3b1a47e`.
The dataset is public and ungated; no license field was found in its dataset
metadata, so no broader redistribution claim is made.

Within each of five tasks, a frozen hash assigned 40 of 50 source-train rows to
training and 10 to calibration. All 100 source-validation rows remained
evaluation-only until provider seal. Twelve normalized timepoints from each
960x192 video were split into left/right/wrist views. A fixed CLIP encoder
produced task-conditioned trajectories. Training-only privileged targets were
dense pseudo-progress, recorded gripper closedness, vertical displacement,
Cartesian speed, and gripper-opening rate. A ridge teacher distilled those five
signals into video-only sequences; an endpoint learner combined their frozen
statistics and causal phase order with the learned visual goal-axis baseline.
Only a split-conformal singleton set could emit success/failure; evaluation
state, action, reward, latent, and success fields never entered the provider.

Teacher transfer on the 50-row calibration role was selective:

| distilled signal | calibration frame-level R2 |
| --- | ---: |
| vertical displacement | 0.746 |
| gripper closedness | 0.725 |
| dense pseudo-progress | 0.620 |
| Cartesian speed | -0.190 |
| gripper-opening rate | -0.207 |

| selector | scope | balanced accuracy | coverage |
| --- | --- | ---: | ---: |
| learned visual goal-axis baseline | all 100 | 61.61% | 100% |
| phase-distilled successor | 61 known rows | 63.22% | 61% |
| baseline on the same 61 known rows | 61 known rows | 54.11% | 61% |

The successor gains `+9.11` points on identical known rows, but records
`SC47_DROID_OOD_PRIVILEGED_PHASE_DISTILLATION_GATE_NOT_MET`: coverage and
absolute balanced accuracy miss the frozen `70%` gates, and success recall is
only `32%` versus failure recall `94.44%`. Thirty-nine non-singleton sets are
balanced by truth (20 failure, 19 success), so abstention did not simply hide
one class. Only ten rows received singleton success. Coefficient mass confirms
that gripper closedness (`1.14`) and dense progress (`0.95`) contribute after
the visual baseline (`1.21`), but phase evidence remains asymmetric. The legal
successor is therefore direct target-bound final-object-state supervision on a
new cohort, not another phase derivative, split, C, or alpha sweep.

The protocol, implementation, backend receipt, visual feature cache, sealed
provider, and result hashes are
`76c879a610a54163a8f8c6ae90ae8503043806554e9ccfbd4acb8fff315b5c84`,
`24b406b391e868e0c07e1c4dfda7c7ed48b0738600203cbad06f410b640389d5`,
`bd036fb84dc96c90d8b9fdd2e1489decdf95efe5a1a3e24e31454824af9c2d4e`,
`ff23f0273c6ac19af7cc0153fea90e5a9d5280b24d1fd38bf21c488d52960edf`,
`6661b1a7a72acfc2d5d85a41264eb5ed5e38ed553601e3623127dbc4fc065fc1`,
and `9ab42595ebc80a40c60a3d3d6b305fea55029162367824e5fbf2991a46c48b55`.
The representative batch selected CPU at `0.249 s` versus verified RTX 5060
CUDA at `0.368 s`; the persisted reason is `CPU_FASTER_MEASURED`, not silent
fallback. Source: [DROID-OOD dataset card](https://huggingface.co/datasets/yilin-wu/droid_ood_data).
This is endpoint-mechanism evidence only. Robot-state signals are privileged
training proxies, pseudo-progress is not physical truth, and no identity,
search, navigation, reachability, human action readiness, arrival, explicit
confirmation, product, or safety claim follows.

### Named-POI facade fingerprint: OCR is optional evidence, not identity authority

The current front-half increment moves from readable indoor labels to named
physical destinations. A frozen Wikidata/Wikimedia source contains 50 images
for ten real Hong Kong entities. Before inference, a visual source audit
admitted six entities with complementary facade, entrance/on-site identity,
and disjoint additional views. It froze 12 public references, six calibration
queries, and 11 evaluation queries; every evaluation image was also paired
with all five wrong admitted goals. No OCR engine or OCR-derived observation
was called.

Three fixed arms separate semantic naming from physical appearance:

| arm | top-1 entity | confirmed correct goal | wrong-goal confirms | balanced accuracy |
| --- | ---: | ---: | ---: | ---: |
| CLIP public name only | 6/11 | 1/11 | 2/55 | 52.73% |
| global CLIP + pooled DINO references | 6/11 | 4/11 | 3/55 | 65.45% |
| global references + mutual-patch affine consistency | 6/11 | 4/11 | **0/55** | **68.18%** |

Local geometry is therefore a useful veto signal: it removed all three
wrong-goal confirmations without OCR. It did not improve the frozen `6/11`
top-1 score, so the predeclared gate failed and the decision is
`NAMED_POI_FACADE_FINGERPRINT_DO_NOT_TUNE_LOCALIZE_REFERENCE_COVERAGE_GAP`.
The opened cohort, weights, threshold reducer, and role split must not be
tuned. The next legal increment changes the information source to a larger
multi-facet target knowledge pack: facade, entrance, logo/sign, architectural
context, and on-site wayfinding references. OCR may later contribute an
independent high-precision branch, but neither OCR nor appearance alone may
emit arrival.

The CPU/GPU probe used the combined CLIP+DINO representative inference. CPU
was measured faster (`0.190 s` versus verified RTX 5060 CUDA `0.469 s`) and was
selected with persisted `CPU_FASTER_MEASURED`; this was not silent fallback.
Protocol and implementation:
`named_poi_facade_fingerprint_protocol_v1.json` and
`named_poi_facade_fingerprint.py`. Result:
`artifacts.local/evidence/l10-r0/named-poi-facade-fingerprint-v1/result.json`,
SHA-256 `349362e6be8b02e4b1a9d45bab2c3f27648ab1c3eafc04b77106e60410e64184`;
backend receipt SHA-256
`3dada8c403d7d293f7cba447d86f96f2094e75dc1ea604342ff58f5b519ed6ce`.
This is small, source-curated image retrieval/confirmation evidence. It does
not establish natural-video search, exact entrance selection, metric
localization, guidance, reacquisition, arrival, product benefit, user benefit,
or safety.

### Multi-facet entrance evidence: scene-level AND is the wrong binding seam

The source was then changed rather than tuning the 11-image facade cohort. A
metadata-prioritized Commons materializer excluded every prior filename and
added 50 new views across Central Station, HKU Main Building, Queen Mary
Hospital, Ruttonjee Hospital, and Hong Kong Sanatorium. Filename metadata
supplied entrance/facade/wayfinding hints before pixel access; CLIP+DINO kept
the frozen non-OCR entity score, while GroundingDINO proposed pedestrian
entrances, doors, gates, stairs, escalators, and passageways. The first join
required scene identity and entrance-view evidence simultaneously.

V1's filename-derived entrance truth is `NOT_EVALUABLE`: all three supposed
false-ready images visibly contain stairs, escalators, or passage structures.
Commons filename hints are useful for acquiring data but are not entrance
visibility truth. No V1 score is promoted as algorithm evidence.

V2 therefore downloaded another 29 source-disjoint images and froze human
entrance visibility before any model call. Eighteen images remained evaluation
after eight calibration views and source-invalid portraits/history objects were
excluded. The unchanged scene-level join produced:

| arm | true ready | false ready | precision | recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| identity-only readiness | 2/8 | 5 | 28.57% | 25.0% | 0.267 |
| identity + CLIP entrance | 1/8 | 1 | 50.0% | 12.5% | 0.200 |
| identity + CLIP/GroundingDINO graph | 0/8 | 0 | 0% | 0% | 0 |

Fresh entity retrieval was `12/18`; seven identities were confirmed and no
wrong goal was confirmed. On the same evaluation views, entrance-only scores
still had usable ranking signal: CLIP AUC `0.7125`, GroundingDINO proposal AUC
`0.675`, and their graph score AUC `0.725`. The collapse occurs at the join:
global identity rejects most HKU/Queen Mary entrance views, and the small
calibration threshold then rejects the surviving target-local evidence. Record
`NAMED_POI_SCENE_LEVEL_IDENTITY_AND_ENTRANCE_GATE_NOT_MET`. Do not tune the
opened roles, thresholds, prompt, or weights.

The successor representation is implemented in
`named_poi_entrance_binding.py`. Every entrance proposal receives an expanded
context crop; that crop, rather than the whole scene, must rank the requested
public target above the map roster. A harmonic target-local edge combines
entrance support and normalized identity margin. The reducer emits `COMMIT`
only for a unique edge, preserves same-target twins as `SET_VALUED`, and returns
`SEARCH` when only a generic high-scoring door belongs to another target. Its
pure mechanism check passes all three cases.

V3 then froze a third prior-file-disjoint source before inference: 20 public
images, four human-boxed actionable entrance views, and 16 negatives. Original-
resolution audit removed HKU image 05 before the model call because its apparent
lower opening is a window, not an entrance. GroundingDINO supplied at least one
proposal at IoU >= 0.30 for all `4/4` positive images, so localized proposal
availability is not the current bottleneck. The one-shot comparison was:

| arm | correct unique commits | false commits | precision | positive recall |
| --- | ---: | ---: | ---: | ---: |
| strongest generic entrance proposal | 0/4 | 20 | 0% | 0% |
| proposal-context target-local binding | 0/4 | **5** | 0% | 0% |

The target-local reducer therefore removes `15/20` false commits (`75%`) and
is retained only as a fail-closed safety filter. It is not retained as a
successful entrance locator: it produced no correct unique commit. Seven views
remained `SET_VALUED`, eight remained `SEARCH`, and five were incorrect
`COMMIT`; only one positive view had truth inside the retained candidate set.
This localizes the next information gap. An expanded crop is still mostly a
generic door/corridor descriptor and does not transport entity evidence to the
specific entrance. The next legal representation is a target-support field:
project reciprocal reference-to-query patch matches into the current image,
then bind entrance proposals by overlap/connectivity to that target support
rather than by another global crop embedding. Do not tune context scale,
thresholds, boxes, prompt, or weights on V3.

V3 implementation and protocol:
`named_poi_target_local_entrance_eval.py` and
`named_poi_target_local_entrance_protocol_v1.json`. Result:
`artifacts.local/evidence/l10-r0/named-poi-target-local-entrance-v1/result.json`,
SHA-256 `5777b52914553913b5016222dab36b5ffda309433f2c46045a67c49c509e4b03`.
The encoder selected measured-faster CPU (`0.173 s` versus verified CUDA
`0.443 s`), while GroundingDINO selected verified RTX 5060 CUDA (`1.093 s`
versus CPU `3.912 s`); receipt SHA-256 values are
`95b1e69b4f83ba6317f08525e5928e8e0f5a24c1c64b851ae63c8a0657052d3a`
and `eeea35e1c457f89841009cde8b45749e5f3c867f65a8ee87ac2c32ce937d827e`.
No OCR calls were made. The human boxes establish only a current-image aperture
or passage assembly, not public access, traversability, accessibility, metric
approach, tracking, guidance, arrival, product benefit, user benefit, or safety.

### Target support, multi-facet references, and active disambiguation

V4 changed representation rather than tuning V3. Reciprocal DINO patch matches
plus affine consistency produced a target-support field, and entrances had to
attach to that field. On 12 new views with five boxed positives, the proposal
oracle covered `4/5`. The generic arm made one correct and 11 false commits;
the 9x9 support field made zero correct and four false commits. Record
`NAMED_POI_COARSE_TARGET_SUPPORT_FIELD_GATE_NOT_MET`: coarse spatial identity
was a useful veto but too diffuse to localize a door.

V5 retained native 16x16 DINO patches and replaced local overlap with downward
support rays: identity patches vote only for entrance proposals in the same
image columns below them. On 25 new views, four boxed entrances, and 21
negatives, the proposal oracle covered `4/4`. The generic arm made one correct
and 24 false commits. Native rays preserved one correct, reduced false commits
to nine, improved precision `4.0% -> 10.0%`, and kept two positives in
`COMMIT / SET_VALUED`. The locator gate still failed because correct commits
did not increase. Record
`NAMED_POI_NATIVE_SUPPORT_RAY_LOCATOR_GATE_NOT_MET_RETAIN_FILTER`.

V6 was source-audited before inference and was not run. Its apparent Central
Library entrance was a Délifrance tenant entrance; apparent IFC/Times Square
entries were Lane Crawford or interior circulation. Record
`NOT_EVALUABLE_NO_TARGET_ENTRANCE_POSITIVES`, not an algorithm negative. The
acquisition bottleneck was then fixed: Commons pagination now inspects up to
500 category files and batches metadata requests instead of silently using
only the first 50.

V7 changed only the reference information source. Twelve previously opened V5
entrance/interior/facade views formed a five-target multi-facet bank; the native
support-ray algorithm and thresholds remained unchanged. Pagination supplied
32 entirely new queries across HKCEC, Central Library, IFC, and Times Square;
six target-building entrances and 26 strong negatives were pixel-boxed before
inference:

| arm | correct unique commits | false commits | precision | positive recall | COMMIT/SET truth coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| strongest generic proposal | 2/6 | 30 | 6.25% | 33.33% | 2/6 |
| **multi-facet native support ray** | **2/6** | **7** | **22.22%** | **33.33%** | **4/6** |

The successor removes `23/30` false commits (`76.7%`), multiplies precision by
`3.56x`, and retains every truth-overlapping proposal made available by the
proposal oracle (`4/6`). It does not pass the locator gate: correct unique
commits stayed at two, while 23/32 frames remained deliberately `SET_VALUED`.
Record
`NAMED_POI_MULTIFACET_SUPPORT_RAY_PROPOSAL_CEILING_REACHED_SINGLE_FRAME_UNIQUENESS_NOT_MET`.
The legal next step is temporal active observation, not another threshold.

`named_poi_active_entrance_belief.py` now lands that seam. `SET_VALUED`
requests `CENTER_AND_APPROACH`; a candidate commits only after remaining
target-bound in two consecutive active views while competitors clear and its
normalized scale does not decrease. A lost commit becomes
`REACQUIRE / SCAN_LAST_BEARING` and resumes tracking when reobserved. The
scale condition prevents geometric persistence alone from promoting an
action-inconsistent distractor. This is runtime-ready mechanics, not temporal
public-video evidence by itself.

V4 result SHA-256:
`c14832f9fad8bd5eb860dbfb529e8c18adad53bc572289ae95b690e34d568c2f`.
V5 result SHA-256:
`2ad6841f7941f96a5692dad91454c69b8d64d0d4d5c9a98458c351a5062e05c1`.
V7 result SHA-256:
`6c097818b134f77120a75743972c5070294b4109ed29d210ecceed45c09f722f`.
For V7 the encoder selected measured-faster CPU (`0.126 s` versus CUDA
`0.281 s`); GroundingDINO selected verified RTX 5060 CUDA (`1.136 s` versus
CPU `3.715 s`). Receipt SHA-256 values are
`524a4f90fddb1d6bbe11dd22dc2569f4ffc6dc416b424a8dc8b7c2bfa6c102ba`
and `a55304b881fd889bd9f9c08b775c1144464266c8bb444b9c4e63006e3e70a488`.
Every run made zero OCR calls. These are curated Development current-image
results; public access, traversability, metric guidance, temporal confirmation,
arrival, product benefit, user benefit, and safety remain unproved.

#### Real ordered portal source and SkyDiscover bridge

An outcome-blind Commons audit checked 455 files across six named-POI
categories and found no video. The IFC Man Cheung Street numbered series did,
however, contain six previously unopened files after the V7 cutoff. They were
frozen from filename metadata, materialized once, and pixel-audited before any
model inference. Frames 06 and 07 show the same exterior glass door bank from a
wide upper-interior view and a closer escalator-aligned view; the remaining four
frames do not preserve that portal. Record
`ADMIT_REVERSE_SIDE_SAME_PORTAL_ACTIVE_VIEW_PROXY`. Source manifest SHA-256 is
`a2c19fbb16325be4f9dc95567fd4819d7a3071fc5de5141b7273cba86dcfd02d`;
the six-image inventory SHA-256 is
`f318ff5503fe58e4892bb26262c48f751e60ed913809edbd42c36b47ac2bfc73`.

The source is deliberately narrow: it is a reverse-side exit approach, filename
order is not measured pose or commanded motion, and it cannot establish outside
entrance finding. Its legal use is one real-perception bridge episode.

The first non-OCR inference on those two frames exposed a real failure rather
than a search opportunity. The generic proposal prompt selected an almost-full
image box and then the foreground escalator; an entrance-only prompt still left
portal-set truth outside the retained candidates in `0/2` frames. The raw
temporal rule falsely produced `SET_VALUED -> COMMIT`. Replaying the frozen
observations with the approach-scale gate produced `SET_VALUED -> SET_VALUED`,
removing this observed false commit but not locating the door bank. Record
`ACTIVE_VIEW_SCALE_GATE_PREVENTS_OBSERVED_FALSE_COMMIT_PROPOSAL_SOURCE_STILL_MISSING`.
Result:
`artifacts.local/evidence/l10-r0/named-poi-active-view-perception-v2/result.json`,
SHA-256 `4562009edb5b3e150644dfc4a727bfd907135b13757bb8f046a7606da4d1e0c8`.
Do not tune prompts, ray thresholds, or policy on this consumed prefix.

The missing representation is now implemented without OCR. A dual-family
functional portal-set proposer groups repeated aligned door posts into one
multi-leaf entrance and pairs aligned vertical handles to infer an aperture;
adjacent leaves therefore no longer compete as distinct target identities. The
first frozen six-building lattice-only cohort retained portal truth in the top
three on `3/6` frames and failed its `>=4/6` gate. After that cohort was consumed,
the handle-pair family raised mechanism-development retention to `5/6`. The
unchanged dual-family proposer then ran once on six previously unopened named
public-building glass entrances: top-one truth retention was `5/6`, top-three
retention was `6/6`, `50/60` stored top-ten candidates and `14/18` top-three
candidates belonged to the frozen functional portal regions. The frozen gate
passed. The earlier Central Station source was correctly marked
`NOT_EVALUABLE` before any algorithm call because five of six images were
exhibition walls and the remaining view was ambiguous.

Record `RETAIN_DUAL_FAMILY_FUNCTIONAL_PORTAL_SET_PROPOSER`. This crosses the
current-image real glass-entrance proposal bottleneck; it does not establish
requested-entity identity, public access, traversability, temporal tracking,
active-view causality, guidance, arrival, user benefit, or safety. Downstream
admission is therefore: separate target-entity lock, then top-three functional
portal proposals, then the scale-gated active entrance belief. A genuinely
ordered or commanded-view source is still required before policy search or
temporal claims. The obsolete oracle benchmark family remains outside this
route.

#### PB1 target-conditioned functional portal binding

PB1 froze the retained dual-family proposer and the existing CLIP/DINOv2
backbones, then trained only a shared candidate MLP plus a permutation-invariant
set summary and explicit `NONE` head. The public-image cohort contains 20
building identities split without overlap into 10 train, four Development, and
six confirmation entities. The confirmation entities were unseen by PB1
training, normalization, checkpoint selection, and baseline selection; their
pixels were previously opened only for the independent proposer confirmation,
so this is algorithm-fresh for PB1 rather than universally fresh imagery.

Development selected the unchanged native support ray as the strongest
baseline. Confirmation results were:

| arm | Top-1 | Top-3 | wrong portal commit | COMMIT/SET truth coverage | wrong-building confirmation |
| --- | ---: | ---: | ---: | ---: | ---: |
| native support ray | 4/6 | 5/6 | 2 | 4/6 | 27/30 |
| learned PB1 head | 4/6 | 6/6 | 0 | **0/6** | **0/30** |

The learned score contains ranking signal and is a perfect wrong-building veto
on this narrow cohort, but it is not a portal binder: every one of 36 test
episodes was reduced to `NONE`, including all `6/6` proposer-available correct
targets. The original promotion expression mechanically passed because it did
not couple false-commit reduction to positive admission coverage. A post-result
integrity adjudication changed no model output, label, or denominator and added
the missing non-degeneracy invariant: correct `COMMIT / SET_VALUED` coverage
must be nonzero and no lower than the selected baseline.

Record
`L10_PB1_FRESH_BUILDING_GATE_NOT_MET_ALL_NONE_STOP_EMBEDDING_FUSION`. Do not tune
weights, thresholds, embeddings, backbones, or fusion on this consumed cohort.
Any successor must change the target-identity information source or
representation and use a new building-disjoint confirmation cohort. L10-AV0 is
not opened: active observation still lacks a correctly admitted real portal.

Protocol and durable summary:
`named_poi_portal_binding_protocol_v1.json` and
`named_poi_portal_binding_result_v1.json`. Raw result SHA-256 is
`fe0b012612860419160407d2768fb6a22536d033bfb1ffebe44cee7483ac1b22`;
adjudication SHA-256 is
`e17cc3e336aa47bf7370cf980dc70d1cede17da52bf306e7ecde27d6ae7fc630`.
The encoder ran on measured-faster CPU (`0.451 s` versus verified RTX 5060 CUDA
`0.522 s` for the frozen batch probe); the small head trained on CPU as
`TASK_NOT_GPU_SUITABLE`. No OCR calls were made. Public access,
traversability, active motion, tracking, guidance, arrival, product benefit,
user benefit, and safety remain unproved.

#### PB2-A specialized VPR place identity

PB2-A changed the identity information source instead of tuning PB1. A
source-only audit rejected 12 invalid or mislabeled Commons rows before any
model call, then froze 12 PB1-disjoint buildings into six Development and six
confirmation entities. Every entity has one reference and four audited query
facets: facade, entrance, side, and partial. Portal proposals, portal crops,
PB1 decisions, tracking, and active observation were not invoked.

Development alone selected the fixed whole-image CLIP+DINO arm as the generic
baseline and official SALAD as the specialized challenger. Confirmation was:

| arm | Recall@1 | Recall@3 | positive acceptance | source-label negative proxy |
| --- | ---: | ---: | ---: | ---: |
| CLIP | 10/24 | 19/24 | 16/24 | 40/120 |
| DINOv2 | 9/24 | 16/24 | 3/24 | 1/120 |
| selected CLIP+DINO | 9/24 | 18/24 | 12/24 | 23/120 |
| selected SALAD | 8/24 | 20/24 | 8/24 | 3/120 |
| MixVPR | 7/24 | 18/24 | 16/24 | 48/120 |

SALAD improved latent Top-3 and rejection, but did not improve Top-1 and lost
four correctly accepted target-present views. MixVPR recovered positive
acceptance only by more than doubling wrong-building confirmations relative to
the selected baseline. Record
`L10_PB2A_SPECIALIZED_VPR_IDENTITY_GATE_NOT_MET_STOP_SINGLE_FRAME_APPEARANCE_ONLY`.
Do not sweep backbones, fusion, thresholds, normalization, seeds, or reference
weights on this consumed cohort. Add a genuinely new information source such as
logo/OCR, map or POI metadata, coarse GPS, or ordered multi-view evidence.
PB2-B and L10-AV0 remain blocked.

#### PB3 metadata-backed lexical identity proof

PB3 changed the information source to public POI aliases plus scene text and
made the evidence join asymmetric. A unique alias proof may accept only its
entity and veto other requested identities. Missing or ambiguous text is
`UNKNOWN`; it preserves the frozen appearance decision and can never create
semantic `NONE`. A source-only audit froze eight entities absent from PB1 and
PB2-A, split four Development / four test, with one identity-bearing and one
context query per entity before any PB3 OCR or embedding call.

Development selected DINOv2 and froze the text score plus uniqueness margin.
On eight test queries:

| Test metric | appearance | metadata text / asymmetric join |
|---|---:|---:|
| Recall@1 | 6/8 | not a ranking arm |
| positive acceptance | 8/8 | **8/8** |
| source-label-negative wrong accepts | 23/24 | **9/24** |
| emitted text proofs | n/a | **5/5 correct** |
| identity-bearing correct proofs | n/a | **2/4** |

The 60.87% relative wrong-accept reduction with unchanged positive acceptance
is a strong precision/veto mechanism signal. It does not pass the frozen branch
gate because identity-bearing correct-proof coverage required at least `3/4`.
Record `L10_PB3_METADATA_BACKED_TEXT_IDENTITY_BRANCH_GATE_NOT_MET`. Canonical
names and metadata aliases both reached `5/5` precision and `5/8` coverage; a
post-result target-blind four-tile diagnostic added zero proofs. Do not tune
aliases, lexical thresholds, OCR models, or crops on this cohort.

The next legal source must add a fresh logo or Chinese identity representation,
or a genuinely ordered executed `APPROACH_TEXT / SWEEP_SIGN` observation with a
before/after receipt. PB2-B and general L10-AV0 remain blocked. The identity-
bearing stratum is deliberately curated and wrong-request rows are source-label
proxies; neither is prevalence or exhaustive physical-absence authority. No
portal ownership, navigation, arrival, product, user-benefit, or safety claim
follows.

#### PB4 Script-Contrastive Identity Lattice

PB4 uses a fresh bilingual identity representation instead of tuning PB3.
SCIL keeps three independent evidence carriers: English word atoms, Han
uni/bi-grams after a fixed traditional/simplified fold, and public marks.
Candidate-pack IDF downweights generic atoms, a small agreement bonus rewards
corroboration, and independently strong carriers that disagree force
`UNKNOWN`. A unique proof authorizes only its own entity; missing evidence
falls back to appearance and cannot emit semantic `NONE`.

Before model access, the source-only audit rejected interior placards,
directories, exhibits, wayfinding maps, neighbor-dominated signs, and an entity
already seen in the PB2-A human rejection audit. The final 24-image source has
four Development and four test entities absent from the PB1/PB2-A/PB3 formal
cohorts, with one identity-bearing and one context query per entity.

| Fresh test metric | appearance | English canonical | flat bilingual alias | **SCIL / join** |
|---|---:|---:|---:|---:|
| Recall@1 | 5/8 | n/a | n/a | n/a |
| positive acceptance | 2/8 | n/a | n/a | **6/8** |
| identity-bearing positive/proof | 0/4 | 0/4 | 0/4 | **4/4** |
| correct text proofs | n/a | 0/8 | 1/8 | **6/8** |
| wrong text proofs | n/a | 0 | 0 | **0** |
| source-label-negative wrong accepts | 1/24 | n/a | n/a | **0/24** |

SCIL produced `6/6` precision, five Han/public-mark-participating correct
proofs, and two proofs with at least two strong script carriers. The
asymmetric join gained four accepted positives (`+50 pp`) while removing the
only appearance wrong accept (`100%` relative, `4.17 pp` absolute). All six
frozen clauses passed. Record
`L10_PB4_SCRIPT_CONTRASTIVE_IDENTITY_LATTICE_GATE_MET`.

The mechanism is promoted only as conditional proof-positive named-place
authority. The next eligible experiment may bind the already frozen functional
portal-set proposer only on a current SCIL-proved frame. `UNKNOWN` remains
blocked and requests `APPROACH_TEXT / SWEEP_SIGN`; the consumed PB4 aliases,
fold, atom weights, OCR, thresholds, and pixels must not be tuned. The curated
identity-bearing stratum is not prevalence, and cross-paired wrong requests are
source-label proxies. No open-world identity, portal ownership, access,
active-view, navigation, arrival, product, user-benefit, or safety claim
follows.

#### PB5 Script-Proved Authority-Carrying Portal Lattice

PB5 tested the PB4-to-portal seam without reading portal truth into runtime
logic. All six test frames with a correct SCIL proof were retained: three
in-scope glass-door positives, two no-portal negative controls, and one visible
monumental entrance recorded as outside the frozen paired-handle
representation. Human labels evaluated output only. The unchanged dual-family
proposer ran on every proof-positive frame.

SP-ACPL hashes the same-frame SCIL proof, requested entity, image, and sealed
PB4 result into an authority token. A proposal could carry that token only if
it passed a truth-blind paired-handle opportunity predicate frozen from the
earlier six source-disjoint positive portal frames. Text-`UNKNOWN`, OOD, and
all `COMMIT` or guidance states remained blocked.

| PB5 composition metric | result |
|---|---:|
| raw portal truth retained at Top-1 / Top-3 | 2/3 / 2/3 |
| identity-bound truth retained at Top-3 | **1/3** |
| no-portal false authorization | **1/2** |
| OOD entrance authorization leakage | 0/1 |
| wrong-request authorization after identity join | 0/18 |
| identity-`UNKNOWN` ownership leakage | 0/2 |

The identity join itself remained asymmetric and precise, but the geometry
opportunity layer failed both positive coverage and no-portal specificity. The
frozen gate therefore records
`L10_PB5_SCRIPT_PROVED_AUTHORITY_CARRYING_PORTAL_LATTICE_GATE_NOT_MET`.
Do not rescue these consumed rows with score, box, family, SCIL, or geometry
threshold sweeps. SCIL remains the conditional place-identity authority and
the dual-family method remains a portal proposal mechanism, but their
geometry-only authorization join is closed on this cohort.

The next legal successor must change the information source or representation:
freeze an independent semantic door/opening evidence layer and evaluate it on
a source-disjoint cohort containing both no-portal and open-aperture controls.
This result is retrospective Development composition evidence, not fresh
confirmation or portal ownership. It establishes no access, traversability,
active-view causality, temporal persistence, guidance, arrival, product,
user-benefit, or safety claim.

#### PB6--PB10 portal information-source ladder

PB6 through PB10 changed the portal information source five times without
reopening PB5 geometry or SCIL identity. The decisive results are:

| representation | positive truth Top-3 | no-portal false | open-mouth leakage | balanced accuracy |
|---|---:|---:|---:|---:|
| PB6 synthetic semantic field, fresh | 0/4 | 1/2 | 1/2 | 0.250 |
| PB7 ADE20K door components, Development | 3/4 | 0/2 | 0/2 | **0.875** |
| PB7 frozen ADE20K, fresh | 0/4 | 0/2 | 0/2 | 0.500 |
| PB8 typed Qwen2-VL grid | 1/4 | 2/2 | 2/2 | 0.125 |
| PB9 specialized door ontology | 0/4 | 0/2 | 0/2 | 0.500 |
| PB10 RDAT + Trans4Trans class-5 | **4/4** | 1/2 | 1/2 | **0.750** |
| PB10 threshold-free glass cut | 1/4 | **0/2** | **0/2** | 0.625 |

PB6 and PB7 each produced an attractive Development result and then failed its
single source-disjoint confirmation. PB8 selected `DOOR` on all eight frames.
PB9 produced no author-final door or doorway candidate. These routes are
closed on their consumed pixels; do not rescue them with prompt, class,
confidence, component, crop, scale, or threshold sweeps.

PB10 supplied the first complementary geometric/semantic pair. A fixed
row-wise far-depth construction retained a truth-member region in RDAT Top-3
on **4/4** positives. Official Trans4Trans class-5 `Glass Door` argmax masks
aligned to a truth-member RDAT proposal on the same **4/4**, with no learned
threshold or morphology. It still fired on one of two no-portal controls and
one of two large open-mouth controls. A preregistered threshold-free
bottom-to-far cut removed all four control errors but collapsed positive
coverage to `1/4`. Record
`L10_PB10_GLASS_DOOR_PLANE_AND_TOPOLOGICAL_CUT_DEVELOPMENT_GATE_NOT_MET`.

Retain RDAT only as a coarse portal proposer and Trans4Trans only as a
glass-door semantic alignment source. PB11 executed the next legal experiment
on a fresh, pre-frozen K-known SUN RGB-D cohort. All `8/8` rows were evaluable,
but its privileged source-depth closure score produced
`min(door)=0.7166`, `max(control)=0.9831`, strict margin `-0.2665`, and ROC AUC
`0.625`. The planar wall-storage and filing-cabinet controls scored `0.9831`
and `0.9570`; the glass double door scored `0.7166`. Record
`L10_PB11_METRIC_PORTAL_CLOSURE_P0_PRIVILEGED_GATE_NOT_MET`.

PB11 therefore closes metric rim-plane closure as door authority. The frozen
protocol requires no DepthART C1 run after this P0 failure, and no cohort,
quadrilateral, rim, interior, relief, quantile, score, or threshold change is
legal on these pixels. The next structural successor is RGB door-part topology
(leaf/frame/handle or hinge), with metric depth retained only as a downstream
geometric consistency check. No portal ownership, access, opening-state,
traversability, active-view, guidance, arrival, product, user-benefit, or safety
claim follows.

PB12 then froze a source-disjoint four-class `door / handle / cabinet door /
refrigerator door` detector and a deterministic smallest-enclosing-parent rule
on eight new SUN RGB-D captures. It authorized only a handle assigned to class
`door`. All `2/2` handled furniture controls and `2/2` large doorless openings
were rejected, but no real door formed an authorized parent-child pair:
architectural-door authorization `0/4`, positive recall `0.0`, control false
positive rate `0.0`, and balanced accuracy `0.500`. Record
`L10_PB12_DOOR_PART_TOPOLOGY_DEVELOPMENT_GATE_NOT_MET`.

This is not an absence-of-signal result. Door parents appeared on `2/4`
positives, visible architectural-door handles on `2/4`, the sliding closet
produced coherent cabinet-door/handle assignments, and both large openings
remained silent. The decisive failure is box-level parent/child co-occurrence
and semantic competition: one true door's handle was enclosed by the competing
cabinet parent, while the other positives lacked either the parent or child.
Do not rescue the consumed cohort with confidence, image size, NMS, crops,
tiles, area, overlap, parent priority, alternate epochs/checkpoints, or fusion.
Change to a distinct pixel-level part source.

The first formal process performed inference but stopped before writing a
result because Ultralytics returned synthetic display paths for a batch-list
input. No detections or metrics were shown and no result file existed. One
mechanical replay removed only that non-scoring path assertion; all frozen
inputs, inference parameters, topology, and gate remained unchanged. The
durable result records both attempts.

PB13 then replaced boxes with Florence-2 referring-segmentation polygons. Six
expressions and eight PB11/PB12-disjoint SUN RGB-D frames from four source
buckets were frozen before output. The first launch failed its environment guard
before model load or cohort image decoding; correcting the guard to the already
frozen `USE_TF=0` value permitted one mechanical replay. The replay produced
three calls on frame 1, then `operation_part` instance 0 component 0 violated the
frozen polygon-shape contract. No frame or metric was adjudicated. Record
`L10_PB13_FLORENCE_PIXEL_PART_TOPOLOGY_DEVELOPMENT_NOT_EVALUABLE_OUTPUT_CONTRACT_FAILURE`.
It is neither a positive nor negative model result. The cohort is consumed, so
parser, prompt, beam, mask, or topology rescue is forbidden; PB14 must use a new
information source and fresh pixels.

PB14 changed to native YOLOE-26n-seg instance masks and froze two scales before
output: one full-frame parent pass, then one exact predicted-parent-box pass for
small operation parts. Eight PB11--PB13-disjoint SUN RGB-D sequences from four
source buckets were sealed with five official file receipts each. The run
completed all eight full-frame calls and nine retained-parent crops in `9.60 s`
with `104,960,512` peak allocated CUDA bytes. It produced no
`architectural_leaf`, `operation_part`, or `hinge` anywhere. Lane 1 instead
returned one `cabinet_door`, two `closet_door`, and six `doorless_opening`
instances; Lane 2 returned no child. Thus positive authorization was `0/4`,
handled-furniture leakage `0/2`, large-opening leakage `0/2`, and balanced
accuracy `0.500`. Record
`L10_PB14_YOLOE_MULTISCALE_PART_TOPOLOGY_DEVELOPMENT_GATE_NOT_MET`.

The falsifier is upstream of topology: the open-vocabulary source did not expose
the required architectural parent or operation-part observables. Close this
exact YOLOE route on the consumed pixels; prompt spelling, threshold, NMS,
image size, parent cap/order, crop expansion, model size, and topology rescue
are forbidden. PB15 changes to two-scale GroundingDINO grounding followed by
SAM2.1 box-conditioned native masks on a fresh source-disjoint cohort.

PB15 froze five parent prompts, five child prompts, GroundingDINO-tiny,
SAM2.1-Hiera-Small, and eight further source-disjoint SUN RGB-D sequences before
model output. Its first launch stopped before model load or cohort RGB decoding:
the GroundingDINO tree-receipt digest had been copied with one missing
hexadecimal character. Correcting only that receipt digest and rebinding the
protocol/cohort left no formal output to overwrite. The valid run then completed
`195` GroundingDINO calls and `37` SAM2.1 calls in `90.90 s`, with
`2,096,144,896` peak allocated and `2,443,182,080` peak reserved CUDA bytes.

The frozen topology authorized positives `1`, `2`, and `4`, missed positive `3`,
rejected furniture `5`, falsely authorized furniture `6`, and rejected both
large-opening controls. Thus positive authorization was `3/4`, handled-furniture
leakage `1/2`, large-opening leakage `0/2`, and balanced accuracy `0.750`.
Record
`L10_PB15_GROUNDED_SAM_MULTISCALE_PART_TOPOLOGY_DEVELOPMENT_GATE_NOT_MET`.

This is not a near-pass suitable for threshold rescue. Multiple handle, knob,
push-bar, panic-bar, and hinge calls returned nearly the entire parent crop, so
SAM2.1 produced whole-object pseudo-part masks. On positive `3`, an architectural
parent existed but the slightly smaller cabinet mask absorbed its pseudo-parts.
On furniture `6`, a slightly smaller architectural mask absorbed whole-cabinet
pseudo-parts and caused the false authorization. The same collapse contaminates
the three nominal true positives. Close the exact PB15 source on these pixels;
prompt text, confidence, parent/child cap, crop geometry, SAM postprocessing,
mask priority, and topology changes are forbidden. PB16 replaces the separate
grounder-and-box-mask chain with native SAM 3.1 text-conditioned concept-instance
masks on a fresh source-disjoint cohort.

PB16 froze the SAM 3.1 multiplex checkpoint, a strict image-only detector
adapter, five parent concepts, five operation-part concepts, and eight more
source-disjoint SUN RGB-D captures. The public SAM 3 image builder failed a
pre-formal synthetic smoke because it constructs a four-scale dual neck while
the SAM 3.1 multiplex detector checkpoint contains a three-scale neck. The
frozen evaluator instead assembled the detector components from Meta's
multiplex-predictor recipe and strictly loaded all `1,166` detector keys with
zero missing or unexpected keys; it instantiated neither tracker nor video
predictor. This is a PB16 custom image-only adapter, not an official supported
SAM 3.1 image API.

The first formal launch stopped before model load, cohort RGB decoding, or
output because WDDM exposed `8,150` rather than the frozen `8,151` MiB. Only the
mechanical same-GPU contract was corrected to a minimum of `8,000` MiB and all
dependent hashes were rebound. The valid run completed `8` image encodes and
`80` text-prompt calls in `34.94 s`, with `4,685,969,920` peak allocated and
`4,922,015,744` peak reserved CUDA bytes.

The route authorized no frame: `0/4` architectural positives, `0/2` handled
furniture false positives, and `0/2` large-opening false positives, for positive
recall `0`, true-negative rate `1`, and balanced accuracy `0.500`. Record
`L10_PB16_SAM3_NATIVE_PART_TOPOLOGY_DEVELOPMENT_GATE_NOT_MET`. The decisive
failure precedes topology: `architectural door` returned zero native instances
on every frame. Child masks appeared on positives `1` through `3`, including
handle, knob, push-bar, and hinge evidence, but no architectural parent existed
to authorize them; positive `4` exposed neither parent nor child. Furniture
concepts remained active and both control families were rejected. Close PB16 on
these pixels without changing concept wording, confidence, cap, mask assignment,
adapter, or topology. A successor must change both supported weight/API and
architectural-parent semantic representation on a fresh source-disjoint cohort.

PB17 materialized the original `facebook/sam3` `model.safetensors` and
`tokenizer.json`, then used the official Transformers image API rather than a
custom source adapter. Its synthetic-only smoke loaded 840,376,374 parameters
with zero missing, unexpected, mismatched, or error keys, reused one vision
embedding for text, and peaked at `3,928,569,856` allocated and `4,276,092,928`
reserved CUDA bytes. The formal representation removed mandatory operation-part
evidence: a frame authorized exactly when the simple noun `door` returned at
least one nonempty native mask. Four competitor prompts were diagnostic only and
could not veto or rescue that decision.

Eight new SUN RGB-D capture sequences were frozen before PB17 cohort output;
they have zero overlap with the 48 distinct PB11--PB16 sequences. The formal run
stopped on the first positive after one image encode and one `door` call because
the returned mask tensor kept native spatial dimensions rather than the frozen
source size. No evaluator result was written and no raw output was persisted.
Input-only diagnosis verified that the official processor supplied `[530,730]`,
exactly the source height and width. The frozen Transformers postprocessor
resizes whenever any mask survives and leaves native dimensions only for a zero
retained count. The failure therefore mechanically establishes one fresh
positive miss, making the preregistered `4/4` conjunction impossible. Record
`L10_PB17_OFFICIAL_SAM3_DOOR_STATE_DEVELOPMENT_GATE_NOT_MET`. The complete
confusion matrix and balanced accuracy remain unevaluable. Close PB17 on these
pixels without repairing the empty-mask contract, prompt, confidence, mask
threshold, processor, or postprocessing; the successor must add a genuinely
different observable information source on another fresh cohort.

#### PB18 Script-Proved Ingress-Connected Portal Graph

PB18 returned to real Named-POI images and changed ownership representation
rather than reopening PB17. Eight entity/file-disjoint Commons frames were
frozen before any OCR, semantic, or proposer call: four plain target portals,
two identity-visible no-portal controls, and two target-plus-tenant portal
ownership controls. The evaluator joined a correct localized SCIL carrier to
the unchanged portal lattice only through the same ADE20K host component and a
walkable-ingress semantic region.

| Fresh PB18 metric | result |
|---|---:|
| SCIL correct / wrong / `UNKNOWN` | 7 / 0 / 1 |
| target portal Top-1 / Top-3 | 1/6 / 4/6 |
| no-portal false authorization | 0/2 |
| tenant-decoy overlap authorization | 1/2 |
| exact target-hit plus tenant-reject ownership pair | 0/2 |
| opportunity-balanced Top-3 accuracy | 0.7083 |

Record
`L10_PB18_SCRIPT_PROVED_INGRESS_CONNECTED_PORTAL_GRAPH_GATE_NOT_MET`.
The graph safely retained both no-portal controls, but host connectivity and
walkable ingress did not express entity ownership: Fu Moon authorized both the
target-building and tenant portals, Midland remained identity-`UNKNOWN`, and
only one target portal ranked first. Close this opened cohort to threshold,
class, rank, host, OCR, proposer, and fusion rescue.

The next admissible information source is an external entity-owned entrance
node joined to provider camera pose and explicit camera-to-entrance occlusion.
This changes the modality from current-image ownership inference to a
geospatially owned endpoint. PB18 establishes no legal/public access,
traversability, active-view causality, directional guidance, metric arrival,
`HANDOFF_READY`, completion, product, user-benefit, or safety claim.

PB19 next implemented that entity-linked entrance-ray representation, but its
Mapillary confirmation never opened. The first 12-way source roster supplied
six mechanically direct, three self-occluded, and three non-target-occluded
panoramas; none of the four preregistered direct frames allowed target portal
truth to be uniquely frozen from marker-blind pixels. A source-only repair then
screened 858 targets, found 74 mechanically valid direct combinations, and
blind-audited 32 complete panoramas from 30 building ways. The first adjudicator
retained four direct frames, but an independent second adjudicator rejected
Markthal Rotterdam because multiple walkable door leaves/openings remained.
Only three frames had overlapping independently frozen portal intervals.

Record
`L10_PB19_MAPILLARY_SOURCE_NOT_EVALUABLE_UNIQUE_PORTAL_TRUTH_3_OF_4`.
No entrance bearing, relative compass angle, predicted x, projection hit, or
formal evaluator call occurred. Do not delete the disputed row, reduce the gate
to `3/3`, or let a visible output marker define truth. The representation may
move unchanged to a separately audited posed-panorama provider; Panoramax is the
next source candidate, conditional on an independently fixed raw-pixel
orientation contract.

#### PB19 Panoramax PanoLab active entrance-ray recovery

Panoramax closed that orientation seam only through a strict source allowlist,
not from provider `view:azimuth` alone. Projection is authorized for a full,
uncropped 2:1 equirectangular JPEG whose dimensions match the declared sensor,
whose owning-instance EXIF supplies true-north `GPSImgDirection`, whose EXIF
direction agrees with `view:azimuth` within one degree, and whose explicit
yaw/pitch/roll and XMP pose are zero and non-conflicting. For true entrance
bearing `B`, image heading `H`, and width `W`, raw horizontal position is
`W * wrap360(B - H + 180) / 360`. Any failed applicability check leaves the
world-bearing-to-pixel projection closed.

The first stratified PanoLab source froze 24 episodes over six POIs and four
causal scenarios, with 42 images from 20 sequences and 20 target ways. Its
marker-blind pixel audit admitted `0/6` tenant-wrong-entrance and `0/6`
multi-entrance episodes, so the required 24-way pixel-portal source could not
open and the remaining 12 occlusion rows stayed unopened. Record
`L10_PANOLAB_STRATIFIED_PIXEL_SOURCE_NOT_EVALUABLE_TENANT_MULTI_0_OF_12`;
there were zero OCR/model, projection-marker, algorithm, or oracle calls. This
is a source ceiling for pixel-owned portal truth, not an active-observation
negative.

A separate metadata-first roster then froze four real provider-adjacent
episodes from four target ways and four sequences: two non-target-building
occlusions and two target-self occlusions. Each episode starts with an exact OSM
entrance ray blocked by the frozen geometry and follows an official reciprocal
`prev`/`next` link to a directly exposed ray. The v1 replay preserved the core
`4/4` recovery effect but failed its action-receipt gate because sanitized source
items omitted those already-frozen links. V2 restored only the provider link
fields; episodes, images, geometry, orientation rule, projection formula,
action, and thresholds were unchanged.

| Fresh PanoLab active-ray metric | result |
|---|---:|
| strict-orientation images | 8/8 |
| initial visibility role correct | 4/4 |
| reciprocal provider action receipts | 4/4 |
| occluded-state false authorization | 0/4 |
| post-action entrance-ray authorization | 4/4 |
| active entrance-ray recovery | 4/4 (100%) |
| mean authority-count delta | +1.0 |
| mean camera displacement | 9.394 m |

Record
`L10_PANOLAB_ACTIVE_ENTRANCE_RAY_RECOVERY_DEVELOPMENT_GATE_MET`. This is the
first real posed-street-sequence Development evidence that a
`SIDESTEP_TO_ENTRANCE_FACE` changes an exact entity-owned entrance ray from
geometry-blocked to geometry-authorized. It does not prove a pixel portal hit,
pixel-derived entrance ownership, public/legal access, walkability,
collision-free motion, arrival, `HANDOFF_READY`, or mobility safety. The active
observation mechanism may advance; any pixel-portal successor still needs an
independent entrance-identity credential rather than another matcher rescue.

PanoLab source, pixel-source audits, strict orientation contract, and entrance
ray evaluators:
`l10_panolab_source_v1.json`, `l10_panolab_protocol_v1.json`,
`l10_panolab.py`, `l10_panolab_source_pixel_audit_v1.json`,
`l10_panolab_stratified_source_pixel_audit_v1.json`,
`l10_panolab_orientation_projection_protocol_v1.json`,
`l10_panolab_entrance_ray.py`, and
`named_poi_entity_linked_entrance_ray.py`.

Active-recovery evaluator and preserved v1/v2 receipts:
`l10_panolab_active_ray_recovery.py`,
`l10_panolab_active_ray_recovery_source_v1.json`,
`l10_panolab_active_ray_recovery_protocol_v1.json`,
`l10_panolab_active_ray_recovery_result_v1.json`,
`l10_panolab_active_ray_recovery_source_v2.json`,
`l10_panolab_active_ray_recovery_protocol_v2.json`, and
`l10_panolab_active_ray_recovery_result_v2.json`. The v2 result SHA-256 is
`abae1ef768ff24ad7771704c34f602d7145b382a8259b4a779b0357960fcd963`.

#### PB20 SEVN address-door PanoLab backend

SEVN 1.0 supplies the missing kind of source truth: each human door polygon is
associated with a street and house number. The adapter verifies the three
original Zenodo metadata objects by their published byte sizes and MD5 hashes,
then converts the legacy Pandas/NetworkX files into separated, dependency-free
public and evaluator JSON. Runtime observations contain only the address
mission, local pose, viewport, action graph, and image locator; door boxes and
binding states remain evaluator-only.

The first deterministic panel contains 24 distinct addresses: eight
`PAN_LEFT`, eight `PAN_RIGHT`, and eight one-edge `APPROACH` recoveries. `HOLD`
is correct-unique on `0/24`; fixed `SWEEP` is `16/24`; the one-step annotation
oracle is `24/24` with zero wrong-unique outcomes. A matching house-number box
is also visible on `22/24`. Record
`L10_SEVN_METADATA_ACTION_ADAPTER_READY_IMAGE_PAYLOAD_PENDING`.

This proves source admission, legacy-format conversion, action receipts, and
truth separation only. It is curated SEVN annotation/graph Development, not a
learned visual result, Panoramax confirmation, continuous sidestep, depth,
arrival, handoff, or safety evidence. The exact next payload is the official
1,861,417,787-byte `images.hdf5`; do not claim pixel replay before its published
MD5 is verified.

Implementation and receipts: `l10_sevn_panolab.py`,
`l10_sevn_panolab_requirements.txt`, `l10_sevn_panolab_protocol_v1.json`,
`l10_sevn_panolab_source_v1.json`, `l10_sevn_panolab_truth_v1.json`, and
`l10_sevn_panolab_result_v1.json`.

#### PB21 SEVN high-resolution pixel address-door canary

The official 27,819,151,974-byte high-resolution panorama archive is now local
and reverified at Zenodo MD5 `1e46ca1de01cdba68b0e9ff7de6dc3df`. The frozen
renderer preserves the existing 135-degree action contract as a 1440-pixel ring
crop from each 3840x1280 panorama. It streams only the 24 designated action-result
members directly from the ZIP, so the formal replay writes no extracted panorama,
viewport, thumbnail, or inference cache.

The canary deliberately isolates perception from action selection. Evaluator
truth selects the already frozen `PAN_LEFT`, `PAN_RIGHT`, or `APPROACH` action;
the runtime then receives only the public target house number and resulting
pixels. Exact RapidOCR tokens are joined to generic doorway/door proposals by a
fixed nearest-box rule. SEVN annotations enter only after the runtime output is
complete.

The one-shot result is `16/22` exact OCR among truth-visible house numbers,
`24/24` with at least one generic portal proposal, `14/24` joined text/portal
proposals, `11/24` correct target-door bindings, `3/24` wrong bindings, and
`10/24` UNKNOWN. By action class, correct/wrong/UNKNOWN is `6/0/2` for left pan,
`2/2/4` for right pan, and `3/1/4` for approach. The OCR and minimum-correct
gates passed, but the frozen maximum of two wrong bindings did not; record
`L10_SEVN_HIGHRES_PIXEL_ADDRESS_DOOR_SIGNAL_ONLY_GATE_NOT_MET`.

This is the first real SEVN pixel mechanism signal, not a learned action policy,
held-out confirmation, or arrival/handoff result. Two wrong-scored proposals
still overlap their target boxes (IoU `0.2341` and `0.1607`) but miss the frozen
center-in-box rule; the third has zero overlap. Do not post-hoc rescore or loosen
the association threshold. A successor should change the representation to
multi-scale number OCR plus mask-level credential-to-portal topology, then freeze
a new protocol.

Implementation and durable result: `l10_sevn_pixel_replay.py`,
`l10_sevn_pixel_replay_protocol_v1.json`, and
`l10_sevn_pixel_replay_result_v1.json`; cleanup is recorded in
`l10_sevn_pixel_replay_cleanup_v1.json`. The result SHA-256 is
`7c8ffe37f141d6770993cb891a6228f48572a7772516cb2eba2fbe337b25fbdf`.

#### PB22 SEVN multi-scale OCR and mask-topology successor

V2 preserves the same 24 episodes, prescribed one-step actions, high-resolution
renderer, OCR weights, and portal model. The single representation change is a
source-blind seven-pass OCR field—one native full view plus six overlapping
`1.6667x` tiles—joined to rasterized instance masks instead of nearest boxes.
Same-portal masks are removed by IoU or containment; an exact house-number token
may bind only inside an adaptive neighborhood of the upper 55% of a door mask.
SEVN annotations remain evaluator-only and enter after runtime output.

The frozen one-shot result improves truth-visible exact OCR from `16/22` to
`19/22`; all three gains are tile-only. Correct/wrong/UNKNOWN changes from
`11/3/10` to `14/2/8`. V2 retains `10/11` V1 correct episodes, turns four V1
non-correct episodes into correct bindings, repairs two of three V1 wrong
bindings, and converts the remaining old wrong binding to UNKNOWN. It also
introduces two new wrong bindings, so this is material progress rather than a
unilateral safety gain.

All six frozen gates pass exactly or better: all 24 episodes rendered, at least
`18` visible signs read (`19`), at least `14` correct bindings (`14`), at most two
wrong (`2`), at least ten V1 correct retained (`10`), and at least four V1
non-correct recovered (`4`). By action class, correct/wrong/UNKNOWN is `6/1/1`
for left pan, `4/1/3` for right pan, and `4/0/4` for approach. Record
`L10_SEVN_MULTISCALE_MASK_TOPOLOGY_DEVELOPMENT_GATE_MET`.

This authorizes the frozen representation as the SEVN Development baseline. It
does not establish autonomous action selection, held-out or cross-source
generalization, arrival, handoff, or safety. The next decision-changing check is
one new address-disjoint SEVN panel with the entire V2 stack frozen; even a pass
there remains same-source evidence before a real-source confirmation.

Implementation and durable evidence: `l10_sevn_pixel_topology_replay.py`,
`l10_sevn_pixel_topology_protocol_v2.json`,
`l10_sevn_pixel_topology_result_v2.json`, and
`l10_sevn_pixel_topology_cleanup_v2.json`. The result SHA-256 is
`b831af55e575030866b4cb7cf8c31d82954630d5c173f39f00409ef8787f5f7e`.

#### PB23 SEVN address- and panorama-disjoint V2 confirmation

The confirmation cohort was selected before opening any new pixel. It excludes
all 24 first-panel addresses and all 32 first-panel panorama frames, and also
prohibits frame reuse across new episodes. An exhaustive metadata-only
constraint search established that eight disjoint `APPROACH` episodes do not
exist under those rules, so the largest balanced panel was frozen at seven
`PAN_LEFT`, seven `PAN_RIGHT`, and seven `APPROACH` episodes. The resulting 21
addresses and 28 frames have zero overlap with the reference panel and zero
cross-episode frame reuse; 20 prescribed action-result views contain a
human-labelled visible target house number.

The already frozen V2 evaluator, renderer, seven OCR passes, exact-token rule,
portal model, mask de-duplication, upper-mask topology, truth scoring, runtime,
and action map were loaded by hash without modification. The confirmation gate
required at least 18 visible-number opportunities, at least `80%` exact OCR,
at least `13/21` correct bindings, at most two wrong bindings, and all three
disjointness checks.

The one-shot result is `12/20` exact OCR (`60%`) and `9/0/12`
correct/wrong/UNKNOWN, so record
`L10_SEVN_V2_ADDRESS_PANORAMA_DISJOINT_SAME_SOURCE_CONFIRMATION_GATE_NOT_MET`.
By action class, correct/wrong/UNKNOWN is `4/0/3` for left pan, `4/0/3` for
right pan, and `1/0/6` for approach. All nine emitted mask-topology proposals
selected the correct target door on this panel, but nine episodes stopped at
`UNKNOWN_TARGET_TEXT` and three more at
`UNKNOWN_NO_ADMISSIBLE_MASK_TOPOLOGY`. The gate therefore fails on OCR recall
and total correct coverage, not wrong-door count.

Do not tune fixed tiles, OCR matching, mask radii, pair ranking, or scoring on
these consumed 21 views. V2 remains a positive first-panel Development result,
but it is not confirmed across fresh SEVN addresses and panoramas. A successor
must change the observation representation on another fresh panel—for example,
portal-conditioned rectified OCR crops instead of fixed viewport tiles—before
any new outcome is observed. It still cannot inherit autonomous action,
cross-source, arrival, handoff, product, or safety authority.

Implementation and durable evidence:
`l10_sevn_address_disjoint_panel.py`,
`l10_sevn_address_disjoint_source_protocol_v1.json`,
`l10_sevn_address_disjoint_source_v1.json`,
`l10_sevn_address_disjoint_truth_v1.json`,
`l10_sevn_address_disjoint_selection_receipt_v1.json`,
`l10_sevn_pixel_topology_confirmation.py`,
`l10_sevn_pixel_topology_confirmation_protocol_v1.json`,
`l10_sevn_pixel_topology_confirmation_result_v1.json`, and
`l10_sevn_pixel_topology_confirmation_cleanup_v1.json`. The result SHA-256 is
`0f6bda18a8a8fa19f9d4d13a2725f89e20cf5a878e9ac858c1787e65ff592c99`.

#### PB24 SEVN portal-private PP-OCRv6 medium witness

The successor sequence used only further fresh, metadata-selected PAN panels.
Portal-conditioned PP-OCRv6 small added one correct and one wrong binding; a
portal-specific witness graph made no change; an independent EasyOCR/CRAFT
full-view fallback added one correct and one wrong; and a global PP-OCRv6 medium
fallback again added one correct and one wrong. Those consumed panels are not
rescored. Their common failure localized the problem: stronger text recognition
without a typed observation edge could still attach a true address token to the
wrong door.

V5 therefore made one structural change. On V2 abstentions only, each detected
portal instance owns a private upper-neighborhood crop. PP-OCRv6 medium reads
that crop, and an exact target token may emit a proposal only when exactly one
portal has the witness. Multiple witnessed portals remain ambiguous and abstain.
The renderer, prescribed PAN action, primary seven-pass OCR, portal weights and
masks, evaluator projection, correctness rule, and all existing successful V2
outputs remain unchanged. Evaluator labels enter only after outputs are sealed.

The source panel contains 40 distinct addresses and 40 distinct frames with no
overlap against the 205 addresses and 220 frames in all earlier SEVN panels and
no frame reuse within the panel. The frozen result improves exact OCR on visible
targets from `22/37` (`59.46%`) to `29/37` (`78.38%`) and improves
correct/wrong/UNKNOWN binding from `16/0/24` to `21/0/19`. It retains all 16
baseline-correct bindings. Five unique portal witnesses emit five correct and
zero wrong bindings; two ambiguous witness sets abstain. Every frozen gate
passes, so record
`L10_SEVN_PPOCRV6_MEDIUM_PORTAL_WITNESS_FRESH_PAN_DEVELOPMENT_GATE_MET`.

This freezes the representation as the strongest SEVN PAN Development backend.
It remains cumulative-reference-disjoint but same-source: it does not establish
a new provider, city, or APPROACH behavior. A portal-private exact-token edge is
not portal ownership, public access, traversability, waypoint, arrival,
`HANDOFF_READY`, user benefit, or safety. The next decision-changing check is an
unchanged replay on a genuinely source-disjoint source with independent
door-instance and address-credential truth plus negative/no-portal controls; do
not tune V5 crops, models, witness uniqueness, or abstention on this panel.

Implementation and durable evidence:
`l10_sevn_fresh_pan_panel.py`,
`l10_sevn_fresh_pan_source_protocol_v5.json`,
`l10_sevn_fresh_pan_source_v5.json`,
`l10_sevn_fresh_pan_truth_v5.json`,
`l10_sevn_fresh_pan_selection_receipt_v5.json`,
`l10_sevn_ppocrv6_medium_fallback.py`,
`l10_sevn_ppocrv6_medium_portal_witness.py`,
`l10_sevn_ppocrv6_medium_portal_witness_protocol_v1.json`, and
`l10_sevn_ppocrv6_medium_portal_witness_result_v1.json`. The result SHA-256 is
`1c56bc1263e36ad357704161b430b4a7ea3557d7d6dabb76f37b6e8fa8b3772a`.

#### PB25 PanoLab component-separated referent-candidate router

The source-disjoint LSAA transfer first established a useful stop boundary for
the SEVN OCR representation. On ten Vienna facade crops, the unchanged V5
field produced only `1/10` exact observations and no bindings; an adaptive
credential-field variant reached `2/10` exact observations and still no
bindings. PP-OCRv6 and independent OCR recognition oracles were `0/10`, and all
four frozen Text Telescope combinations were `0/10`. The selected facade pixels
did not expose enough target text; this is an observation-reachability failure,
not evidence that a looser join would generalize.

The active Panoramax branch therefore kept the geometry and referent-candidate
questions separate. On four fresh target-way/item-disjoint episodes across
three cities, the frozen long/short exact-token bank found `0/4` target names
while the post-action entrance ray remained valid on `4/4`; the target names
were absent from the selected entrance windows. A naive ordered appearance
memory bank then ranked all three consumed targets correctly but did not beat
the single after frame: mean truth margin changed `0.11823 -> 0.11602`. Record
`L10_PANOLAB_ORDERED_APPEARANCE_BANK_POSTHOC_DEVELOPMENT_GATE_NOT_MET` and do
not tune temporal max pooling on those rows.

The next component router preserved lexical `UNKNOWN` and attached a fixed
facade fingerprint only as a `SEARCH_PRIORITY_ONLY` fallback. The fingerprint
is `0.35` CLIP global cosine + `0.25` DINOv2 pooled cosine + `0.40` mutual-patch
affine consistency against two references per target. On the four consumed
positive rows it changed lexical `0/0/4` correct/wrong/UNKNOWN into `4/0/0`,
with `0/12` wrong-goal candidates and zero ownership bindings. This
source-extended posthoc result is not confirmation.

The open-set check froze its accept thresholds before viewing any negative
pixels: minimum top-1 score `0.6617196786` and minimum margin
`0.0346412068`, copied from the weakest of the four positive calibration rows.
Four new target-absent Panoramax images across three cities, with zero target
way/item/collection overlap, were all rejected as `UNKNOWN`; false accepts were
`0/4`. Record
`L10_PANOLAB_OPEN_SET_APPEARANCE_ABSTENTION_FRESH_NEGATIVE_DEVELOPMENT_GATE_MET`.
This small fixed-negative result is neither an open-world guarantee nor
conformal/FDR control.

The decisive fresh-positive panel then froze the only two remaining named
targets in the local metadata pool that had strict DIRECT views from two
collections. Rumillat and Halle Rebatet were target-way/item-unseen; each used
two reference images from one collection and a query from another, for four
query-reference-disjoint collections and zero prior item overlap. Ranked
against all six targets, the unchanged gate accepted Halle Rebatet and returned
`UNKNOWN` for Rumillat: `1/0/1` correct/wrong/UNKNOWN, `50%` positive coverage,
and `0/10` wrong-goal candidates. Record
`L10_PANOLAB_CROSS_COLLECTION_FRESH_POSITIVE_ROUTER_DEVELOPMENT_GATE_NOT_MET`.
The precision boundary held, but cross-condition positive recall did not.

A single frozen sequence hypothesis tested whether reciprocal neighbours could
repair the failure without threshold rescue. For both consumed targets it took
exact provider `prev/anchor/next` frames and averaged the highest two of three
per-frame scores for each candidate. The result stayed `1/0/1`; Rumillat's one
clear next frame ranked correctly, but the two-view aggregate still ranked a
distractor higher. Record
`L10_PANOLAB_RECIPROCAL_TEMPORAL_QUERY_ROUTER_POSTHOC_DEVELOPMENT_GATE_NOT_MET`
and close that aggregation on FP01/FP02.

The final structural change routed orthogonal evidence instead of changing the
appearance score. It ran the previously frozen PP-OCRv6 medium exact-token bank
over the same three frames. If and only if one roster name matched, it emitted a
lexical search-priority candidate; on `NO_MATCH` it reused the sealed appearance
result, while conflicts returned `UNKNOWN`. Rumillat was uniquely witnessed as
`rumillat` at OCR score `0.91` in the reciprocal previous frame; Halle Rebatet
had no lexical match and retained its correct appearance candidate. The combined
router changed the sealed appearance `1/0/1` into `2/0/0`, with `0/10`
wrong-target lexical matches and zero ownership bindings. Record
`L10_PANOLAB_TEMPORAL_LEXICAL_APPEARANCE_ROUTER_POSTHOC_DEVELOPMENT_GATE_MET`
for consumed posthoc mechanism Development only.

That result proved a useful mechanism: active observation can expose an
orthogonal credential that a cross-condition facade embedding misses, and the
router can preserve abstention instead of weakening the appearance gate. It
did not prove that the token belonged to the target entity, facade, or entrance,
and the exact-token branch still failed when OCR produced a near spelling or a
contracted token.

#### PB26 conservative distinctive-token router and fresh confirmation

A new metadata ledger first extended the same-provider closed roster to two
Caen targets whose anchors had not entered the router. Euromaster and
Bibliothèque de Caen - Maladrerie both routed correctly by lexical evidence and
by temporal appearance (`2/0/0` for each), with zero wrong-goal candidates.
Those four reference anchors had been materialized for an earlier portal-source
audit, so this is router/OCR-unseen extension evidence, not fully pixel-unseen
confirmation.

The next panel enforced the stronger boundary. Before any selected pixel,
human review, appearance, or OCR call, it chose one target per available capture
producer from the remaining strict metadata ledger: Département du Calvados
(nlehuby), Crèche des Petits Lutins (Nzau), and Église Saint-Pierre
(Carto'Cité). All target ways and 15 items were disjoint from the prior router;
references and reciprocal `prev/anchor/next` queries came from different
collections. Human truth froze all three as valid before inference. The fixed
appearance router safely returned `1/0/2`; the exact temporal token bank added
nothing (`0/0/3`), leaving combined `1/0/2`. Thirty wrong-target lexical trials
and 30 appearance controls emitted no wrong candidate.

The failures exposed two concrete representation errors, not a reason to lower
the appearance threshold. OCR read high-confidence `Colvados` for target token
`calvados`, and read `Creche` plus `p'tits` for `creche + petits`. One posthoc
successor was frozen before a single replay: OCR row score at least `0.90`,
normalized target token length at least six, edit distance at most one, and two
distinctive evidence units; a target token of length at least eight contributes
two units, while length six/seven contributes one. A two- or three-fragment OCR
row may also be concatenated, so `p + tits -> ptits`. The branch emits only when
exactly one roster target meets the evidence threshold. On the consumed three
rows it recovered Calvados and Petits Lutins, retained Saint-Pierre through the
sealed appearance fallback, and changed `1/0/2 -> 3/0/0`, with `0/30`
wrong-target matches and zero ownership bindings. Record
`L10_PANOLAB_CONSUMED_DISTINCTIVE_EDIT_TOKEN_MECHANISM_DEVELOPMENT_GATE_MET` as
posthoc mechanism evidence only.

The decisive confirmation froze two further targets using only metadata name
capability and producer strata before downloading ten selected pixels: Monoprix
(nlehuby) and Maison de Quartier Chemin Vert (Nzau). Both target ways and all
items were router-unseen, and reference/query collections were disjoint. Human
truth froze both before model calls. Against the expanded 13-target roster, the
unchanged distinctive-token branch uniquely recovered Maison de Quartier from
`maison + chemin`, returned `NO_MATCH` for Monoprix, and produced `0/24`
wrong-target matches. The router therefore invoked the unchanged appearance
branch only for Monoprix; its fixed score/margin gate accepted the correct
target with `0/12` wrong-goal candidates. Combined result: `2/0/0`, one lexical
route, one appearance route, zero `UNKNOWN`, and zero ownership bindings.
Record
`L10_PANOLAB_FRESH_DISTINCTIVE_TOKEN_CONDITIONAL_APPEARANCE_ROUTER_DEVELOPMENT_GATE_MET`.

The added edit tolerance was then attacked without tuning. The four previously
frozen exact-roster-absent controls span Grenoble, Rezé, and Arcueil. Their
pixels had already entered the appearance evaluation, but no governed OCR result
referenced them before this protocol. Across 4 queries x 13 targets, the fixed
branch produced `0/52` matches and kept `UNKNOWN` `4/4`. Record
`L10_PANOLAB_DISTINCTIVE_EDIT_TOKEN_OPEN_SET_NEGATIVE_DEVELOPMENT_GATE_MET` for
pre-existing-pixel, first-governed-OCR negative Development only. It is not
fresh-pixel, conformal, FDR-controlled, calibrated-probability, or general
open-world evidence; `UNKNOWN` is not known-safe.

The staged successor result is therefore `3/3` consumed posthoc plus `2/2`
separately frozen fresh positives, with the authority split retained rather
than collapsed into a fictitious `5/5` confirmation. Exa literature points to
three orthogonal future upgrades: predict when temporal matching helps
([SMR](https://arxiv.org/abs/2503.06840)), allocate observation length instead
of fixing it ([dynamic sequence length](https://arxiv.org/abs/2407.00863)), and
add patch verification/calibration only after an independent calibration split
([SafeVPR](https://arxiv.org/abs/2605.28048),
[To Match or Not to Match](https://arxiv.org/abs/2504.06116)). The current tiny
curated panels cannot authorize a learned selector or conformal/FDR claim.

One consumed efficiency replay then applied a progressive evidence-first stop
rule to DF01/DF02. Both routes already had sufficient evidence in the first
`prev` frame: Maison exited on its two lexical witnesses and Monoprix exited on
appearance score `0.703153`, margin `0.050528`, above the unchanged gates. The
derived outcome remained `2/0/0` while observations fell `6 -> 2` (`-66.67%`),
with one lexical and one appearance early exit and no new model call. Record
`L10_PANOLAB_CONSUMED_PROGRESSIVE_EVIDENCE_EARLY_EXIT_MECHANISM_DEVELOPMENT_GATE_MET`.
Because per-frame evidence was inspected before the rule was frozen, this is
posthoc mechanism evidence, not online latency, compute, energy, motion, or
fresh dynamic-length confirmation.

The repository also contained a genuinely different provider boundary: four
Mapillary spherical panoramas in Rotterdam and Den Haag, retained and human-
reviewed during PB19 but never passed to governed OCR or a tested learning
model. They were copied hash-identically into canonical `artifacts.local` and
evaluated once through the unchanged distinctive-token branch. The crop was
conditioned on the earlier human facade/portal interval, so the test is not
pixel-unseen or an automatic entrance-ray result. Against a 17-target roster it
returned `0/0/4/0` correct/wrong/`UNKNOWN`/ambiguous, `0/64` wrong-target
matches, and zero ownership bindings. Markthal yielded only `HAL@0.937`; the
explicit Ontmoetingskerk sign yielded `ONTMOETINOBKERK@0.972`, outside the
frozen one-edit rule; Oude Kerk yielded only numbers and Kievitkerk no OCR row.
Record
`L10_MAPILLARY_DISTINCTIVE_EDIT_TOKEN_PROVIDER_TRANSFER_DEVELOPMENT_GATE_NOT_MET`.
This is a provider/city-disjoint first-OCR observation-reachability failure,
not evidence that a looser join is safe and not a full combined-router failure.

#### PB27 public-source admission and goal-conditioned background-IDF verifier

A metadata-frozen KartaView/Jakarta successor first materialized eight exact,
hash-verified images across four sequences and four contributors. Direct visual
audit found either selected target in `0/6` query frames, so no OCR or router was
run. The official public `mcp-karta-view` source was then pinned at commit
`37ac5c062c9ace66bcc540086236127a3fb86bf0`. Its public `nearby_photos` tool
calls only the geometric photo endpoint; `object_search` returns an empty result
and `KartaView token is required` before dispatch without a user-owned
`x-karta-token`. No dependency install, MCP launch, new KartaView cohort, or
credential request was made. Record
`L10_KARTAVIEW_PUBLIC_MCP_SOURCE_SIDE_VISIBILITY_AUTHORITY_NOT_ADMITTED`.

The separate RoadTextVQA/RoadText-1K source supplies ten sampled OCR frames per
driving video plus independently authored name-question answers. Video 19 was
the sole development exposure: exact `CAVA` was visible to OCR but unusable by
the legacy minimum-length-six rule. A first 30-video, OCR-unseen protocol added
roster-unique exact four/five-letter tokens. Correct rows rose `9 -> 15` and
UNKNOWN fell `15 -> 3`, but the result also contained one wrong and 11 ambiguous
episodes, so
`L10_ROADTEXTVQA_ROSTER_INFORMATION_ROUTER_DEVELOPMENT_GATE_NOT_MET`.

The failure showed that roster uniqueness is not environmental rarity. A sealed
background scan therefore computed video-document frequency for 64,090 OCR
tokens across all 2,635 training videos (`26,350` sampled-frame files). The next
router required eight information bits, target-token uniqueness in the current
roster, exact matching for length four/five, and at most edit distance one for
longer tokens. On a second disjoint 30-video panel it improved correct
`10 -> 15` and UNKNOWN `19 -> 10`, but still produced one wrong and four
ambiguous episodes. Inspection showed the decisive issue was not a threshold:
the supposedly competing names were often genuinely co-visible businesses
(`Hema` with `Hello`, `Vboard` with `Jayini`, and others). Record
`L10_ROADTEXTVQA_BACKGROUND_IDF_ROUTER_DEVELOPMENT_GATE_NOT_MET`.

Ten-meter assistance already has the user-selected destination, so the third
successor removed the invalid global-mutual-exclusion assumption. It applies the
same frozen background-IDF evidence only to the conditioned goal and can emit
only that target or `UNKNOWN`; co-visible names cannot redirect it. On a third,
fully disjoint 30-video OCR-unseen panel, correct/`UNKNOWN` improved
`6/24 -> 19/11` (`20.0% -> 63.3%`, `+43.3 pp`, `+13` correct). All 30
hash-derived absent-query canaries abstained, wrong-goal candidates and
identity/portal bindings were zero, and accepted episodes needed a mean `5.158`
of ten sampled frames. Record
`L10_ROADTEXTVQA_GOAL_CONDITIONED_BACKGROUND_IDF_VERIFIER_DEVELOPMENT_GATE_MET`.
A cyclic cross-video diagnostic accepted `1/30`: `canary` was repeatedly visible
in the challenged video. Because the dataset does not independently assert its
absence there, this is collision evidence and not an authorized false-positive
rate.

Exa literature review converged on the remaining representation gap. VPRText
removes repetitive nearby-place words and combines layout with edit similarity
([paper](https://www.mdpi.com/2414-4088/6/11/102)); a later text-VPR system uses
word boxes plus spatial-temporal coherence to suppress unrelated text matches
([paper](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2024.1424883/full));
scene-text retrieval itself is explicitly conditioned on a text query
([ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/papers/Lluis_Gomez_Single_Shot_Scene_ECCV_2018_paper.pdf)).
The local result is narrower: it validates goal conditioning and background
rarity but has no text-to-sign ownership or localization authority.

One structural follow-up therefore retained the IDF verifier unchanged and
added only a same-line compact-phrase branch over the source OCR boxes. It joins
two to four tokens when consecutive boxes overlap vertically by at least 50%
and their gap is at most three local text heights; compact aliases of length at
most five remain exact-only, longer aliases permit one edit. On the consumed
third panel it recovered `U + HAUL` in four frames and `HIGH + END` in two,
changing `19/30 -> 21/30` with `0/30` negatives. Record
`L10_ROADTEXTVQA_CONSUMED_LAYOUT_PHRASE_MECHANISM_DEVELOPMENT_GATE_MET` as
posthoc mechanism evidence only.

The fixed branch then ran once on a fourth, fully video/OCR-unseen panel of 30
multi-token answers. The IDF branch already accepted `20/30`; layout recovered
only `VAPOR + IN`, producing `21/30` (`70.0%`) and `0/30` generated two-token
negative accepts. The effect was real but below the preregistered minimum gain
of two, so record
`L10_ROADTEXTVQA_FRESH_GOAL_CONDITIONED_LAYOUT_PHRASE_VERIFIER_DEVELOPMENT_GATE_NOT_MET`.
Freeze its overlap, gap, token-count, edit, and information thresholds.

Freeze PB26, Mapillary, KartaView/Jakarta, and all four RoadTextVQA fresh panels.
The
next decision-changing check needs a
new provider/city-disjoint sequence cohort with independent exact-target truth,
reference views for the visual fallback, and exact-roster-absent controls. A
goal-conditioned background-IDF text observation is now the fixed lexical
successor, but generated canaries and cross-video collisions cannot replace
human target-absent pixel truth. The MR images cannot tune edit distance, suffix
rules, crop scale, or OCR preprocessing. A search-priority candidate remains neither
exact-instance/facade/portal ownership nor access, traversability, waypoint,
arrival, handoff, user benefit, or safety evidence.

Primary results and SHA-256:

- `l10_panolab_ordered_appearance_bank_result_v1.json`:
  `f9a0f01125a660da1f8062762ca69d4f9803c9cca555f68a653ce9c384bca847`
- `l10_panolab_component_router_result_v1.json`:
  `a8546c907e1d1d4c913e9699838c9fe35ea0de47dbde2865b8a1b884fe0e738f`
- `l10_panolab_open_set_router_result_v1.json`:
  `3974a36dbe411c91136b23e64ca75d386e7934b7e71d5487630ea747cb92db1b`
- `l10_panolab_cross_collection_positive_router_result_v1.json`:
  `db5103503a07c79e300b97e7ec7b00d5b3361743eb2639874717236926df04bd`
- `l10_panolab_temporal_query_router_result_v1.json`:
  `be8b307b85ea943ecef6d93b3c9d1b74174f750413632e43dac02493a4efb71b`
- `l10_panolab_temporal_lexical_appearance_router_result_v1.json`:
  `3deada55268a3bdc300e5f9f4b77b0a5f46351cb5219db4ea2848dee21faa4e4`
- `l10_panolab_producer_stratified_appearance_router_result_v1.json`:
  `3dda8517e564eecc9db35848a76b87b9cd5813f06aeb664eec2467108015b8d4`
- `l10_panolab_producer_stratified_lexical_appearance_router_result_v1.json`:
  `39225b9b69ddc2b708ce84966cdea74836096bd8aaba0d2444901e9ef68d988a`
- `l10_panolab_distinctive_edit_token_posthoc_result_v1.json`:
  `73734c0025be44e5aa289fd31fb27482d31f1eaed2257df508119bdff77b85f0`
- `l10_panolab_distinctive_token_fresh_confirmation_result_v1.json`:
  `84850bfc6fb68a46fbe43a31f2d8c67ef8e573d168ca43694c0e7e27305a5e45`
- `l10_panolab_distinctive_token_fresh_combined_router_result_v1.json`:
  `2856134e8d7f37b37dad0c68db93d74a4f5edbfb3e16f07d45c1172ee610e18c`
- `l10_panolab_distinctive_token_open_set_negative_result_v1.json`:
  `8b6f8b271badbe20cdd04d0109ddc49a395313ca5770aa6cbd7e5dc20fea0880`
- `l10_panolab_progressive_evidence_early_exit_posthoc_result_v1.json`:
  `a41722c776ca1be6ad578355696f3dbeb2de886ffb6634147f9cedb2232c7636`
- `l10_mapillary_distinctive_token_provider_transfer_result_v1.json`:
  `6333af29d64d2d26968ba6e7369bc605c97eab93e103d58d30a8149061bd6f4a`
- `l10_kartaview_public_mcp_capability_preflight_v1.json`:
  `8b3641805e1e69f9456784cb3c93b77ea633ae3a424ffd09dad6ebb8ff6450d2`
- `l10_roadtextvqa_roster_information_router_result_v1.json`:
  `5d7b035873b71b9782288a470d15a4727519c9f67d6d677e70ce2df1260a592b`
- `l10_roadtextvqa_background_idf_router_result_v1.json`:
  `3027dcae41930eac796e5d266cb2c4b71ed8476b68d8c77a098eadd217ae916c`
- `l10_roadtextvqa_goal_conditioned_idf_verifier_result_v1.json`:
  `aba065110339c355f62c963e4ccd25192475a870b112cd740ef6cc43028178bf`
- `l10_roadtextvqa_layout_phrase_posthoc_result_v1.json`:
  `7f4048f6e0e5c2c348f7db440ac7deee3b7de375f86ba08e239197ebf628cbf2`
- `l10_roadtextvqa_layout_phrase_fresh_result_v1.json`:
  `015b5925c16bb0bca324e2a22c7451ef73b3439b7132ae3cf57773613b166a86`

LychSim remains a conditional secondary synthetic lab, not this route's
replacement. The source/hardware screen and one bounded go/no-go test are in
`L10_LYCHSIM_FEASIBILITY_2026-08-31.md`.

#### PB19 Panoramax entrance-credential successors

The first successor changed the source from facade appearance to exact OSM
entrance-node text. Its frozen Development panel contained eight nodes, seven
target ways, and 22 real Panoramax views. Scene-single-view and
ray-single-view proofs were both `0/8`; source-blind multiview accumulation
proved only `1/8`, with zero wrong entrance proofs and `1/2` same-building
distinct credentials. One selected way had two pixels exposed to an inventory
agent after metadata selection froze, so the result remains curated
Development only. Record
`L10_PANOLAB_NODE_CREDENTIAL_FIELD_DEVELOPMENT_GATE_NOT_MET` and do not tune
OCR, crop, field order, or thresholds on those pixels.

The next successor replaced text with a query-independent Panoramax sequence
for the same exact entrance node. Four new target ways supplied eight strict
views and four cross-collection pairs. A frozen reciprocal native-grid DINO
field plus portal lattice retained a ray-containing portal in Top-3 on `3/4`,
but produced only `1/4` unique correct reference-bound portal tokens and
`9/12` wrong-reference ray bindings. Record
`L10_PANOLAB_CROSS_SEQUENCE_REFERENCE_PORTAL_DEVELOPMENT_GATE_NOT_MET`.
The whole viewport is place/facade evidence, not exact-door identity; do not
tune DINO, viewport, portal lattice, support binding, model, or fusion on this
cohort.

The exact-portal-patch successor then reserved the final three cross-sequence
ways from the frozen candidate pool. Reference and query pixels were given to
disjoint annotators. Strict orientation was `6/6` and cross-sequence pairing
was `3/3`, but reference source admission was `1/3`, query source admission was
`0/3`, and joint admission was `0/3`. Query truth was therefore not created and
EfficientLoFTR calls were exactly zero. Record
`L10_PANOLAB_EXACT_PORTAL_PATCH_SOURCE_NOT_EVALUABLE_REFERENCE_1_OF_3_QUERY_0_OF_3_NO_MATCHER_CALL`.
The `DIRECT` geometry label does not imply a uniquely visible portal in pixels.
The close-range successor then scanned the 111-row unconsumed metadata source
and froze all five remaining way-disjoint strict-DIRECT cross-collection
entrances before pixels. Its ten views were `7.290-22.319 m` from the entrance
(mean `13.997 m`), all passed strict orientation, and all five pairs were from
different collections. The role-separated audit nevertheless admitted `0/5`
reference portals and `1/5` query portals, hence `0/5` joint portals. The
formal three-way source gate failed, the matcher stayed at zero calls, and no
eligible unconsumed way remains in this local pool. Record
`L10_PANOLAB_CLOSE_PORTAL_SOURCE_NOT_EVALUABLE_REFERENCE_0_OF_5_QUERY_1_OF_5_JOINT_0_OF_5_NO_MATCHER_CALL`.
Distance alone did not turn the exact OSM node into visible pixel-portal truth.
Do not resample this pool or reopen a matcher rescue. The next Panoramax action
requires a genuinely new independent collection for a near strict-DIRECT seed,
or federated per-collection discovery in new coverage, frozen before pixels.

That bounded federated successor is now complete. It added five high-coverage
French cities and, before opening selected pixels, froze the official Panoramax
web-viewer 5.2.0 effective-zero pose semantics while retaining the full-sensor
2:1, true-north heading, azimuth-agreement, distance, global-DIRECT and exact
entrance-ray rules. Across 215 frozen named-building targets, strict projection
found 25 cross-collection targets and the viewer-equivalent contract found 32.
Global geometry retained 18 distinct cross-collection DIRECT ways and froze the
first five, with zero search or geometry errors. Record
`L10_PANOLAB_FEDERATED_VIEWER_EQUIVALENT_PORTAL_METADATA_GATE_MET`.

The single role-separated pixel audit did not convert that coverage into a
transfer source: reference admission was `0/5`, query admission was `2/5`, joint
admission was `0/5`, and matcher calls remained zero. Record
`L10_PANOLAB_FEDERATED_PORTAL_SOURCE_NOT_EVALUABLE_REFERENCE_0_OF_5_QUERY_2_OF_5_JOINT_0_OF_5_NO_MATCHER_CALL`.
Federated coverage and optional-pose metadata are therefore no longer the active
bottleneck. Do not mine more Panoramax cities, widen the ray, or resample these
opened rows. The next fresh cohort must add an independently measured entrance
extent or portal interval and freeze that geometry before pixels; Panoramax may
remain an image provider but is not the missing entrance-identity source.

#### Independent portal extent with Panoramax imagery

Two source-changing successors tested that requirement without touching the
matcher. The first paired the SAVeNoW/TUM LoD3 model with the local Panoramax
catalog. The frozen model supplied 61 unique door polygons, but the 74 catalog
images across four collections supplied zero near, directionally visible door
pairs: the minimum camera-to-door distance was `68.866 m`, the minimum
directionally possible distance was `175.052 m`, and pixel requests remained
zero. Record
`L10_LOD3_PANORAMAX_DOOR_REACHABILITY_GATE_NOT_MET_ZERO_NEAR_VISIBLE_DOOR_PAIRS`.
The published `1-3 cm` LoD3 figure is relative modeling accuracy, not global
Door-to-Panoramax registration. Also preserve the declared CRS split: the
combined CityGML2 file is `EPSG:25832`, while the individual files declare
`EPSG:32632`; never silently treat them as identical.

The second successor replaced the point ray with a frozen width interval from
an independently mapped OSM entrance width and host-wall tangent. Across three
building-disjoint targets, the metadata replay reduced 359 returned items to
194 perspective-projectable and 40 robust under each image's full declared
horizontal-accuracy circle, then froze six images from six distinct
collections. Record
`L10_WIDTH_FIRST_PERSPECTIVE_PORTAL_METADATA_GATE_MET_3_OF_3_BUILDINGS_6_DISTINCT_COLLECTIONS`.
The one role-separated pixel audit admitted `0/3` reference portals and `0/3`
query portals, hence `0/3` joint portals and zero matcher calls. Record
`L10_WIDTH_FIRST_PERSPECTIVE_PORTAL_SOURCE_NOT_EVALUABLE_REFERENCE_0_OF_3_QUERY_0_OF_3_JOINT_0_OF_3_NO_MATCHER_CALL`.
The mapped width improved metadata reachability, but the declared `4-5 m`
camera-position envelopes remained too broad and some views were grazing or
off-facade. Do not narrow those declared envelopes or reselect the consumed
cohort after pixels. The next source change must add sub-metre
camera-to-portal registration or a directly posed portal mask/mesh. Panoramax
may carry the pixels only when that independent pose/registration is supplied;
generic Panoramax mining is closed.

#### Directly posed portal transfer ceiling

The next source change replaced unanchored outdoor pixels with a provider-posed
reference door surface. Hypersim `ai_034_001` supplied three independent camera
trajectories, 300 public semantic-instance masks, and nine door instance IDs.
Before opening RGB, the source selector froze the first three door-disjoint
instances satisfying one fixed mask-size/margin rule, visibility in two camera
trajectories, and at least one query distractor door. The selected targets were
instance IDs `32`, `33`, and `36`; no RGB, depth, pose, or model output was used
to select them.

One frozen mechanism then lifted each reference target contour through the
provider world-position image and projected it into the independent query
trajectory with Hypersim's official per-scene `M_proj` and query pose. It filled
one convex portal envelope, ranked every visible query door by envelope IoU,
and emitted the projected centroid as one observation ray endpoint. Query RGB,
query semantic masks, other-door truth, and preview overlays remained
evaluator-only; query depth and query world positions were not inputs.

| Posed-portal metric | HP01 | HP02 | HP03 | aggregate |
|---|---:|---:|---:|---:|
| target / selected instance | 32 / 32 | 33 / 33 | 36 / 36 | 3/3 correct |
| camera baseline | 3.843 m | 1.051 m | 0.940 m | 0.940-3.843 m |
| envelope IoU | 0.7063 | 0.4944 | 0.7528 | 0.7063 median |
| precision / recall | 1.0000 / 0.7063 | 0.5951 / 0.7451 | 0.9493 / 0.7843 | -- |
| centroid error | 8.65 px | 34.95 px | 16.51 px | 20.04 px mean |
| wrong-door commit | 0 | 0 | 0 | 0/3 |
| centroid inside target | yes | yes | yes | 3/3 |

Record `L10_HYPERSIM_POSED_PORTAL_TRANSFER_DEVELOPMENT_GATE_MET`. This is the
first positive result after the Panoramax source ceiling because it adds the
missing camera-to-portal registration instead of tuning the consumed matcher.
It remains a privileged synthetic indoor mechanism ceiling: semantic door
surface is not doorway aperture or traversability, and the run establishes no
real/outdoor, named-entrance, access, approach, waypoint, arrival,
`HANDOFF_READY`, product, user-benefit, or safety authority. The next
confirmation should use real posed door geometry, preferably ScanNet++ after
access and materialization. Outdoor confirmation still needs sub-metre imagery
co-registered with the existing LoD3 doors. A Panoramax-only relative SfM
canary is insufficient and stays parked as
`RELATIVE_REGISTRATION_ONLY / NOT_PORTAL_BOUND`.

#### Real posed RGB-D confirmation

SceneNN then replaced the synthetic world-position credential with real
handheld RGB-D, reconstructed camera poses, an instance-labelled mesh, and
three door-disjoint scenes (`014`, `249`, and `521`). The first protocol asked
for the complete target door inside a four-pixel image margin. Across all
`9,000` poses it found zero qualifying frames, so record
`L10_SCENENN_FULL_DOOR_SOURCE_NOT_EVALUABLE_ZERO_FULLY_CONTAINED_TARGET_FRAMES`.
Before any ONI frame was opened, v2 made one source-level change: carry the
image-clipped target surface. It froze door IDs `814659`, `889447`, and
`1024952`, one ordered reference/query pair per scene, and the same plane,
transfer, ranking, and gate rules.

The one six-frame replay fitted a robust plane from real reference depth and
projected its contour through the held-out query pose. Query RGB, query depth,
query mesh labels, and other-instance envelopes did not enter the prediction.

| Real posed-portal metric | SN01 / 014 | SN02 / 249 | SN03 / 521 | aggregate |
|---|---:|---:|---:|---:|
| target / selected instance | 814659 / 60040 | 889447 / 889447 | 1024952 / 1024952 | 2/3 correct |
| camera baseline | 2.277 | 1.147 | 1.010 | 1.147 median |
| visible transferred envelope | no | yes | yes | 2/3 |
| envelope IoU | 0.0000 | 0.4719 | 0.2647 | 0.2647 median |
| precision / recall | 0.0000 / 0.0000 | 0.8760 / 0.5085 | 0.2924 / 0.7110 | -- |
| centroid inside target | no | yes | no | 1/3 |
| world-centroid error | 0.985 | 0.683 | 1.457 | 0.985 median |
| reference-plane normal error | 82.47 deg | 4.55 deg | 16.60 deg | 34.54 deg mean |

Record
`L10_SCENENN_REAL_RGBD_PARTIAL_METRIC_PORTAL_TRANSFER_CONFIRMATION_GATE_NOT_MET`.
The decisive error is source visibility, not matcher capacity. In SN01 the
mesh-projected reference envelope was behind a visible occluder; a plane with
only `2.5 mm` median residual was nevertheless `82.47 deg` from the target
door plane and projected entirely outside the query image. SN02 and SN03 still
selected the correct physical door, but their geometry was not accurate enough
for the frozen IoU, centroid, or nominal metric-world gates.

Do not reselect these scenes or frames, change the 70-percent plane fit, or
tune the gates. The next admissible source change is a fresh real-scene cohort
with source-side visibility authority: a provider 2D door-instance mask or a
z-buffered semantic triangle surface checked against synchronized depth before
forming the reference credential. Keep query-side RGB/depth evaluator-only and
retain Panoramax as closed until an independent absolute portal anchor exists.

The strict-triangle v3 successor made exactly that one source change on six
fresh geometry-only candidates. Before opening RGB-D it froze the top three
visibility-qualified episodes: `246:3153->4108`, `032:706->2439`, and
`073:401->4650`. The selector rasterized only faces whose three vertex labels
equalled the target door ID, then admitted pixels that survived the full-scene
z-buffer. The implementation, imported v2 helper, synchronized sparse-frame
extractor source, official SceneNN MD5 values, and all selected masks were
bound into the cohort. An extractor that reproduces the official 33 ms
lag-only synchronization loop was byte-for-byte and pixel-for-pixel identical
to the existing sealed `014` RGB-D pair before the one new replay ran.

| Visible-source metric | SV01 / 246 | SV02 / 032 | SV03 / 073 | aggregate |
|---|---:|---:|---:|---:|
| target / selected instance | 1107814 / 1107814 | 483 / 483 | 445 / 258 | 2/3 correct |
| camera baseline | 2.175 | 2.323 | 1.141 | 2.175 median |
| reference visible pixels | 20,443 | 17,763 | 10,365 | -- |
| reference bbox width | 118 | 66 | 67 | all touch right image edge |
| visible transferred envelope | yes | yes | no | 2/3 |
| envelope IoU | 0.1197 | 0.0408 | 0.0000 | 0.0408 median |
| precision / recall | 0.8395 / 0.1215 | 0.5531 / 0.0232 | 0.0000 / 0.0000 | -- |
| centroid inside target | yes | yes | no | 2/3 |
| world-centroid error | 0.659 | 0.973 | 0.685 | 0.685 median |
| reference-plane normal error | 15.97 deg | 1.24 deg | 6.77 deg | 8.00 deg mean |

Record `L10_SCENENN_VISIBLE_METRIC_PORTAL_TRANSFER_DEVELOPMENT_GATE_NOT_MET`.
The z-buffer change was nevertheless diagnostic: mean reference-plane normal
error fell from v2's `34.54 deg` to `8.00 deg`, so it removed the foreground
plane contamination that broke SN01. The remaining information gap is portal
extent. Every frozen reference credential was an unoccluded but image-edge-
clipped sliver only `66-118 px` wide; it supplied an accurate local plane but
not the complete door centre or boundary. The two visible predictions covered
only `12.15%` and `2.32%` of the query target envelope, while the third
projected outside the query image.

Do not add a post-result image margin, reselect these scenes, or tune the
visibility, plane, contour, or decision thresholds. 3RScan E0 made the next
source change: freeze a door instance shared by a reference scan and rescan,
use the official scan-to-reference transform and stable instance mapping, and
evaluate a reference-derived extent against separate rescan geometry. This
changes the missing extent source rather than rescuing consumed SceneNN.

The one same-source ablation retained the positive-edge 25 percent of each
reference door in the partial arm and carried the complete provider instance in
the registered arm. Across three train-split endpoints from distinct reference
scenes, both arms kept target Top-1 at `3/3`, wrong commits at `0/3`, and
centroid-inside at `3/3`. Complete extent raised median planar IoU from `0.2727`
to `0.7688` (`+0.4961`) and reduced median world-centroid error from `0.3760 m`
to `0.1428 m` (`-0.2332 m`), meeting every frozen E0 condition. Record
`L10_3RSCAN_REGISTERED_ENDPOINT_EXTENT_DEVELOPMENT_GATE_MET`.

This effect is aggregate rather than universal: on RE03 the complete arm had
`0.3168 m` centroid error versus the partial arm's `0.1570 m`, although its IoU
rose `0.4046 -> 0.4898` and it retained the correct door. E0 therefore supports
complete registered extent as the missing information source on this narrow
real-geometry Development cohort; it does not establish per-door dominance or
an RGB method.

The first pixel successor froze the only new stable door among the 12 locally
materialized scans whose complete instance was inside both reference and rescan
images. The selector read `4,828` pose/depth members and zero RGB members before
freezing PF01. It then used a frozen DINOv2-S `36 x 20` feature grid and trained
one reference-only linear mask plus canonical-coordinate head per arm. Query
pose, depth, instance geometry, labels, and registration remained evaluator-only
until both RGB predictions were sealed.

Complete reference supervision raised held-out rescan IoU from `0.0706` to
`0.3312` (`+0.2607`), kept the correct instance Top-1, and retained `47.45%` of
the `0.6980` registered-geometry ceiling. This was a real signal, but not a gate
pass: IoU remained below `0.35`, the predicted centroid fell outside the target,
and its evaluator-lifted world error was `3.159 m`. The learned query coordinate
range covered only part of the canonical width, so the global homography
extrapolated far outside supported pixels. Record
`L10_3RSCAN_REFERENCE_PIXEL_ENDPOINT_FIELD_DEVELOPMENT_CANARY_NOT_MET`.

Do not tune the consumed PF01 images, linear head, threshold, or homography.
BRM01 made the requested structural change on fresh target doorframe 47. DINO
mutual matches became fixed positive/negative prompts for frozen SAM2.1; the
extent was the convex hull of one query-mask component, its endpoints were
actual supported pixels, and neither coordinate regression nor homography was
present. Dispersed positive matches still caused a room-scale mask: complete
IoU was `0.1606`, ceiling retention `17.93%`, centroid-inside failed, and metric
error was `1.427 m`. Record
`L10_3RSCAN_BOUNDED_REFERENCE_ENDPOINT_MASK_DEVELOPMENT_CANARY_NOT_MET`. A
first evaluator execution aborted before SAM because the partial arm had only
one mutual positive prompt; the one mechanical retry merely serialized that
arm as an empty prediction and kept all frozen decision inputs unchanged.

SRM01 then used a different reference scene and froze target doorframe 27 at
frames `213 / 174`, with `0.881 / 0.841` depth-visible ratios and zero RGB reads
before freeze. It replaced dispersed prompts with a dense foreground-minus-
background DINO field, selected one connected component, and bounded the SAM
mask by the component's fixed expanded ROI. Complete IoU reached `0.5403`,
centroid-inside passed, metric error fell to `0.422 m`, and both endpoints were
supported query pixels. The result remained below gate because it retained
`56.30%` of the `0.9597` registered ceiling and overlapping instance 11 scored
`0.6053` versus target 27 at `0.5403`. The partial arm reached `0.6908`, target
Top-1, and `0.077 m`, demonstrating a strong bounded localization effect but
not complete-reference dominance. Record
`L10_3RSCAN_SPATIAL_REFERENCE_ENDPOINT_MASK_DEVELOPMENT_CANARY_NOT_MET`.

Do not tune either consumed bounded cohort. Target 27's registered-geometry
projection itself overlaps instance 11 at `0.8842`; the remaining information
gap is exact-instance or set-valued portal binding. A new successor must add a
legal public binding or preserve the ambiguous portal set before affordance and
active observation. Another SAM, embedding, threshold, fusion, or projective
endpoint decoder is not authorized. Generic Panoramax remains closed without
an independent sub-metre portal anchor.

Registered-extent E0 protocol, executable, frozen cohort, and result:
`l10_3rscan_registered_extent_protocol_v1.json`,
`l10_3rscan_registered_extent_ceiling.py`,
`l10_3rscan_registered_extent_cohort_v1.json`, and
`l10_3rscan_registered_extent_result_v1.json`.

Reference-conditioned pixel-field protocol, executable, frozen cohort, and
result: `l10_3rscan_reference_pixel_field_protocol_v1.json`,
`l10_3rscan_reference_pixel_field.py`,
`l10_3rscan_reference_pixel_field_cohort_v1.json`, and
`l10_3rscan_reference_pixel_field_result_v1.json`.

Bounded mutual-match/SAM protocol, executable, frozen cohort, and result:
`l10_3rscan_bounded_reference_mask_protocol_v1.json`,
`l10_3rscan_bounded_reference_mask.py`,
`l10_3rscan_bounded_reference_mask_cohort_v1.json`, and
`l10_3rscan_bounded_reference_mask_result_v1.json`.

Spatially coherent bounded-field protocol, executable, frozen cohort, and
result: `l10_3rscan_spatial_reference_mask_protocol_v1.json`,
`l10_3rscan_spatial_reference_mask.py`,
`l10_3rscan_spatial_reference_mask_cohort_v1.json`, and
`l10_3rscan_spatial_reference_mask_result_v1.json`.

Independent-extent protocols, source builders, audits, and results:
`l10_lod3_panoramax_door_reachability_result_v1.json`,
`l10_width_first_perspective_portal_protocol_v1.json`,
`l10_width_first_perspective_portal_source.py`,
`l10_width_first_perspective_portal_source_result_v1.json`,
`l10_width_first_perspective_portal_source_v1.json`,
`l10_width_first_perspective_portal_materialize.py`,
`l10_width_first_perspective_portal_reference_audit_v1.json`,
`l10_width_first_perspective_portal_query_audit_v1.json`,
`l10_width_first_perspective_portal_adjudicate.py`, and
`l10_width_first_perspective_portal_source_admission_result_v1.json`.

Posed-portal protocol, frozen pre-RGB cohort, one replay implementation, and
result: `l10_hypersim_posed_portal_protocol_v1.json`,
`l10_hypersim_posed_portal_cohort_v1.json`,
`l10_hypersim_posed_portal_transfer.py`, and
`l10_hypersim_posed_portal_result_v1.json`.

Real posed-portal source audit, protocols, frozen cohort, replay implementation,
and result: `l10_scenenn_full_door_source_result_v1.json`,
`l10_scenenn_real_posed_portal_protocol_v1.json`,
`l10_scenenn_real_posed_portal_protocol_v2.json`,
`l10_scenenn_real_posed_portal_cohort_v2.json`,
`l10_scenenn_real_posed_portal_transfer.py`, and
`l10_scenenn_real_posed_portal_result_v2.json`. The strict visible-surface
successor adds `l10_scenenn_visible_portal_protocol_v3.json`,
`l10_scenenn_visible_portal_cohort_v3.json`,
`l10_scenenn_visible_portal_transfer.py`,
`l10_scenenn_extract_selected_sync.cpp`,
`l10_scenenn_visible_portal_rgbd_receipt_v3.json`, and
`l10_scenenn_visible_portal_result_v3.json`.

Credential successor protocols, evaluators, frozen sources, and results:
`l10_panolab_node_credential_protocol_v1.json`,
`l10_panolab_node_credential.py`,
`l10_panolab_node_credential_source_v1.json`,
`l10_panolab_node_credential_result_v1.json`,
`l10_panolab_reference_portal_protocol_v1.json`,
`l10_panolab_reference_portal.py`,
`l10_panolab_reference_portal_source_v1.json`,
`l10_panolab_reference_portal_result_v1.json`,
`l10_panolab_exact_portal_patch_protocol_v1.json`,
`l10_panolab_exact_portal_patch.py`,
`l10_panolab_exact_portal_patch_source_v1.json`,
`l10_panolab_exact_portal_reference_annotations_v1.json`,
`l10_panolab_exact_portal_query_source_audit_v1.json`, and
`l10_panolab_exact_portal_patch_result_v1.json`, plus
`l10_panolab_close_portal_source_protocol_v1.json`,
`l10_panolab_close_portal_source.py`,
`l10_panolab_close_portal_freeze_v1.json`,
`l10_panolab_close_portal_source_v1.json`,
`l10_panolab_close_portal_reference_audit_v1.json`,
`l10_panolab_close_portal_query_audit_v1.json`, and
`l10_panolab_close_portal_source_result_v1.json`, plus
`l10_panolab_viewer_equivalent_projection_protocol_v1.json`,
`l10_panolab_viewer_equivalent_projection.py`,
`l10_panolab_federated_portal_source_protocol_v1.json`,
`l10_panolab_federated_portal_source.py`,
`l10_panolab_federated_portal_source_result_v1.json`,
`l10_panolab_federated_portal_materialize.py`,
`l10_panolab_federated_portal_source_v1.json`,
`l10_panolab_federated_portal_reference_audit_v1.json`,
`l10_panolab_federated_portal_query_audit_v1.json`,
`l10_panolab_federated_portal_adjudicate.py`, and
`l10_panolab_federated_portal_source_admission_result_v1.json`.

Durable PB10 summary:
`named_poi_trans4trans_glass_door_plane_development_result_v1.json`.

PB11 protocol, frozen cohort, source audit, evaluator, and durable summary:
`named_poi_metric_portal_closure_protocol_v1.json`,
`named_poi_metric_portal_closure_development_cohort_v1.json`,
`named_poi_sunrgbd_metric_closure_source_audit.py`,
`named_poi_metric_portal_closure.py`, and
`named_poi_metric_portal_closure_development_result_v1.json`.

PB12 protocol, frozen cohort, evaluator, and durable summary:
`named_poi_door_part_topology_protocol_v1.json`,
`named_poi_door_part_topology_development_cohort_v1.json`,
`named_poi_door_part_topology.py`, and
`named_poi_door_part_topology_development_result_v1.json`.

PB13 protocol, frozen cohort, evaluator, and durable NOT_EVALUABLE receipt:
`named_poi_florence_pixel_part_topology_protocol_v1.json`,
`named_poi_florence_pixel_part_topology_development_cohort_v1.json`,
`named_poi_florence_pixel_part_topology.py`, and
`named_poi_florence_pixel_part_topology_development_result_v1.json`.

PB14 protocol, frozen cohort, evaluator, and durable negative result:
`named_poi_yoloe_multiscale_part_topology_protocol_v1.json`,
`named_poi_yoloe_multiscale_part_topology_development_cohort_v1.json`,
`named_poi_yoloe_multiscale_part_topology.py`, and
`named_poi_yoloe_multiscale_part_topology_development_result_v1.json`. The
result SHA-256 is
`a729b850c21bd14976aebbaf32c654da2e81b1c07b12f42401dae5b586c7c887`.

PB15 protocol, frozen cohort, evaluator, and durable negative result:
`named_poi_grounded_sam_multiscale_part_topology_protocol_v1.json`,
`named_poi_grounded_sam_multiscale_part_topology_development_cohort_v1.json`,
`named_poi_grounded_sam_multiscale_part_topology.py`, and
`named_poi_grounded_sam_multiscale_part_topology_development_result_v1.json`.
The result SHA-256 is
`76d172298d9a1d98371b481ae2db808db5ed1329ec0cbc3b18d9bbf1b50c5c76`.

PB16 protocol, frozen cohort, evaluator, and durable negative result:
`named_poi_sam3_native_part_topology_protocol_v1.json`,
`named_poi_sam3_native_part_topology_development_cohort_v1.json`,
`named_poi_sam3_native_part_topology.py`, and
`named_poi_sam3_native_part_topology_development_result_v1.json`. The result
SHA-256 is
`c56116c95cc10ba3c7b1c9f4f07489b5e4ac42f61de2a842c27e4a713d8fb9ab`.

PB17 protocol, frozen cohort, evaluator, and durable partial adjudication:
`named_poi_sam3_official_door_state_protocol_v1.json`,
`named_poi_sam3_official_door_state_development_cohort_v1.json`,
`named_poi_sam3_official_door_state.py`, and
`named_poi_sam3_official_door_state_development_result_v1.json`. The result
SHA-256 is
`ce644f3763363b201dcf9d4a7f17e2d7bcd05e634ae949fbf5044bab0cd12d6f`.

PB18 source selection, frozen cohort, protocol, evaluator, and durable result:
`named_poi_scil_ingress_portal_graph_source_v1.json`,
`named_poi_scil_ingress_portal_graph_fresh_cohort_v1.json`,
`named_poi_scil_ingress_portal_graph_protocol_v1.json`,
`named_poi_scil_ingress_portal_graph.py`, and
`named_poi_scil_ingress_portal_graph_result_v1.json`. The formal raw result is
SHA-256
`03f8b3b59bb1eec21e706d07b751f3edd345b95a998076681d363ba8ed60fbec`.

The `x/120` column is not physical target-absence authority. The source audit
did not exhaustively freeze every gallery building co-visible in dense
same-city frames, so physical target-absent rejection is `NOT_EVALUABLE`. A
post-result integrity adjudication changed no model output, positive label,
threshold, denominator, or decision: Recall@1 and target-present acceptance
already fail the gate independently.

Protocol and durable summary are
`named_poi_place_identity_protocol_v1.json` and
`named_poi_place_identity_result_v1.json`; the authority correction is
`named_poi_place_identity_adjudication_v1.json` (SHA-256
`ed08e330df0902bd78e5a74510ef664adc0a6f8888103182a79d796e9d7f9523`).
Raw result SHA-256 is
`1c40187e7b0269fc48b77b77d64c2f657f0a14e265c46b0c78ffd023443a03fd`.
The generic baseline, SALAD, and MixVPR all selected verified RTX 5060 CUDA
after measured CPU/GPU probes. This curated confirmation does not reject VPR
generally and establishes no portal ownership, access, traversability, active
motion, navigation, arrival, product benefit, user benefit, or safety.

V2 result:
`artifacts.local/evidence/l10-r0/named-poi-multifacet-entrance-v2/result.json`,
SHA-256 `dff6fcd89460f185fc3785549131800869ce7661d35d5119b3f5dd4d58488f51`.
Encoder and GroundingDINO backend receipts are
`f9771304ef3d2665b6e8fbada9330aa118b690fe5def8c8b1a85abc8c54c6771`
and `f5a29bcbcc3fac435fda39a25cf4abe16cd99d06d16e670358bcd2dd230b6c92`.
The encoder selected measured-faster CPU (`0.125 s` versus CUDA `0.314 s`);
GroundingDINO selected verified RTX 5060 CUDA (`1.186 s` versus CPU `3.788 s`).
No OCR calls were made. Image-level human entrance visibility still does not
establish exact entrance pixels, public access, traversability, accessibility,
metric approach, arrival, product benefit, user benefit, or safety.

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

## HierText exhaustive-truth goal-carrier replay

The official HierText test release supplies 1,634 previously unused images and
complete word transcripts/polygons. Its `test.tgz` is 537,066,598 bytes with
SHA-256 `3ad9bbaad4e03df33898700febac701fe7aa6bcc88fc14634dfb6e9b007a97df`.
The source is independent of RoadTextVQA and the earlier HierText geometry audit,
which consumed validation annotations only. Official documentation states that
test annotations were released on 2024-12-02, images come from Open Images, and
the dataset is CC BY-SA 4.0:
<https://github.com/google-research-datasets/hiertext>.

Three disjoint, GT-only-selected 30-image panels were consumed in order. The
first fixed multiscale branch changed correct/wrong/`UNKNOWN`
`9/0/21 -> 13/2/15` and accepted `0/30` queries proven absent by complete truth;
it missed the no-wrong-regression gate. Audit showed that exact target words
were often merged into longer OCR lines. The second successor projected a
unique exact query substring onto its character span along the OCR
quadrilateral. It changed `7/1/22 -> 22/3/5`, again with `0/30` absent accepts,
but exposed text-identical instance ambiguity and one truth fragment contained
inside a longer word.

The third protocol fixed both the evaluator referent contract and runtime
abstention before opening any further pixels. A selected target must occur once
across every annotated word node and cannot be a substring of a longer truth
token. Runtime preserves query-span projection, but returns `UNKNOWN` when
best-rank OCR observations form multiple spatial carrier components. On 30
further images it changed correct/wrong/`UNKNOWN` `11/0/19 -> 22/0/8`
(`36.7% -> 73.3%`, `+36.7 pp`, `+11` correct) with `0/30` complete-truth-
absent accepts. Every frozen gate passed, so record
`L10_HIERTEXT_TEST_UNIQUE_REFERENT_SPAN_CARRIER_DEVELOPMENT_GATE_MET`.

Evidence:

- first multiscale result:
  `l10_hiertext_test_multiscale_goal_verifier_result_v1.json`, SHA-256
  `306c16d62dd095809eb584ab6dc0d1cfffd0eb748ad52aa4d82e5007715100c9`;
- query-span result:
  `l10_hiertext_test_span_carrier_verifier_result_v1.json`, SHA-256
  `22805d161e769b3b4dd996f9d39b3b4d28381f531d0bbe622706a255e0204dd0`;
- unique-referent query-span result:
  `l10_hiertext_test_unique_referent_span_verifier_result_v1.json`, SHA-256
  `b661e72459498dfc4a4e434a6618f0232c562bbf13d288347ce0e4524f0beed9`.

This is a static OCR carrier Development result. Complete-truth uniqueness is an
evaluator authority, not an open-world instance-identity solution. A localized
text carrier is not a facade, sign, portal, entrance ownership, access,
traversability, waypoint, arrival, handoff, user-benefit, deployment, or safety
claim.

## FSNS source-distinct multi-view goal evidence

The official French Street Name Signs source moves the text mechanism from one
static image to multiple observations of the same physical sign. Each sample is
a `600x150` image containing up to four `150x150` crops intended to come from
different positions and/or times, with a normalized canonical street-name label.
The source and format are documented by the
[official TensorFlow model repository](https://github.com/tensorflow/models/tree/master/research/attention_ocr)
and the [FSNS paper](https://arxiv.org/abs/1702.03970).

The 7,904,079-byte official testdata file was used only for mechanism
development. Its 50 examples include 39 four-view samples. On a GT-only-selected
30-example panel, the frozen single-first-view exact branch reached `26/30`; the
multi-view goal-conditioned branch reached `30/30`, with `0/30` canonical-label-
disjoint challenge accepts and `0/30` synthetic-negative accepts. Because the
development protocol required a gain of five while the observed baseline left
only four possible recoveries, record
`L10_FSNS_TESTDATA_MULTIVIEW_GOAL_MECHANISM_GATE_NOT_MET`; do not rewrite that
gate after the result.

One official validation shard was then admitted without opening its selected
pixels. It contains 252 physical-sign samples, including 201 with four views:

- TFRecord: 39,890,103 bytes, SHA-256
  `b8eb695d267ea5a0ae9d7ce86f270d614b15fe5e7b09bb68043e3696e2e67ea4`;
- lossless adapter manifest: 252 rows, SHA-256
  `258dcd60b1bc3bdfa53729823f837ab31c66314eb30bdac2517e62c76362c9a7`;
- source adapter: `l10_fsns_tfrecord_adapter.py`, SHA-256
  `dc9d2a9f8444cd2906145ff6a8b40ca560056c5937c7a6195e6a9c4f9f2de7c8`.

The formal cohort contains 40 four-view samples in source order. Every image
hash, normalized canonical label, and distinctive goal token is disjoint from
all 50 testdata examples. The algorithm is unchanged: OCR each view at the same
3x scale, accept an exact conditioned distinctive token from any view, and allow
edit distance one only if the same target token recurs in at least two different
views. The result changed correct/`UNKNOWN` `34/6 -> 40/0`
(`85% -> 100%`, `+15 pp`, `+6` correct). All 40 accepted by exact evidence; the
six new rows first accepted at view 2 (`3`) or view 3 (`3`). Canonical-label-
disjoint challenges and synthetic negatives both remained `0/40`. Every frozen
gate passes, so record
`L10_FSNS_FRESH_MULTIVIEW_GOAL_EVIDENCE_DEVELOPMENT_GATE_MET`.

Evidence:

- protocol: `l10_fsns_multiview_goal_verifier_protocol_v1.json`, SHA-256
  `2f22876af176f031d5567a72554abf37b73642ce506d2707dc9a102b89010194`;
- result: `l10_fsns_multiview_goal_verifier_result_v1.json`, SHA-256
  `54a3389fa22fd5fc615118ae23a7cc1db1d50dfcc0dda519354069995ce53e1e`;
- OCR cache SHA-256
  `52d7a3c26bdbeae67027da2c4226a6ff1de657bc9c2287f312fbabf040637ff8`.

This establishes a source-distinct multi-view canonical sign-reading mechanism.
FSNS does not provide exhaustive pixel text, per-tile target boxes, camera poses,
the observation action that produced a view, metric distance, facade or entrance
association, portal ownership, access, arrival, handoff, deployment, or safety.
The label-disjoint challenges are collision diagnostics, not a pixel-absence
false-positive rate.

## CATALIST controlled-action goal recovery

Exa discovery identified CATALIST as the first admitted public text source in
this route that records a deliberately executed observation transformation
rather than only passive multi-view association. The
[official site](https://catalist-2021.github.io/) and
[paper](https://www.cse.iitb.ac.in/~ganesh/papers/catalist2021.pdf) document
2,322 real 1920x1080 videos recorded at 25 fps on a tripod under translation,
roll, tilt, pan, and zoom. The official annotation provides a manually verified
video text label, transformation class, and optional start/end times. Its
Cloud-Vision per-frame masks are explicitly unverified and were not used. The
official Kaggle API reports the license as unknown, so source videos remain
local and are not redistributed.

The formal selector read only the official train/validation text files. It
froze 30 validation videos at six per action class, source-order-first per
label/action pair. The cohort contains 12 verified labels; both every full label
and each selected distinctive goal token are absent from the 1,595 valid
training rows. The evaluator transferred the FSNS evidence contract unchanged:
first-frame exact goal evidence is the baseline; the successor retains it and
samples 25/50/75/100 percent of the annotated controlled transformation,
accepting exact evidence from any checkpoint or edit distance one only across
two different checkpoints. Three pre-result transport attempts are recorded in
the protocol: the first stopped before model import; the next two exposed
unreliable MP4 random seeking and false container frame counts. No cache,
result, metric, or OCR output was persisted or surfaced. The final transport
counts decodable frames first and projects the unchanged fractional checkpoints
onto that interval.

The result changed first-frame correct/`UNKNOWN` `27/3 -> 29/1`
(`90.0% -> 96.7%`, `+2`, `+6.667 pp`). Both recoveries were pan episodes; the
other four action classes started at `5/6` or `6/6` and added none. All 30
canonical-label-disjoint challenges and all 30 synthetic negatives abstained.
The preregistered minimum gain was five, so record
`L10_CATALIST_CONTROLLED_ACTION_GOAL_RECOVERY_DEVELOPMENT_GATE_NOT_MET`.
A posthoc two-token check on the consumed OCR cache changed `27 -> 27`; it did
not justify opening the six remaining validation labels for the same mechanism.

Evidence:

- evaluator: `l10_catalist_controlled_action_goal_recovery.py`, SHA-256
  `38bf563f955a253832881c616b3fa8a71836dd9fe79044335f0371dfd2fa136d`;
- protocol: `l10_catalist_controlled_action_goal_recovery_protocol_v1.json`,
  SHA-256 `1bbcb3abede51262ae3e297dad995eae7cf6453a02da45e88b705dc8ff053b4b`;
- result: `l10_catalist_controlled_action_goal_recovery_result_v1.json`,
  SHA-256 `3ed202edc9d72d24076db6fb073ad57215db45d7d20fbd26d5566f103a65a840`;
- OCR cache SHA-256
  `e1f8ee8ec4fb4089aff01ca069ec3ad033c2b2a80f7837bded4d4a5d1d632b71`.

This is controlled-tripod Development evidence. CATALIST does not provide
transformation direction, metric camera pose, randomized stationary
counterfactuals, exhaustive per-frame pixel text, business/facade/sign
ownership, a door or entrance, public access, traversability, arrival, handoff,
user benefit, or safety. Recovery after a labelled transformation is
intervention-linked evidence, not proof that a live L10 controller selected the
action or recovered an exact target entrance in an open street.

## 3D Street View center-target lock

Exa discovery identified the official
[3D Street View project](http://3drepresentation.stanford.edu/) and
[repository](https://github.com/amir32002/3D_Street_View). Its test contract
labels whether two observations share the exact physical target point and
places that target at the optical center. The 1 GB official matching archive was
rate-limited, so this replay freezes the first 181,018,624 bytes instead of
pretending the source is complete. That prefix contains 6,220 fully decodable
JPEG members. Intersecting them with `verpairs.txt`, then placing endpoint ID
prefixes into a SHA-256-derived group split, yielded 2,180 training pairs and
426 prefix-disjoint test pairs; 957 cross-partition pairs were discarded.

The baseline is global/global cosine from the local immutable DINOv2-small
snapshot. The frozen successor uses the provider geometry directly: 101x101
target patches remain unchanged; full 640x640 images receive centered 1.0, 0.5,
and 0.25 crops, and the score selects scale only on the full-image side. On the
held-out groups, AUROC improved `0.949859 -> 0.984860` (`+3.5001 pp`), balanced
accuracy `0.895773 -> 0.926341` (`+3.0568 pp`), average precision
`0.954445 -> 0.982243`, and retrieval Top-1 across 32 eligible anchors
`0.843750 -> 1.000000` (`+15.625 pp`). The image/patch subgroup improved AUROC
`0.874644 -> 0.966274`, image/image improved `0.987794 -> 0.998644`, and the
already cropped patch/patch subgroup stayed `0.981692`. Record
`L10_3DSTREETVIEW_CENTER_TARGET_LOCK_DEVELOPMENT_GATE_MET`.

Evidence:

- evaluator: `l10_3dstreetview_center_target_lock.py`, SHA-256
  `1abb07f11e1ac5e1c30d11ccc3aca1f88362e0bd4c330cfbf69f549d551d0f81`;
- protocol: `l10_3dstreetview_center_target_lock_protocol_v1.json`, SHA-256
  `3a7938e9839daa240513a397d14e79c87b82431db5475f3e75b0284ed25789b6`;
- cohort: `l10_3dstreetview_center_target_lock_cohort_v1.json`, SHA-256
  `f81a76e8a45ade7ff2b4a9a18cb32feb35d04b942ced24950e4caf788d128847`;
- result: `l10_3dstreetview_center_target_lock_result_v1.json`, SHA-256
  `12c2d5cb7b26720b72c6d8c6354914c9693064e77e925482de60a3b95e6e24ee`.

This is real-image, same-provider, archive-prefix Development evidence for
locking a provider-verified physical point under scale/view change. The test
truth does not say that the point is a door or entrance, belongs to a named
venue, is publicly accessible or traversable, supplies an approach waypoint,
proves arrival or `HANDOFF_READY`, generalizes across providers, or improves
user benefit or mobility safety.

## SceneNN and 3RScan truth-proposed door transfer

The first transfer kept the 3D Street View model, centered scales, and
`0.25/0.75` score unchanged and reused the three already consumed SceneNN door
episodes. Evaluator-authoritative mesh labels and poses supplied six square-
padded door proposals. All six doors sat at image edges and the proposals were
severe partial slivers. Global cosine reached AUROC/AP/Top-1
`0.500000/0.577778/2-of-6`; the center-scale score reached
`0.444444/0.555556/2-of-6`, with minimum margin `-0.221899`. Record
`L10_SCENENN_CENTER_TARGET_DOOR_RETRIEVAL_TRANSFER_DEVELOPMENT_GATE_NOT_MET`.
Freeze the crops: they falsify transfer under partial visibility, not general
cross-provider appearance matching.

A new 3RScan protocol fixed the resulting information contract before RGB. It
required `98%` of target vertices inside the source image, at least `2,000 px`
projected area, at least `32 px` image margin, and depth-consistent visibility.
The first diversity rule—one target per reference scan—found only two eligible
targets and opened zero RGB. V2 changed only that pre-RGB rule to at most two
distinct physical targets per reference scan. It froze three doors across two
reference scan families after reading `8,185` pose and depth members and zero RGB
members. The six selected source-frame margins were at least `36.267 px`.

On the single frozen replay, global cosine reached pair AUROC/AP/Top-1
`0.944444/0.916667/5-of-6`; the center-scale score regressed to
`0.833333/0.833333/4-of-6`. Both errors were a collision between two visually
similar doors from the same scan family. Record
`L10_3RSCAN_CENTER_TARGET_DOOR_RETRIEVAL_TRANSFER_DEVELOPMENT_GATE_NOT_MET`.
This shows that local target emphasis alone cannot resolve a finite sibling-door
roster.

The sole structural successor is parameter-free: construct the complete 3x3
reference/query score matrix and maximize total score under a one-to-one
constraint with `linear_sum_assignment`. On the consumed matrix it changed the
center-scale result from independent `4/6` to an equivalent bidirectional `6/6`;
the selected complete assignment beat second-best by `0.246219`. This is
explicit posthoc mechanism Development.

Before confirmation, a first target-triple-disjoint freeze was rejected without
RGB because one row reused the same physical reference target through another
rescan. V2 strengthened exclusion to every prior physical
`(reference_scan_id,target_instance_id)`. It then froze doorframes 36, 37, and
28, with minimum source-frame margin `32.705 px`, after `8,933` pose/depth and
zero RGB members. On their single replay, both global and center-scale pair
AUROC/AP were `1.0/1.0`; independent retrieval was `6/6`; the center-scale
one-to-one assignment was `3/3` (bidirectional equivalent `6/6`) and its complete-
assignment margin was `0.419960`. Record
`L10_3RSCAN_ROSTER_ASSIGNMENT_TARGET_DISJOINT_DEVELOPMENT_GATE_MET`.

Full-frame visual audit showed that the raw 3RScan camera frames carry roughly
90-degree roll: the apparently horizontal square-padded proposals correspond to
complete doors/doorframes when viewed in scene orientation, rather than isolated
door lintels. The audit does not add semantic authority beyond provider labels.

The closed roster did not answer whether a candidate should remain unmatched.
An open-roster protocol therefore froze four new physical targets, two views per
target per side, and four scenarios before RGB: closed, query-extra, reference-
extra, and a balanced missing-plus-extra swap. The unchanged appearance score,
symmetric multiview bottleneck, and strict reciprocal row/column maxima retained
only `4/12` true matches, emitted `5` false matches, missed `8`, and reached F1
`0.380952`; the balanced swap produced `0` true and `2` false matches. Record
`L10_3RSCAN_OPEN_ROSTER_ZERO_ASSIGNMENT_TARGET_DISJOINT_DEVELOPMENT_GATE_NOT_MET`.
Relative appearance rank is now a frozen open-roster `NEGATIVE_CONTROL`, not an
existence test.

The next protocol selected four further physical targets from one registered
reference/rescan family before reading RGB or depth. It applied the official
rigid transform and scored each pair by symmetric median nearest-surface
distance. Complete surface assignment and reciprocal surface ranking both
recovered all `12/12` true correspondences, but each emitted one adjacent
doorframe-to-door false match in the balanced swap (`TP=12, FP=1, FN=0`, F1
`0.96`; exact unmatched sets `3/4`). Record
`L10_3RSCAN_REGISTERED_SURFACE_ZERO_ASSIGNMENT_TARGET_DISJOINT_DEVELOPMENT_GATE_NOT_MET`.
Registered surface distance is a strong `COMPONENT`, but relative mutual rank
alone is not absence authority.

A third protocol froze four more physical targets from the same scan family and
predeclared the first two as registration witnesses before computing distances.
Their symmetric surface residuals were `0.038178 m` and `0.038756 m`; the maximum,
with no multiplier, slack, or selected quantile, became the absolute rejector
ceiling. On the two evaluated targets, the unchanged registered-surface score,
strict reciprocity, and witness ceiling reached `TP=12, FP=0, FN=0`, F1 `1.0`,
and exact unmatched sets `4/4`, versus complete-assignment `TP=12, FP=1, FN=0`,
F1 `0.96`. Record
`L10_3RSCAN_WITNESS_CALIBRATED_ZERO_ASSIGNMENT_TARGET_DISJOINT_DEVELOPMENT_GATE_MET`.
Rank-only surface matching also reached F1 `1.0` on this fresh cohort, however,
so the witness is a `CHALLENGER`; its incremental value is not established here.

A severe incremental protocol next excluded every consumed reference scan family
and physical target, then froze the first available six same-class doors from
`6bde60c0... -> 6bde60c4...`. Two witnesses and four evaluated targets were
declared before distances; one closed, two unequal-size, and all twelve ordered
balanced swaps supplied `64` truth correspondences and `26` unmatched nodes.
Rank-only surface matching reached `TP=64, FP=2, FN=0`, F1 `0.984615`, with
exact unmatched sets in `13/15` scenarios. The unchanged two-witness ceiling was
`0.376774 m`: it removed both false matches but reduced true matches to `48`,
introduced `16` misses, lowered F1 to `0.857143`, and made only `2/15`
unmatched sets exact. Record
`L10_3RSCAN_SCAN_FAMILY_DISJOINT_WITNESS_INCREMENTAL_DEVELOPMENT_GATE_NOT_MET`.
The raw maximum of two witness residuals is `DEAD_FOR_THIS_ROLE` as a scan-
family-general absolute rejector; no slack, multiplier, quantile, or witness
reselection may rescue the consumed result.

Exa review of partial-registration primary work consistently separated overlap
support from global distance or correspondence rank. The next protocol therefore
changed information rather than threshold: for each reference target it reused
the deterministic upright portal frame, projected the reference and transformed
query surfaces into horizontal-by-vertical coordinates, formed convex hulls, and
required strictly positive intersection area. No IoU magnitude threshold exists.
It froze the first six still-unconsumed same-class `doorframe` targets from the
same family before surface scores or overlaps and evaluated one closed, two
unequal-size, and all thirty ordered balanced swaps (`136` truth correspondences,
`62` unmatched nodes).

Complete assignment reached `TP=102, FP=64, FN=34`, F1 `0.675497`. Strict
rank-only zero assignment reached `TP=136, FP=4, FN=0`, F1 `0.985507`, and
exact unmatched sets in `29/33` scenarios. Positive registered extent support
retained all `136` true matches, removed all `4` false matches, reached F1
`1.0`, and recovered exact unmatched sets in all `33/33` scenarios. Record
`L10_3RSCAN_REGISTERED_EXTENT_SUPPORT_ZERO_ASSIGNMENT_TARGET_DISJOINT_DEVELOPMENT_GATE_MET`.
The support predicate is `RETAINED_CORE` for this privileged registered-
geometry partial-roster responsibility.

For scan-family confirmation, the full metadata inventory was filtered before
geometry access. The first eligible unconsumed family was
`422885e5... -> 422885e7...`, with four stable `door` identities. Only its two
`semseg.v2.json` and two `labels.instances.annotated.v2.ply` files
(`2.85 MiB` total) were materialized through the previously authorized official
downloader; no RGB, depth, or sequence archive was downloaded. Before any surface
score or overlap, the protocol froze the four targets, one closed, two unequal-
size, and all twelve ordered balanced swaps (`34` truth correspondences and
`26` unmatched nodes).

With every algorithmic surface unchanged, complete assignment reached
`TP=34, FP=12, FN=0`, F1 `0.85`. Rank-only zero assignment reached
`TP=34, FP=2, FN=0`, F1 `0.971429`, and exact unmatched sets in `13/15`
scenarios. Positive registered extent support retained all `34` true matches,
removed both false matches, reached F1 `1.0`, and made all `15/15` unmatched
sets exact. Record
`L10_3RSCAN_EXTENT_SUPPORT_SCAN_FAMILY_DISJOINT_DEVELOPMENT_GATE_MET`.

### SceneNN observed-surface carrier and partial-view veto boundary

The next source admission selected SceneNN scene 096 before RGB-D access: four
provider-labelled doors, eight fixed reference/query anchors, one closed, two
unequal-size, and twelve ordered balanced-swap scenarios (`34` truth
correspondences, `26` unmatched nodes). All `8/8` synchronized RGB-D members
were extracted and hash-sealed. The frozen single-frame carrier retained only
`159` and `120` depth-consistent points for SO01/SO04 query against the unchanged
256-point minimum. No surface score or overlap matrix was computed. Record
`L10_SCENENN_SINGLE_FRAME_OBSERVED_EXTENT_SUPPORT_PROVIDER_DISJOINT_NOT_EVALUABLE_SOURCE_SUPPORT`;
this is missing carrier support, not an algorithm negative.

Exa review of registered RGB-D reconstruction motivated one structural repair:
fuse exact trajectory offsets `[-5,0,+5]` in the provider world frame while
leaving 0.08 m depth consistency, the 256-point fused minimum, 4,096 caps,
surface score, reciprocity, and support unchanged. A transparent transport
repair allowed an individual frozen neighbor to contribute zero points, because
the protocol constrains only the fused observation. It did not refreeze the
cohort or open another RGB-D member. Every fused role then supplied at least
`931` points. Rank-only reached `TP=34, FP=3, FN=0`, F1 `0.957746`, with
`12/15` exact scenarios. Positive convex-hull intersection reached only
`TP=25, FP=2, FN=9`, F1 `0.819672`, `6/15` exact; true-pair IoUs were
`0.285715, 0, 0.197744, 0.263963`. Record
`L10_SCENENN_TEMPORAL_FUSED_EXTENT_SUPPORT_POSTHOC_DEVELOPMENT_GATE_NOT_MET`.
The three-frame carrier repaired evaluability, but the partial-view
convex-hull veto is `DEAD_FOR_THIS_ROLE`.

Two further candidates were frozen from primary colored-registration and
instance-retrieval evidence, without fitting a threshold or weight. Saturation-
weighted Hue reciprocal agreement removed all three geometry false positives
but retained only two true matches: consensus `TP=2, FP=0, FN=32`, F1
`0.111111`, `0/15` exact. Frozen local DINOv2-S foreground feature averaging
used the exact three visible-target masks per role, continuous patch-mask
weights, and max cosine across all `3x3` view pairs. It reached appearance-only
`TP=7, FP=21, FN=27`; geometry consensus was `TP=7, FP=1, FN=27`, F1
`0.333333`, `0/15` exact. Record
`L10_SCENENN_TEMPORAL_HUE_GEOMETRY_CONSENSUS_POSTHOC_DEVELOPMENT_GATE_NOT_MET`
and
`L10_SCENENN_TEMPORAL_DINOV2_FFA_GEOMETRY_CONSENSUS_POSTHOC_DEVELOPMENT_GATE_NOT_MET`.
Both generic appearance veto roles are closed on this consumed scene.

A distinct source admission then selected the four smallest still-unconsumed
one-door SceneNN recordings (`021`, `273`, `011`, `279`) before RGB or model
access. Geometry froze eight observations and the same fifteen open-roster
scenarios. Fixed EfficientLoFTR used the truth-visible target box expanded by
`1.25`, score threshold `0.2`, at least eight matches, six-pixel homography
RANSAC, and at least six inliers; unsupported pairs became NONE. Same-crop
DINOv2 reciprocal assignment reached `TP=6, FP=24, FN=28`, F1 `0.1875`, and
`1/15` exact scenarios. EfficientLoFTR improved to `TP=11, FP=18, FN=23`, F1
`0.349206`, but supported only LF01/LF03 true diagonals and reached `0/15`
exact scenarios. Record
`L10_SCENENN_EFFICIENTLOFTR_FRESH_NONE_DEVELOPMENT_GATE_NOT_MET`. The frozen
homography-inlier hard gate is `DEAD_FOR_THIS_ROLE`; its positive F1 delta is
retained only as evidence that local correspondence is more useful than global
patch averaging. Exa review of RoMa and DKM admits a separately fixed dense-
certainty and balanced-sampling successor on fresh targets. No consumed crop,
threshold, or RANSAC parameter may be tuned.

That successor used the official `romatch 0.1.2` wheel, RoMa indoor and DINOv2-
L/14 weights, symmetric `560 -> 864` dense warps, the official `0.05`
certainty, six-pixel cycle consistency, at least one-percent directional cycle
coverage, and majority cycle purity. Four further SceneNN one-door scenes
(`700`, `276`, `234`, `237`) were selected by XML labels and ONI byte order
before geometry, RGB, or model access. Geometry then froze the minimum
qualifying `0.30 m` pair in each scene. Same-crop DINOv2 reciprocal rank reached
`TP=9, FP=23, FN=25`, F1 `0.268657`, and `4/15` exact scenarios. RoMa absolute
support reached `TP=9, FP=0, FN=25`, precision `1.0`, F1 `0.418605`, but only
RA02 had a supported true diagonal and exact assignment remained `0/15`.
Record `L10_SCENENN_ROMA_ACTIVE_NONE_DEVELOPMENT_GATE_NOT_MET`. Retain the
zero-false-positive dense-cycle branch as
`COMPONENT_OR_CHALLENGER / COMPONENT`, not standalone full-recall authority.

The failed pairs exposed an action-policy defect independent of matcher
thresholds: minimum camera baseline could select a temporally distant revisit.
A consumed posthoc protocol therefore changed only pair ranking to minimum
qualifying frame gap before baseline. Original gaps `105, 90, 185, 525` became
`20, 25, 35, 20` frames while displacement remained `0.30-0.37 m`; scenes,
targets, crop, model, certainty, cycle, support, assignment, scenarios, and gate
were unchanged. DINOv2 rose to `TP=34, FP=2, FN=0`, F1 `0.971429`, and `13/15`
exact. RoMa rose to `TP=27, FP=0, FN=7`, precision `1.0`, F1 `0.885246`, and
`8/15` exact, with TL01-TL03 supported and TL04 still unsupported. Record
`L10_SCENENN_ROMA_TEMPORAL_LOCAL_POSTHOC_DEVELOPMENT_GATE_NOT_MET`. The large
single-change effect retains the temporally local active-observation policy as
a Development component, but the consumed replay cannot regain fresh authority.
The next valid check freezes this action policy unchanged on new physical
targets; fitting a DINOv2 margin, RoMa purity, cycle threshold, or fusion on
TL01-TL04 is forbidden.

That confirmation admitted the next four unconsumed one-door SceneNN recordings
by XML labels and official ONI byte order (`043`, `082`, `213`, `207`) before
opening their mesh, trajectory, ONI, RGB, visibility, or model scores. The
unchanged temporal-local action rule selected gaps `20, 25, 50, 35` at
`0.30-0.34 m`, so the action repair generalized. On the new fifteen-scenario
roster, same-crop DINOv2 reached `TP=25, FP=4, FN=9`, precision `0.862069`,
recall `0.735294`, F1 `0.793651`, and `5/15` exact. Frozen RoMa cycle support
again removed every false positive and retained precision `1.0`, but supported
only TF01 and TF04: `TP=16, FP=0, FN=18`, recall `0.470588`, F1 `0.64`, and
`2/15` exact. TF02 had abundant cycle coverage but only `0.35-0.39` cycle
purity; the edge-clipped TF03 view had approximately one-percent cycle support.
Record `L10_SCENENN_ROMA_TEMPORAL_LOCAL_FRESH_DEVELOPMENT_GATE_NOT_MET`. The
posthoc F1 `0.885246` does not confirm cross-scene. Preserve temporal-local
selection and RoMa as components; use a consumed-cohort geometric
viewpoint-normalization or multi-frame carrier before spending another fresh
cohort. Threshold, purity, crop, scenario, or fusion fitting on TF01-TF04 is
forbidden.

Two frozen consumed-cohort repairs then isolated what did not work. Exact
target-mesh PCA plane rectification changed no assignment metric: RoMa remained
`TP=16, FP=0, FN=18`, F1 `0.64`, `2/15` exact, and `2/4` supported; TF04 purity
rose to about `0.90`, but TF02 remained about `0.34` and TF03 produced zero
cycles. One pre-RGB geometry-selected midpoint per episode also changed no
metric under a strict both-edge support requirement. Record
`L10_SCENENN_ROMA_PLANAR_RECTIFIED_POSTHOC_DEVELOPMENT_GATE_NOT_MET` and
`L10_SCENENN_ROMA_MIDPOINT_BRIDGE_POSTHOC_DEVELOPMENT_GATE_NOT_MET`; both
standalone recall-repair roles are closed.

An Exa review of OAMatcher and O-MaMa instead motivated one spatial-domain
change: preserve each complete `640x480` image for RoMa context, then count a
cycle only when it starts inside the source provider target-visible mask and
lands inside the paired target-visible mask. Nearest-neighbour mask resize,
source-mask denominators, RoMa weights/resolutions, certainty `0.05`, normalized
six-pixel cycle error, bilateral `0.01` cycle fraction, `0.5` purity, scenarios,
and strict reciprocal assignment were frozen unchanged. On the consumed fresh
cohort this recovered all four diagonals and reached `TP=34, FP=0, FN=0`,
precision/recall/F1 `1.0`, and `15/15` exact scenarios, improving the cropped
predecessor by `+18 TP`, `-18 FN`, `+0.36 F1`, and `+13` exact scenarios with no
false-positive cost. Record
`L10_SCENENN_ROMA_FULL_CONTEXT_MASK_POSTHOC_DEVELOPMENT_GATE_MET`. This is a
`COMPONENT_OR_CHALLENGER / CHALLENGER`, not confirmation: the next valid run
must freeze the identical full-context/mask-gated rule on newly admitted scenes.
Provider masks and complete SceneNN geometry remain privileged and establish no
raw-phone proposal, portal ownership, access, waypoint, arrival, or handoff.

The unchanged confirmation then selected the final four unconsumed SceneNN
recordings containing exactly one `^door[0-9]*$` label (`074`, `109`, `005`,
`076`) by official ONI size before opening selected geometry, RGB-D, visibility,
or any model score. Frozen temporal-local selection produced `20-60` frame gaps
at `0.30-0.32 m`. Across the same fifteen closed, extra, missing, and ordered
swap scenarios, full-context mask-gated RoMa again reached `TP=34, FP=0, FN=0`,
precision/recall/F1 `1.0`, `15/15` exact NONE-aware assignments, and `4/4`
true pairs with absolute support. Same-crop DINOv2 on these scenes was
`25/4/9`, F1 `0.793651`, and `5/15` exact. Record
`L10_SCENENN_ROMA_FULL_CONTEXT_MASK_FRESH_CONFIRMATION_GATE_MET`; retain the
unchanged correspondence mechanism as
`COMPONENT_OR_CHALLENGER / COMPONENT` within same-provider Development.

The first proposal-carrier bridge then changed only the mask source. On all
eight consumed confirmation frames, one frozen GroundingDINO-Tiny call with the
single prompt `door`, strict `0.4/0.3` box/text thresholds, and deterministic
top-one selection supplied a box without truth access. Frozen SAM2.1-Hiera-Small
converted that box to one native full-frame mask with no IoU-score selection or
morphology. Proposal truth IoU ranged from `0.60` to `0.95`, yet unchanged
full-context RoMa retained `TP=34, FP=0, FN=0`, precision/recall/F1 `1.0`,
`15/15` exact NONE-aware assignments, and `4/4` true support. Record
`L10_SCENENN_ROMA_GROUNDED_SAM_PROPOSAL_POSTHOC_DEVELOPMENT_GATE_MET`.

This is the first direct evidence that exact provider mask boundaries are not
required by the correspondence mechanism, but it remains
`COMPONENT_OR_CHALLENGER / CHALLENGER`: the source is consumed, every scene has
exactly one labelled door, and GroundingDINO/SAM2.1 ran on desktop GPU. Freeze
the prompt, thresholds, top-one rule, SAM postprocess, RoMa thresholds, and all
eight frames. The next test must add source-disjoint multi-door target selection
or provider-disjoint proposals; EdgeSAM-style deployment is a separate runtime
question. None of these results proves raw-phone deployment, aperture, named
entrance, ownership, access, traversability, waypoint, arrival,
`HANDOFF_READY`, user benefit, reliability, or safety.

The frozen multi-door/provider-distinct 3RScan transfer then exposed the
carrier's actual boundary before correspondence. The cohort selected three
physical targets across two scan families and six full `960x540` frames, each
containing `11-17` provider-labelled doors. On the first required reference
frame, the unchanged single prompt `door` at box/text thresholds `0.4/0.3`
retained zero GroundingDINO boxes despite the evaluation-only provider target
box. Execution therefore stopped after one GroundingDINO call and before any
SAM2 or RoMa call. Record
`L10_3RSCAN_ROMA_GROUNDED_SAM_PROPOSAL_POSTHOC_SOURCE_NOT_EVALUABLE`. This is
not a negative for SAM2, RoMa, identity assignment, or NONE handling; it is a
frozen negative control for high-confidence category-prompt reachability on
this consumed source. Do not lower thresholds or alter the text prompt.

An Exa review of cycle-consistent cross-view mask prediction, Predictive Cycle
Consistency, and MESA/DMESA instead motivates a structural successor: use an
already-bound reference mask to constrain dense correspondence, retain only
bidirectionally cycle-consistent points, and derive the target SAM2 prompt from
that target-conditioned support. This transfers object information instead of
asking a category detector to choose among many doors. Target provider boxes
remain evaluation-only. The literature supports the representation choice but
does not validate BlindAssist or this implementation:
[CCMP](https://arxiv.org/html/2602.18996v2),
[Predictive Cycle Consistency](https://openaccess.thecvf.com/content/CVPR2025/html/Baade_Self-Supervised_Cross-View_Correspondence_with_Predictive_Cycle_Consistency_CVPR_2025_paper.html),
and [MESA/DMESA](https://arxiv.org/html/2408.00279).

The first frozen reference-conditioned implementation rasterized the privileged
initial reference bbox, kept the largest eight-connected set of high-certainty
bidirectional RoMa cycles, fitted one ordinary least-squares affine map, and
projected the reference extent into the query as the sole SAM2 box. Query truth
opened only for evaluation. All three doors were localized: prompt IoU was
`0.638-0.838` and native query-mask bbox IoU was `0.584-0.916`. The full gate
remained unmet at `2/3` bilateral supports because DR01 forward cycle purity was
`0.494961`, just below the unchanged `0.5` gate. No threshold was changed.

One structural ablation replaced the full reference rectangle with one native
SAM2 reference mask. That restored absolute support to `3/3` and raised DR01
forward purity to `0.690015`, but the same mask covered only `0.352211` IoU of
the bound DR01 rectangle, reducing minimum prompt IoU to `0.309975` and query-
mask bbox IoU to `0.487915`. This showed that semantic identity support and
complete object extent were different responsibilities.

The frozen dual-surface successor therefore used the native reference SAM2 mask
only to select correspondence support and score bilateral cycles, while the
original already-bound rectangle supplied only the four complete extent corners
projected by the same affine map. It passed every frozen gate: prompt IoU
`0.637-0.837`, query-mask bbox IoU `0.582-0.909`, and unchanged bilateral
absolute support `3/3`. DR01 forward purity was `0.634361`; no model, certainty,
cycle, component, SAM postprocess, frame, target, or gate changed. Record
`L10_3RSCAN_ROMA_CYCLE_PROMPT_DUAL_SURFACE_POSTHOC_DEVELOPMENT_GATE_MET` and
retain it as `COMPONENT_OR_CHALLENGER / CHALLENGER`.

The first pre-RGB physical-target-disjoint cohort selected instances `36/37/28`
after excluding every physical target in five consumed cohorts. Transported
extent IoU was `0.767-0.920` and bilateral support passed `3/3`, but DR02 native
query SAM undersegmentation produced bbox IoU `0.371`. This failed the frozen
single-mask geometric-output gate while preserving target selection. A zero-
model-call four-surface posthoc split assigned reference/query SAM masks only to
identity support and reference/transported boxes only to geometric extent.

The four-surface rule then ran unchanged on a second pre-RGB cohort containing
new instances `35/15/4`. All extents reached IoU `0.738-0.793`; all query SAM
support bboxes reached `0.724-0.870`. Legacy global high-certainty purity still
supported only `2/3`: DR03 was `0.473/0.422` despite a `17,624`-pixel coherent
cycle component and low affine residual. This froze whole-mask global purity as
a negative control rather than lowering its `0.5` gate.

The structural successor made support spatial: keep the unchanged `0.01`
reference cycle opportunity and require the largest 8-connected component to
contain a majority (`>=0.5`) of all valid bidirectional cycles. On the opened
cohort it reached `3/3`, with minimum cycle fraction `0.242` and dominance
`0.653`, versus legacy `2/3`. The unchanged rule then ran on a third pre-RGB
physical-target-disjoint cohort containing instances `18/34/12`. It passed:
extent IoU `0.532-0.851`, query-support bbox IoU `0.542-0.915`, minimum cycle
fraction `0.508`, minimum component dominance `0.825`, and coherent support
`3/3`; legacy global purity again reached only `2/3`. Record
`L10_3RSCAN_CYCLE_COMPONENT_SUPPORT_PHYSICAL_TARGET_DISJOINT_CONFIRMATION_GATE_MET`
and retain the four-surface/coherent-cycle rule as
`COMPONENT_OR_CHALLENGER / COMPONENT` for same-provider Development.

Across the three successive target-disjoint panels, nine new physical targets
were selected before RGB/model access. A frozen consumed-cohort open-set test
then retained the third panel's three positives and crossed scene families to
construct four queries where the exact physical reference target was absent,
although other doors could remain visible. The unchanged `0.01` cycle-
opportunity and `0.5` dominant-component gates retained `3/3` positives and
rejected `4/4` negatives with zero false commits. Positive cycle opportunity
was `0.507526-0.644900`; negatives were `0`, `0.002674`, `0`, and `0.002821`.
The latter two nonzero negatives could still have component dominance `0.604`
and `0.528`, showing why cycle opportunity and spatial dominance play distinct
roles. Record
`L10_3RSCAN_CYCLE_COMPONENT_OPEN_SET_POSTHOC_DEVELOPMENT_GATE_MET`.

The first execution stopped before adjudication because a zero-cycle negative
raised `NO_REFERENCE_CYCLES`. Its failure receipt is retained. The only repair
mapped zero cycles to the already frozen non-commit decision; models, pairs,
thresholds and gates remained unchanged. This result is consumed posthoc,
cross-scene exact-target-absence Development evidence. It is not a same-scene
sibling-door test. Scan families/backgrounds still overlap for the positives,
and initial reference boxes remain privileged. This does not establish raw-
camera initial referent discovery, provider-independent generalization, named
entrance, ownership, public access, traversability, waypoint, arrival,
`HANDOFF_READY`, Android deployment, user benefit, reliability, or safety.
Stop tuning positive 3RScan targets; the next claim-changing source must address
initial binding or frozen same-scene sibling-door rejection.

The consumed sibling panel next combined the retained local bilateral binding
with either target-excluded global epipolar majority or bidirectional active-
query majority. This complementary rule preserved `3/3` positives and rejected
both siblings. Its first physical-target-disjoint source was not evaluable:
FC30 had no active query above `0.591093` target completeness versus the frozen
`0.98` gate. A successor froze FC31 and FC08 as positives and kept FC30 only as
an exact-target-absent bound before opening RGB or models. Both positives passed
the local bilateral and primary/active extent gates. They were nevertheless
rejected because global epipolar support was only `0.415395/0.362869`, while
primary-to-active/reverse paired coverage was `0.297609/0.435110` and
`0.287409/0.183586`, all below the inherited `0.5` majority. The FC30 negative
had zero primary and active local cycles and was correctly rejected. Record
`L10_3RSCAN_COMPLEMENTARY_CORROBORATION_PARTIAL_PHYSICAL_TARGET_CONFIRMATION_GATE_NOT_MET`:
the specialist corroborators did not transfer as a universal hard gate.

Following the NeurIPS 2023 cascade-deferral analysis found through Exa, one
zero-model Development successor made specialist verification conditional. An
inherited local non-commit rejects immediately; a local commit with at least
one-quarter primary coherent-cycle coverage exits directly; only a lower-
coverage local commit requests the unchanged complementary corroborator. The
quarter-coverage landmark was selected once after seeing the failed panel, with
no sweep. Recomposition over the consumed sibling and physical-target panels
restored `5/5` positives, retained `0/3` false commits and committed precision
`1.0`. Only the `0.087404` DR01-to-DR02 ambiguity requested corroboration, so
the decision graph requested the extra branch for `1/8` rows versus `8/8`
under universal verification (`87.5%` counterfactual branch avoidance). Record
`L10_3RSCAN_SELECTIVE_CORROBORATION_CASCADE_POSTHOC_DEVELOPMENT_GATE_MET`.
This is decision-logic and counterfactual-demand evidence on consumed rows, not
measured runtime savings or confirmation. Freeze `0.25`; the next admissible
check is a pre-model source with positive and exact-target-absent rows. Same-
provider correspondence still does not establish raw-camera initial binding,
named entrance, ownership, access, traversability, waypoint, arrival, handoff,
deployment, user benefit, reliability, or safety.

The first frozen cascade challenge then selected never-consumed target SC34
from `1c211546 -> 1c211548`, a scan family not used by the coherent-cycle
mechanism. Both reference and query target geometry were fully in-frame before
RGB/model access; two new negative pairings reused FC31/FC08 query pixels only
as different-scene controls for the new SC34 referent. SC34 produced strong
identity-cycle statistics (`0.442978` opportunity and `0.981006` component
dominance), while both negatives produced zero cycles. It nevertheless failed
the unchanged local commit because affine extent IoU was `0.065238`; the prompt
and SAM query mask locked a different repeated structure. The extent gate was
therefore decision-critical and correctly blocked the frozen `0.25` direct
exit. Record
`L10_3RSCAN_SELECTIVE_CORROBORATION_CASCADE_SCAN_FAMILY_CHALLENGE_GATE_NOT_MET`.

One consumed posthoc geometry intervention replaced only the affine fit with an
OpenCV USAC-MAGSAC homography over the same largest coherent-cycle component.
It reused the frozen six-pixel cycle-error scale as its sole reprojection scale,
required majority inliers, and performed no threshold sweep. The fit itself was
strong (`0.942384` inliers, `1.452 px` mean and `5.997 px` maximum inlier
residual), but target extent IoU improved only `0.065238 -> 0.081832`. Record
`L10_3RSCAN_PROJECTIVE_EXTENT_POSTHOC_DEVELOPMENT_GATE_NOT_MET` and freeze both
affine and projective geometry as insufficient for this repeated-structure
failure. Following that attribution, the official downloader added the
`422885e5/422885e7` sequence archives (about `137 MB`) to `artifacts.local`.
The unchanged strict pre-RGB selector admitted `0/7` available door/doorframe
candidates, so no model calls were spent on that family. The next admissible
change is observation reachability or an independent instance-bearing cue, not
another geometric estimator, local threshold, or reinterpretation of strong
cycles as identity truth.

Evidence:

- partial physical-target confirmation implementation/protocol/result SHA-256:
  `280a6641455c6da6d184ee4422243370ab2945130bfed03bc9621de1880e418d`,
  `3680014ca75676e9f7216fcfbd62fc0c48d275541b41f8fd4485e47ca9ecf47c`,
  and `ebd787a6963ccfc65b3ce0369789304caa8fcb4ef3bb0c3150c5f31ba82ddfac`;
- selective-cascade implementation/protocol/result SHA-256:
  `57f07210286a61af35a0f657fa6a1a7b152d1bfb833472ee5ea45842849882aa`,
  `4205875135bc8fdf847c76d9831f26f217f286750422422743557ce247897e0c`,
  and `9c835d51daaafaf6b7eb4517bec255a42b914305e6dde3119302916b512ecc31`;
- SC34 freeze implementation/protocol/cohort SHA-256:
  `47c25519663d2d216fc755f3de722c267bf0b18d2158b16fabd97d17a6a46302`,
  `c763bb2d7ff5fca4c3f6d3306e71bf0474bd4f1a749e4568f0a2e3771dc9354a`,
  and `2675e7724d0ad8207a6211efcc062a5be87094a3120662f472b8bd23f83646e8`;
- SC34 local-carrier protocol and selective confirmation
  implementation/protocol/result SHA-256:
  `41a26f2a4a7bf41920dc30d735630c30d60437df67144e927a4d423ee5c95eda`,
  `84015b72db646ded6158cd38203b4a80dd94bafd500e22bbc9ffe1eed17c41e4`,
  `f832f13ba63e8e1ff0088f04424525b266e44bf3b9cac8011c0787b35643d7b2`,
  and `304d770640f55e5031fa2120223494887838f54bf89430e758207a3af3ffbb30`;
- projective-extent implementation/protocol/result SHA-256:
  `9f8c7b90c7852d7bff01f101894bcafd3917f15c2372d66face6161676235358`,
  `264becf3d7badfb91e71f3998c7fc7cfb92e961c9ef59ac27f8b0ffa29b3e0f2`,
  and `a96a03c0274c3fd952c131a9bdcf978c0a617bf6b5e5ea9e45b4bc36651634f3`;
- added official 3RScan `422885e5/422885e7` sequence SHA-256:
  `b11adb2295d7a8c97c87810dfce68a052058fcc3a7567e07629bd629ad50b6f1`
  and `55817b707056b6cd36d722432aab93ab5c9769bda8ce5fab7dbf8661ef73b2ee`;
- SceneNN transfer result:
  `l10_scenenn_center_target_door_retrieval_result_v1.json`, SHA-256
  `5850b222eec2b1fbd870706ea8580dc973f871c718546771eab6ebeb1e48e282`;
- 3RScan pairwise result:
  `l10_3rscan_center_target_door_retrieval_result_v2.json`, SHA-256
  `6b2a9c2729b64f8c6de052894f6c58a7c7f32dd219d69324757d3b86ae62a4f8`;
- assignment posthoc result:
  `l10_3rscan_roster_assignment_posthoc_result_v1.json`, SHA-256
  `c679db740615e45c61f4a20d035a31d250994f97322b81050f86844e40e1fc79`;
- physically target-disjoint assignment protocol/cohort/result SHA-256:
  `11eacf310a9d2ae310de321be4de152232fc8cdcb71458b04e0f29689895d699`,
  `c5e2c69fa6a0132494302b4a786909dfea5484eb156355c1d0f5d78e49cac72d`,
  and `46025f243e4c108271e0776d801cd3789680738986e2afdce367a842350c149f`;
- confirmation evaluator SHA-256:
  `973a009919a259327eaaf65eaf92721ca6c4d50b44ed244ef407bdbde2ffebd3`;
- 3RScan Grounded-SAM source-reachability protocol/implementation/result
  SHA-256:
  `e25c707d22a8b2b1a297bc79defebb3ae790ed8a05e47a9a0d6d4fa3d77cabae`,
  `b1bd1aa1dfcea82850cc0e5141ce01e4007104ecaea29667036a055bc6eb78e7`,
  and `db07154fa30bc1904b87a723adbd8e91a957039662068ae7f7b88cd4e4f0a0c2`;
- 3RScan rectangle cycle-prompt implementation/protocol/result SHA-256:
  `a417f3f87ebcc45a063dfdbd0378eb705baafce5cf8cb651b6f9f0463aded5af`,
  `22e766bd744341b70f66572ad00ffd053a0125c4307aefc2e9d91255ec33b1d0`,
  and `daeb7185189bf8de8eb318acd62ff3ae9d7fb6f78f8edeb98b2d7bb5deeeb1ac`;
- 3RScan reference-SAM-mask implementation/protocol/result SHA-256:
  `9a8c5dd3bcb7a437ce3ad397d98a81961f045f3dee0db0e838f77bdb17bec0d0`,
  `8f40345fde9206bb236d0a0ea90639019dbc63017f08976e4100e4da538ff1e3`,
  and `52641980e558d8a028777021919dc438d169e27719a30a2082902644b811e5f8`;
- 3RScan dual-surface implementation/protocol/result SHA-256:
  `c2abf48e21a984b5085ccc696f6a9f01f3f0f818f8d7e9bb1d2feb4bbb83b841`,
  `6229bebf245622a9e10e50f4df1dfa6fed3c66ff87f8ded7033e7a41a72f5b31`,
  and `a85344095cc4dc5789ec4bba18055fd24075ed644a2a827589857508a131e740`;
- first target-disjoint source protocol/cohort and dual-surface confirmation
  implementation/protocol/result SHA-256:
  `5452888bbacb384d223c409893b2175f395f07131ec3a0a20fc9e2a7b7e3a5ab`,
  `a99eb59bcf2959e1bbc3b3aaddaa7a7c8cf4f285a3eee5b6f554657989501198`,
  `0fe49882bbdfedaa373ffa1c219f635af27d61abd47c12d3cf0a34399cd68ea8`,
  `664b5e0638ecea9bbc7be70f7f38d843c120facc4eb70f4dbe3cea36459a45c0`,
  and `e6903068ac35f812d9b362ffaf3a12a0205744b5b3a3789267f32a1c224d92c1`;
- four-surface posthoc implementation/protocol/result SHA-256:
  `c6a74499414bccafd36079bd9ddf948834beff912d775461394ac31e04ad9b2f`,
  `a1f1e1c200c847aec9d62ead68953906a03d182e8d1315e81efca8636ce782dd`,
  and `19f5c390c7d252c5908c43613cfc610e5a704195a2bc119459a11eeb041ecc34`;
- second target-disjoint source protocol/cohort and four-surface confirmation
  implementation/protocol/result SHA-256:
  `a0544a49bdd1c8cb399d26aeac732b2e2a784826874707b42d8be6334917f63f`,
  `9aa5addeabea11e7bda7765b20b99fc7f6e44cea0c3fa300f1cc63600a66bd3d`,
  `e3f2040d031b04d6d9420d8d9b1ba21a4d76e0e6c4fd8610f7df18dc31db3072`,
  `0aa217491a11551a2fc41c2ec34990023be67e0660cbb1d4099ab20fddffa5eb`,
  and `01ac4318e70ee5e92e2eb058ad25168d9e11416fc39fdd86bff2eda376bb79e6`;
- coherent-cycle posthoc implementation/protocol/result SHA-256:
  `a04ac3bb438895a209979649661df99a92122bf73fa80fc81fc9b4e13e4e00d0`,
  `c12ab9d319514db87e9135898345eb04767e9b30146e0e2084200a4cb3333528`,
  and `1ac8e33fe87f489140f8519d628e77e4fb737f9239f520099fd2dda1b5d3b840`;
- third target-disjoint source protocol/cohort and coherent-cycle confirmation
  implementation/protocol/result SHA-256:
  `009702146e61f33f6f2e9fd531ede652fca2d36bb6b2b1fcd291f7d1ce24ea06`,
  `ee1f57ff7a6d71e0fcba0b27ccea86a1b5af11a100b8c1ee5fc1e83e22a7aae1`,
  `9716169ba05b6457e9ef4a966b384d8d48917a30c292959c942a479171a0cc3a`,
  `7407631cfc9f5354e9614a8dc9a7357638ffba55409f4ae34c012ac7de901c90`,
  and `b8d8a7b47d08f5998afc3bdca9b03bac74ee39eb5c65a04d63e37619d61a06d0`;
- coherent-cycle open-set implementation/protocol/result and initial execution-
  failure receipt SHA-256:
  `812f5bf5372389aeab462a6e8339e9aae78629dce7187ab74bc32e8b6f53b4c4`,
  `d13e625f16a3e492343565b43c9ca74602924e255bb9f883a2b26e4a82773ec8`,
  `22a423e589af3f120b2d9ff4d84921fbf274f27d00aa74e4d9710f4d4e21ed2e`,
  and `1184126b73c44bee32ec006780b80c41a9f92815c30557b73e5282d8096d7b60`;
- open-roster appearance protocol/cohort/result/evaluator SHA-256:
  `7813213a2dee008f868131a6772cd1e85d047f64e02ed8be42b2cbb0bcc5cbaf`,
  `4a96770b7939d4f32cf3ab84bb235dbd7d4082a41f224b457cb6f9e51a9044e0`,
  `0980fe447bbd6552d830d8bad5df65edc7d9ec36dda2aa00b03e01e3c0c82328`,
  and `a194b86ec4b495b44fd487eaebd875d4da20868468882a2fc4b476ed7650cf4c`;
- registered-surface protocol/cohort/result/evaluator SHA-256:
  `5b48458676b0c3a98576fa58c0d84ff8b1ab243b56d04dcf04522632fedd4fd6`,
  `a4dcfc909f13562f38c90b1b4f94e08d3f1007531d97d3f959f0c2a570597e84`,
  `fbdf4c1bca054a2591e1975c5e42705d05ffc7e5cf05b92950a3358084b5fe6d`,
  and `a69b20c3c6f1d73ba0458e7ea183bcc5189104a529f7512f4b3e4543134e9a6b`;
- witness-calibrated protocol/cohort/result/evaluator SHA-256:
  `bc6a898fdc0d24994f0f1e22c50dd92082ab74b550fd655f767c61ad433c12fb`,
  `99b51776038a091026a793dce62e00e5c55226a04d4706a535336d4b9a43741d`,
  `38ef5051cd93ce04f347837831142aeb1ebb1eb8f9761ca28f6f765bb666876c`,
  and `7e2a448b286aab43da9442f8fc9c13d9ceac6d4411d5532da093f6447d64bd79`;
- scan-family-disjoint witness protocol/cohort/result/evaluator SHA-256:
  `78e3989dac20d98c340318d01f5470f85a3c7be6a127922a7b6467ba7743576f`,
  `051669620123c9fc30f4cd58828961908546b0e6ea29e914173b712d657064cd`,
  `7c25f8f08a41ba9f72aef26fa262bc79a558b5de756a3763639259baa9383a0e`,
  and `109d36814509f916cfba3bccedecd8ff7fff6bcb916c9a26cb9d8ca41b144c0d`;
- registered-extent-support protocol/cohort/result/evaluator SHA-256:
  `55784575cf6a3e7365ceaba10a3bdcc6154555705bc860b5823570198322595f`,
  `e1aa3cc56beab733dc636cf63cd8291de4fb00645289fd5a9014eb5d13b32bd7`,
  `6e4c3b3bbff823af097106b7915b08bf88b92c3f0f4534a9c5886b34a0beb37b`,
  and `a03f3903dabad275f784683fa09e31fbf1133e3caed367a3a325fd750518ce2b`;
- scan-family confirmation protocol/cohort/result/evaluator SHA-256:
  `cb71bd1ab0abc7e7a689b64ec51f96c537064899a0c99b96dd6694afdf601a8a`,
  `af364a3a8f68110e0be1d2c5a198182dd712da1e0f74cebd9dde98e04396dbf3`,
  `01720cd11f5c387725255eef15ea796e3fc2cca2f580f450c0b67d0d005f6f7e`,
  and `e4ae41608a6171b857e11dbc90ed46ddffc0aed0ef321a2769e9abce97920a4b`;
- SceneNN single-frame observed-surface protocol/cohort/receipt/result SHA-256:
  `deb0be3fe25ff5384b0344c3accfbab3074049a18c577e22a376cb5b595a8b7c`,
  `38981c8940927b44202b3afea7be70fcf8e077615f039efb1ad532c18ad7c817`,
  `25eb367897508fff9b060fe82dfe9c5b8d2d168480652ecafe43058c39eb6dc5`,
  and `37405323ebe2c55d7aa8b25a89c31a8b488b4b65789f4b88ecfda726f874894e`;
- SceneNN temporal protocol/cohort/receipt/result SHA-256:
  `496e1893380e9d50aeec2ad9bffcca31a9378f08cad5d088761cb4366b4f7b7b`,
  `b6f41771806454769565a7592c6ce32dbd8f628b55f0432f790ae6e7aa079a1d`,
  `2e1378a037900363d2f14bfd6d5f2ecab9ddfa627473d1212c36f12984f70274`,
  and `0acfa5bd2aa8ff3bb4d4dad75c63e916f0a7064b082a138ae225d62b173bcd27`;
- SceneNN Hue protocol/result SHA-256:
  `e0fe8535b5ec75179926aeda136d91223fd453697bc18e32faada734da3f88dc`
  and `a42c3a913fac3b3b9fab9cf81c0a4f5544dfdbd1cf3b5ab204d15d637aea4c67`;
- SceneNN DINOv2-FFA protocol/result SHA-256:
  `e02f8acbcfe3fb6026b00634b6ac59cabf0444dd8a51890af57b1d8eb9ec19a1`
  and `a66d9b5f45dee1317e28852bd92f676c534abe6d3af4c4b2c9edab97f8fdc8eb`;
- SceneNN EfficientLoFTR fresh protocol/cohort/RGB receipt/result SHA-256:
  `f61d54ce43d05c72258dbe92f4e48efea621002c822cd89f66dc2b4689e16503`,
  `b6952b081d3476db1c5f3d600fb18aa54fd8858bad7c13b02e8ed08860c76dd4`,
  `6cf8384a27bd95e9646bfc18fa3d3a421d03a8a4e8a4ba02216efcd9b357230b`,
  and `4fe0eff3200b538b685554a0373ce52e160b97470a3b59b780e66bb146fa4115`;
- SceneNN RoMa active protocol/cohort/ONI receipt/RGB receipt/result SHA-256:
  `667c5e8e6bdc766f1d1c3f119ef8b065f7594b9d20923d9278f27953e99e6889`,
  `caa1803dff654f09801b6b230e9012841aa0fc7065bfd399549650a38b5401bd`,
  `0f5f08397ffc634f0dbe334c341b25bf4715ffb283b326e104ed8a976f943a14`,
  `6108bd2d75275e44dc3cc932c8c59f978a02567eb6f50b4dc63cd8435d419aa3`,
  and `49acb6e611f8b305bc00003ebca179894fe8ed2195a6973b13acce1e43888e3a`;
- SceneNN temporal-local posthoc protocol/cohort/RGB receipt/result SHA-256:
  `de814db8b09ca487f2717a0b521328fd97c24a6eb61fbe4efdfb463432a77ff5`,
  `3ef8cf33a1cad646c4c76464e11af085c32f2fdd553f50764ce2605035e3b2b1`,
  `58b3bbefa720c0c6b39b7b5c48cd254abd590b20a500a05dabca40e58a54eb59`,
  and `c833ab86a17a35d590508a667ed1fa0000a239169a4e3d589d4e42c2e62a8e7d`;
- SceneNN temporal-local fresh protocol/admission/cohort/ONI receipt/RGB
  receipt/result SHA-256:
  `c740bcae2cd7baeb8eeee2aeb79bde6f680cc25f62e8a1be641fd5b387a49449`,
  `0dfd728e52b2c21e27b3849823afebcabd4b430d97a86d26a795d424ff45ba4e`,
  `836cf4e9b7d79ed73b5c628fd0fb1b442f17c2ec69735a2b719f22acbccb0e79`,
  `7ef564e483d94e06a29e27c67c6b7bf47fa55f60c277164685d9c9594c00590e`,
  `e17a13e2e0a1a3376b3cf0c7d96faa1af1f0a68f6e65ade24a288644ef0e813b`,
  and `d2d049d55ab40d8b660dc212e9bc12092086cb0c5872b736b2070f4750852b8f`;
- SceneNN planar-rectified posthoc protocol/result SHA-256:
  `eb65527dd7777397916baf57a42303900fffcde0acd4e56132d2c36bb783a7ca`
  and `eb8824847e07d8edb93fe97c373e665e57db25510f7e4e7e99b8479762f45429`;
- SceneNN midpoint-bridge posthoc protocol/admission/RGB receipt/result SHA-256:
  `cdcb736840c3b102ae40687e09b2ee67d40203dea12203ae5888c67b5151814f`,
  `f0bb69c8aaf1c694f6463f89577532400afe5108c63ecfd7a95ccc89dab1f83c`,
  `8e58d58ca734b9f1ba6ec9406608d71e912331b63c859423dbb513e0068dbd24`,
  and `7d5cdb9c37eb839b97ce700255065fbf3d1af53fa68ecd666ff0293f7357f331`;
- SceneNN full-context mask-gated posthoc protocol/result SHA-256:
  `f7f177f838d5dd2667a75fe58a07b327caffdc96ce1f2700758f84686b4723f7`
  and `38cb51d184308dfe01b2cba955a596e5e9ffa28e4f8e2699a33f0bbe8b560b09`;
- visual audits:
  `artifacts.local/results/l10-3rscan-center-target-door-retrieval-v2/proposal-crops-montage.jpg`
  and
  `artifacts.local/results/l10-3rscan-roster-assignment-confirmation-v2/full-frame-audit-montage.jpg`.

The latest positive result removes the complete-roster assumption but still
requires truth proposals, complete surfaces, and the official rigid transform
from one provider. It is not raw-image or phone-side unknown rejection, door
detection, a doorway aperture, named entrance, venue ownership, public access,
traversability, approach waypoint, arrival, `HANDOFF_READY`, provider-independent
confirmation, user benefit, deployment, or mobility-safety evidence.

Inheritance disposition:

- `l10_3dstreetview_center_target_lock.py` is `RETAINED_CORE` for provider-
  centered real-image point scoring below any portal claim;
- the fixed SceneNN edge-sliver cohort is `NEGATIVE_CONTROL` for partial-
  visibility source admission;
- the unchanged center-scale pair score is
  `COMPONENT_OR_CHALLENGER / COMPONENT` inside the closed-roster successor; its
  standalone sibling-door failure remains frozen;
- parameter-free one-to-one assignment is `RETAINED_CORE` only for an externally
  justified complete closed roster;
- appearance-only reciprocal open-roster matching is `NEGATIVE_CONTROL` for the
  false assumption that relative rank establishes match existence;
- registered target-surface scoring is
  `COMPONENT_OR_CHALLENGER / COMPONENT` for partial-roster identity, not
  standalone absence authority;
- the raw maximum of two witness residuals is `DEAD_FOR_THIS_ROLE` as a scan-
  family-general absolute rejector after the frozen incremental test removed two
  false matches but killed sixteen true matches;
- strictly positive registered planar-extent support is `RETAINED_CORE` for
  privileged registered-geometry partial rosters after fresh incremental
  `136/4/0 -> 136/0/0` evidence and unchanged scan-family-disjoint
  `34/2/0 -> 34/0/0` confirmation;
- the one-reference-per-target v1 shortage and the exact-triple-only v1 freeze
  opened zero RGB and remain unclassified working material, not
  `DEAD_FOR_THIS_ROLE`.

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
