# L10-R0 Goal-Lock Copilot

Status: `ACTIVE`

- **Core controller:** SC1W--SC2 seek/guide/reacquire and the SC14 causal
  action-belief handoff guard remain implemented; the later SceneFun3D ordinal
  source line terminates at `SC40_NOT_EVALUABLE_NO_FRESH_DEPTH_VISIBLE_ACTIVE_VIEW`.
- **Active observation:** PanoLab passed
  `L10_PANOLAB_ACTIVE_ENTRANCE_RAY_RECOVERY_DEVELOPMENT_GATE_MET` (`4/4`
  reciprocal `SIDESTEP_TO_ENTRANCE_FACE` recoveries). This authorizes an exact
  entrance ray geometrically, not a pixel portal.
- **SEVN backend:** the address-door metadata adapter is ready on 24 distinct
  addresses (`8` left-pan, `8` right-pan, `8` graph-approach episodes). The
  annotation oracle changes `0/24` HOLD bindings to `24/24` correct-unique, but
  the 1.86 GB image payload is not downloaded, so this is not pixel-model evidence.
- **Pixel portal:** generic Panoramax mining is closed after the width-first
  source admitted `0/3` reference, `0/3` query, and `0/3` joint portals. It may
  carry imagery only with independent sub-metre portal registration.
- **Posed transfer:** Hypersim met the synthetic Development gate; SceneNN is
  terminal at
  `L10_SCENENN_REAL_RGBD_PARTIAL_METRIC_PORTAL_TRANSFER_CONFIRMATION_GATE_NOT_MET`
  because reference-side visibility was not authoritative.

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
`l10_scenenn_real_posed_portal_result_v2.json`.

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
