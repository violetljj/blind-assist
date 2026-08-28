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
