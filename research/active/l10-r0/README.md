# L10-R0 Goal-Lock Copilot

Status: `ACTIVE / L10-SC1W SEMANTIC IDENTITY + VISUAL CONTINUITY + CTC WORD CARRIER`

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
