# L10-R0 Goal-Lock Copilot

Status: `ACTIVE / CONTROLLED CLOSED-LOOP DEVELOPMENT`

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

Mechanism references: [GoMatching long/short matching](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2d66a70c770de7835678f1c1e65fe5e1-Abstract-Conference.html),
[QueryNLT joint language/visual tracking](https://openaccess.thecvf.com/content/CVPR2024/html/Shao_Context-Aware_Integration_of_Language_and_Visual_References_for_Natural_Language_Tracking_CVPR_2024_paper.html),
and [AECNav evidence consolidation](https://arxiv.org/abs/2608.10817).
