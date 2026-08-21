# P1-AMRM0 adaptive multi-view referent memory

状态：`MATCHED_CANARY_TERMINAL=P1_AMRM0_MEMORY_POISONING_FAIL / FAILURE_AUTOPSY_ONLY / DEFAULT_APP_UNCHANGED`

This stdlib-only module now owns the P1-AMRM0 main experiment path. P1-VF0 remains its verifier foundation. AMRM0 asks
whether a compact bank of independently verified distance/viewpoint/context observations can improve same-instance
reacquisition and reduce wrong-instance reacquisition relative to continuous 2D correspondence. This is the highest
priority hypothesis to test, not an established result. It does not modify or resume consumed W1-T0/W2 cohorts.

The stable surface in `core.py` provides:

- a `GoalContract` with `UNIQUE / SET_VALUED / AMBIGUOUS`, explicit rebinding, arrival, and safety semantics;
- an immutable, bounded `ReferentLedger` with `H_other`, parent/child-slot anchoring, distractor registry, current goal
  validity, and append-only evidence receipts;
- verifier decisions including `CONFIRMED_VISIBLE`, `VERIFYING`, `AMBIGUOUS`, `STALE`, explicit set-valued rebinding,
  and `DISPROVED`;
- independent prediction plus distractor-exclusion authority, while appearance is capped and never confirms identity;
- observability-gated negative evidence and rotation/context-only evidence requests. No request authorizes translation.

`memory.py` adds the adaptive multi-view layer:

- immutable refs for target crop, context crop, full frame, orientation, distance, viewpoint, scale, and context anchor;
- strictly separate tentative and verified stores; retrieval exposes verified entries only;
- promotion only when the matching verifier receipt independently confirms identity;
- no writes while out of view, occluded, stale, disproved, or rebound to another referent;
- novelty admission over distance × viewpoint × scale plus stable context, with bounded diversity-preserving eviction.

Only a verified candidate may append to the identity gallery. Multiple supported hypotheses remain ambiguous. Identity
confirmation and current goal validity are separate, so a confirmed entity can be `DISPROVED` for the present task.

## Matched Development canary

`run_matched_canary.py` fixes the first scientific canary to the consumed P1-D0 15-episode cohort and the exact P1-A4
candidate stream (`1,724` frames, `926` candidate-available frames, `798` explicit candidate absences). The sealed A4
output is the continuous-correspondence baseline. AMRM can only commit or abstain on the same bbox; added search,
candidate, frame, prompt, VIO, SLAM, VLM, or threshold sweep is forbidden.

Target and masked-context matching inherit the unchanged consumed P1-A2 DINOv2-S dense-consensus gate. A new keyframe
may enter the verified bank only when it remains directly supported by both the original binding target and original
masked context. The private evaluator opens only after both predictions and contribution trace are written. Positive
AMRM value requires improved identity precision, reduced wrong commitments, non-trivial identity coverage, at least one
true reacquisition, no increased false reacquisition, zero poisoning, and at least one true reacquisition supported by a
newly accumulated verified keyframe. Snapshot-only verifier gain has its own non-AMRM terminal.

Focused check:

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest `
  scripts/research/goal_copilot_bridge/p1_verifier_first/test_core.py `
  scripts/research/goal_copilot_bridge/p1_verifier_first/test_memory.py `
  scripts/research/goal_copilot_bridge/p1_verifier_first/test_matched_canary.py
```

The 31 tests establish contract mechanics only. Physical viewpoint truth is unavailable; the canary reports a public
2D bearing-change proxy and marks physical viewpoint reacquisition `NOT_EVALUABLE`.

Executed result: identity precision `9.48% -> 10.65%` at coverage `96.87% -> 80.13%`, but wrong-instance
reacquisition increased `12 -> 38`, verified-bank poisoning was 17, and newly accumulated keyframes contributed zero
true reacquisitions. This is `P1_AMRM0_MEMORY_POISONING_FAIL`, not AMRM value. The only successor is a read-only,
outcome-preserving poisoning autopsy; no threshold tuning or AMRM/geometry expansion is authorized.
