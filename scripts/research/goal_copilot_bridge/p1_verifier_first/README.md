# P1-AMRM0 adaptive multi-view referent memory

状态：`MAIN_EXPERIMENT_PATH / HIGHEST_PRIORITY_HYPOTHESIS / OUTCOME_BLIND_MECHANICS_IMPLEMENTED / SYNTHETIC_CONTRACT_TESTED / REAL_EXPERIMENT_NOT_RUN / DEFAULT_APP_UNCHANGED`

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

Focused check:

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest `
  scripts/research/goal_copilot_bridge/p1_verifier_first/test_core.py `
  scripts/research/goal_copilot_bridge/p1_verifier_first/test_memory.py
```

Claim ceiling: synthetic contract mechanics only. The 26 tests do not establish AMRM utility. This module contains no RGB provider, model, threshold search, real
cohort, performance result, VIO, SLAM, metric 3D, POMDP, active translation, arrival-distance promise, Android path, or
product/safety authority. The next experiment must first establish a causal data adapter and matched correspondence
control; minimal geometry remains conditional on a measured translation failure.
