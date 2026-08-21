# P1-VF0 verifier-first referent ledger

状态：`OUTCOME_BLIND_MECHANICS_IMPLEMENTED / SYNTHETIC_CONTRACT_TESTED / PERFORMANCE_NOT_RUN / DEFAULT_APP_UNCHANGED`

This stdlib-only module is the newly authorized, versioned successor to the closed P1 work. It does not modify or
resume the consumed W1-T0/W2 cohorts. Detector, tracker, matcher, VLM, and geometry inputs are proposal sources only;
opaque entity hypothesis IDs are not evaluator truth.

The stable surface in `core.py` provides:

- a `GoalContract` with `UNIQUE / SET_VALUED / AMBIGUOUS`, explicit rebinding, arrival, and safety semantics;
- an immutable, bounded `ReferentLedger` with `H_other`, parent/child-slot anchoring, distractor registry, current goal
  validity, and append-only evidence receipts;
- verifier decisions including `CONFIRMED_VISIBLE`, `VERIFYING`, `AMBIGUOUS`, `STALE`, explicit set-valued rebinding,
  and `DISPROVED`;
- independent prediction plus distractor-exclusion authority, while appearance is capped and never confirms identity;
- observability-gated negative evidence and rotation/context-only evidence requests. No request authorizes translation.

Only a verified candidate may append to the identity gallery. Multiple supported hypotheses remain ambiguous. Identity
confirmation and current goal validity are separate, so a confirmed entity can be `DISPROVED` for the present task.

Focused check:

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest `
  scripts/research/goal_copilot_bridge/p1_verifier_first/test_core.py
```

Claim ceiling: synthetic contract mechanics only. This module contains no RGB provider, model, threshold search, real
cohort, performance result, VIO, SLAM, metric 3D, POMDP, active translation, arrival-distance promise, Android path, or
product/safety authority. Any data-adequacy or Development execution requires separate user authorization.
