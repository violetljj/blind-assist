# DTR CARLA X93 consumed eleven-cohort Development

Date: 2026-09-01

Decision: `DTR_CARLA_X93_CONSUMED_ELEVEN_COHORT_PRECISION_EFFECT_POSITIVE`

## Change

X93 unifies two existing collision-authority limits. X89 established that
receding branch consensus is insufficient when correspondence is
underdetermined; X90 established that lateral-dominant motion supplies
cross-route kinematics but not positive-time collision timing without an X75
cross-representation credential. X93 applies those limits when every carrier's
transport history is itself contradicted. A contradiction remains useful
diagnostic evidence against a stable association, but cannot substitute for a
collision credential. X93 clears only positive-time future entries whose every
confirmed carrier is an uncredentialed, contradicted surface branch and whose
carrier set is either entirely receding or entirely lateral-dominant. Current
overlap, closing or mixed motion, credentialed parents, non-surface carriers,
and zero-contradiction histories retain X92. The rule is relational and adds no
fitted numeric, detector, route, weather, lighting, or class threshold.

## Result

The replay applied X93 to sealed X92 predictions and opened only the already
consumed evaluators from C26/C27/C28/C32/C34/C35/C36/C37/C39/C40/C41.

| Cohort | X92 TP/FP/FN | X93 TP/FP/FN | Delta vs X92 | Release frames/tracks |
|---|---:|---:|---:|---:|
| C26 | 136/5/37 | 136/5/37 | 0/0/0 | 0/0 |
| C27 | 134/3/39 | 134/3/39 | 0/0/0 | 0/0 |
| C28 | 128/1/45 | 128/1/45 | 0/0/0 | 0/0 |
| C32 | 130/2/42 | 130/2/42 | 0/0/0 | 0/0 |
| C34 | 135/7/37 | 135/7/37 | 0/0/0 | 0/0 |
| C35 | 132/8/40 | 132/8/40 | 0/0/0 | 0/0 |
| C36 | 143/16/29 | 143/16/29 | 0/0/0 | 0/0 |
| C37 | 133/19/39 | 133/17/39 | 0/-2/0 | 2/2 |
| C39 | 137/10/35 | 137/10/35 | 0/0/0 | 0/0 |
| C40 | 129/15/43 | 129/4/43 | 0/-11/0 | 11/13 |
| C41 | 135/11/37 | 135/11/37 | 0/0/0 | 0/0 |

Pooled X92 is `1,472 TP / 97 FP / 423 FN` at
`93.82/77.68/84.99%` precision/recall/F1. Pooled X93 is
`1,472 TP / 84 FP / 423 FN` at `94.60/77.68/85.31%`, or
`0 TP / -13 FP / +0.32 pp F1`. All thirteen releases are false positives;
the other nine cohorts are classification identical to X92. C40 precision
improves from `89.58%` to `96.99%`. Every contact-recall, safe-segment,
full-arm, and required authority-invariant check passes.

## Evidence

- Replay output suffix: `x93-consumed-development-20260901-224500`
- C37 summary SHA-256:
  `010777D6C66401BE081237DB3574EBB7136A47858734CDDD54D64A1FFC9741D6`
- C40 summary SHA-256:
  `2A39AC141A333F6725A1D61881884EB579EBBDB798D49EF227F8EA2CC7926B9C`
- X93 predictor SHA-256:
  `69E7390801DD7FB1F65A06E7103C2D95B23706FB595E10AEE31BFC334622D85A`
- X93 runner SHA-256:
  `EA10FD7C9BF50C1C34CA9200E686D2581580884D6F7A6ABC4C8E2CAC7AFBA028`

## Claim boundary

X93 was designed after all eleven evaluator truths and transport histories
were opened. The result is consumed, post-hoc synthetic Development evidence
for a correspondence and collision-authority mechanism, not fresh
confirmation or an estimate of natural-distribution performance. X73 retains
the latest complete source-disjoint confirmation authority. Promotion of X93
requires a new preregistered source that exercises the same contradicted,
uncredentialed non-closing partition while preserving frozen recall, contact,
safe-segment, and authority constraints. This is not real-world, deployment,
reliability, user-benefit, or safety evidence.
