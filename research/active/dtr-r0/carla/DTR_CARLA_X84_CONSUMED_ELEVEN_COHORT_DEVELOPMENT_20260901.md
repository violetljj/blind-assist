# DTR CARLA X84 consumed eleven-cohort Development

Date: 2026-09-01

Decision: `DTR_CARLA_X84_CONSUMED_ELEVEN_COHORT_PRECISION_EFFECT_POSITIVE`

## Change

X84 closes an authority gap in inherited metric-credentialed parent
continuation. A held carrier may preserve direction after metric dropout, but
direction alone cannot identify which closing branch owns route risk when the
authorized branch hypotheses outnumber direct transport-anchor pairs. X84
therefore clears route risk only when every confirmed carrier is a held,
forward-closing, direction-consistent continuation in that branch-overloaded
partition. Occupancy-peak anchored carriers, non-closing carriers, and
continuations whose anchors cover their active branches are retained. Track
identity, geometry, motion, lineage, and suppressed cross-representation
evidence remain available. The rule is class-independent and adds no fitted
metric, detector, route, weather, or lighting threshold.

## Result

The replay applied X84 to sealed X83 predictions and opened only the already
consumed evaluators from C26/C27/C28/C32/C34/C35/C36/C37/C39/C40/C41.

| Cohort | X83 TP/FP/FN | X84 TP/FP/FN | Delta vs X83 | Release frames |
|---|---:|---:|---:|---:|
| C26 | 136/10/37 | 136/10/37 | 0/0/0 | 0 |
| C27 | 134/5/39 | 134/5/39 | 0/0/0 | 0 |
| C28 | 128/10/45 | 128/10/45 | 0/0/0 | 0 |
| C32 | 130/3/42 | 130/3/42 | 0/0/0 | 0 |
| C34 | 135/8/37 | 135/8/37 | 0/0/0 | 0 |
| C35 | 132/11/40 | 132/11/40 | 0/0/0 | 0 |
| C36 | 143/23/29 | 143/23/29 | 0/0/0 | 0 |
| C37 | 133/19/39 | 133/19/39 | 0/0/0 | 0 |
| C39 | 137/16/35 | 137/16/35 | 0/0/0 | 0 |
| C40 | 129/22/43 | 129/22/43 | 0/0/0 | 0 |
| C41 | 135/15/37 | 135/12/37 | 0/-3/0 | 3 |

Pooled X83 is `1,472 TP / 142 FP / 423 FN` at
`91.20/77.68/83.90%` precision/recall/F1. Pooled X84 is
`1,472 TP / 139 FP / 423 FN` at `91.37/77.68/83.97%`, or
`0 TP / -3 FP / +0.07 pp F1`. C41 alone improves from
`90.00/78.49/83.85%` to `91.84/78.49/84.64%`. The three released frames are
exactly its branch-overloaded held closing continuation; the 13 held
continuation true positives observed elsewhere remain protected by occupancy
peaks, non-closing motion, or anchor coverage. Every contact-recall,
safe-segment, full-arm, and required authority-invariant check passes.

## Evidence

- Output directory suffix beside each cohort's existing X83 run:
  `x84-consumed-development-20260901-182000`
- C41 summary SHA-256:
  `847CB9BB9BE34C20A8BCF95A96371CF0232F07BBF9B8675D04AD43264D5C2574`
- X84 predictor SHA-256:
  `56D4FE10987B932130D4EB9E7C0CCD58A9F53D2F43D02581E29404E2E2F539BD`
- X84 runner SHA-256:
  `24EB3AC7BCB609D27831A9ACF7246D189922AE63B4D8FFA9E9F0AF1FAE00176F`

## Claim boundary

X84 was designed after C41 truth and X83 structure were opened. All eleven
cohorts are therefore consumed, post-hoc synthetic Development evidence for
X84. The zero-TP-loss separation is a falsifiable mechanism result on these
cohorts, not fresh confirmation or an estimate of natural-distribution
performance. X73 retains the latest complete source-disjoint confirmation
authority. Promotion of X84 requires a new preregistered source that exercises
the branch-overloaded continuation release while preserving the frozen recall,
contact, safe-segment, and authority constraints. This is not real-world,
deployment, reliability, user-benefit, or safety evidence.
