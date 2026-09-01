# DTR CARLA X94 consumed eleven-cohort Development

Date: 2026-09-01

Decision: `DTR_CARLA_X94_CONSUMED_ELEVEN_COHORT_RECALL_EFFECT_POSITIVE`

## Change

X94 closes a narrow continuity gap left by X93. When both detector candidates
and metric footprints disappear for one observation, X94 may transport the
immediately previous confirmed rigid surface carrier only if a current held row
proves the same parent identity, the valid issued-plan receipt is unchanged,
the elapsed time remains inside the inherited X24 `0.60 s` hold window, and no
current release is active. The transported carrier cannot reseed itself, so the
exception is limited to exactly one next observation. Absence alone never
creates risk, and no detector, route, class, weather, lighting, or fitted
numeric threshold is added.

## Result

The replay applied X94 to sealed X93 predictions and opened only the already
consumed evaluators from C26/C27/C28/C32/C34/C35/C36/C37/C39/C40/C41.

| Cohort | X93 TP/FP/FN | X94 TP/FP/FN | Delta vs X93 | Continuity frames/tracks |
|---|---:|---:|---:|---:|
| C26 | 136/5/37 | 136/5/37 | 0/0/0 | 0/0 |
| C27 | 134/3/39 | 135/3/38 | +1/0/-1 | 1/1 |
| C28 | 128/1/45 | 130/1/43 | +2/0/-2 | 2/2 |
| C32 | 130/2/42 | 131/2/41 | +1/0/-1 | 1/1 |
| C34 | 135/7/37 | 136/7/36 | +1/0/-1 | 1/1 |
| C35 | 132/8/40 | 132/8/40 | 0/0/0 | 0/0 |
| C36 | 143/16/29 | 143/16/29 | 0/0/0 | 0/0 |
| C37 | 133/17/39 | 133/17/39 | 0/0/0 | 0/0 |
| C39 | 137/10/35 | 137/10/35 | 0/0/0 | 0/0 |
| C40 | 129/4/43 | 129/4/43 | 0/0/0 | 0/0 |
| C41 | 135/11/37 | 136/11/36 | +1/0/-1 | 1/1 |

Pooled X93 is `1,472 TP / 84 FP / 423 FN / 3,389 TN` at
`94.60/77.68/85.31%` precision/recall/F1. Pooled X94 is
`1,478 TP / 84 FP / 417 FN / 3,389 TN` at `94.62/77.99/85.51%`, or
`+6 TP / 0 FP / -6 FN / +0.20 pp F1`. All six changed frames are true
positives. Every contact-recall, safe-segment, full-arm, and required
authority-invariant check passes; the five required authority defect counts
remain zero.

## Evidence

- Replay output root:
  `artifacts.local/evidence/dtr-carla-x94-eleven-cohort-development`
- C27 summary SHA-256:
  `CA198297728F4BB162DE68AE6FC48A0F598B413DEDAC2F1DE2DDAF3726DA06D9`
- C28 summary SHA-256:
  `A93122C08D3BE21C7F63A85BC0C740D6F176CD08603CFD7AD59EE706BAAA93A0`
- C32 summary SHA-256:
  `272A26C25D1E2CDA20FF3CA5047BDD42A84361313858372D76DF685D42FF0EEE`
- C34 summary SHA-256:
  `39B84D7A4BB3772874736A773087AFDB82DE4AE9C9433170B12E1BBC7515810B`
- C41 summary SHA-256:
  `F9DAAD235BE6C21273E910B6D342739A241096FFDCED5837DE13407508FF0405`
- X94 predictor SHA-256:
  `8A58C1387513AD80B4E4AC474B8D4B02E4E1E5183EF13B78894D15BF46C8E5F3`
- X94 runner SHA-256:
  `29BF61356BFE6C53F0FA28CAF4F3F504F97A2E0CE39F076811D7E6C29F8B2525`

## Claim boundary

X94 was designed after all eleven evaluator truths and histories were opened.
This is consumed, post-hoc synthetic Development evidence for a one-frame
evidence-continuity mechanism, not fresh confirmation or an estimate of
natural-distribution performance. X73 retains the latest complete
source-disjoint confirmation authority. Promotion of X94 requires a new
preregistered source that exercises the same full-dropout continuity partition
while preserving frozen precision, recall, contact, safe-segment, and authority
constraints. This is not real-world, deployment, reliability, user-benefit, or
safety evidence.
