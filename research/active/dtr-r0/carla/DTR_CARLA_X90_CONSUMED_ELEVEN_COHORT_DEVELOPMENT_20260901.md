# DTR CARLA X90 consumed eleven-cohort Development

Date: 2026-09-01

Decision: `DTR_CARLA_X90_CONSUMED_ELEVEN_COHORT_PRECISION_EFFECT_POSITIVE`

## Change

X90 extends X79's collision-credential distinction from exactly pure-lateral
motion to lateral-dominant motion without adding a fitted ratio. A small
route-forward component does not by itself synchronize a surface carrier's
cross-route motion with the issued wearer plan. X90 therefore clears a
positive-time future entry only when every confirmed carrier is a surface
branch whose absolute lateral velocity is greater than its absolute
route-forward velocity, whose transport history has zero contradictions, and
whose parent never obtained the inherited X75 cross-representation collision
credential. Current overlap, credentialed parents, contradicted histories,
forward-dominant or static motion, and non-surface carriers retain the X89
decision. Track, geometry, motion, lineage, and diagnostic history remain
available. The rule is ordinal and adds no fitted numeric, detector, route,
weather, lighting, or class threshold.

## Result

The replay applied X90 to sealed X89 predictions and opened only the already
consumed evaluators from C26/C27/C28/C32/C34/C35/C36/C37/C39/C40/C41.

| Cohort | X89 TP/FP/FN | X90 TP/FP/FN | Delta vs X89 | Release frames/tracks |
|---|---:|---:|---:|---:|
| C26 | 136/8/37 | 136/8/37 | 0/0/0 | 0/0 |
| C27 | 134/4/39 | 134/4/39 | 0/0/0 | 0/0 |
| C28 | 128/10/45 | 128/6/45 | 0/-4/0 | 4/4 |
| C32 | 130/2/42 | 130/2/42 | 0/0/0 | 0/0 |
| C34 | 135/8/37 | 135/8/37 | 0/0/0 | 0/0 |
| C35 | 132/9/40 | 132/9/40 | 0/0/0 | 0/0 |
| C36 | 143/21/29 | 143/21/29 | 0/0/0 | 0/0 |
| C37 | 133/19/39 | 133/19/39 | 0/0/0 | 0/0 |
| C39 | 137/14/35 | 137/14/35 | 0/0/0 | 0/0 |
| C40 | 129/20/43 | 129/20/43 | 0/0/0 | 0/0 |
| C41 | 135/11/37 | 135/11/37 | 0/0/0 | 0/0 |

Pooled X89 is `1,472 TP / 126 FP / 423 FN` at
`92.12/77.68/84.28%` precision/recall/F1. Pooled X90 is
`1,472 TP / 122 FP / 423 FN` at `92.35/77.68/84.38%`, or
`0 TP / -4 FP / +0.10 pp F1`. All four releases are the C28
zero-contradiction, uncredentialed lateral-dominant partition. The other ten
cohorts are classification identical to X89. Every contact-recall,
safe-segment, full-arm, and required authority-invariant check passes.

## Evidence

- Replay output suffix: `x90-consumed-development-20260901-204500`
- C28 summary SHA-256:
  `4B890F8175ECEF759E449AB3581EA118ED80F3D10C1F5ABC9EA984B60FA9F1A2`
- X90 predictor SHA-256:
  `35F34F872B87C339402CDF35331C44691F43D511BA0E2EC0F1D0EA41D30470BC`
- X90 runner SHA-256:
  `603D11467706AEE436F88CDB7A185AB5CF75861F0A9B2B8B7392D8A0414F7C28`

## Claim boundary

X90 was designed after all eleven evaluator truths and transport histories
were opened. The result is consumed, post-hoc synthetic Development evidence
for a relational collision-authority mechanism, not fresh confirmation or an
estimate of natural-distribution performance. X73 retains the latest complete
source-disjoint confirmation authority. Promotion of X90 requires a new
preregistered source that exercises the same uncredentialed lateral-dominant
partition while preserving frozen recall, contact, safe-segment, and authority
constraints. This is not real-world, deployment, reliability, user-benefit,
or safety evidence.
