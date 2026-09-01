# DTR CARLA X82 consumed ten-cohort Development

Date: 2026-09-01

Decision: `DTR_CARLA_X82_CONSUMED_TEN_COHORT_PRECISION_EFFECT_POSITIVE`

## Change

X82 separates route-risk authority from multiplicity of stale association
proxies. X72 boundary-completion carriers preserve useful rigid identity,
geometry, and motion, but multiple carried (`HOLD`) completions are not
multiple current observations. X82 clears route risk only when every confirmed
carrier is an X72 completion, at least two distinct proxy parents are present,
and every proxy is `HOLD`. A single proxy, any current `MEASURED` completion,
or any mixed direct carrier retains X81's conservative decision. The rule adds
no detector, distance, route, time, weather, class, or fitted metric threshold.

## Result

The frozen replay used the already sealed X81 predictions and evaluator truth
from C26/C27/C28/C32/C34/C35/C36/C37/C39/C40. It did not rerun perception or
CARLA.

| Cohort | X81 TP/FP/FN | X82 TP/FP/FN | X82 P/R/F1 | Delta vs X81 | Release frames |
|---|---:|---:|---:|---:|---:|
| C26 | 136/10/37 | 136/10/37 | 93.15/78.61/85.27% | 0/0/0 | 0 |
| C27 | 134/5/39 | 134/5/39 | 96.40/77.46/85.90% | 0/0/0 | 0 |
| C28 | 128/10/45 | 128/10/45 | 92.75/73.99/82.32% | 0/0/0 | 0 |
| C32 | 130/3/42 | 130/3/42 | 97.74/75.58/85.25% | 0/0/0 | 0 |
| C34 | 135/8/37 | 135/8/37 | 94.41/78.49/85.71% | 0/0/0 | 0 |
| C35 | 132/11/40 | 132/11/40 | 92.31/76.74/83.81% | 0/0/0 | 0 |
| C36 | 143/23/29 | 143/23/29 | 86.14/83.14/84.62% | 0/0/0 | 0 |
| C37 | 133/19/39 | 133/19/39 | 87.50/77.33/82.10% | 0/0/0 | 0 |
| C39 | 137/16/35 | 137/16/35 | 89.54/79.65/84.31% | 0/0/0 | 0 |
| C40 | 129/25/43 | 129/22/43 | 85.43/75.00/79.88% | 0 TP / -3 FP / +0.74 pp F1 | 3 |

Pooled X81 is `1,337 TP / 130 FP / 386 FN` at
`91.14/77.60/83.82%` precision/recall/F1. Pooled X82 is
`1,337 TP / 127 FP / 386 FN` at `91.33/77.60/83.90%`, or
`0 TP / -3 FP / +0.08 pp F1`. The mechanism released six proxy tracks on
three C40 frames and changed no other cohort. Every contact-recall,
safe-segment, full-arm, and required authority-invariant check passed. In C40,
X82 crosses the previously missed 85% precision floor without changing recall.

## Evidence

- Output directory suffix in each cohort's existing evidence run:
  `x82-consumed-development-20260901-165300`
- C40 summary SHA-256:
  `77C083EEB9302EECD633AFB95F60D67A91D3E232CD864760C77C2DF4CF5253D8`
- X82 predictor SHA-256:
  `12F4052CDD27DE41430F5974CB1B1C8AEF07EC1C24369D5E6137CB11FC989CC1`
- X82 runner SHA-256:
  `B69EECCC9B302FC811118B732E7D910CF1A3C582AA5EBDE87C3198D1F8DBC437`

## Claim boundary

X82 was designed after C40 truth was opened. C40 and all earlier cohorts are
therefore consumed, post-hoc synthetic Development evidence for X82; crossing
C40's old gate does not retroactively create fresh confirmation. X73 retains
the latest complete source-disjoint confirmation authority. X82 is the
strongest current ten-cohort Development arm and is eligible to be frozen for
one later preregistered, source-disjoint confirmation. This is not real-sensor,
deployment, reliability, user-benefit, or safety evidence.
