# DTR CARLA X79 consumed cross-cohort Development

Date: 2026-09-01

Decision: `DTR_CARLA_X79_CONSUMED_CROSS_COHORT_PRECISION_EFFECT_POSITIVE`

## Structural change

A surface track can establish object existence and lateral motion without
independently establishing that its crossing is synchronized with the issued
route. X75 already records a parent collision credential only when the surface,
X25 rigid footprint, and X24 metric point all agree on route risk.

X79 keeps every identity and motion row but clears route risk when every
confirmed carrier is a conflict-free `WORLD_OCCUPANCY_COMPONENT` with zero
longitudinal and nonzero lateral velocity whose parent never obtained that
triple collision credential. Credentialed lateral branches, static obstacles,
longitudinal motion, transport contradictions, and mixed-carrier frames remain
conservative. The zero/nonzero tests use the inherited numeric epsilon; X79
adds no learned or tuned numeric threshold.

Across the eight consumed cohorts, the retained X78 arm contained 22
lateral-only route-risk frames: `19 FP / 3 TP`. The prior cross-representation
credential and contradiction history protected all three true positives. The
remaining 15 uncredentialed, conflict-free frames were all false positives and
occurred in five cohorts.

## Eight-cohort result

All sources were consumed before X79 was designed. These are Development
results, not fresh confirmation.

| Cohort | X78 TP/FP/FN | X79 TP/FP/FN | X79 P/R/F1 | Delta vs X78 | Release frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| C26 | 136/12/37 | 136/12/37 | 91.89/78.61/84.74% | neutral | 0 |
| C27 | 134/7/39 | 134/5/39 | 96.40/77.46/85.90% | 0 TP, -2 FP, +0.55 pp F1 | 2 |
| C28 | 128/12/45 | 128/10/45 | 92.75/73.99/82.32% | 0 TP, -2 FP, +0.53 pp F1 | 2 |
| C32 | 130/3/42 | 130/3/42 | 97.74/75.58/85.25% | neutral | 0 |
| C34 | 135/13/37 | 135/8/37 | 94.41/78.49/85.71% | 0 TP, -5 FP, +1.34 pp F1 | 5 |
| C35 | 132/11/40 | 132/11/40 | 92.31/76.74/83.81% | neutral | 0 |
| C36 | 143/25/29 | 143/23/29 | 86.14/83.14/84.62% | 0 TP, -2 FP, +0.50 pp F1 | 2 |
| C37 | 133/23/39 | 133/19/39 | 87.50/77.33/82.10% | 0 TP, -4 FP, +1.00 pp F1 | 4 |

Pooled X78 was `1,071 TP / 106 FP / 308 FN` at
`90.99/77.66/83.80%`. X79 is `1,071 TP / 91 FP / 308 FN` at
`92.17/77.66/84.30%`: `0 TP / -15 FP / +0.49 pp F1`.
Every released frame was a false positive. Every required authority invariant
remained zero, and every contact-recall, safe-segment, and full-arm reference
check passed in all eight cohorts.

## Evidence identity

- X79 predictor SHA-256:
  `537C7BC5ECB842548583282EEFD1FB28F6A630CD457075795AF62332D7C88F93`
- Consumed runner SHA-256:
  `507414F9820E3D5F771DE681A2172B76057F7BA604C37E1A1AAB3036C1AF82FA`
- Summary SHA-256 by cohort:
  - C26: `1079691A32B201E121E5540D2CD5B513D452272F765BCEB98673626F33A67998`
  - C27: `9D012A8D7B83F1B647FF8110D1DEF974BF7D85C2D1A38F6DFA9BC36F54D02837`
  - C28: `44007C3CD0E7836C30A49C229E6CF6F17182630619D3FB3E2C061DEDF567F372`
  - C32: `A3B5611A262B867C56CBFF48E2AC01DD746F3F84873E5513C703D5537F3D0ABD`
  - C34: `EEF2BA826E51A62D40051AA1135724E0DA5D61850CD67DA1711176B50444E38F`
  - C35: `87F4E1F9F60D37DF21CEBC9B0F780AB2FA337A70756C33DDEDD71FD112B249E9`
  - C36: `B2B81555D1581D730271DBB886C0E8DFA999082AB61AA9267E098F0A27017DBA`
  - C37: `D21D1BB03DD6D2141D3FF686DD74EBA979796930A8FD8BAB032FEBFD7BF406A5`

## Claim boundary and next decision

X79 is the strongest current eight-cohort CARLA Development arm by pooled F1.
Its effect is visible in five cohorts and non-regressing in the other three,
but every cohort was consumed before design. X79 therefore does not replace
X73's C35 source-disjoint confirmation authority.

Freeze X79 unchanged before admitting a later source-disjoint CARLA cohort.
The fresh gate should require mechanism exercise, zero TP loss versus X78, all
full-arm reference checks, and all authority invariants zero. This is synthetic
Development evidence, not real-world, natural-distribution, product-default,
deployment, user-benefit, reliability, or safety evidence.
