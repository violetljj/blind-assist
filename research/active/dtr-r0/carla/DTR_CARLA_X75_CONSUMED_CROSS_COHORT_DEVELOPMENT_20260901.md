# DTR CARLA X75 consumed cross-cohort Development

Date: 2026-09-01

Decision: `DTR_CARLA_X75_CONSUMED_CROSS_COHORT_PRECISION_EFFECT_POSITIVE`

## Structural change

X46 object permanence can retain an already authorized surface risk without a
fixed timeout while its propagated footprint continues to enter the route.
C36 exposed a render-domain failure where one uncorroborated surface identity
continued for 19 false-positive `OBJECT_PERMANENCE_BELIEF_HOLD` frames.

X75 separates existence memory from collision-risk authority. A surface parent
earns a collision credential only when current confirmed surface, X25
rigid-footprint, and X24 metric-point route risk spatially agree under the
inherited 1.5 m association radius. An occupancy-peak-anchored permanence
belief that never earned this credential and has zero recorded transport
contradictions may remain in the track state, but cannot independently carry
route risk. Any nonzero transport contradiction protects the conservative
belief path, as do all credentialed histories and non-permanence carriers.

X75 can only clear an existing alert. It adds no detector, score, distance,
duration, route, weather, lighting, or other numeric threshold.

## Seven-cohort result

All seven sources were consumed before the formal X75 score. C36 had already
completed its one-shot X74 evaluation and was used only as post-confirmation
diagnosis. These are Development results, not fresh X75 confirmation.

| Cohort | X74 TP/FP/FN | X75 TP/FP/FN | X75 P/R/F1 | Delta vs X74 | Release frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| C26 | 136/16/37 | 136/16/37 | 89.47/78.61/83.69% | neutral | 0 |
| C27 | 134/7/39 | 134/7/39 | 95.04/77.46/85.35% | neutral | 0 |
| C28 | 128/12/45 | 128/12/45 | 91.43/73.99/81.79% | neutral | 0 |
| C32 | 130/3/42 | 130/3/42 | 97.74/75.58/85.25% | neutral | 0 |
| C34 | 135/13/37 | 135/13/37 | 91.22/78.49/84.38% | neutral | 0 |
| C35 | 132/12/40 | 132/12/40 | 91.67/76.74/83.54% | neutral | 0 |
| C36 | 143/44/29 | 143/25/29 | 85.12/83.14/84.12% | 0 TP, -19 FP, +4.45 pp F1 | 19 |

Pooled across the seven equal-size cohorts, X74 was
`938 TP / 107 FP / 269 FN` at `89.76/77.71/83.30%`
precision/recall/F1. X75 is `938 TP / 88 FP / 269 FN` at
`91.42/77.71/84.01%`: `0 TP / -19 FP / +0.71 pp F1`. All 19 releases belonged
to one C36 permanence identity and all were false-positive frames. The other
six cohorts were classification-neutral. Every required authority invariant
remained zero, and every contact-recall and safe-segment constraint passed.

## Evidence identity

- X75 predictor SHA-256:
  `2A21A794EC52ED30D15D45BE88FC5E0846735FA06B65839BEBC365ED5E992808`
- Consumed runner SHA-256:
  `BF17C93D8989D7DBB72E9E727B6A5FDF108C337CF070CDCAD86365EA73308784`
- C26 summary / X75 prediction SHA-256:
  `D46975DE97510A76B51DCD32E22EAEB32BFC72A6464313CB0F984EEB00D8D5D2` /
  `8B4FABA34D4300F501D1D8D210B1D477034CC13F773CAE3C5D1802BDA26DA880`
- C27 summary / X75 prediction SHA-256:
  `F8DC673341BCCC94A634185AE952735E2F570B93D23BAB35DA38C23B4C9B8031` /
  `8B7383B28926AA37DD9980B4476F873B5F84480B62560DED957883D9C0BDF9C2`
- C28 summary / X75 prediction SHA-256:
  `12CD56742A34B7D9AB7FCF27CC8A860D096051D899D7905FE559EF01B84745FD` /
  `1CED4FE3CEF19D06CCCAF480BC050D7DF94C7DF3344EA3B0A1A519F641CF8A82`
- C32 summary / X75 prediction SHA-256:
  `70BB73958AF6A190CD67E8AE44A8C6BE9E13ABE8598E4A76DF287635081A60A4` /
  `C682E1A375826679249C564784CA9FB3A3428981703A6B9068686269BA61AAE8`
- C34 summary / X75 prediction SHA-256:
  `889D7F37903521E5546D0B805E633B0952F24EEDF30E9F8A0C4B746274208B16` /
  `45876A02CCB5404BDC4E30B925E6799BA90EBAF04D0DC87303303F502CF957AF`
- C35 summary / X75 prediction SHA-256:
  `5FBC4585CC8899C3F6044A6E631B6937B81B000D6E8A4ED2460E3387C4C83944` /
  `02EEA69BFFCD962C095BCC23634648237E6E93D26A9967F15023F32ED3B7FAB1`
- C36 summary / X75 prediction SHA-256:
  `C2F9AD9B36B1D9041AEAA23FD40EE883D5D06F7D0090130D18C9AF351CBD99FB` /
  `12E32831181CE77EEB6DDE100F81ABC0DE95BE5E6014E45D229E4648B70B878F`

## Claim boundary and next decision

X75 is the strongest current seven-cohort CARLA Development arm by pooled F1.
Its visible effect is large on the newly consumed C36 render domain and exactly
neutral on six earlier cohorts. It also restores C36 above the 85% precision
floor without sacrificing its high recall. However, only C36 exercised the
release and C36 was used to design it, so this is not generalization evidence.

One genuinely new frozen source may test unchanged X75. Positive incremental
confirmation requires an exercised release, zero TP loss versus X74, a strict
FP reduction, all required authority invariants zero, and the inherited
contact/safe constraints. Do not tune X75 after new pixels or outcomes open.

This is synthetic Development evidence, not real-world, natural-distribution,
product-default, deployment, user-benefit, reliability, or safety evidence.
