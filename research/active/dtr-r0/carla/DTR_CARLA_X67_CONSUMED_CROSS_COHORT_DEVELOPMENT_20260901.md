# DTR CARLA X67 consumed cross-cohort Development

Date: 2026-09-01

Decision: `DTR_CARLA_X67_CONSUMED_CROSS_COHORT_NONREGRESSION_EFFECT_POSITIVE`

## Structural change

X67 separates object existence memory from actionable collision authority after
a complete observed lifecycle. A confirmed X65 risk is released only when all
of its carriers were previously reactivated from dormancy, have disappeared
again for at least the already-existing X24 measurement hold horizon, retain
only direction-consistent surface transport, and carry a receding longitudinal
velocity. The track and its existence evidence remain in the frame.

The first version used only the hold-horizon and receding conditions. It
removed four C34 false positives but also removed eight C26 and one C27 true
positives. That falsified the assumption that the forward-axis sign alone has
cross-source receding semantics. The retained version adds the discrete
reactivation receipt: it distinguishes a second loss after reacquisition from a
first occlusion interval without introducing a new score or numeric threshold.

No detector, route, distance, speed, duration, weather, or score threshold was
added. The only time boundary is the inherited X24 `0.6 s` measurement hold
horizon.

## Five-cohort result

All five sources had been scored before X67 was designed. These are consumed
Development results only.

| Cohort | X65 TP/FP | X65 F1 | X67 TP/FP | X67 P/R/F1 | Delta vs X65 | Release frames |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C26 | 128/20 | 79.75% | 128/20 | 86.49/73.99/79.75% | identical | 0 |
| C27 | 129/15 | 81.39% | 129/15 | 89.58/74.57/81.39% | identical | 0 |
| C28 | 125/14 | 80.13% | 125/14 | 89.93/72.25/80.13% | identical | 0 |
| C32 | 127/7 | 83.01% | 127/7 | 94.78/73.84/83.01% | identical | 0 |
| C34 | 127/26 | 78.15% | 127/22 | 85.23/73.84/79.13% | 0 TP, -4 FP, +0.97 pp F1 | 4 |

Pooled across the five equal-size cohorts, X65 was `636 TP / 82 FP / 227 FN`
at `88.58/73.70/80.46%` precision/recall/F1. X67 was
`636 TP / 78 FP / 227 FN` at `89.08/73.70/80.66%`: `0 TP / -4 FP / +0.20 pp
F1`. No cohort regressed. All authority invariants remained zero, every contact
recall floor remained satisfied, and the safe-segment constraints were
unchanged.

On consumed C34, the four released frames were exactly ep_03 samples 48-51.
They were all false positives and formed the post-contact tail that caused X65
to miss the frozen 85% precision floor. X67 removes that tail without changing
any contact-frame decision, raising C34 precision above the floor.

## Evidence identity

- X67 predictor SHA-256:
  `4E9719E605501F1DE078B71A93A19DDA065CDBF4B9DF059F9290BB89E06380B2`
- Consumed runner SHA-256:
  `C1A97C96AC5907DB3362DAB4A3201B4B5642B092F5D8B0A3CAD2339769F3E79C`
- C26 summary SHA-256:
  `1DCAB0DF68C2C6F7B4E87B45170C223F1E92664716A86BC62DBFEB0FAE7F3978`
- C27 summary SHA-256:
  `A286A402C7C09A4B81308D46D68C32E134106F1BB3A31203202303C820604FFC`
- C28 summary SHA-256:
  `284E71E16E1A883F513EE010B49BBF76BC7E120BC0F0572A66EAD4BA507E6F92`
- C32 summary SHA-256:
  `721D71F67EF598803FCBEEAAD39ECB97595F47D745ECCD566F6A8A7A0B25045D`
- C34 summary SHA-256:
  `B35AC1CDFD17627114B4E43276F3E5B30F2F0888E429068F50E2F68B4096E302`

## Claim boundary and next decision

X67 is the strongest current cross-cohort CARLA Development arm because it
improves one consumed cohort and is framewise identical on four others. C34 was
fresh for frozen X65, but became consumed before X67 was designed; therefore
the C34 improvement is not fresh confirmation or promotion authority.

The remaining C34 error is `45 FN / 22 FP`. The next Development change should
target the object-local disagreement between surface collision geometry and the
metric near-miss trajectory. Freeze a new source only after another structural
mechanism produces visible cross-cohort effect without losing contact recall.

This is synthetic Development evidence, not real-world, natural-distribution,
product-default, deployment, user-benefit, reliability, or safety evidence.
