# DTR CARLA X68 consumed cross-cohort Development

Date: 2026-09-01

Decision: `DTR_CARLA_X68_CONSUMED_CROSS_COHORT_NONREGRESSION_EFFECT_POSITIVE`

## Structural change

X68 corrects a representation mismatch rather than changing an alert
threshold. The surface arm retains the obstacle footprint needed for route
contact, but its lattice transport quantizes lateral velocity. For each current
measured surface risk, X68 spatially matches a current measured X24 metric
track and recomputes inherited route contact with that object's metric velocity.

The metric velocity is admitted only when its dot product with the surface
velocity is positive and its route-right magnitude is no greater than the
surface estimate. It can therefore remove spurious lateral lattice motion but
cannot introduce a new crossing direction or more lateral motion. Mixed frames
containing a metric-handback carrier without a surface footprint remain exactly
unchanged. No detector, route, distance, speed, duration, weather, or score
threshold was added; matching reuses the X24 `1.5 m` association distance.

A pre-formal counterfactual without the direction and lateral-nonexpansion
conditions removed 16 false positives but also lost two true positives. Those
losses falsified unrestricted metric substitution. The retained one-sided rule
preserves both true positives while removing 15 false positives. An initial
runner implementation also attempted to refine mixed handback frames and
aborted four cohorts before scoring; it was corrected before the v2 scored
outputs below, so inherited handback authority is outside X68's scope.

## Five-cohort result

All five sources had been scored before X68 was designed. These are consumed
Development results only.

| Cohort | X67 TP/FP | X68 TP/FP | X68 P/R/F1 | Delta vs X67 | Refined/release frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| C26 | 128/20 | 128/17 | 88.28/73.99/80.50% | 0 TP, -3 FP, +0.75 pp F1 | 11/7 |
| C27 | 129/15 | 129/10 | 92.81/74.57/82.69% | 0 TP, -5 FP, +1.30 pp F1 | 10/5 |
| C28 | 125/14 | 125/14 | 89.93/72.25/80.13% | classification-neutral | 16/3 |
| C32 | 127/7 | 127/5 | 96.21/73.84/83.55% | 0 TP, -2 FP, +0.55 pp F1 | 12/6 |
| C34 | 127/22 | 127/17 | 88.19/73.84/80.38% | 0 TP, -5 FP, +1.25 pp F1 | 20/9 |

Pooled across the five equal-size cohorts, X67 was `636 TP / 78 FP / 227 FN`
at `89.08/73.70/80.66%` precision/recall/F1. X68 was
`636 TP / 63 FP / 227 FN` at `90.99/73.70/81.43%`: `0 TP / -15 FP / +0.77 pp
F1`. Four cohorts improved and one was classification-neutral; no cohort
regressed. All required authority invariants remained zero and every contact
recall and safe-segment reference constraint remained satisfied.

Relative to the original frozen X65 C34 result, X68 removes nine false
positives with no true-positive loss. C34 rises from
`83.01/73.84/78.15%` to `88.19/73.84/80.38%`, and its safe-episode segment
counts change from `0/3/1/1` to `0/2/1/0` for ep_02/04/06/08.

## Evidence identity

- X68 predictor SHA-256:
  `48B354246BEEF6287AFB961880B81FF54BD2423059385FA485FC599FDC4A9D1E`
- Consumed runner SHA-256:
  `6C3072404248DFF0E8FAF18583B7F6A0E5F93266C22EF2DC10264C3CF7FC07BB`
- C26 summary / prediction SHA-256:
  `E66EA3C57B5DBD4C28F95F61026A826EE82404EC7E406465B020F7042BD54FE4` /
  `4F6E7877AEE31DAD72E6ACFBCE7BD90ECDAFF451C9F34A6E5084DA72F3983D2C`
- C27 summary / prediction SHA-256:
  `4D6E0AB0E608640FB40DB79461334C57119FF3CF0D825258DC9D73358AC6AE98` /
  `4FBB9B2833525A82C4CD0110039179F52BDEB08C0F8A943726804CBFED36DCEA`
- C28 summary / prediction SHA-256:
  `91F77CEAB9050A444B70F2BA01E1BCE45454DAE6D8258C7DBE0C81F084E5BB2E` /
  `64AEB2A9813C4EEE443DC75281BF33E22C5097FBF8AE029F260DD5646675E463`
- C32 summary / prediction SHA-256:
  `1DCC6C15E4EA20AF3CC956742548B9360E6EA82E263BB962B791E9870BB18E62` /
  `ACAE53C56D083C12FE882593A18EF9BD6BD79F2F39029EF4B0A2E8A70E13EA78`
- C34 summary / prediction SHA-256:
  `1C83E92230898CE589FB1D89B3E150FDCD61800C1FC24957BF76636018B7F7CC` /
  `53DD8C2A598885A17B06AAB778D699513861CB4ED53B15D1BF1C6907F6AA384D`

## Claim boundary and next decision

X68 is the strongest current cross-cohort CARLA Development arm: its mechanism
was exercised in every cohort, it improved four independently generated
cohorts, and it did not regress the fifth. This is broader evidence than the
single-cohort X67 effect, but all five sources were consumed before X68 was
designed. It is not fresh confirmation or promotion authority.

The remaining pooled error is `227 FN / 63 FP`; C34 remains `45 FN / 17 FP`.
The next Development change should target missed contact onset without
reintroducing lateral-quantization tails, then require another visible
cross-cohort effect before freezing a new confirmation source.

This is synthetic Development evidence, not real-world, natural-distribution,
product-default, deployment, user-benefit, reliability, or safety evidence.
