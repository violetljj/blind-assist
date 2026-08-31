# DTR CARLA X72 consumed cross-cohort Development

Date: 2026-09-01

Decision: `DTR_CARLA_X72_CONSUMED_CROSS_COHORT_NONREGRESSION_EFFECT_POSITIVE`

## Structural change

X72 completes a fragmented surface object's extent with a current X25 rigid
footprint. A surface parent first earns a collision credential through an
existing X71 route-risk decision. While that same parent remains observable,
X72 may restore route risk only when a current confirmed X25 collision
footprint intersects at least one current measured fragment of the credentialed
parent and the footprint's rigid center lies inside none of that parent's
current fragments.

The parent-wide center exclusion is the key separator. It treats boundary-only
intersection as evidence that the surface representation has split the object
and lets the detector-mask footprint complete its missing extent. If any
current fragment already contains the rigid center, X72 rejects the handback as
a contained same-object near miss. An explicit X69 mature rigid-contradiction
release clears credentials before completion.

Polygon intersection uses the convex separating-axis theorem and inherited
floating-point epsilon. No detector, route, class, distance, weather, time, or
other numeric threshold was added.

A broader credential-plus-overlap counterfactual recovered nine true positives
but added seven false positives. Aggregating all current fragments by parent
and requiring boundary-only overlap removed all seven false positives while
retaining eight of the nine true positives. The discarded C32 frame had rigid
center containment and was not admitted selectively.

## Five-cohort result

All five sources had been scored before X72 was designed. These are consumed
Development results only.

| Cohort | X71 TP/FP | X72 TP/FP | X72 P/R/F1 | Delta vs X71 | Completion frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| C26 | 130/16 | 133/16 | 89.26/76.88/82.61% | +3 TP, 0 FP, +1.10 pp F1 | 3 |
| C27 | 129/7 | 129/7 | 94.85/74.57/83.50% | neutral | 0 |
| C28 | 125/12 | 126/12 | 91.30/72.83/81.03% | +1 TP, 0 FP, +0.38 pp F1 | 1 |
| C32 | 128/3 | 128/3 | 97.71/74.42/84.49% | neutral | 0 |
| C34 | 131/13 | 135/13 | 91.22/78.49/84.38% | +4 TP, 0 FP, +1.46 pp F1 | 4 |

Pooled across the five equal-size cohorts, X71 was `643 TP / 51 FP / 220 FN`
at `92.65/74.51/82.59%` precision/recall/F1. X72 is
`651 TP / 51 FP / 212 FN` at `92.74/75.43/83.19%`: `+8 TP / +0 FP / +0.60 pp
F1`. The mechanism was positive in three cohorts and classification-neutral in
two. All required authority invariants remained zero, and every contact-recall
and safe-segment constraint remained satisfied.

C34 explicitly rejected six center-contained safe frames and admitted four
boundary-only contact frames. Relative to frozen X65 on C34, X72 removes 13
false positives and recovers eight true positives. C34 rises from
`83.01/73.84/78.15%` to `91.22/78.49/84.38%` precision/recall/F1.

## Evidence identity

- X72 predictor SHA-256:
  `30F2E21893D34EF71FC4D8D74DFA80C3C38A6047F3B0E3BC64FB7D818C23CE82`
- Consumed runner SHA-256:
  `60BC7365F252E9D715C144E1A2F7194399C591E516DF1A73868AF60CFACB9750`
- C26 summary / X72 prediction / X25 rigid prediction SHA-256:
  `E05747923BA2ADBA4C5D71555840ED9B75C6BC21A919C07C40E97DDD511E42EE` /
  `DE8C992FEA5773527C3CD3E480A480ACB39337160DB673FDDC1CD3D0DCEC04F4` /
  `0B1D5904057C8E6E0751F00F97546163A51FED33C04A4CAD47F9659571183868`
- C27 summary / X72 prediction / X25 rigid prediction SHA-256:
  `B4A55D39F9A126604EA42363D19B7A501DD387724BBB7B3655BD3A4446665EC7` /
  `207A7A05D3BEF56F22F1467C065F278B7D6939FA0B2803E9A11D87985F029828` /
  `A2C9FFA93F6219B89DADEBE9A00CAE6B3285870F77931063F4488EE78BA94E0B`
- C28 summary / X72 prediction / X25 rigid prediction SHA-256:
  `B8652DDE109E062DB5E6DA8EC11DD59A2AC043ABCFF0413BDCBFC6C9EBA33BC5` /
  `6DD9362C93EF82CE4AD15A26890609A84A8FDEF64D25C5D7B207F9E673E26908` /
  `A9BECAC8047AB3AC61CDA2064F941F9E4CAE5F5047A28E1212BFBF5621B5E7C3`
- C32 summary / X72 prediction / X25 rigid prediction SHA-256:
  `13A8D1F9EFA0D5FDAAA78599A29D83C8C18B8D602837B2A0F93FD8A8E3C6A86C` /
  `602DF661F9DF93888B362705BFF2246E344F42FC1ECE7C241C55DE7F6170299A` /
  `844DF9E132E5EF90F063D4DA45C01D62EF4638A190363AF1FBD10E1314057ED3`
- C34 summary / X72 prediction / X25 rigid prediction SHA-256:
  `F34361A76C31021586FFCCB11AD99EE222851902AB0C606F27284E49A41BD987` /
  `02849BEB14C878F7891D01ED6EF1B340E1982CA6DEF33B0DD7C5D0560422CC0B` /
  `15C792B658406F5919616FA1A45968DEF7AFD8B56FBABF070282599E3EFDB69C`

## Claim boundary and next decision

X72 is the strongest current cross-cohort CARLA Development arm. Its positive
effect in three cohorts shows that parent-level surface identity plus current
boundary geometry can recover object extent lost to component fragmentation
without accepting the contained-overlap false-positive mode. All sources were
consumed before X72 was designed, so this is not fresh confirmation or
promotion authority.

The remaining pooled error is `212 FN / 51 FP`; C34 remains `37 FN / 13 FP`.
The next recall mechanism should address frames with no live credentialed
surface parent or no confirmed X25 collision footprint, rather than weakening
the boundary-only separator. A new confirmation source should be frozen only
after another visible structural gain or to adjudicate unchanged X72.

This is synthetic Development evidence, not real-world, natural-distribution,
product-default, deployment, user-benefit, reliability, or safety evidence.
