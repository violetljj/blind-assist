# DTR CARLA X70 consumed cross-cohort Development

Date: 2026-09-01

Decision: `DTR_CARLA_X70_CONSUMED_CROSS_COHORT_NONREGRESSION_EFFECT_POSITIVE`

## Structural change

X70 adds a narrow object-identity handback across a current surface dropout.
An X25 detector-mask RGB-D rigid track first earns a collision credential only
when current X69 surface risk, current X25 rigid-footprint risk, and current X24
metric-point risk spatially agree on the same object. The same continuously
confirmed X25 identity may then restore route risk on a frame where X69 is
clear only if no current measured risk-eligible surface row spatially matches
that identity.

An explicit X69 mature rigid-contradiction release clears credentials before
continuation, so X70 cannot reverse X69's false-positive reduction. Matching
reuses X24's inherited `1.5 m` association distance and the X25 track's existing
lifetime. No detector, route, score, distance, weather, or new numeric threshold
was added.

A broader two-channel counterfactual that credentialed X25 directly from X69
and X25 recovered eight true positives but added five false positives. It was
rejected before formalization. Requiring independent X24 metric-point agreement
retained four recovered true positives while eliminating all five added false
positives.

## Five-cohort result

All five sources had been scored before X70 was designed. These are consumed
Development results only.

| Cohort | X69 TP/FP | X70 TP/FP | X70 P/R/F1 | Delta vs X69 | Handback frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| C26 | 128/16 | 130/16 | 89.04/75.14/81.50% | +2 TP, 0 FP, +0.75 pp F1 | 2 |
| C27 | 129/7 | 129/7 | 94.85/74.57/83.50% | neutral | 0 |
| C28 | 125/12 | 125/12 | 91.24/72.25/80.65% | neutral | 0 |
| C32 | 127/3 | 128/3 | 97.71/74.42/84.49% | +1 TP, 0 FP, +0.38 pp F1 | 1 |
| C34 | 127/13 | 128/13 | 90.78/74.42/81.79% | +1 TP, 0 FP, +0.38 pp F1 | 1 |

Pooled across the five equal-size cohorts, X69 was `636 TP / 51 FP / 227 FN`
at `92.58/73.70/82.06%` precision/recall/F1. X70 is
`640 TP / 51 FP / 223 FN` at `92.62/74.16/82.37%`: `+4 TP / +0 FP / +0.30 pp
F1`. The mechanism was positive in three cohorts and classification-neutral in
two. All required authority invariants remained zero, and every contact-recall
and safe-segment constraint remained satisfied.

Across all cohorts, 57 triple-credential births produced only four admitted
surface-dropout handback frames: two held X25 observations in C26, one measured
observation in C32, and one held observation in C34. This narrow exercise is
the visible effect; credentials did not create any safe-frame alert.

Relative to frozen X65 on C34, X70 removes 13 false positives and recovers one
true positive. C34 rises from `83.01/73.84/78.15%` to
`90.78/74.42/81.79%` precision/recall/F1.

## Evidence identity

- X70 predictor SHA-256:
  `7D438E9F852CDA2380CC8E95D95E89C6FB341D9F30582042B8B8DF55539D5245`
- Consumed runner SHA-256:
  `53EC41A9B61215179480C636D9063C567F7FF3F3F3FABCC0284982C893BC5358`
- C26 summary / X70 prediction / X25 rigid prediction SHA-256:
  `5A2D91A8C387BE9560ED466692F03318147DB109B18B0E037190921743D53B03` /
  `905638618701E45646B9A4EC26D08BE1658F7E3FA2E739C97C9CF5FE23B7E380` /
  `0B1D5904057C8E6E0751F00F97546163A51FED33C04A4CAD47F9659571183868`
- C27 summary / X70 prediction / X25 rigid prediction SHA-256:
  `DA2952D29C65309C4F3CD84465CC30A4322215CF837180683CF9C722BCA31F1E` /
  `C33681C68680A45E8AE6D4B4A81DFDD06F6165F9388F44097077A2F0666C1963` /
  `A2C9FFA93F6219B89DADEBE9A00CAE6B3285870F77931063F4488EE78BA94E0B`
- C28 summary / X70 prediction / X25 rigid prediction SHA-256:
  `486223822672C9AEE17BD6BCE777F45AD8EC65042CB875FDEF1F534753A4B195` /
  `567CEB751D11C5257E29000BDCBAFCB6C4C4FCCD92113F7A88D31009420F6E9C` /
  `A9BECAC8047AB3AC61CDA2064F941F9E4CAE5F5047A28E1212BFBF5621B5E7C3`
- C32 summary / X70 prediction / X25 rigid prediction SHA-256:
  `3B5DF7DEBC16C49A87E23CD618CD0C3188A4CB9C9C612665B9D05D09EE272CC4` /
  `44E594367BB00019DF28B2FB3B06EF09303845A52F7467D83974CA8E7689CA5C` /
  `844DF9E132E5EF90F063D4DA45C01D62EF4638A190363AF1FBD10E1314057ED3`
- C34 summary / X70 prediction / X25 rigid prediction SHA-256:
  `8E8D125FDDCBDB48719093B48383C12476F2054B04448D1F4CBBA4F1EFE658FA` /
  `81A9709F309289939FDB04B0404A25BF11AEFC6ECA642C39864A8438D52C9905` /
  `15C792B658406F5919616FA1A45968DEF7AFD8B56FBABF070282599E3EFDB69C`

## Claim boundary and next decision

X70 is the strongest current cross-cohort CARLA Development arm. Its three-way
credential birth and X69 release precedence show that a small amount of recall
can be recovered across representation dropout without reopening the false
positives removed by X69. All five sources were nevertheless consumed before
X70 was designed, so this is not fresh confirmation or promotion authority.

The remaining pooled error is `223 FN / 51 FP`; C34 remains `44 FN / 13 FP`.
Most opened misses still lack a surface route candidate or jointly agreeing
metric candidate. The next recall mechanism should therefore improve
observation reach or object-local occupancy birth, not relax credential birth.
A new confirmation source should be frozen only after another visible
structural gain or to adjudicate unchanged X70.

This is synthetic Development evidence, not real-world, natural-distribution,
product-default, deployment, user-benefit, reliability, or safety evidence.
