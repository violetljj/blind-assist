# DTR CARLA X74 consumed cross-cohort Development

Date: 2026-09-01

Decision: `DTR_CARLA_X74_CONSUMED_CROSS_COHORT_PRECISION_EFFECT_POSITIVE`

## Structural change

X57 can restore an X24 metric route-risk track when the retained occupancy core
has zero eligible tracks. C35 exposed a concrete identity failure: the handed
back track remained a route-risk `truck`, while the nearest current X25 rigid
footprint was a stable, non-route `person`. X74 adds object-local
cross-representation class contradiction to the X73 line.

Release is allowed only when every confirmed carrier is an X57 metric handback,
the nearest current measured X25 footprint lies inside the inherited 1.5 m X24
association radius, that footprint is not itself an X25 route candidate, and
its current detector class differs from the metric carrier class. The nearest
match is used so an unrelated route-risk object elsewhere in the frame cannot
block or trigger the decision. X74 can only clear an existing metric handback;
it cannot birth or prolong risk. No detector, route, association, score,
duration, weather, lighting, or other numeric threshold changed.

## Six-cohort result

All six sources were consumed before the formal X74 score. C35 had already
served as the one-shot X73 confirmation source and was used here only as
post-confirmation diagnosis. These are Development results, not fresh X74
confirmation.

| Cohort | X73 TP/FP/FN | X74 TP/FP/FN | X74 P/R/F1 | Delta vs X73 | Release frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| C26 | 136/16/37 | 136/16/37 | 89.47/78.61/83.69% | neutral | 0 |
| C27 | 134/7/39 | 134/7/39 | 95.04/77.46/85.35% | neutral | 0 |
| C28 | 128/12/45 | 128/12/45 | 91.43/73.99/81.79% | neutral | 0 |
| C32 | 130/3/42 | 130/3/42 | 97.74/75.58/85.25% | neutral | 0 |
| C34 | 135/13/37 | 135/13/37 | 91.22/78.49/84.38% | neutral | 0 |
| C35 | 132/18/40 | 132/12/40 | 91.67/76.74/83.54% | 0 TP, -6 FP, +1.56 pp F1 | 6 |

Pooled across the six equal-size cohorts, X73 was
`795 TP / 69 FP / 240 FN` at `92.01/76.81/83.73%`
precision/recall/F1. X74 is `795 TP / 63 FP / 240 FN` at
`92.66/76.81/83.99%`: `0 TP / -6 FP / +0.27 pp F1`. All six releases belonged
to one C35 metric identity and all six were false-positive frames. The other
five cohorts were bitwise classification-neutral. Every required authority
invariant remained zero, and every contact-recall and safe-segment constraint
remained satisfied.

## Evidence identity

- X74 predictor SHA-256:
  `52558F7999258B4966C43A6473793E364D170111C90BD41BAC3FDE55F033289E`
- Consumed runner SHA-256:
  `8F5A1C3EC6CA340402561DFA100579402B71FE968FB086F307AB13A3DA3E0EE3`
- C26 summary / X74 prediction SHA-256:
  `09BDC2F5FB226E1A86DF25AB103F1C7681028DCEA1D5C38A4EDBBEE0C8575A0E` /
  `CDFA55979CDFFA1E5E886346CA2BA2F7893B984AAF30BAE5E4A698AA1148A697`
- C27 summary / X74 prediction SHA-256:
  `FA3633119A3F8AD6ACDC517D95B75E57ADC95B4015FEB70D4949094948917F17` /
  `5ACB047897901BA676816A94B6CBB7FE26EF0184455EDD9A8CB14184CBAE1D2F`
- C28 summary / X74 prediction SHA-256:
  `0548F87AFDAC4EBBA8D31F5AF7E3A58AC6579373800FC9FA2305F5B2C3D252EA` /
  `DDFD2FC55A70749B3C3C7CB0332FE9D88B0F26228D896F60BC65F4612A9A887A`
- C32 summary / X74 prediction SHA-256:
  `284C9D271ACBA624DA8EF40AD54E597BFEEEE3869538214E0D18BCC29E322187` /
  `48D5457559A184781AFC5EA1BC3E36ECE93E23E94DFFE230FB03ECB90A532D04`
- C34 summary / X74 prediction SHA-256:
  `94C9D2AD0827E11FA3262B6DDFE6BF1264B084622A799A67C3FDECE484C81A65` /
  `67AD441CE4A14E346F7496BE4B8011597014048EA607E79330C6483EA49B55E3`
- C35 summary / X74 prediction SHA-256:
  `928D2DF657AB3C65D6CAC77A6E9E2D628155A026389CF6BFFE8142D49B4997A1` /
  `C9EDABDB622C6076157B627DB47F311DD661492CF9BB9AB4DC4E4573AC967852`

## Claim boundary and next decision

X74 is the strongest current six-cohort CARLA Development arm by pooled
precision and F1. The visible effect is narrow but falsifiable: it repairs one
cross-representation identity failure without changing any other scored frame.
Because only C35 exercised the rule and C35 was consumed during its design,
this is not evidence that class contradiction will generalize to a new source.

One genuinely new frozen source may test unchanged X74. Before that score,
freeze the exact predictor and require at least one exercised contradiction,
zero true-positive loss versus X73, no authority-invariant violation, and the
existing contact/safe constraints. Do not tune X74 after the new pixels or
outcome are opened.

This is synthetic Development evidence, not real-world, natural-distribution,
product-default, deployment, user-benefit, reliability, or safety evidence.
