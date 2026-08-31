# DTR CARLA X69 consumed cross-cohort Development

Date: 2026-09-01

Decision: `DTR_CARLA_X69_CONSUMED_CROSS_COHORT_NONREGRESSION_EFFECT_POSITIVE`

## Structural change

X69 adds a one-way contradiction release between two already-existing geometry
representations. X68 owns the surface-risk decision. X25 supplies an
object-local detector-mask RGB-D rigid footprint. X69 may clear an X68 risk only
when every confirmed surface carrier is currently measured, has nonzero
cross-route lattice transport, spatially matches a current measured X25 rigid
track, and X25 has no current route-contact candidate.

The X25 negative is admitted only after the surface carrier has accumulated
transport ambiguity for the full inherited X24 `1.0 s` track-history window.
This separates a persistent lattice tail from a genuine early crossing whose
rigid footprint has not yet matured. Matching reuses the X24 `1.5 m`
association distance. No detector, route, score, distance, weather, or new
numeric threshold was added, and X69 cannot birth or prolong an alert.

Two broader counterfactuals were rejected before formalization. Parent-level
surface convex recomposition recovered one true positive but introduced six
false positives by filling real safe gaps. Unmatured rigid contradiction
removed 14 false positives but lost one early C26 crossing true positive. The
retained inherited-history maturity condition preserves that true positive.

## Five-cohort result

All five sources had been scored before X69 was designed. These are consumed
Development results only.

| Cohort | X68 TP/FP | X69 TP/FP | X69 P/R/F1 | Delta vs X68 | Release frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| C26 | 128/17 | 128/16 | 88.89/73.99/80.76% | 0 TP, -1 FP, +0.25 pp F1 | 1 |
| C27 | 129/10 | 129/7 | 94.85/74.57/83.50% | 0 TP, -3 FP, +0.80 pp F1 | 3 |
| C28 | 125/14 | 125/12 | 91.24/72.25/80.65% | 0 TP, -2 FP, +0.52 pp F1 | 2 |
| C32 | 127/5 | 127/3 | 97.69/73.84/84.11% | 0 TP, -2 FP, +0.55 pp F1 | 2 |
| C34 | 127/17 | 127/13 | 90.71/73.84/81.41% | 0 TP, -4 FP, +1.03 pp F1 | 4 |

Pooled across the five equal-size cohorts, X68 was `636 TP / 63 FP / 227 FN`
at `90.99/73.70/81.43%` precision/recall/F1. X69 was
`636 TP / 51 FP / 227 FN` at `92.58/73.70/82.06%`: `0 TP / -12 FP / +0.63 pp
F1`. Every cohort improved, all required authority invariants remained zero,
and every contact-recall and safe-segment constraint remained satisfied.

Relative to frozen X65 on C34, X69 removes 13 false positives with no
true-positive loss. C34 rises from `83.01/73.84/78.15%` to
`90.71/73.84/81.41%` precision/recall/F1.

## Evidence identity

- X69 predictor SHA-256:
  `4DA147DCF99CB45BEFA79AAF63D33D3C2798A73DCA7A9B2B91A01912DFE95E60`
- Consumed runner SHA-256:
  `D118DECE2AC7AED00EF83BF3C96F7BA5B3D5C18E707F7B8FBC27C0F81AF31F14`
- C26 summary / X69 prediction SHA-256:
  `992245DAE36CE8A568456742A407A9052766CA17ACF22D252D5973B4AFE2A65F` /
  `165ECB16C64E5886BE0636DA909DCB3B931902075786BD3FE850EFD79F762D84`
- C27 summary / X69 prediction SHA-256:
  `D3FD0F1F4DA85523DD1D849E649A438AAC827B54515B60F81C4FAB5ADBC4B0AD` /
  `BD6E233F6F9B5C7AD18D882F9DB207423F4DAC5730EAE24A2F61A92DC7BA6D41`
- C28 summary / X69 prediction SHA-256:
  `F7180E334A908909C3A50124604F6CCDFCCDA99EA5F38C9D917BE412C4E39731` /
  `EC8552C8BDD69EC9F3DC386A472C2B291E1CB5A61EBAD972250749E67BE2CA1F`
- C32 summary / X69 prediction SHA-256:
  `3CB239C9406929BEEA6B5AD23FCE503B1717C3A54C81A97B40D845AD760BC86B` /
  `B47560798D0CA0647A5ED8FBFF79FE58CBB6F1650EA68A5AFA319728F37C3F57`
- C34 summary / X69 prediction SHA-256:
  `49DF9EFE35D2CDEBC305D181DADDAB1B25CB86908D2E9D6A414A8B1C247AF628` /
  `84580B972947FB788DAB2ED5EABE5206CB581CC9DF159BB45B70F99466C924DF`

## Claim boundary and next decision

X69 is the strongest current cross-cohort CARLA Development arm. Unlike a
posthoc episode selector, the same structural contradiction mechanism was
exercised and positive in all five independently generated cohorts. All five
sources were nevertheless consumed before X69 was designed, so this is not
fresh confirmation or promotion authority.

The remaining pooled error is `227 FN / 51 FP`; C34 remains `45 FN / 13 FP`.
The opened FN diagnosis shows that most misses have neither a surface route
candidate nor a jointly agreeing metric candidate. Further recall gains should
therefore improve observation reach or object-local occupancy birth rather than
relax route confirmation. A new confirmation source should be frozen only after
another visible structural gain or to adjudicate unchanged X69.

This is synthetic Development evidence, not real-world, natural-distribution,
product-default, deployment, user-benefit, reliability, or safety evidence.
