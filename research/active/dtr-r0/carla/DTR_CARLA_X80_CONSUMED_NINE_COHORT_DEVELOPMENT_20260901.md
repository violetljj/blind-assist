# DTR CARLA X80 consumed nine-cohort Development

Date: 2026-09-01

Decision: `DTR_CARLA_X80_CONSUMED_NINE_COHORT_PRECISION_EFFECT_POSITIVE`

## Structural change

X71 can birth route risk when an X24 metric point and an X25 rigid footprint
agree at their predicted route-entry time. That establishes a shared object and
motion hypothesis, but it does not by itself establish that the rigid occupancy
shape supports a cross-route carrier.

X80 adds one ordinal spatial credential. An otherwise uncredentialed X71 birth
can authorize cross-route occupancy only when its rigid footprint's lateral
span strictly exceeds its route-forward span. If every confirmed carrier lacks
that credential, X80 retains the identity, footprint, and motion rows but clears
route risk. Collision-credentialed, cross-route-elongated, and mixed-carrier
frames remain conservative. The rule compares axis order with the inherited
numeric epsilon; it adds no fitted magnitude, class, weather, detector, route,
or timing threshold.

The rule was designed after C39 was opened. C39 and every earlier source in
this replay are therefore consumed Development material.

## Nine-cohort result

| Cohort | X79 TP/FP/FN | X80 TP/FP/FN | X80 P/R/F1 | Delta vs X79 | Release frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| C26 | 136/12/37 | 136/12/37 | 91.89/78.61/84.74% | neutral | 0 |
| C27 | 134/5/39 | 134/5/39 | 96.40/77.46/85.90% | neutral | 0 |
| C28 | 128/10/45 | 128/10/45 | 92.75/73.99/82.32% | neutral | 0 |
| C32 | 130/3/42 | 130/3/42 | 97.74/75.58/85.25% | neutral | 0 |
| C34 | 135/8/37 | 135/8/37 | 94.41/78.49/85.71% | neutral | 0 |
| C35 | 132/11/40 | 132/11/40 | 92.31/76.74/83.81% | neutral | 0 |
| C36 | 143/23/29 | 143/23/29 | 86.14/83.14/84.62% | neutral | 0 |
| C37 | 133/19/39 | 133/19/39 | 87.50/77.33/82.10% | neutral | 0 |
| C39 | 137/22/35 | 137/16/35 | 89.54/79.65/84.31% | 0 TP, -6 FP, +1.53 pp F1 | 6 |

Across all nine consumed cohorts, X79 was `1,208 TP / 113 FP / 343 FN` at
`91.45/77.89/84.12%`. X80 is `1,208 TP / 107 FP / 343 FN` at
`91.86/77.89/84.30%`: `0 TP / -6 FP / +0.18 pp F1`.

Every released frame was a C39 false positive. X80 changed no classification in
the other eight cohorts. Every required authority invariant remained zero, and
every contact-recall, safe-segment, and full-arm reference check passed in all
nine cohorts.

## Evidence identity

- X80 predictor SHA-256:
  `FDC1417CBBA1641E790D04E240499B7760ECBAD433F872C00E55E881D89DD0E8`
- Consumed runner SHA-256:
  `4D4E9577D5DE7A3118885EB3A5E984BB570BC7969CD69D48057E11C71643FBA1`
- C39 consumed-reuse alias SHA-256:
  `DF4708436E9A0854E03AAB80C847E98A87FD5B182F470DF5B2432707EC14F794`
- The C39 alias changed only the outer authority/status metadata; its episode
  predictions and fixed constants are identical to the single formally scored
  X79 file.
- Summary SHA-256 by cohort:
  - C26: `85DE883E98C67CC6800E301868C3B7E6C37A512452856C9EB48DF25A76E9A3BA`
  - C27: `D391EBEEE6EE8048C8976C657216607374661D48B0F103E9EE564A31381A61B3`
  - C28: `100DBF39E5B2A888312E0822DA3FDA4AE52273810F725D17742A7D895370630C`
  - C32: `06B95B71DEA30EF9503FCAEB62C0A17974A76E3C99C86146060AE793E03E0734`
  - C34: `89BECC5F65B3A1FB03FA3488CB5D1D0FBA9E2525451B2540841EECADEF90EB76`
  - C35: `598B229F0F7750FBBB685431E712D7BF869C6532DEFBDF1B4AC0327FDB8A5156`
  - C36: `D6FC3854BD9F4F8994371553D04344ACB705135C270F5DF679DF7AFB8F929DBC`
  - C37: `A9A09A9B79ED338A2DB28CD274496F874295DC9C98D9510172D3399EFEE4D58F`
  - C39: `0325FA4F248A9B0A8E1E3A66DAA468574C326343AFB66BBF6B410ED1FBCA8E1E`

## Claim boundary and next decision

X80 is the strongest current nine-cohort CARLA Development arm by pooled F1,
but its only incremental effect was designed and measured on consumed C39.
It does not upgrade the C39 outcome or replace X73's positive C35 fresh
confirmation authority.

Freeze X80 unchanged before admitting a later source-disjoint cohort. A fresh
gate should require this exact shape mechanism to exercise, zero TP loss versus
X79, at least one FP reduction, all full-arm checks, and every authority
invariant zero. This is scripted synthetic CARLA Development evidence, not
natural-distribution, real-world, product-default, deployment, user-benefit,
reliability, or safety evidence.
