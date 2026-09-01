# DTR CARLA X76 consumed cross-cohort Development

Date: 2026-09-01

Decision: `DTR_CARLA_X76_CONSUMED_CROSS_COHORT_PRECISION_EFFECT_POSITIVE`

## Structural change

C37 exposed ten consecutive false-positive frames carried only by X73's
credentialed parent current-fragment hull. In every frame, the carrier's
transport state was `ZERO_SHIFT_SURFACE_SUPPORT_CARRY` with zero recorded
contradictions, while its reconstructed velocity remained nonzero. Across the
other seven consumed cohorts, all 18 true-positive parent-hull frames used a
non-zero-shift transport state.

X76 rejects only this internal motion contradiction: if every confirmed carrier
is a credentialed parent current-fragment hull, every carrier declares
zero-shift and zero contradictions, yet at least one velocity component is
nonzero under the inherited numeric epsilon, the arm clears route risk. It does
not change the detector, tracking, geometry, route, weather, score, distance, or
duration threshold. Contradicted histories, stationary hulls, non-zero-shift
hulls, and every non-hull carrier are retained.

## Eight-cohort result

All sources were consumed before X76 was designed. These are Development
results, not fresh confirmation.

| Cohort | X75 TP/FP/FN | X76 TP/FP/FN | X76 P/R/F1 | Delta vs X75 | Release frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| C26 | 136/16/37 | 136/16/37 | 89.47/78.61/83.69% | neutral | 0 |
| C27 | 134/7/39 | 134/7/39 | 95.04/77.46/85.35% | neutral | 0 |
| C28 | 128/12/45 | 128/12/45 | 91.43/73.99/81.79% | neutral | 0 |
| C32 | 130/3/42 | 130/3/42 | 97.74/75.58/85.25% | neutral | 0 |
| C34 | 135/13/37 | 135/13/37 | 91.22/78.49/84.38% | neutral | 0 |
| C35 | 132/12/40 | 132/12/40 | 91.67/76.74/83.54% | neutral | 0 |
| C36 | 143/25/29 | 143/25/29 | 85.12/83.14/84.12% | neutral | 0 |
| C37 | 133/34/39 | 133/24/39 | 84.71/77.33/80.85% | 0 TP, -10 FP, +2.38 pp F1 | 10 |

Pooled across the eight equal-size cohorts, X75 was
`1,071 TP / 122 FP / 308 FN` at `89.77/77.67/83.28%`.
X76 is `1,071 TP / 112 FP / 308 FN` at `90.53/77.67/83.61%`:
`0 TP / -10 FP / +0.33 pp F1`. Every release was a C37 false positive. The
other seven cohorts were classification-neutral. Every required authority
invariant remained zero, and every contact-recall and safe-segment constraint
passed. C37 precision rose by 5.07 points but remains 0.29 points below the 85%
reference floor.

## Evidence identity

- X76 predictor SHA-256:
  `517E50196CC85EEA55A494EA505DAB6D48F1F26235DA4B67553A8836905E7865`
- Consumed runner SHA-256:
  `51BBD97970087ED5AB37C10A8A45963D634CF1527406DF084298D36566EAB4EA`
- C26 summary / X76 prediction SHA-256:
  `6CA721B053C7E18B7AE263E362F93C50B8C0A3C17638561D1F8DDB8BC40428E9` /
  `6513672DBF9AC437B8DD9813727E8F048A934440576255C915B5E243B29AE0E5`
- C27: `8737CA046F475ABEC8C5AB851DC72E47A6892AF57C4287AB56ED661471093E71` /
  `6F5B0A6A2DBD2A6504D8B121C81521EEA4702D34760077ABA72F52C0C93707EC`
- C28: `1BE4ABCA5C49661DAAF9D7E84D6DC9D972ADED7B73D9A2B902EE6167956188D1` /
  `95EA14BAC93736E2EA3793F83BCA76D2EEC83FA29FECF77BBF7F1B0734CEBA91`
- C32: `F20EF4B881800FD135C2093D53EB758F11B4550EC3DD29E1D8FC0B479B8BBC32` /
  `CD88206C7F27443DA8D6A7D4BEA44F9D55F85EF09FA3A81A485FD940781FE7FF`
- C34: `936CDAF02805C59269FD8D52A487D76C3F8967ABABC0E3EFD70A3CEB3359663E` /
  `4DE62C8F0E8B32CA792B8D3FE29499730B6E8CC29C95D56642DE79A84EF56AB6`
- C35: `D138CDB285495CBCB77A5774EED3DA61EC30FA997FB85931703EE61D365AA6DD` /
  `F115968AA9C6E5C898EDB4F15BD83DB4C25AB9AC49BABFF1333F8E4E1B095BEB`
- C36: `CF27A0B3A0AF01F77F6BF5A1F92D02573725BE7C37EE99A70C1622AE02B6258C` /
  `6B70BBEAD85DE80772C273EDD8EFCF28CD8CD06FCF7BB4A03DC9A4121A2741F7`
- C37: `758E14D0AAA0D7EC2C90255AB51C44127DDF27430BA25DFF6929253BD53AD11D` /
  `8EB2D48D240F754EFFE2084689245CB029F259CEDD905836B72A467A42D7C72D`

## Claim boundary and next decision

X76 is the strongest current eight-cohort CARLA Development arm by pooled F1.
Its exact structural effect is visible and non-regressing across every consumed
cohort, but C37 was used to design it. X76 is therefore not fresh confirmation
and does not replace X73's C35 source-disjoint confirmation authority.

Before any new source, the remaining C37 false-positive modes may be used for
one more consumed-only structural successor. Any successor must replay all
eight cohorts without TP loss. A later fresh test must freeze the unchanged
arm before admitting pixels.

This is synthetic Development evidence, not real-world, natural-distribution,
product-default, deployment, user-benefit, reliability, or safety evidence.
