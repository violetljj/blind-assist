# DTR CARLA X77 consumed cross-cohort Development

Date: 2026-09-01

Decision: `DTR_CARLA_X77_CONSUMED_CROSS_COHORT_PRECISION_EFFECT_POSITIVE`

## Structural change

Across the eight consumed cohorts, all seven true-positive frames whose sole
carrier was an X24 metric temporal handoff had negative forward velocity: the
obstacle was approaching. All six false-positive frames with that carrier had
positive forward velocity: the obstacle was receding.

X77 clears route risk only when every confirmed carrier is an X24 metric
temporal handoff and its forward velocity is positive under the inherited
numeric epsilon. It adds no numeric threshold and changes no detector,
tracking, geometry, route, weather, score, distance, or duration threshold.
Approaching and stationary handoffs and every non-handoff carrier are retained.

## Eight-cohort result

All sources were consumed before X77 was designed. These are Development
results, not fresh confirmation.

| Cohort | X76 TP/FP/FN | X77 TP/FP/FN | X77 P/R/F1 | Delta vs X76 | Release frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| C26 | 136/16/37 | 136/12/37 | 91.89/78.61/84.74% | 0 TP, -4 FP, +1.04 pp F1 | 4 |
| C27 | 134/7/39 | 134/7/39 | 95.04/77.46/85.35% | neutral | 0 |
| C28 | 128/12/45 | 128/12/45 | 91.43/73.99/81.79% | neutral | 0 |
| C32 | 130/3/42 | 130/3/42 | 97.74/75.58/85.25% | neutral | 0 |
| C34 | 135/13/37 | 135/13/37 | 91.22/78.49/84.38% | neutral | 0 |
| C35 | 132/12/40 | 132/11/40 | 92.31/76.74/83.81% | 0 TP, -1 FP, +0.27 pp F1 | 1 |
| C36 | 143/25/29 | 143/25/29 | 85.12/83.14/84.12% | neutral | 0 |
| C37 | 133/24/39 | 133/23/39 | 85.26/77.33/81.10% | 0 TP, -1 FP, +0.25 pp F1 | 1 |

Pooled across the eight equal-size cohorts, X76 was
`1,071 TP / 112 FP / 308 FN` at `90.53/77.67/83.61%`.
X77 is `1,071 TP / 106 FP / 308 FN` at `90.99/77.66/83.80%`:
`0 TP / -6 FP / +0.20 pp F1`. Every released frame was a false positive.
C37 crossed the 85% precision reference floor without recall loss. Every
required authority invariant remained zero, and every contact-recall and
safe-segment constraint passed.

## Evidence identity

- X77 predictor SHA-256:
  `1F7BF820AB3048C394923E3CA7A23F10BBDB9C8813AE8C78A86D201318AEC167`
- Consumed runner SHA-256:
  `0D784D1935F5AAAB5ED9331D398C579DDB70D2B8BCC3695663EEA7751EDC282F`
- C26 summary / X77 prediction SHA-256:
  `020E1593FF45337A04FF3192843689CA90059B43B1FD33C62399678A73DDDA41` /
  `325110B20A9760AD6DE8D74E39460FA8A29294FB75763E30C6F165E6866B5231`
- C27: `0828780DDDC06586C68A66A1F448443A240C26764A65E7A6F0FEBEA9D7A8F4C0` /
  `79FDCE9E62B1F98CA5E54CA2698EA58ACF2D8EEE797A15DAE821F605311EC918`
- C28: `9E821C7325BD99838D2EF6419DBB31142075DDE4426E0D5DB349795781277865` /
  `3DA19EAA539C33AD2AF7794D9A5025F7D8A481E6368A6FE6B81AD8356A7661B2`
- C32: `71CA75DBA3E3C934799B5A2B7FF55573AE13B62EA1A62BC9A76E1E5195FDECAF` /
  `CA03A2749B7C35E001CCE6A142F19FCDC0E96F06AF415951C43E860201411235`
- C34: `3B33D001E443A5769EE6F469EECFD614E6C19640F80CF94A4379456C9A3D6698` /
  `31666573F79DA88D58273EE4471DE221E30F0975C54924FDB2BBFAC011F17C46`
- C35: `609083629D8FC3D3DC28E075568EA6BD1C28BF2356F0E1449B451AD59835DBE5` /
  `A1890828AB499FF5CA42EF6F92ACB7C7F1D3B20E9C01063DF8E747EFA418A85A`
- C36: `81175B78CC00A71CB96C5C5E568CE0F1707922D2CF6A143C7CDDA478DE267353` /
  `D86D16E339010CD2047A7F1AFCC01D848AD6FDE256776A8F924EA5199ED81FE4`
- C37: `B31023DF11EF75DAF4016414D7A3DA6220295FC2078BB1E0D6226FC820BC99AB` /
  `195B74E43235F2161CF87BE91BE6F494D5F1F54264C1D1EB5ADD88E1F69EC7F9`

## Claim boundary and next decision

X77 is the strongest current eight-cohort CARLA Development arm by pooled F1.
Its sign-based structural effect is visible in three cohorts and non-regressing
in all eight, but every cohort was consumed before design. X77 therefore does
not replace X73's C35 source-disjoint confirmation authority.

Freeze unchanged X77 before admitting any later source-disjoint CARLA cohort.
The fresh test should require the X77 mechanism to be exercised, zero TP loss
versus X76, all full-arm reference checks, and all authority invariants zero.

This is synthetic Development evidence, not real-world, natural-distribution,
product-default, deployment, user-benefit, reliability, or safety evidence.
