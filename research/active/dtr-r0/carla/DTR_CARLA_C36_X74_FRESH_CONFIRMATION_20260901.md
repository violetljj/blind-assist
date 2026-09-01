# DTR CARLA C36 X74 fresh confirmation

Date: 2026-09-01

Decision: `DTR_CARLA_C36_X74_MECHANISM_NOT_EXERCISED`

## Frozen identity

- Cohort: `DTR_CARLA_C36_X74_FRESH_CONFIRMATION_V1`
- Seed: `361074`
- Frozen protocol SHA-256:
  `4EF6BE6B4E1625556A4F78E4DFBC8F981E5C89DC1EB73EDA0C991F7DFECD83EA`
- Confirmation runner SHA-256:
  `FFCA914FA9F2BD75FDF001C177D98DECE2AE31AC61B8CCD648E6044E5C575DEE`
- Unchanged X74 predictor SHA-256:
  `52558F7999258B4966C43A6473793E364D170111C90BD41BAC3FDE55F033289E`
- Source result SHA-256:
  `3877F53C33D65FB86C430DD7E927AD991045B9FDD7F4558FDE5CDD2342CB236C`
- Formal summary SHA-256:
  `7A1060660FD5EC465BAD1409BEED14EE48E26E6044BADDEE43E6815AC8A899BA`

## Source and execution

C36 admitted new pixels under seed `361074` and render assignments
`ClearSunset / HardRainSunset / WetCloudyNoon / SoftRainNoon`. Instance,
wearable, depth, and witness capture each completed with 728 PNGs at 1280x720.
All source checks passed, 73 actual blueprint types were present, and no CARLA
or capture process remained after the storage lease was released.

The frozen detector produced 5,289 candidates over all 728 frames. X24 was
frozen and predicted before the confirmation runner sealed X25, X73, and X74
predictions. Evaluator-bearing material was opened only after all three files
were written. This was the sole scored X74 invocation on C36 and must not be
rerun.

## Result

| Arm | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| X24 | 74 | 16 | 98 | 82.22% | 43.02% | 56.49% |
| X73 | 143 | 44 | 29 | 76.47% | 83.14% | 79.67% |
| X74 | 143 | 44 | 29 | 76.47% | 83.14% | 79.67% |

X74 changed no scored frame: `0 TP / 0 FP / 0.00 pp F1` versus X73. There
were zero class-contradiction release frames and zero contradicted tracks, so
the preregistered incremental mechanism requirement was not exercised. X73
parent-hull reconstruction was also not exercised on C36.

Contact recall for episodes 01/03/05/07 was
`100.00/84.44/82.61/70.83%`. Safe episodes 02/04/06/08 produced `0/1/3/3`
false-alert segments, seven total. Every required authority invariant remained
zero. Recall, F1, contact, safe-segment, and authority checks passed; precision
was below the frozen 85% floor. The required X74 FP reduction did not occur.

## Evidence identity

- C36 model manifest SHA-256:
  `A15B1CC26B3B6FF1F16A0293ED7CD1EDD4B7C7BB4F13A16A9F716D5429931585`
- X24 freeze / prediction SHA-256:
  `18E435A611686E6B42D2C9E26BA3E8E187CB6ABCF1576489CC039727E631F69E` /
  `665DB3546561EEFDFA82F6B107BEF6734E33E1A9FC18E9E35FFBAF3C5472BC1E`
- X25 rigid / X73 / X74 prediction SHA-256:
  `BD63AB0A08A89923E90DDD954A0B5ED4C83ED4C9C732D1883538448E255817B1` /
  `4351C1E0B36B4DEA97FBF79D4853663985E8DB00458EA8180DE8D99A02C7AE56` /
  `BC56C7A217D9F78D742E7CFCAA6097924C96927FB721E077AD7AE2915B6FE1FC`

## Claim boundary and next action

C36 does not confirm or falsify the incremental X74 class-contradiction
mechanism because the frozen trigger never occurred. It does provide
source-disjoint synthetic evidence that the unchanged X74 composition is
classification-identical to X73 on this source, with strong recall but poor
precision. The 44 false-positive frames are now consumed diagnosis material
for a successor; C36 cannot be reused as fresh confirmation.

X73 retains its positive C35 source-disjoint confirmation. X74 remains a
consumed Development candidate with a narrow C35 precision effect and no fresh
mechanism confirmation. A successor should target C36's render-domain false
alerts without losing its 143 true positives, then replay all consumed cohorts
before any new-source claim.

C36 is not unseen-map, open-world traffic, natural-distribution, real-sensor,
Android, product-default, deployment, user-benefit, reliability, or safety
evidence.
