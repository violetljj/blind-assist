# DTR CARLA C37 X75 fresh confirmation

Date: 2026-09-01

Decision: `DTR_CARLA_C37_X75_GENERALIZATION_GATE_NOT_MET`

## Frozen identity

- Cohort: `DTR_CARLA_C37_X75_FRESH_CONFIRMATION_V1`
- Seed: `371075`
- Frozen protocol SHA-256:
  `81AE71541B2DB5C6CD7DD29A97A360D0B4D9F02E9E19E5F759D428D8EDF9BF80`
- Confirmation runner SHA-256:
  `D7E96CC9C58961C606C541F8AFF88F7487408534B7708ABA9E9C9BCB1688636C`
- Unchanged X75 predictor SHA-256:
  `2A21A794EC52ED30D15D45BE88FC5E0846735FA06B65839BEBC365ED5E992808`
- Source result SHA-256:
  `581509BCDF6AEC07CFA932449414558F020AEB25F2349A621BBF2688810BD134`
- Formal summary SHA-256:
  `EAF77CE2081F82FFE32AF04DD6FBF9959E831DEE3F794F0D07157BB79BEA602C`

## Source and execution

C37 admitted new pixels under seed `371075` and render assignments
`MidRainSunset / WetNight / ClearNight / SoftRainSunset`. Instance, wearable,
depth, and witness capture each completed with 728 PNGs at 1280x720. All source
checks passed and no CARLA or capture process remained after the storage lease
was released.

The frozen YOLO11n-seg detector produced 5,451 candidates over all 728 frames.
X24 was frozen and predicted before the confirmation runner sealed X25, X74,
and X75 predictions. Evaluator-bearing material was opened only after all three
files were written. This was the sole scored X75 invocation on C37 and must not
be rerun.

## Result

| Arm | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| X24 | 79 | 27 | 93 | 74.53% | 45.93% | 56.83% |
| X74 | 133 | 35 | 39 | 79.17% | 77.33% | 78.24% |
| X75 | 133 | 34 | 39 | 79.64% | 77.33% | 78.47% |

X75 exercised one existence-only permanence release and removed one X74 false
positive with zero TP loss: `0 TP / -1 FP / +0.23 pp F1`. Seven collision
credential births protected independently supported permanence rows.

Across consumed C26/C27/C28/C32/C34/C35/C36/C37, X75 is now
`1,071 TP / 122 FP / 308 FN` at `89.77/77.67/83.28%`, versus X74 at
`1,071 / 142 / 308` and `88.29/77.67/82.64%`. The cumulative X75 effect is
therefore `0 TP / -20 FP / +0.64 pp F1` over eight cohorts.

Contact recall for episodes 01/03/05/07 was
`100.00/75.56/63.04/77.08%`. Safe episodes 02/04/06/08 produced `0/0/0/3`
false-alert segments. Every required authority invariant remained zero. The
incremental release, delta, recall, F1, contact, safe-segment, and authority
checks passed. Full-arm precision was `79.64%`, below the preregistered 85%
floor, so the primary generalization gate did not pass.

## Evidence identity

- C37 model manifest SHA-256:
  `F68F8A427B390FC8AFF59CC34495FB26EE20002219AA7F7AEB9CB3AEE9D667ED`
- X24 freeze / prediction SHA-256:
  `27DC11708E16956CD42D91D85704424BACB0B4DFFD9851A156C3862C7022888A` /
  `CC355A6D11F84F79BB234B2243C1DEB646075B1E420E2B53CAE7E107306C6D51`
- X25 rigid / X74 / X75 prediction SHA-256:
  `032D16EC22452DE40E0D2B4D09A24F7D80EAFE6512923454178532AF77823AAF` /
  `6C80F1C5D44DC61345B7D1704EC9CC8A894973BDDBA3E0832B7B82922B3AAE51` /
  `9D249A574F4062BB744AADC740F52A3A430240EA34CE52E1595F0553A547401A`

## Claim boundary and next action

C37 is positive fresh evidence for the direction of X75's incremental effect:
the frozen mechanism was exercised and strictly reduced false positives without
losing a true positive. It is not a positive full-arm confirmation because the
predeclared precision floor failed. X75 therefore remains the strongest
eight-cohort Development arm but does not replace X73's C35 source-disjoint
confirmation authority.

C37 is now consumed diagnosis material for the remaining 34 false-positive and
39 false-negative frames. It cannot be reused as a fresh X75 confirmation or
rescored after tuning.

C37 is not unseen-map, open-world traffic, natural-distribution, real-sensor,
Android, product-default, deployment, user-benefit, reliability, or safety
evidence.
