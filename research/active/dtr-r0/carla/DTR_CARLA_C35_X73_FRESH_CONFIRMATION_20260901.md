# DTR CARLA C35 X73 fresh confirmation

Date: 2026-09-01

Decision: `DTR_CARLA_C35_X73_GENERALIZATION_GATE_MET`

## Frozen identity

- Cohort: `DTR_CARLA_C35_X73_FRESH_CONFIRMATION_V1`
- Seed: `351073`
- Frozen protocol SHA-256:
  `53E52FC4318E0ECD4F60870E3999B878DF18CF6C84C7890D8434C043B6718A7E`
- Confirmation runner SHA-256:
  `FF8F4B333708A41AEFC0B004A7EB89880D4CC324951CDC92101CACFE66560B88`
- Unchanged X73 predictor SHA-256:
  `8722FAB54E441459EDE6E1EBE61CE1BE0FD7E8956BB2C9B139BF67E3BF51BBD2`
- Source result SHA-256:
  `8F302B7D58FA800FE1883D3E8AE3DAC9E3DE9305A92A197636530D19192B660E`
- Formal summary SHA-256:
  `5DD32A549A8214CDBD1F0EC80174277EDFD293DC948730726C4050545B30E4D2`

## Source and execution

C35 admitted genuinely new pixels under capture seed `351073` and the four
render assignments `HardRainNoon / WetSunset / CloudyNight / ClearNoon`.
Instance, wearable, depth, and witness capture each completed with 728 PNGs at
1280x720. Every source check passed, including all eight physical-occlusion
contracts, cross-sensor pose replay, RGB/depth frame alignment, sanitized model
root, blueprint diversity, and immutable issued-plan receipts.

The frozen YOLO checkpoint produced all 728 truth-blind candidate records. X24
was frozen and predicted before the confirmation runner sealed X25, X72, and
X73 predictions. Evaluator-bearing material was opened only after those files
were written successfully. This was the sole scored X73 invocation on C35 and
must not be rerun.

## Result

| Arm | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| X24 | 87 | 28 | 85 | 75.65% | 50.58% | 60.63% |
| X72 | 126 | 18 | 46 | 87.50% | 73.26% | 79.75% |
| X73 | 132 | 18 | 40 | **88.00%** | **76.74%** | **81.99%** |

X73 improved unchanged X72 by **`+6 TP / +0 FP / +2.24 pp F1`**. Parent-hull
reconstruction was exercised on six frames and produced six added
true-positive frames. The current rigid-center veto rejected four ambiguous
parent groups; five other credentialed parent hulls had no current route entry.

Contact recall for episodes 01/03/05/07 was
`87.88/84.44/63.04/75.00%`. Safe episodes 02/04/06/08 produced `0/1/4/4`
false-alert segments, nine total. Every required authority invariant remained
zero. All primary transfer checks passed.

The stretch target was not met: precision was below 92%, recall was 0.26 pp
below 77%, and F1 was below 84%. This does not change the preregistered primary
gate result.

## Evidence identity

- C35 model manifest SHA-256:
  `AB8F8ACCCB9504823F28B013DE469A8F918C22BC53C1A9470BF09876A8A4EF03`
- X24 freeze / prediction SHA-256:
  `6D92DC3F6634BB8180B26F123B92F1ED8D1E941BD16FD0167CE56C26EB3D2921` /
  `1A0B71A82E9F018232FBB6A2995B0539854A61C7BD74DEDEBC2C6E232E1D97E7`
- X25 rigid / X72 / X73 prediction SHA-256:
  `E80B2965315912EDBE45EDB899F407A524FD3168B0A42CB5CF857BCA04B9E04E` /
  `171086D5A988A1FE3E64B721A27082E858252172209F4BF5A92F6DC6D01507C3` /
  `78923A3CE3E488FB500C7946F7C1D29398C1A9BDB2561FB4720C1B37C2967BC6`

## Claim boundary and next action

C35 is positive, source-disjoint synthetic Development confirmation that the
unchanged X73 parent-hull mechanism can add collision frames without adding
false-positive frames under a new seed, weather assignment, and pixel set. It
also confirms that the composed X65-to-X73 line transfers at
`88.00/76.74/81.99%` within the same map, route layouts, detector, scripted
motion profiles, and evaluator contract.

Do not rerun or tune C35 as confirmation. It is now consumed and may be used
only for post-confirmation diagnosis. The next algorithm should target the
remaining `40 FN / 18 FP` without weakening X69 release or X73 rigid-center
containment. Any successor needs a visible multi-cohort Development effect and
a later new source before it can inherit confirmation authority.

C35 is not unseen-map, open-world traffic, natural-distribution, real-sensor,
Android, product-default, deployment, user-benefit, reliability, or safety
evidence.
