# DTR CARLA C38 X77 fresh confirmation

Date: 2026-09-01

Decision: `DTR_CARLA_C38_X77_MECHANISM_NOT_EXERCISED`

## Frozen identity

- Cohort: `DTR_CARLA_C38_X77_FRESH_CONFIRMATION_V1`
- Seed: `381077`
- Render assignments: `CloudySunset / HardRainNight / WetCloudySunset /
  DustStorm`
- Frozen protocol SHA-256:
  `B11E8C0B138D075FEF9A74295AA8E4A3F730350C42F1237A453130A6838DD31D`
- Confirmation runner SHA-256:
  `CC852847A124FDF2F1C2E1A3AA95179DF86282F1BBF01F429B1D978B10ED481C`
- Unchanged X77 predictor SHA-256:
  `1F7BF820AB3048C394923E3CA7A23F10BBDB9C8813AE8C78A86D201318AEC167`
- Source result SHA-256:
  `FAD3A86EEAC5C47E85E669843F4DDBE3F02E0F596CEDFD25AE0A5B93C81F134B`
- Formal summary SHA-256:
  `88C74F48BEB67DC99CB0C0DE270EDD5395AA5D69C22779DF7F8495D3993C0D54`

## Source and execution

C38 admitted genuinely new pixels using seed `381077` and the four render
assignments above. Instance, wearable, depth, and witness capture each
completed with 728 frames at 1280x720. The join sealed 2,912 raw sensor
payloads, 6,041 evidence files, and 1,484 truth-blind model files. All source,
alignment, replay, occlusion, asset, and payload checks passed. The task-owned
CARLA process, ports, and storage leases were released after capture.

The unchanged YOLO11n-seg detector produced 4,753 candidates over all 728
frames. X24 was frozen and predicted before the single-use runner generated and
sealed X25, X76, and X77 predictions. Evaluator-bearing material was opened
only after those files were written. This was the sole scored X77 invocation on
C38 and must not be rerun as confirmation.

## Result

| Arm | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| X24 | 69 | 13 | 103 | 84.15% | 40.12% | 54.33% |
| X76 | 124 | 48 | 48 | 72.09% | 72.09% | 72.09% |
| X77 | 124 | 48 | 48 | 72.09% | 72.09% | 72.09% |

X77 recorded zero receding metric temporal-handoff release frames and changed
no classification relative to X76: `0 TP / 0 FP / 0.00 pp F1`. The frozen
incremental mechanism was therefore not exercised.

Contact recall for episodes 01/03/05/07 was
`100.00/80.00/52.17/64.58%`. Safe episodes 02/04/06/08 produced `0/1/1/3`
false-alert segments. Every required authority invariant remained zero. Recall,
safe-segment, and authority checks passed, but the per-contact floor failed on
episode 05; precision, F1, incremental FP reduction, and mechanism-exercise
checks also failed.

## Evidence identity

- C38 model manifest / candidate manifest SHA-256:
  `22892592290513B2CD881ECAEFD698903CD704D52C14E45B46CBB1ADD4C6744A` /
  `921C5F134F54D51317D7B763A84F652DCB014CE4DB7F20D15ED5F900EBFD45A0`
- X24 freeze / prediction SHA-256:
  `2A7CE573B4A79172736E262F3563EE5A2D5E6077BD14BB67EDBF247AEF0898FB` /
  `97DF209C751501DC19556D5956943FAFC987754552242CEDDEADD04187416865`
- X25 rigid / X76 / X77 prediction SHA-256:
  `2487919F4E8D434AD59E1E491E91FD49F38540DD265E98779AF6B7CBC1336714` /
  `37853D186A53AC1497C6B543301D0B52F7240BAAA162F11DAA4BB0A2EC698244` /
  `5C2A73EAD2F01A9DE08421B730978B845DDCEF01A87C6666AED888C4326455A0`

## Claim boundary and next action

C38 is neither positive nor negative incremental confirmation of X77 because
the frozen release did not occur. It does expose a full-line render-transfer
gap: the absolute X76/X77 precision and F1 floors failed and one contact episode
fell below the recall floor. X73 therefore retains the line's source-disjoint
synthetic confirmation authority from C35.

C38 may be inspected only as consumed diagnosis material. It cannot be reused
as a fresh X77 confirmation, and no tuned successor may be promoted from a C38
rescore. The next structural direction is to preserve track identity memory
while separating route-risk authority through measurement-backed existence and
uncertainty, then validate that direction on already consumed Development
cohorts before freezing any later fresh source.

C38 is not unseen-map, open-world traffic, natural-distribution, real-sensor,
Android, product-default, deployment, user-benefit, reliability, or safety
evidence.
