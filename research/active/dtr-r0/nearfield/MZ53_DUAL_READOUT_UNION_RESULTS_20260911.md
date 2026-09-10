# MZ53: fixed dual-readout union result

2026-09-11. One registered CPU-only, explicitly post hoc Development composition pass. Every union uses the exact existing OPEN OR GATED decisions, uniformly across frames, queries and profiles. No cutoff, fit, model inference or native access was added.

Primary practical gate: **PASS**. OLD_NEG union has 463 nonfit true events and 20 false events; it adds 16 old noncalibration false events over MZ37. The declared bounds are TP >=453, FP <=21, and old added FP <=19. This is a new composition result; MZ51's individual OLD_NEG GATED gate remains failed.

| DROP_CLOSE group | fixed MZ50 union TP/FP/FN | NEW_NEG union TP/FP/FN | OLD_NEG union TP/FP/FN |
| --- | ---: | ---: | ---: |
| legacy_noncal | 2725/50/223 | 2724/41/224 | 2722/45/226 |
| mz48_fit | 698/38/502 | 676/38/524 | 743/38/457 |
| mz48_nonfit | 455/21/537 | 443/23/549 | 463/20/529 |
| all_noncal | 3878/109/1262 | 3843/102/1297 | 3928/103/1212 |

legacy_noncal excludes original DEV and includes relation2000, distance1000, older rich44 and all400 MZ36 attempts. MZ48 nonfit is heldout-site640 plus withheld-family384. all_noncal includes fit1280 once and is not wholly held-out evidence.

| DROP_CLOSE nonfit | OPEN-only TP (native winner) | GATED-only TP (native winner) | both TP | union added TP with accepted native winner | no-gated-candidate TP |
| --- | ---: | ---: | ---: | ---: | ---: |
| MZ50 | 2 (2) | 69 (50) | 2 | 54/73 | 2 |
| NEW_NEG | 1 (1) | 60 (47) | 0 | 48/61 | 1 |
| OLD_NEG | 21 (21) | 58 (40) | 2 | 63/81 | 9 |

Native columns reuse source-bound saved winning flags; they are not new native-depth validation. A correct event is not automatically a grounded local witness. Detailed per-query exclusive counts are retained below.

| OLD_NEG nonfit query | OPEN-only TP / FP | GATED-only TP / FP | shared new TP / FP |
| --- | ---: | ---: | ---: |
| BODY_NEAR | 1/0 | 26/0 | 0/0 |
| BODY_FAR | 0/0 | 32/0 | 0/0 |
| HEAD_NEAR | 18/0 | 0/0 | 2/0 |
| HEAD_FAR | 2/0 | 0/0 | 0/0 |

| DROP_CLOSE OLD_NEG union versus | group | TP gained/lost | FP added/removed | exact frames gained/lost |
| --- | --- | ---: | ---: | ---: |
| MZ50/UNION | legacy_noncal | 4/7 | 13/18 | 21/16 |
| NEW_NEG/UNION | legacy_noncal | 5/7 | 14/10 | 14/18 |
| OLD_NEG/GATED/candidate | legacy_noncal | 2/0 | 11/0 | 2/10 |
| MZ50/UNION | mz48_nonfit | 22/14 | 0/1 | 17/6 |
| NEW_NEG/UNION | mz48_nonfit | 27/7 | 0/3 | 23/2 |
| OLD_NEG/GATED/candidate | mz48_nonfit | 21/0 | 0/0 | 14/0 |
| MZ50/UNION | mz48_fit | 50/5 | 0/0 | 39/4 |
| NEW_NEG/UNION | mz48_fit | 70/3 | 0/0 | 53/2 |
| OLD_NEG/GATED/candidate | mz48_fit | 56/0 | 0/0 | 43/0 |

| profile | fixed MZ50 union nonfit TP/FP | NEW_NEG union nonfit TP/FP | OLD_NEG union nonfit TP/FP |
| --- | ---: | ---: | ---: |
| IDEAL | 677/34 | 668/37 | 655/36 |
| MERGE_CLOSE | 673/21 | 666/21 | 668/21 |
| DROP_CLOSE | 455/21 | 443/23 | 463/20 |

Independent scalar OR truth-table audit passed for 251,424 known event checks across 54 cohort/profile/union combinations. 1,320 individual count rows exactly reproduce sealed MZ51 scoring; 262 original MZ50 arrays match. All MZ37 positives, 1280 source pairs, MZ36's400 attempts and80 UNKNOWN bits remain intact.

Interpretation: the matched unions measure composition and negative-source coverage separately from a generic OR benefit. Any retained gain is a challenger result on these consumed controlled sources. A remaining false-alert cost prevents a no-cost or default-promotion claim. The experiment supplies no fresh-confirmation, natural-scene, calibrated-hardware or safety evidence.

Evidence: `F:\ba-data\blindassist-artifacts-20260805\work\mz53-dual-readout-union-20260911/score-v1/result.json`, `audit.json`, and the final `receipt.json`. The receipt binds this report, the brief, implementation, source receipts and saved outputs. No source data, previous score, cutoff or MZ51 gate was changed.
