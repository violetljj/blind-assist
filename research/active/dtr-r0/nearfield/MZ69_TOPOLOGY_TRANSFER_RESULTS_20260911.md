# MZ69 frozen topology transfer: actual results

Frozen MZ64 GEOMETRY and MZ68 NULL transfer to all 4096 MZ67 frames, with no fit or recut. HELD DROP G 628/71/396 -> NULL 636/71/388; ALL_INVALID G 0/0/1024 -> NULL 0/0/1024 (TP/FP/FN). Run 76.231s, score 3.466s; RGB4096 once, three shared views, no dense cache. New-shape missing HEAD_NEAR ranking remains weak: NULL HELD AUC0.582977/AP0.337211; prioritize new-source training over a calibration-only explanation.

All results are consumed, curated controlled Development. All 4096 frames and original roles remain; no source role was used for fitting or calibration.

| Profile | Group | OLD_NEG TP/FP/FN | GEOMETRY TP/FP/FN | NULL TP/FP/FN |
| --- | --- | --- | --- | --- |
| IDEAL | all | 2542/185/1554 | 2801/212/1295 | 2867/205/1229 |
| IDEAL | role/TRAIN_CANDIDATE | 1240/94/808 | 1369/108/679 | 1407/104/641 |
| IDEAL | role/CALIBRATION | 635/47/389 | 700/55/324 | 713/53/311 |
| IDEAL | role/HELDOUT_GEOMETRY | 667/44/357 | 732/49/292 | 747/48/277 |
| MERGE_CLOSE | all | 2718/220/1378 | 2934/220/1162 | 3058/220/1038 |
| MERGE_CLOSE | role/TRAIN_CANDIDATE | 1325/114/723 | 1436/114/612 | 1499/114/549 |
| MERGE_CLOSE | role/CALIBRATION | 682/49/342 | 733/49/291 | 765/49/259 |
| MERGE_CLOSE | role/HELDOUT_GEOMETRY | 711/57/313 | 765/57/259 | 794/57/230 |
| DROP_CLOSE | all | 2164/280/1932 | 2447/280/1649 | 2468/280/1628 |
| DROP_CLOSE | role/TRAIN_CANDIDATE | 1057/135/991 | 1208/135/840 | 1215/135/833 |
| DROP_CLOSE | role/CALIBRATION | 545/74/479 | 611/74/413 | 617/74/407 |
| DROP_CLOSE | role/HELDOUT_GEOMETRY | 562/71/462 | 628/71/396 | 636/71/388 |
| ALL_INVALID | all | 0/0/4096 | 1/0/4095 | 1/0/4095 |
| ALL_INVALID | role/TRAIN_CANDIDATE | 0/0/2048 | 1/0/2047 | 0/0/2048 |
| ALL_INVALID | role/CALIBRATION | 0/0/1024 | 0/0/1024 | 1/0/1023 |
| ALL_INVALID | role/HELDOUT_GEOMETRY | 0/0/1024 | 0/0/1024 | 0/0/1024 |

| Profile / HELD query | G TP/FP | NULL TP/FP | TP gained/lost | FP added/removed | Gained NULL native / known-wrong / UNKNOWN |
| --- | --- | --- | --- | --- | --- |
| IDEAL/BODY_NEAR | 208/0 | 208/0 | 0/0 | 0/0 | 0/0/0 |
| IDEAL/BODY_FAR | 225/19 | 225/19 | 0/0 | 0/0 | 0/0/0 |
| IDEAL/HEAD_NEAR | 223/21 | 218/20 | 0/5 | 0/1 | 0/0/0 |
| IDEAL/HEAD_FAR | 76/9 | 96/9 | 21/1 | 0/0 | 9/0/12 |
| MERGE_CLOSE/BODY_NEAR | 208/0 | 208/0 | 0/0 | 0/0 | 0/0/0 |
| MERGE_CLOSE/BODY_FAR | 231/26 | 231/26 | 0/0 | 0/0 | 0/0/0 |
| MERGE_CLOSE/HEAD_NEAR | 234/12 | 232/12 | 0/2 | 0/0 | 0/0/0 |
| MERGE_CLOSE/HEAD_FAR | 92/19 | 123/19 | 31/0 | 0/0 | 18/0/13 |
| DROP_CLOSE/BODY_NEAR | 186/0 | 182/0 | 0/4 | 0/0 | 0/0/0 |
| DROP_CLOSE/BODY_FAR | 156/37 | 156/37 | 0/0 | 0/0 | 0/0/0 |
| DROP_CLOSE/HEAD_NEAR | 213/15 | 205/15 | 0/8 | 0/0 | 0/0/0 |
| DROP_CLOSE/HEAD_FAR | 73/19 | 93/19 | 20/0 | 0/0 | 8/0/12 |
| ALL_INVALID/BODY_NEAR | 0/0 | 0/0 | 0/0 | 0/0 | 0/0/0 |
| ALL_INVALID/BODY_FAR | 0/0 | 0/0 | 0/0 | 0/0 | 0/0/0 |
| ALL_INVALID/HEAD_NEAR | 0/0 | 0/0 | 0/0 | 0/0 | 0/0/0 |
| ALL_INVALID/HEAD_FAR | 0/0 | 0/0 | 0/0 | 0/0 | 0/0/0 |

The following flags describe native transfer and false-bit cost separately. They do not authorize replacement or erase true-positive losses.

| Primary profile | At least one new true native winner over G | No new false bits |
| --- | --- | --- |
| DROP_CLOSE | True | True |
| ALL_INVALID | False | True |

Retain this complete transfer evaluation as a COMPONENT, with any sensitivity/error tradeoffs explicit. MZ64 remains its original challenger; the MZ68 retention failure and MZ66 projection negative control retain their original scope. This evaluation neither retrains nor promotes a model.

UNKNOWN remains [0, 0, 0, 0] query bits, 12370308 full-raster cells and 11700402 angular cells. Native is an evaluator lookup at a saved winner; it does not establish causal use of that surface. Full logits were not re-encoded to recompute argmax.

[Source results](MZ67_TOPOLOGY_SOURCE_RESULTS_20260911.md) retain the exact count/presence collision evidence. These shapes were curated in existing controlled sites. This is not proof of unseen natural categories, sensor fidelity, a trained ceiling or safe clearance.

The earlier MZ68 saved-output diagnostic found that 37 of 42 DROP HEAD_FAR gains were cutoff-only sufficient; the old-G-cut counterfactual also exposed missing-input BODY_NEAR/BODY_FAR raw-score gains. No counterfactual cutoff was adopted.

A separate saved-output MZ68 ranking check uses the same 256 positive/768 negative HELD HEAD_NEAR opportunities under ALL_INVALID. G to NULL AUC rises 0.778788 to 0.835434 and grouped-tie AP 0.611537 to 0.704470, despite zero detections at the original cuts. Ranking overlap remains: these averages do not prove a useful near-zero-FP threshold. This diagnostic adopts no cutoff and provides no native-localization evidence. Its receipt is `mz68-missing-input-coverage-20260911/ranking-diagnostic-v1/receipt.json`, SHA `2e9dc02664b4fb0d9886745f88ed147b14ad3b16e813e28809a374cd406a7b62`.

Dai et al. study a related missing-input/full-input tradeoff in audio-visual speech recognition and use modality-specific adapters for entirely missing video. This motivates a separately tested missing-state readout, but its benefit here remains unproved. [CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Dai_A_Study_of_Dropout-Induced_Modality_Bias_on_Robustness_to_Missing_CVPR_2024_paper.pdf).

Conformal Risk Control requires exchangeable monotone loss functions for its basic expected-risk guarantee. Our dependent synthetic support pairs and geometry shift do not supply that assumption; a mixed FP-plus-FN objective is generally non-monotone in one threshold. No conformal guarantee is claimed. [ICLR 2024 paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf).

[Execution and exact artifacts](MZ69_EXECUTION_20260911.md) bind every method/profile, role/family/range/support/site/relation partition, all changed event IDs, support-pair effects and independent scalar checks.

The following subsequent saved-output diagnostic uses the same known, MZ37-negative, mutually supported opportunity set for both models. It adopts no threshold. AUC gives ties half credit; AP groups tied scores. The source roles remain descriptive.

| Role / profile / query | Positive/negative opportunity | G AUC/AP | NULL AUC/AP |
| --- | --- | --- | --- |
| TRAIN_CANDIDATE/DROP_CLOSE/BODY_NEAR | 355/1536 | 0.877481/0.745043 | 0.878576/0.732619 |
| TRAIN_CANDIDATE/DROP_CLOSE/BODY_FAR | 191/1465 | 0.850047/0.543490 | 0.833851/0.478341 |
| TRAIN_CANDIDATE/DROP_CLOSE/HEAD_NEAR | 193/1508 | 0.971547/0.914322 | 0.968482/0.895991 |
| TRAIN_CANDIDATE/DROP_CLOSE/HEAD_FAR | 380/1502 | 0.885502/0.775609 | 0.853136/0.736493 |
| TRAIN_CANDIDATE/ALL_INVALID/BODY_NEAR | 512/1536 | 0.717796/0.595899 | 0.742404/0.637634 |
| TRAIN_CANDIDATE/ALL_INVALID/BODY_FAR | 512/1536 | 0.802733/0.603064 | 0.807800/0.654019 |
| TRAIN_CANDIDATE/ALL_INVALID/HEAD_NEAR | 512/1536 | 0.530059/0.275945 | 0.576191/0.319383 |
| TRAIN_CANDIDATE/ALL_INVALID/HEAD_FAR | 512/1536 | 0.593268/0.351533 | 0.581262/0.371791 |
| HELDOUT_GEOMETRY/DROP_CLOSE/BODY_NEAR | 185/768 | 0.927759/0.844051 | 0.929540/0.848970 |
| HELDOUT_GEOMETRY/DROP_CLOSE/BODY_FAR | 111/732 | 0.823007/0.580832 | 0.823254/0.558930 |
| HELDOUT_GEOMETRY/DROP_CLOSE/HEAD_NEAR | 98/753 | 0.980175/0.900644 | 0.972545/0.868933 |
| HELDOUT_GEOMETRY/DROP_CLOSE/HEAD_FAR | 190/749 | 0.872089/0.739320 | 0.840665/0.688207 |
| HELDOUT_GEOMETRY/ALL_INVALID/BODY_NEAR | 256/768 | 0.732427/0.643820 | 0.752665/0.667637 |
| HELDOUT_GEOMETRY/ALL_INVALID/BODY_FAR | 256/768 | 0.801707/0.607160 | 0.802175/0.654330 |
| HELDOUT_GEOMETRY/ALL_INVALID/HEAD_NEAR | 256/768 | 0.545817/0.294350 | 0.582977/0.337211 |
| HELDOUT_GEOMETRY/ALL_INVALID/HEAD_FAR | 256/768 | 0.594264/0.359399 | 0.577754/0.377621 |

This new-source result changes the next decision. Under ALL_INVALID, HEAD_NEAR HELD AUC/AP only rise 0.545817/0.294350 to 0.582977/0.337211 (positive prevalence 0.25), with similar weakness on the descriptive TRAIN partition. DROP HEAD_FAR has more accepted positives while ranking falls 0.872089/0.739320 to 0.840665/0.688207. Thus the new-shape gap includes poor raw separation; it cannot be explained by calibration alone. Cross-source AUC differences are descriptive, not a paired causal estimate.

Prioritize a separately declared learning comparison that actually uses MZ67 TRAIN, preserves old comparators and tests retention and weak-input behavior. Evaluate existing new-source training opportunity before another untargeted capture batch. Add shape or sensing-state coverage when the learning comparison identifies a remaining gap. No MZ67 fit or new calibration has happened in MZ69.
