# MZ68 missing-input coverage — actual results

The registered retaining gate **FAIL (6/11)**. In the tested mode this candidate is a scoped **NEGATIVE_CONTROL**; MZ64 remains a challenger and MZ66 remains a negative control.

MZ61 held ALL_INVALID TP/FP/FN changes from 43/4/981 to 114/11/910; DROP changes from 795/38/229 to 826/39/198. All values below are saved-output controlled, consumed Development results.

The sole intervention replaces zero-based step%4==3 with canonical all-zero ranges/all-false validity on both native8 and OLD8 replay inputs:384 ALL_INVALID and384 each original profile. All1,536 original frame batches, initial MZ62 CONTROL weights, original loss, Adam/seed151 and original1,256-row DROP calibration rule remain fixed. No projection, distillation or MZ67 source is used. ALL_INVALID reaches1,192/2,048 unique MZ61 TRAIN IDs; this is not complete per-ID missing exposure.

| Clause | Result |
| --- | --- |
| legacy_noncal_tp_retained | PASS |
| legacy_noncal_fp_not_increased | FAIL |
| mz48_nonfit_tp_retained | PASS |
| mz48_nonfit_fp_not_increased | PASS |
| mz55_held640_tp_retained | PASS |
| mz55_held640_fp_not_increased | FAIL |
| DROP_CLOSE_held_total_tp_retained | PASS |
| DROP_CLOSE_held_no_new_fp_bits | FAIL |
| ALL_INVALID_held_total_tp_greater | PASS |
| ALL_INVALID_held_no_new_fp_bits | FAIL |
| held_native_body_near_retained | FAIL |

TP/FP/FN totals across all four queries; ratios below are counts, not rates.

| MZ61 profile | Group | Method | TP/FP/FN |
| --- | --- | --- | --- |
| IDEAL | all | OLD_NEG/UNION | 3514/274/582 |
| IDEAL | all | MZ57/GLOBAL_ANCHOR/UNION | 3514/274/582 |
| IDEAL | all | MZ62/CONTROL/UNION | 3535/313/561 |
| IDEAL | all | MZ64/CONTROL/UNION | 3565/348/531 |
| IDEAL | all | MZ64/GEOMETRY/UNION | 3739/351/357 |
| IDEAL | all | MZ66/PROJECT/UNION | 3674/333/422 |
| IDEAL | all | MZ68/NULL_COVERAGE/candidate | 3830/325/266 |
| IDEAL | all | MZ68/NULL_COVERAGE/UNION | 3908/331/188 |
| IDEAL | HELDOUT_GEOMETRY | OLD_NEG/UNION | 865/70/159 |
| IDEAL | HELDOUT_GEOMETRY | MZ57/GLOBAL_ANCHOR/UNION | 865/70/159 |
| IDEAL | HELDOUT_GEOMETRY | MZ62/CONTROL/UNION | 871/80/153 |
| IDEAL | HELDOUT_GEOMETRY | MZ64/CONTROL/UNION | 882/94/142 |
| IDEAL | HELDOUT_GEOMETRY | MZ64/GEOMETRY/UNION | 921/91/103 |
| IDEAL | HELDOUT_GEOMETRY | MZ66/PROJECT/UNION | 908/85/116 |
| IDEAL | HELDOUT_GEOMETRY | MZ68/NULL_COVERAGE/candidate | 941/84/83 |
| IDEAL | HELDOUT_GEOMETRY | MZ68/NULL_COVERAGE/UNION | 962/85/62 |
| MERGE_CLOSE | all | OLD_NEG/UNION | 3513/118/583 |
| MERGE_CLOSE | all | MZ57/GLOBAL_ANCHOR/UNION | 3513/118/583 |
| MERGE_CLOSE | all | MZ62/CONTROL/UNION | 3521/126/575 |
| MERGE_CLOSE | all | MZ64/CONTROL/UNION | 3551/125/545 |
| MERGE_CLOSE | all | MZ64/GEOMETRY/UNION | 3816/119/280 |
| MERGE_CLOSE | all | MZ66/PROJECT/UNION | 3707/118/389 |
| MERGE_CLOSE | all | MZ68/NULL_COVERAGE/candidate | 3878/128/218 |
| MERGE_CLOSE | all | MZ68/NULL_COVERAGE/UNION | 3956/133/140 |
| MERGE_CLOSE | HELDOUT_GEOMETRY | OLD_NEG/UNION | 860/31/164 |
| MERGE_CLOSE | HELDOUT_GEOMETRY | MZ57/GLOBAL_ANCHOR/UNION | 860/31/164 |
| MERGE_CLOSE | HELDOUT_GEOMETRY | MZ62/CONTROL/UNION | 863/32/161 |
| MERGE_CLOSE | HELDOUT_GEOMETRY | MZ64/CONTROL/UNION | 874/33/150 |
| MERGE_CLOSE | HELDOUT_GEOMETRY | MZ64/GEOMETRY/UNION | 938/31/86 |
| MERGE_CLOSE | HELDOUT_GEOMETRY | MZ66/PROJECT/UNION | 914/31/110 |
| MERGE_CLOSE | HELDOUT_GEOMETRY | MZ68/NULL_COVERAGE/candidate | 948/36/76 |
| MERGE_CLOSE | HELDOUT_GEOMETRY | MZ68/NULL_COVERAGE/UNION | 971/37/53 |
| DROP_CLOSE | all | OLD_NEG/UNION | 2985/154/1111 |
| DROP_CLOSE | all | MZ57/GLOBAL_ANCHOR/UNION | 2985/154/1111 |
| DROP_CLOSE | all | MZ62/CONTROL/UNION | 3034/155/1062 |
| DROP_CLOSE | all | MZ64/CONTROL/UNION | 3068/155/1028 |
| DROP_CLOSE | all | MZ64/GEOMETRY/UNION | 3244/154/852 |
| DROP_CLOSE | all | MZ66/PROJECT/UNION | 3168/154/928 |
| DROP_CLOSE | all | MZ68/NULL_COVERAGE/candidate | 3284/156/812 |
| DROP_CLOSE | all | MZ68/NULL_COVERAGE/UNION | 3383/159/713 |
| DROP_CLOSE | HELDOUT_GEOMETRY | OLD_NEG/UNION | 740/38/284 |
| DROP_CLOSE | HELDOUT_GEOMETRY | MZ57/GLOBAL_ANCHOR/UNION | 740/38/284 |
| DROP_CLOSE | HELDOUT_GEOMETRY | MZ62/CONTROL/UNION | 756/38/268 |
| DROP_CLOSE | HELDOUT_GEOMETRY | MZ64/CONTROL/UNION | 761/39/263 |
| DROP_CLOSE | HELDOUT_GEOMETRY | MZ64/GEOMETRY/UNION | 795/38/229 |
| DROP_CLOSE | HELDOUT_GEOMETRY | MZ66/PROJECT/UNION | 775/38/249 |
| DROP_CLOSE | HELDOUT_GEOMETRY | MZ68/NULL_COVERAGE/candidate | 793/39/231 |
| DROP_CLOSE | HELDOUT_GEOMETRY | MZ68/NULL_COVERAGE/UNION | 826/39/198 |
| ALL_INVALID | all | OLD_NEG/UNION | 17/16/4079 |
| ALL_INVALID | all | MZ57/GLOBAL_ANCHOR/UNION | 17/16/4079 |
| ALL_INVALID | all | MZ62/CONTROL/UNION | 71/16/4025 |
| ALL_INVALID | all | MZ64/CONTROL/UNION | 78/16/4018 |
| ALL_INVALID | all | MZ64/GEOMETRY/UNION | 248/16/3848 |
| ALL_INVALID | all | MZ66/PROJECT/UNION | 202/16/3894 |
| ALL_INVALID | all | MZ68/NULL_COVERAGE/candidate | 539/28/3557 |
| ALL_INVALID | all | MZ68/NULL_COVERAGE/UNION | 539/29/3557 |
| ALL_INVALID | HELDOUT_GEOMETRY | OLD_NEG/UNION | 5/4/1019 |
| ALL_INVALID | HELDOUT_GEOMETRY | MZ57/GLOBAL_ANCHOR/UNION | 5/4/1019 |
| ALL_INVALID | HELDOUT_GEOMETRY | MZ62/CONTROL/UNION | 10/4/1014 |
| ALL_INVALID | HELDOUT_GEOMETRY | MZ64/CONTROL/UNION | 18/4/1006 |
| ALL_INVALID | HELDOUT_GEOMETRY | MZ64/GEOMETRY/UNION | 43/4/981 |
| ALL_INVALID | HELDOUT_GEOMETRY | MZ66/PROJECT/UNION | 36/4/988 |
| ALL_INVALID | HELDOUT_GEOMETRY | MZ68/NULL_COVERAGE/candidate | 114/11/910 |
| ALL_INVALID | HELDOUT_GEOMETRY | MZ68/NULL_COVERAGE/UNION | 114/11/910 |

Old retention groups, each retaining all three input profiles:

| Profile | Old group | Method | TP/FP/FN |
| --- | --- | --- | --- |
| IDEAL | legacy_noncal | MZ64/GEOMETRY/UNION | 2899/66/49 |
| IDEAL | legacy_noncal | MZ66/PROJECT/UNION | 2897/61/51 |
| IDEAL | legacy_noncal | MZ68/NULL_COVERAGE/UNION | 2905/64/43 |
| IDEAL | mz48_nonfit | MZ64/GEOMETRY/UNION | 752/45/240 |
| IDEAL | mz48_nonfit | MZ66/PROJECT/UNION | 740/42/252 |
| IDEAL | mz48_nonfit | MZ68/NULL_COVERAGE/UNION | 771/43/221 |
| IDEAL | mz55_held640 | MZ64/GEOMETRY/UNION | 581/98/135 |
| IDEAL | mz55_held640 | MZ66/PROJECT/UNION | 578/95/138 |
| IDEAL | mz55_held640 | MZ68/NULL_COVERAGE/UNION | 580/97/136 |
| MERGE_CLOSE | legacy_noncal | MZ64/GEOMETRY/UNION | 2886/51/62 |
| MERGE_CLOSE | legacy_noncal | MZ66/PROJECT/UNION | 2881/48/67 |
| MERGE_CLOSE | legacy_noncal | MZ68/NULL_COVERAGE/UNION | 2890/58/58 |
| MERGE_CLOSE | mz48_nonfit | MZ64/GEOMETRY/UNION | 771/24/221 |
| MERGE_CLOSE | mz48_nonfit | MZ66/PROJECT/UNION | 751/23/241 |
| MERGE_CLOSE | mz48_nonfit | MZ68/NULL_COVERAGE/UNION | 804/25/188 |
| MERGE_CLOSE | mz55_held640 | MZ64/GEOMETRY/UNION | 579/85/137 |
| MERGE_CLOSE | mz55_held640 | MZ66/PROJECT/UNION | 577/84/139 |
| MERGE_CLOSE | mz55_held640 | MZ68/NULL_COVERAGE/UNION | 585/87/131 |
| DROP_CLOSE | legacy_noncal | MZ64/GEOMETRY/UNION | 2748/62/200 |
| DROP_CLOSE | legacy_noncal | MZ66/PROJECT/UNION | 2740/59/208 |
| DROP_CLOSE | legacy_noncal | MZ68/NULL_COVERAGE/UNION | 2749/67/199 |
| DROP_CLOSE | mz48_nonfit | MZ64/GEOMETRY/UNION | 554/25/438 |
| DROP_CLOSE | mz48_nonfit | MZ66/PROJECT/UNION | 547/24/445 |
| DROP_CLOSE | mz48_nonfit | MZ68/NULL_COVERAGE/UNION | 571/24/421 |
| DROP_CLOSE | mz55_held640 | MZ64/GEOMETRY/UNION | 487/86/229 |
| DROP_CLOSE | mz55_held640 | MZ66/PROJECT/UNION | 481/84/235 |
| DROP_CLOSE | mz55_held640 | MZ68/NULL_COVERAGE/UNION | 491/88/225 |

HELD paired changes versus MZ64 GEOMETRY, preserving additions and removals separately:

| Profile | Query | TP gained | TP lost | FP added | FP removed |
| --- | --- | --- | --- | --- | --- |
| IDEAL | BODY_NEAR | 0 | 2 | 0 | 0 |
| IDEAL | BODY_FAR | 0 | 0 | 0 | 0 |
| IDEAL | HEAD_NEAR | 0 | 0 | 1 | 8 |
| IDEAL | HEAD_FAR | 43 | 0 | 1 | 0 |
| MERGE_CLOSE | BODY_NEAR | 0 | 2 | 0 | 0 |
| MERGE_CLOSE | BODY_FAR | 0 | 0 | 0 | 0 |
| MERGE_CLOSE | HEAD_NEAR | 0 | 0 | 0 | 0 |
| MERGE_CLOSE | HEAD_FAR | 36 | 1 | 6 | 0 |
| DROP_CLOSE | BODY_NEAR | 0 | 11 | 0 | 0 |
| DROP_CLOSE | BODY_FAR | 1 | 0 | 1 | 0 |
| DROP_CLOSE | HEAD_NEAR | 0 | 1 | 0 | 0 |
| DROP_CLOSE | HEAD_FAR | 42 | 0 | 0 | 0 |
| ALL_INVALID | BODY_NEAR | 10 | 6 | 0 | 0 |
| ALL_INVALID | BODY_FAR | 9 | 0 | 0 | 0 |
| ALL_INVALID | HEAD_NEAR | 0 | 0 | 0 | 0 |
| ALL_INVALID | HEAD_FAR | 58 | 0 | 7 | 0 |

HELD new true positives beyond OLD_NEG split by the new readout winner, evaluator-only:

| Profile | Query | Native winner | Known non-native | UNKNOWN winner |
| --- | --- | --- | --- | --- |
| IDEAL | BODY_NEAR | 6 | 0 | 0 |
| IDEAL | BODY_FAR | 0 | 0 | 0 |
| IDEAL | HEAD_NEAR | 5 | 0 | 0 |
| IDEAL | HEAD_FAR | 62 | 1 | 23 |
| MERGE_CLOSE | BODY_NEAR | 3 | 0 | 0 |
| MERGE_CLOSE | BODY_FAR | 0 | 0 | 0 |
| MERGE_CLOSE | HEAD_NEAR | 2 | 0 | 0 |
| MERGE_CLOSE | HEAD_FAR | 71 | 1 | 34 |
| DROP_CLOSE | BODY_NEAR | 17 | 1 | 1 |
| DROP_CLOSE | BODY_FAR | 1 | 0 | 0 |
| DROP_CLOSE | HEAD_NEAR | 2 | 1 | 1 |
| DROP_CLOSE | HEAD_FAR | 44 | 3 | 15 |
| ALL_INVALID | BODY_NEAR | 38 | 4 | 0 |
| ALL_INVALID | BODY_FAR | 9 | 0 | 0 |
| ALL_INVALID | HEAD_NEAR | 0 | 0 | 0 |
| ALL_INVALID | HEAD_FAR | 43 | 0 | 15 |

DROP held native BODY_NEAR additions beyond OLD_NEG: G30 → NULL17. Held shallow-awning audit retains 64 known BN-positive frames; new native additions = 4. The complete winner/margin/union states are retained per row; a native winner labels the saved predicted location and does not establish causal feature use.

All11 clauses and all methods remain reported even on failure. Source role/family/range/support and TRAIN/CAL/HELD breakdowns, attempted-frame UNKNOWN, margin distributions, support-pair effects and full gained/lost/false-bit IDs are in [complete results](../../../../artifacts.local/work/mz68-missing-input-coverage-20260911/score-v1/result.json) and [paired events](../../../../artifacts.local/work/mz68-missing-input-coverage-20260911/score-v1/paired-events.json).

ALL_INVALID is a missing-input boundary, not a sensor-noise simulator, detection probability, real-device range limit or clearance signal. Low-range availability counts are not accuracy and do not establish that RGB cannot recognize those frames. No natural-scene, Android, trained-ceiling or safety claim is made. This observed Development comparison is not fresh confirmation.

[Independent score receipt](../../../../artifacts.local/work/mz68-missing-input-coverage-20260911/score-v1/receipt.json) SHA `7fdbde94f6ee0cd713d47be07422a334b1a789ad9dedc41cb4834fb75df745ed`; [result](../../../../artifacts.local/work/mz68-missing-input-coverage-20260911/score-v1/result.json) SHA `819d925f2ab3e012ed56c01f9b39efe08f9632b49a0af04716f7345bf75b1a44`.
