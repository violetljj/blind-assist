# MZ62 complete profile coverage

Primary retaining check: **FAIL**.

| Clause | Result |
| --- | --- |
| more_native_body_near | FAIL |
| held640_tp_retained | FAIL |
| held640_fp_not_added | PASS |
| legacy_noncal_tp_retained | FAIL |
| legacy_noncal_fp_not_added | PASS |
| mz48_nonfit_tp_retained | FAIL |
| mz48_nonfit_fp_not_added | PASS |

{'control_native_body_near': 17, 'coverage_native_body_near': 6, 'held640_tp': {'CONTROL': 480, 'COVERAGE': 461}, 'held640_fp': {'CONTROL': 88, 'COVERAGE': 82}, 'legacy_noncal_tp': {'CONTROL': 2732, 'COVERAGE': 2726}, 'legacy_noncal_fp': {'CONTROL': 57, 'COVERAGE': 50}, 'mz48_nonfit_tp': {'CONTROL': 537, 'COVERAGE': 522}, 'mz48_nonfit_fp': {'CONTROL': 24, 'COVERAGE': 24}}

| DROP group | Method | TP | FP | FN | Exact |
| --- | --- | ---: | ---: | ---: | ---: |
| MZ55 held640 | OLD_NEG/UNION | 451 | 79 | 265 | 410 |
| MZ55 held640 | MZ59/DIVERSE/UNION | 454 | 81 | 262 | 411 |
| MZ55 held640 | MZ60/WITNESS/UNION | 453 | 83 | 263 | 406 |
| MZ55 held640 | MZ62/CONTROL/candidate | 474 | 67 | 242 | 419 |
| MZ55 held640 | MZ62/CONTROL/UNION | 480 | 88 | 236 | 409 |
| MZ55 held640 | MZ62/COVERAGE/candidate | 450 | 61 | 266 | 414 |
| MZ55 held640 | MZ62/COVERAGE/UNION | 461 | 82 | 255 | 408 |
| Legacy noncal | OLD_NEG/UNION | 2722 | 45 | 226 | 3175 |
| Legacy noncal | MZ59/DIVERSE/UNION | 2723 | 51 | 225 | 3170 |
| Legacy noncal | MZ60/WITNESS/UNION | 2722 | 48 | 226 | 3172 |
| Legacy noncal | MZ62/CONTROL/candidate | 2723 | 41 | 225 | 3177 |
| Legacy noncal | MZ62/CONTROL/UNION | 2732 | 57 | 216 | 3170 |
| Legacy noncal | MZ62/COVERAGE/candidate | 2715 | 34 | 233 | 3180 |
| Legacy noncal | MZ62/COVERAGE/UNION | 2726 | 50 | 222 | 3174 |
| MZ48 nonfit | OLD_NEG/UNION | 463 | 20 | 529 | 523 |
| MZ48 nonfit | MZ59/DIVERSE/UNION | 505 | 21 | 487 | 562 |
| MZ48 nonfit | MZ60/WITNESS/UNION | 518 | 25 | 474 | 569 |
| MZ48 nonfit | MZ62/CONTROL/candidate | 470 | 24 | 522 | 537 |
| MZ48 nonfit | MZ62/CONTROL/UNION | 537 | 24 | 455 | 589 |
| MZ48 nonfit | MZ62/COVERAGE/candidate | 450 | 24 | 542 | 518 |
| MZ48 nonfit | MZ62/COVERAGE/UNION | 522 | 24 | 470 | 575 |

Matched total frame presentations and budget; profile assignment/order changes together. Original loss and calibration. One complete pass per frame/profile is not a trained ceiling. No MZ61 training or hardware claim.
