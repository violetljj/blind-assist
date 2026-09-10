# MZ60 native positive-query objective

Primary retaining check: **FAIL**.

| Clause | Result |
| --- | --- |
| native_body_near_exceeds_one | FAIL |
| heldout_tp_at_least_454 | FAIL |
| heldout_fp_at_most_79 | FAIL |
| legacy_tp_at_least_2722 | PASS |
| legacy_fp_at_most_45 | FAIL |
| mz48_nonfit_tp_at_least_514 | PASS |
| mz48_nonfit_fp_at_most_23 | FAIL |

Values: {'native_new_family_heldout_body_near': 1, 'heldout_tp': 453, 'heldout_fp': 83, 'legacy_tp': 2722, 'legacy_fp': 48, 'nonfit_mz48_tp': 518, 'nonfit_mz48_fp': 25}

| DROP group | Method | TP | FP | FN | Exact |
| --- | --- | ---: | ---: | ---: | ---: |
| MZ55 all | OLD_NEG/UNION | 1832 | 317 | 1032 | 1629 |
| MZ55 all | MZ59/CONTROL/UNION | 1836 | 319 | 1028 | 1628 |
| MZ55 all | MZ59/DIVERSE/UNION | 1852 | 322 | 1012 | 1635 |
| MZ55 all | MZ60/WITNESS/candidate | 1739 | 285 | 1125 | 1567 |
| MZ55 all | MZ60/WITNESS/UNION | 1847 | 336 | 1017 | 1613 |
| MZ55 heldout640 | OLD_NEG/UNION | 451 | 79 | 265 | 410 |
| MZ55 heldout640 | MZ59/CONTROL/UNION | 451 | 79 | 265 | 410 |
| MZ55 heldout640 | MZ59/DIVERSE/UNION | 454 | 81 | 262 | 411 |
| MZ55 heldout640 | MZ60/WITNESS/candidate | 440 | 62 | 276 | 410 |
| MZ55 heldout640 | MZ60/WITNESS/UNION | 453 | 83 | 263 | 406 |
| Legacy noncal | OLD_NEG/UNION | 2722 | 45 | 226 | 3175 |
| Legacy noncal | MZ59/CONTROL/UNION | 2722 | 45 | 226 | 3175 |
| Legacy noncal | MZ59/DIVERSE/UNION | 2723 | 51 | 225 | 3170 |
| Legacy noncal | MZ60/WITNESS/candidate | 2710 | 32 | 238 | 3177 |
| Legacy noncal | MZ60/WITNESS/UNION | 2722 | 48 | 226 | 3172 |
| MZ48 nonfit | OLD_NEG/UNION | 463 | 20 | 529 | 523 |
| MZ48 nonfit | MZ59/CONTROL/UNION | 514 | 23 | 478 | 569 |
| MZ48 nonfit | MZ59/DIVERSE/UNION | 505 | 21 | 487 | 562 |
| MZ48 nonfit | MZ60/WITNESS/candidate | 449 | 25 | 543 | 513 |
| MZ48 nonfit | MZ60/WITNESS/UNION | 518 | 25 | 474 | 569 |

All original arrays, groups, UNKNOWN attempts and fixed comparisons remain. Full per-query/profile/source-role/family/site/relation/context metrics, paired losses and native winner counts are in result.json.

Single positive-query-pooling change with identical source/schedule/model/initialization. No threshold selection. Native winners are supporting localization evidence, not causal proof or hardware observability. Failed retaining clauses remain failures; no default-App or safety claim.
