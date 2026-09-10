# MZ59 matched training coverage

Primary gate: **FAIL**.

DIVERSE versus CONTROL: more native-winning BODY_NEAR TP added over OLD_NEG/UNION on MZ55 heldout new-family480 DROP frames; no higher final OLD_NEG/UNION OR candidate FP on all MZ55 heldout640 or original legacy noncalibration. Candidate FP is reported separately.

Native new-family heldout BODY_NEAR gains: CONTROL 0, DIVERSE 1.

| DROP group | Method | TP | FP | FN | Exact |
| --- | --- | ---: | ---: | ---: | ---: |
| MZ55 all | OLD_NEG/UNION | 1832 | 317 | 1032 | 1629 |
| MZ55 all | MZ57/GLOBAL_ANCHOR/UNION | 1835 | 319 | 1029 | 1630 |
| MZ55 all | MZ59/CONTROL/candidate | 1723 | 268 | 1141 | 1577 |
| MZ55 all | MZ59/DIVERSE/candidate | 1746 | 271 | 1118 | 1591 |
| MZ55 all | MZ59/CONTROL/UNION | 1836 | 319 | 1028 | 1628 |
| MZ55 all | MZ59/DIVERSE/UNION | 1852 | 322 | 1012 | 1635 |
| MZ55 heldout640 | OLD_NEG/UNION | 451 | 79 | 265 | 410 |
| MZ55 heldout640 | MZ57/GLOBAL_ANCHOR/UNION | 453 | 79 | 263 | 412 |
| MZ55 heldout640 | MZ59/CONTROL/candidate | 436 | 58 | 280 | 412 |
| MZ55 heldout640 | MZ59/DIVERSE/candidate | 441 | 60 | 275 | 415 |
| MZ55 heldout640 | MZ59/CONTROL/UNION | 451 | 79 | 265 | 410 |
| MZ55 heldout640 | MZ59/DIVERSE/UNION | 454 | 81 | 262 | 411 |
| MZ55 new-family heldout480 | OLD_NEG/UNION | 320 | 76 | 236 | 279 |
| MZ55 new-family heldout480 | MZ57/GLOBAL_ANCHOR/UNION | 320 | 76 | 236 | 279 |
| MZ55 new-family heldout480 | MZ59/CONTROL/candidate | 305 | 56 | 251 | 281 |
| MZ55 new-family heldout480 | MZ59/DIVERSE/candidate | 309 | 57 | 247 | 284 |
| MZ55 new-family heldout480 | MZ59/CONTROL/UNION | 320 | 76 | 236 | 279 |
| MZ55 new-family heldout480 | MZ59/DIVERSE/UNION | 321 | 77 | 235 | 279 |
| Old noncal | OLD_NEG/UNION | 2722 | 45 | 226 | 3175 |
| Old noncal | MZ57/GLOBAL_ANCHOR/UNION | 2722 | 48 | 226 | 3172 |
| Old noncal | MZ59/CONTROL/candidate | 2710 | 29 | 238 | 3180 |
| Old noncal | MZ59/DIVERSE/candidate | 2711 | 35 | 237 | 3175 |
| Old noncal | MZ59/CONTROL/UNION | 2722 | 45 | 226 | 3175 |
| Old noncal | MZ59/DIVERSE/UNION | 2723 | 51 | 225 | 3170 |
| MZ48 nonfit | OLD_NEG/UNION | 463 | 20 | 529 | 523 |
| MZ48 nonfit | MZ57/GLOBAL_ANCHOR/UNION | 487 | 20 | 505 | 547 |
| MZ48 nonfit | MZ59/CONTROL/candidate | 437 | 23 | 555 | 509 |
| MZ48 nonfit | MZ59/DIVERSE/candidate | 428 | 21 | 564 | 500 |
| MZ48 nonfit | MZ59/CONTROL/UNION | 514 | 23 | 478 | 569 |
| MZ48 nonfit | MZ59/DIVERSE/UNION | 505 | 21 | 487 | 562 |

All three profiles, per-query/group counts, paired gains/losses, source support pairs and native winner events are retained in result.json/native-added-events.json. MZ55 roles remain descriptive and its calibration-named rows did not set cutoffs; MZ36 retains all400 attempts and80 UNKNOWN bits.

Matched coverage continuation. Both candidates and final fixed ORs reported; no selected winner or cutoff change. Native winning cell is localization evidence, not proof of causal use or sensor detectability. No hardware/natural-scene/clearance/safety claim.
