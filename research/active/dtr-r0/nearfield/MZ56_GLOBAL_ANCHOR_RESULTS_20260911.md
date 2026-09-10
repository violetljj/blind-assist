# MZ56: global measured-context continuation

Primary DROP_CLOSE gate: **PASS**. Added nonfit BODY_NEAR TP with a native winner outside 45 degrees: GLOBAL 26, LOCAL 11. Nonfit FP GLOBAL/LOCAL: 20/20; old noncal FP: 32/36.

Pre-fit operational clarification: count nonfit BODY_NEAR true events added over MZ37 by this arm passing its existing cutoff, with a native winning cell outside the measured 45-degree field. Compare GLOBAL_ANCHOR > LOCAL_ONLY, with no higher nonfit or legacy noncal FP. Inherited MZ37 positives do not count as new-path evidence; report all final TP separately.

| DROP_CLOSE group | LOCAL TP/FP/FN | GLOBAL TP/FP/FN | SUPPRESSED TP/FP/FN | MZ54 FULL TP/FP/FN | fixed OLD_NEG union TP/FP/FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| legacy_noncal | 2710/36/238 | 2710/32/238 | 2710/33/238 | 2710/31/238 | 2722/45/226 |
| mz48_fit | 612/38/588 | 637/38/563 | 620/38/580 | 592/39/608 | 743/38/457 |
| mz48_nonfit | 394/20/598 | 410/20/582 | 404/20/588 | 384/20/608 | 463/20/529 |
| all_noncal | 3716/94/1424 | 3757/90/1383 | 3734/91/1406 | 3686/90/1454 | 3928/103/1212 |

Old noncalibration excludes DEV. Nonfit is 640 heldout-site + 384 withheld-family frames; all_noncal includes fit 1280 once and is not wholly held-out evidence.

| nonfit DROP query | LOCAL TP/FP/FN | GLOBAL TP/FP/FN | SUPPRESSED TP/FP/FN | MZ54 FULL TP/FP/FN | OLD_NEG union TP/FP/FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| BODY_NEAR | 50/0/174 | 66/0/158 | 60/0/164 | 41/0/183 | 66/0/158 |
| BODY_FAR | 146/5/110 | 146/5/110 | 146/5/110 | 146/5/110 | 178/5/78 |
| HEAD_NEAR | 145/5/111 | 145/5/111 | 145/5/111 | 145/5/111 | 165/5/91 |
| HEAD_FAR | 53/10/203 | 53/10/203 | 53/10/203 | 52/10/204 | 54/10/202 |

| profile nonfit | LOCAL TP/FP | GLOBAL TP/FP | SUPPRESSED TP/FP | MZ54 FULL TP/FP | OLD_NEG union TP/FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| IDEAL | 577/31 | 596/31 | 586/31 | 567/31 | 655/36 |
| MERGE_CLOSE | 631/18 | 648/18 | 642/18 | 622/18 | 668/21 |
| DROP_CLOSE | 394/20 | 410/20 | 404/20 | 384/20 | 463/20 |

| DROP GLOBAL versus | group | TP gained/lost | FP added/removed |
| --- | --- | ---: | ---: |
| MZ56/LOCAL_ONLY/candidate | legacy_noncal | 0/0 | 0/4 |
| MZ56/GLOBAL_SUPPRESSED/candidate | legacy_noncal | 0/0 | 1/2 |
| MZ54/FULL_RASTER/candidate | legacy_noncal | 0/0 | 1/0 |
| OLD_NEG/UNION | legacy_noncal | 0/12 | 3/16 |
| MZ56/LOCAL_ONLY/candidate | mz48_nonfit | 16/0 | 0/0 |
| MZ56/GLOBAL_SUPPRESSED/candidate | mz48_nonfit | 6/0 | 0/0 |
| MZ54/FULL_RASTER/candidate | mz48_nonfit | 26/0 | 0/0 |
| OLD_NEG/UNION | mz48_nonfit | 24/77 | 0/0 |
| MZ56/LOCAL_ONLY/candidate | mz48_fit | 30/5 | 0/0 |
| MZ56/GLOBAL_SUPPRESSED/candidate | mz48_fit | 18/1 | 0/0 |
| MZ54/FULL_RASTER/candidate | mz48_fit | 45/0 | 0/1 |
| OLD_NEG/UNION | mz48_fit | 42/148 | 0/0 |

| nonfit BODY_NEAR attribution | added TP over MZ37 | added native winner | added native outside 45 | all final native outside 45 |
| --- | ---: | ---: | ---: | ---: |
| MZ56/LOCAL_ONLY | 11 | 11 | 11 | 27 |
| MZ56/GLOBAL_ANCHOR | 27 | 26 | 26 | 44 |
| MZ56/GLOBAL_SUPPRESSED | 21 | 21 | 21 | 39 |

GLOBAL_SUPPRESSED uses the same learned GLOBAL weights and cutoff with its global vector zeroed. Its changes isolate inference use of that path; GLOBAL versus LOCAL also includes different learned responses. Global context is an average of actual valid in-field zone tokens, not a measured range for an outside object. The native label and sensor coverage are evaluator-only attribution. Other query/profile regressions remain costs.

Validation: 790 MZ54 arrays preserved exactly; 2040 previous individual/union count rows agree; 251,424 scalar known-bit checks and 92,160 native-winning lookups. Two existing cutoffs recomputed only for parity. The exact 600-step schedules, 1280 pairs, 2560-frame partition, all MZ37 positives, and MZ36 400 attempts/80 UNKNOWN remain. The run separately records 16 actual TRAIN-feature initialization checks against completed MZ54, initial and learned all-missing zeros, read-only cache identity, and 5734 uncached full RGB extractions.

Scored source: run-v2. The earlier run-v1 stopped before fit or feature extraction because an extra GPU bitwise-equality check was stricter than the registered numerical tolerance. Its directory, traceback, original runner bytes and handle-release receipt remain preserved in preflight-failure-v1. The actual run uses the same predeclared atol 2e-5 / rtol 1e-6 and retains exact initial parameter equality.

This is one fixed controlled Development continuation on consumed sources. It cannot establish hardware calibration, natural-scene generalization, clearance or default-App suitability. A failure rejects this design and budget, not every possible use of metric context.

Evidence: `F:\ba-data\blindassist-artifacts-20260805\work\mz56-global-anchor-20260911/score-v1/result.json`, `audit.json`, and final `receipt.json`. The receipt seals this report, scorer, brief, run outputs, and source/prior-score bindings.
