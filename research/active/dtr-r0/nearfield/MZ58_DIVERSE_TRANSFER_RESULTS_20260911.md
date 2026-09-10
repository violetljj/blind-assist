# MZ58 fixed diverse-shape transfer

All 2560 MZ55 frames, 10 methods, 3 fixed profiles. All source roles are descriptive; no new training or calibration.

| DROP_CLOSE method | TP | FP | FN | Exact frames |
| --- | ---: | ---: | ---: | ---: |
| MZ37 | 1710 | 266 | 1154 | 1569 |
| OLD_NEG/OPEN/candidate | 1727 | 270 | 1137 | 1583 |
| OLD_NEG/GATED/candidate | 1823 | 313 | 1041 | 1622 |
| OLD_NEG/UNION | 1832 | 317 | 1032 | 1629 |
| MZ56/LOCAL_ONLY/candidate | 1717 | 268 | 1147 | 1574 |
| MZ56/GLOBAL_ANCHOR/candidate | 1718 | 268 | 1146 | 1575 |
| MZ56/GLOBAL_SUPPRESSED/candidate | 1715 | 267 | 1149 | 1573 |
| MZ57/LOCAL_ONLY/UNION | 1836 | 319 | 1028 | 1631 |
| MZ57/GLOBAL_ANCHOR/UNION | 1835 | 319 | 1029 | 1630 |
| MZ57/GLOBAL_SUPPRESSED/UNION | 1834 | 318 | 1030 | 1630 |

| Fixed union versus OLD_NEG/UNION | TP gained | FP added | Gate |
| --- | ---: | ---: | --- |
| MZ57/LOCAL_ONLY/UNION | 4 | 2 | FAIL |
| MZ57/GLOBAL_ANCHOR/UNION | 3 | 2 | FAIL |
| MZ57/GLOBAL_SUPPRESSED/UNION | 2 | 1 | FAIL |

Each fixed union independently: DROP_CLOSE all 2560 TP gain > 0 and added FP = 0 versus OLD_NEG/UNION. No winner selection.

Query UNKNOWN bits: 0; fullframe UNKNOWN cells: 7,417,741; angular UNKNOWN cells: 6,422,779. Query completeness does not imply local completeness.

The result retains all roles, sites, families, relations, ranges, settings, support contexts and per-query counts. Actual native positives, including extra far events and intent mismatches, remain in their original denominators. Each changed union TP/FP has an exact frame/query, existing cutoff, native winner and sensor/crop coverage record in paired-events.json.

GLOBAL versus SUPPRESSED uses the same learned GLOBAL weights and cutoff. A native winning cell is supporting localization evidence, not proof that the network used that surface causally. Inherited MZ37 positives are excluded from new-path gain attribution.

Consumed controlled Development, no MZ55 fitting/calibration or fresh confirmation. Native winners are indexed saved argmax outputs; complete fields are not saved to re-evaluate argmax. Coverage does not establish outside-object ranging. No natural-scene, hardware, clearance or safety claim.

CPU saved-output composition adds no inference here. Deployment retains crop and full RGB feature/head paths; duplicate encoder work and Android latency/memory remain unmeasured.
