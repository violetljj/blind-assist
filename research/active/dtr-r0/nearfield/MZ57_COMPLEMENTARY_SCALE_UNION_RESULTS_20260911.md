# MZ57: fixed complementary-scale union result

POSTHOC consumed Development: GLOBAL24TP gained/77lost against the retained union, and old3FP added/16removed, were known before registration. This is not fresh confirmation. All three unions below use unchanged existing decisions; no fit, new cutoff or selection.

| DROP group | retained OLD_NEG union TP/FP/FN | +LOCAL | +GLOBAL | +GLOBAL_SUPPRESSED |
|---|---:|---:|---:|---:|
| legacy_noncal | 2722/45/226 | 2722/52/226 | 2722/48/226 | 2722/49/226 |
| mz48_fit | 743/38/457 | 769/38/431 | 785/38/415 | 776/38/424 |
| mz48_nonfit | 463/20/529 | 474/20/518 | 487/20/505 | 482/20/510 |
| all_noncal | 3928/103/1212 | 3965/110/1175 | 3994/106/1146 | 3980/107/1160 |

legacy_noncal excludes old DEV; nonfit=640heldout-site+384withheld-family. all_noncal includes1280fit frames once and is not wholly heldout.

| DROP nonfit query | retained TP/FP/FN | +LOCAL | +GLOBAL | +SUPPRESSED |
|---|---:|---:|---:|---:|
| BODY_NEAR | 66/0/158 | 77/0/147 | 90/0/134 | 85/0/139 |
| BODY_FAR | 178/5/78 | 178/5/78 | 178/5/78 | 178/5/78 |
| HEAD_NEAR | 165/5/91 | 165/5/91 | 165/5/91 | 165/5/91 |
| HEAD_FAR | 54/10/202 | 54/10/202 | 54/10/202 | 54/10/202 |

| Union | descriptive utility | nonfit addedTP/FP | old addedFP vs retained / MZ37 | native newTP / outside45 |
|---|---|---:|---:|---:|
| LOCAL_ONLY | PASS | 11/0 | 7/23 | 11/11 |
| GLOBAL_ANCHOR | PASS | 24/0 | 3/19 | 23/23 |
| GLOBAL_SUPPRESSED | PASS | 19/0 | 4/20 | 19/19 |

Historical MZ50 GATED adds19oldnoncalFP over MZ37; this remains a reference, not a newly tuned gate or budget. Report all old costs even if nonfit descriptive utility passes.

| Nonfit profile | retained TP/FP | +LOCAL | +GLOBAL | +SUPPRESSED |
|---|---:|---:|---:|---:|
| IDEAL | 655/36 | 665/36 | 678/36 | 672/36 |
| MERGE_CLOSE | 668/21 | 679/21 | 692/21 | 687/21 |
| DROP_CLOSE | 463/20 | 474/20 | 487/20 | 482/20 |

GLOBAL versus SUPPRESSED unions uses the same learned GLOBAL weights/cutoff, with only the global vector suppressed. GLOBAL versus LOCAL also includes learned-weight differences. All fixed comparisons and query-wise paired changes are in result.json.

Saved-output CPU composition needs no new inference here. Deployment combines retained crop/ensemble and full-RGB anchor paths; Android latency/memory and removal of duplicate encoder work are unmeasured. CPU composition time is not deployment inference time.

Validation: 251424 scalar known-bit OR/count checks; 2400 prior metric rows equal; 790 original arrays preserved; 92160 independently checked MZ52 winners. MZ36 remains400attempts/20excluded/80UNKNOWN. No native depth, RGB, model or cutoff search was used.

Native attribution applies only to newly accepted MZ56 constituents beyond the retained union. An8x8 cell containing a native query point is not direct ToF observation outside its field. No hardware, natural-scene, calibrated-uncertainty, clearance or default-App claim follows.
