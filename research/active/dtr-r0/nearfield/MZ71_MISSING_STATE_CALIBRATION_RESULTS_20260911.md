# MZ71 missing-state calibration: actual results

MZ67 HELD ALL_INVALID CONTROL: original cut 3/1/1021 → state cut 0/1/1024 (TP/FP/FN); TP gained/lost 0/3, false bits added/removed 0/0.
MZ67 HELD ALL_INVALID DIVERSE: original cut 58/0/966 → state cut 30/1/994 (TP/FP/FN); TP gained/lost 1/29, false bits added/removed 1/0.

Retain the original MZ70 cuts and both trained arms; record these two missing-state cutoff vectors as NEGATIVE_CONTROL for the proposed replacement role, without denying their query-specific tradeoffs. DIVERSE loses29 new-source BODY_NEAR true bits (26 native,2 known-wrong,1 UNKNOWN), gains one native HEAD_FAR bit and adds one HEAD_FAR false bit with an UNKNOWN winner. On old MZ61 HELD ALL_INVALID, DIVERSE103TP/5FP becomes72/7:53 near-body losses versus13 far-body,1 near-head and8 far-head gains; the near-head gain is known-wrong and7 far-head gains are locally UNKNOWN. CONTROL gains58 old-source BODY_FAR bits (54 native) but loses120 BODY_NEAR,9 HEAD_NEAR and6 HEAD_FAR. All original-profile changes occur only on naturally missing rows, including DIVERSE DROP legacy2747/51 to2746/55, MZ48 nonfit571/21 to570/21 and MZ55 held501/83 to493/83. The missing-state rule raises DIVERSE BODY_NEAR cutoff4.069206 to4.831364 while lowering its HEAD_NEAR cutoff4.233131 to3.043964 without recovering new-source HEAD_NEAR. Thus matching the calibration sensing state alone does not solve the tested problem. All eight cutoff maxima come from original DEV negative rows; excellent within-MZ67 BODY_NEAR AUC does not establish cross-source negative-tail separation. No raw score, weight or nonmissing decision changes, and no real-sensor or safety claim follows.

Frozen MZ70 heads, original-cut versus missing-state calibration on the same original1256 calibration IDs under ALL_INVALID. At application time missing means no valid packet slot, regardless of profile name. Every nonmissing row preserves the original decision bytes; all historical arrays remain untouched. No fit, ranking, selected operating-point sweep or overall promotion gate.

A missing row has no valid measured slot in its actual packet. New state cutoffs apply to such rows under every profile, including naturally missing IDEAL/MERGE/DROP rows. Other rows retain original decision bytes. This is a same-model calibration contrast, not raw learning or a new representation. No overall promotion gate or model-selection conclusion is inferred.

Each model uses the same original DEV1000 + MZ48 CAL256 identities, now evaluated under ALL_INVALID, with the unchanged scalar zero-added rule. MZ55/MZ61/MZ67 CAL or HELD do not set the cuts. All decisions in the tables are final OLD_NEG OR candidate; detailed outputs also retain candidate-only costs.

| Query | CONTROL original/state cut | DIVERSE original/state cut |
| --- | --- | --- |
| BODY_NEAR | 4.25687313/6.1407814 | 4.06920576/4.83136415 |
| BODY_FAR | 6.67741108/4.37822628 | 5.22085333/4.16248798 |
| HEAD_NEAR | 2.99686217/3.21121955 | 4.23313093/3.04396439 |
| HEAD_FAR | 4.12095499/4.2980504 | 2.95513535/2.65751219 |

| HELD source/profile | Query | C old TP/FP/FN | C state TP/FP/FN | D old TP/FP/FN | D state TP/FP/FN | C gained native/wrong/UNKNOWN | D gained native/wrong/UNKNOWN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mz67/ALL_INVALID | BODY_NEAR | 3/0/253 | 0/0/256 | 57/0/199 | 28/0/228 | 0/0/0 | 0/0/0 |
| mz67/ALL_INVALID | BODY_FAR | 0/0/256 | 0/0/256 | 0/0/256 | 0/0/256 | 0/0/0 | 0/0/0 |
| mz67/ALL_INVALID | HEAD_NEAR | 0/0/256 | 0/0/256 | 0/0/256 | 0/0/256 | 0/0/0 | 0/0/0 |
| mz67/ALL_INVALID | HEAD_FAR | 0/1/256 | 0/1/256 | 1/0/255 | 2/1/254 | 0/0/0 | 1/0/0 |
| mz67/DROP_CLOSE | BODY_NEAR | 185/0/71 | 185/0/71 | 216/0/40 | 214/0/42 | 0/0/0 | 0/0/0 |
| mz67/DROP_CLOSE | BODY_FAR | 156/37/100 | 156/37/100 | 156/37/100 | 156/37/100 | 0/0/0 | 0/0/0 |
| mz67/DROP_CLOSE | HEAD_NEAR | 221/15/35 | 221/15/35 | 223/15/33 | 223/15/33 | 0/0/0 | 0/0/0 |
| mz67/DROP_CLOSE | HEAD_FAR | 102/19/154 | 102/19/154 | 158/19/98 | 158/19/98 | 0/0/0 | 0/0/0 |
| mz61/ALL_INVALID | BODY_NEAR | 139/0/117 | 19/0/237 | 80/0/176 | 27/0/229 | 0/0/0 | 0/0/0 |
| mz61/ALL_INVALID | BODY_FAR | 3/0/253 | 61/0/195 | 3/0/253 | 16/2/240 | 54/0/4 | 9/0/4 |
| mz61/ALL_INVALID | HEAD_NEAR | 12/0/244 | 3/0/253 | 0/0/256 | 1/0/255 | 0/0/0 | 0/1/0 |
| mz61/ALL_INVALID | HEAD_FAR | 58/7/198 | 52/6/204 | 20/5/236 | 28/5/228 | 0/0/0 | 1/0/7 |
| mz61/DROP_CLOSE | BODY_NEAR | 235/0/21 | 234/0/22 | 229/0/27 | 228/0/28 | 0/0/0 | 0/0/0 |
| mz61/DROP_CLOSE | BODY_FAR | 222/10/34 | 222/10/34 | 225/10/31 | 225/10/31 | 0/0/0 | 0/0/0 |
| mz61/DROP_CLOSE | HEAD_NEAR | 243/5/13 | 243/5/13 | 239/5/17 | 239/5/17 | 0/0/0 | 0/0/0 |
| mz61/DROP_CLOSE | HEAD_FAR | 158/23/98 | 158/23/98 | 139/23/117 | 139/23/117 | 0/0/0 | 0/0/0 |

| HELD source/profile | Query | C TP gain/loss; FP added/removed | D TP gain/loss; FP added/removed | C lost native/wrong/UNKNOWN | D lost native/wrong/UNKNOWN |
| --- | --- | --- | --- | --- | --- |
| mz67/ALL_INVALID | BODY_NEAR | 0/3; 0/0 | 0/29; 0/0 | 3/0/0 | 26/2/1 |
| mz67/ALL_INVALID | BODY_FAR | 0/0; 0/0 | 0/0; 0/0 | 0/0/0 | 0/0/0 |
| mz67/ALL_INVALID | HEAD_NEAR | 0/0; 0/0 | 0/0; 0/0 | 0/0/0 | 0/0/0 |
| mz67/ALL_INVALID | HEAD_FAR | 0/0; 0/0 | 1/0; 1/0 | 0/0/0 | 0/0/0 |
| mz67/DROP_CLOSE | BODY_NEAR | 0/0; 0/0 | 0/2; 0/0 | 0/0/0 | 2/0/0 |
| mz67/DROP_CLOSE | BODY_FAR | 0/0; 0/0 | 0/0; 0/0 | 0/0/0 | 0/0/0 |
| mz67/DROP_CLOSE | HEAD_NEAR | 0/0; 0/0 | 0/0; 0/0 | 0/0/0 | 0/0/0 |
| mz67/DROP_CLOSE | HEAD_FAR | 0/0; 0/0 | 0/0; 0/0 | 0/0/0 | 0/0/0 |
| mz61/ALL_INVALID | BODY_NEAR | 0/120; 0/0 | 0/53; 0/0 | 114/2/4 | 52/1/0 |
| mz61/ALL_INVALID | BODY_FAR | 58/0; 0/0 | 13/0; 2/0 | 0/0/0 | 0/0/0 |
| mz61/ALL_INVALID | HEAD_NEAR | 0/9; 0/0 | 1/0; 0/0 | 9/0/0 | 0/0/0 |
| mz61/ALL_INVALID | HEAD_FAR | 0/6; 0/1 | 8/0; 0/0 | 3/0/3 | 0/0/0 |
| mz61/DROP_CLOSE | BODY_NEAR | 0/1; 0/0 | 0/1; 0/0 | 1/0/0 | 1/0/0 |
| mz61/DROP_CLOSE | BODY_FAR | 0/0; 0/0 | 0/0; 0/0 | 0/0/0 | 0/0/0 |
| mz61/DROP_CLOSE | HEAD_NEAR | 0/0; 0/0 | 0/0; 0/0 | 0/0/0 | 0/0/0 |
| mz61/DROP_CLOSE | HEAD_FAR | 0/0; 0/0 | 0/0; 0/0 | 0/0/0 | 0/0/0 |

Native partitions inspect the same saved winner before and after the cut change. An unchanged OLD_NEG detection is not a newly recovered branch detection. Known-wrong and UNKNOWN false-alert winners, all per-family/role/range/support partitions, and exact changed event IDs remain in the detailed result. Native lookup does not establish causal use or independently recompute dense argmax.

| Original condition | Missing/admitted-known/attempted frames | C natural-missing TP gain/loss; FP added/removed | D natural-missing TP gain/loss; FP added/removed |
| --- | --- | --- | --- |
| DEV/IDEAL | 70/1000/1000 | 0/0; 0/0 | 0/0; 0/0 |
| relation10000/IDEAL | 184/2000/2000 | 0/0; 0/1 | 0/0; 0/0 |
| distance5000/IDEAL | 0/1000/1000 | 0/0; 0/0 | 0/0; 0/0 |
| rich/IDEAL | 4/44/44 | 0/2; 0/0 | 0/1; 0/0 |
| mz36/IDEAL | 34/380/400 | 0/0; 2/0 | 0/0; 2/0 |
| mz48/IDEAL | 204/2560/2560 | 0/13; 0/0 | 0/6; 0/0 |
| mz55/IDEAL | 0/2560/2560 | 0/0; 0/0 | 0/0; 0/0 |
| mz61/IDEAL | 15/4096/4096 | 0/3; 0/0 | 0/3; 0/0 |
| mz67/IDEAL | 53/4096/4096 | 0/0; 0/0 | 0/10; 0/0 |
| DEV/MERGE_CLOSE | 70/1000/1000 | 0/0; 0/0 | 0/0; 0/0 |
| relation10000/MERGE_CLOSE | 184/2000/2000 | 0/0; 0/1 | 0/0; 0/0 |
| distance5000/MERGE_CLOSE | 0/1000/1000 | 0/0; 0/0 | 0/0; 0/0 |
| rich/MERGE_CLOSE | 4/44/44 | 0/2; 0/0 | 0/1; 0/0 |
| mz36/MERGE_CLOSE | 34/380/400 | 0/0; 2/0 | 0/0; 2/0 |
| mz48/MERGE_CLOSE | 204/2560/2560 | 0/13; 0/0 | 0/6; 0/0 |
| mz55/MERGE_CLOSE | 0/2560/2560 | 0/0; 0/0 | 0/0; 0/0 |
| mz61/MERGE_CLOSE | 15/4096/4096 | 0/3; 0/0 | 0/3; 0/0 |
| mz67/MERGE_CLOSE | 53/4096/4096 | 0/0; 0/0 | 0/10; 0/0 |
| DEV/DROP_CLOSE | 101/1000/1000 | 0/1; 0/0 | 0/1; 0/0 |
| relation10000/DROP_CLOSE | 227/2000/2000 | 0/0; 0/1 | 0/0; 2/0 |
| distance5000/DROP_CLOSE | 0/1000/1000 | 0/0; 0/0 | 0/0; 0/0 |
| rich/DROP_CLOSE | 9/44/44 | 0/2; 0/0 | 0/1; 0/0 |
| mz36/DROP_CLOSE | 41/380/400 | 0/0; 2/0 | 0/0; 2/0 |
| mz48/DROP_CLOSE | 396/2560/2560 | 4/20; 0/0 | 5/7; 0/0 |
| mz55/DROP_CLOSE | 73/2560/2560 | 0/34; 0/0 | 1/25; 0/0 |
| mz61/DROP_CLOSE | 102/4096/4096 | 2/19; 0/0 | 0/8; 0/0 |
| mz67/DROP_CLOSE | 231/4096/4096 | 0/0; 0/0 | 1/12; 0/0 |

All changes in the preceding table are confined to missing rows; a zero net FP difference does not erase added and removed false bits. MZ36 includes all400 attempts, of which380 are admitted; its20 excluded frames retain80 query-UNKNOWN bits and are not counted as nonmissing measurements.

| Old profile/group | C old TP/FP/FN | C state TP/FP/FN | D old TP/FP/FN | D state TP/FP/FN | C TP gain/loss; FP added/removed | D TP gain/loss; FP added/removed |
| --- | --- | --- | --- | --- | --- | --- |
| IDEAL/legacy_noncal | 2905/70/43 | 2903/71/45 | 2903/60/45 | 2902/62/46 | 0/2; 2/1 | 0/1; 2/0 |
| IDEAL/mz48_nonfit | 783/39/209 | 781/39/211 | 792/36/200 | 791/36/201 | 0/2; 0/0 | 0/1; 0/0 |
| IDEAL/mz55_held640 | 579/94/137 | 579/94/137 | 592/91/124 | 592/91/124 | 0/0; 0/0 | 0/0; 0/0 |
| MERGE_CLOSE/legacy_noncal | 2889/50/59 | 2887/51/61 | 2889/49/59 | 2888/51/60 | 0/2; 2/1 | 0/1; 2/0 |
| MERGE_CLOSE/mz48_nonfit | 796/21/196 | 794/21/198 | 830/21/162 | 829/21/163 | 0/2; 0/0 | 0/1; 0/0 |
| MERGE_CLOSE/mz55_held640 | 578/82/138 | 578/82/138 | 597/82/119 | 597/82/119 | 0/0; 0/0 | 0/0; 0/0 |
| DROP_CLOSE/legacy_noncal | 2753/58/195 | 2751/59/197 | 2747/51/201 | 2746/55/202 | 0/2; 2/1 | 0/1; 4/0 |
| DROP_CLOSE/mz48_nonfit | 567/21/425 | 563/21/429 | 571/21/421 | 570/21/422 | 0/4; 0/0 | 0/1; 0/0 |
| DROP_CLOSE/mz55_held640 | 503/84/213 | 493/84/223 | 501/83/215 | 493/83/223 | 0/10; 0/0 | 0/8; 0/0 |
| ALL_INVALID/legacy_noncal | 697/68/2251 | 689/29/2259 | 673/27/2275 | 700/32/2248 | 28/36; 7/46 | 37/10; 13/8 |
| ALL_INVALID/mz48_nonfit | 178/2/814 | 197/0/795 | 193/0/799 | 212/2/780 | 45/26; 0/2 | 33/14; 2/0 |
| ALL_INVALID/mz55_held640 | 162/6/554 | 126/4/590 | 131/3/585 | 126/3/590 | 19/55; 0/2 | 28/33; 0/0 |

All 3363 original MZ70 arrays are byte-preserved. Fixed weights, each same-arm raw/support/winner array and all nonmissing decision bytes are unchanged. This statement does not imply that old-cohort ALL_INVALID inference existed before MZ71. Local UNKNOWN cells: `{"mz48":7562077,"mz55":7417741,"mz61":12188903,"mz67":12370308}`.

All rankings and raw winners are unchanged within each same-arm calibration comparison. No new model training or model selection; no hardware, sensor-fidelity, natural-scene, safe-clearance, dense-argmax or trained-ceiling claim. Missing-state routing uses packet validity, including naturally missing original-profile rows; profiles are not predictor inputs.

Complete results and all historical methods: [score-v1/result.json](../../../../artifacts.local/work/mz71-missing-state-calibration-20260911/score-v1/result.json). Exact exchanges: [score-v1/paired-events.json](../../../../artifacts.local/work/mz71-missing-state-calibration-20260911/score-v1/paired-events.json). [Execution and bindings](MZ71_EXECUTION_20260911.md). Root may add scoped interpretation after reviewing these unselected effects; this renderer assigns no inheritance or promotion.

Parallel CPU/worker preparation produced a4,096-row source draft:3,072 fixed-physical-size frames and1,024 angular-size controls, with wood slats, grounded stair frames, supported ducts and Birch_h tree/branch geometry. Roles2048/1024/1024 and64 fixed canary rows are declared. Both-host installed asset checks passed, but no execution spec or capture was launched here. Birch material evidence is opaque bark/tree geometry, not verified transparent foliage. New placement visibility and credible support require the fixed canary; source intent must permit additional actual native event bits. [Preparation and limits](../../../../artifacts.local/work/mz71-missing-state-calibration-20260911/next-source-design-v1/REPORT.md); preparation receipt SHA `b2e03374fd7ed6c32248b3777f68347fee9581ecb0734e31f5c1fb31040fc61d`.
