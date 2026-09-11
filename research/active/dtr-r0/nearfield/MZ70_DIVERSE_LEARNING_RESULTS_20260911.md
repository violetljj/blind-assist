# MZ70 matched diverse-source learning: actual results

MZ67 HELD ALL_INVALID: CONTROL 3/1/1021 → DIVERSE 58/0/966 (TP/FP/FN). Paired TP gained/lost 58/3; false bits added/removed 0/1. HEAD_NEAR gained native winners 0, lost native winners 0. These are separate effects and costs, not a blanket promotion gate.

Retain both trained arms as scoped CHALLENGERS, with MZ64, MZ68 and MZ66 original dispositions unchanged. Diverse exposure improves new-shape raw ranking in all four missing-input queries and actual DROP decisions, but does not produce all-missing HEAD_NEAR detections. The new-source ALL_INVALID gains include38 BODY_NEAR native winners,13 known-wrong and6 UNKNOWN, with3 native BODY_NEAR losses; the sole HEAD_FAR gain is native. DROP HEAD_NEAR gains4 native winners and loses3. Old-source exposure was halved at fixed total updates, so the old-source decline does not isolate catastrophic forgetting from reduced old training. Better global ranking is not a useful low-FP operating point or a model replacement claim. BODY_NEAR missing ranking is nearly perfect on new geometry (AUC0.999873), yet only57/256 positives pass the original DROP-calibrated cutoff; this makes sensing-state calibration a concrete remaining question alongside head representation and source balance.

Both arms start from the identical MZ64 GEOMETRY checkpoint and run 4096 updates at the same four-profile allocation, optimizer, loss, RGB normalization and original 1256-row DROP calibration rule. CONTROL presents MZ61 TRAIN twice per profile; DIVERSE presents MZ61 TRAIN and MZ67 TRAIN once each. MZ48 and selected OLD_NEG replay are shared. The matched comparison tests source allocation at fixed budget. G and NULL are retained historical references with different budgets; their differences are descriptive.

G = frozen MZ64 GEOMETRY; NULL = frozen MZ68 NULL_COVERAGE. All tables below use the final OLD_NEG OR candidate decisions unless stated otherwise. Full candidate-only costs and every original method remain in the detailed result. Consumed controlled Development; source CAL/HELD do not participate in fit or new-source calibration.

| HELD source/profile | Query | G TP/FP/FN | NULL TP/FP/FN | CONTROL TP/FP/FN | DIVERSE TP/FP/FN |
| --- | --- | --- | --- | --- | --- |
| mz67/ALL_INVALID | BODY_NEAR | 0/0/256 | 0/0/256 | 3/0/253 | 57/0/199 |
| mz67/ALL_INVALID | BODY_FAR | 0/0/256 | 0/0/256 | 0/0/256 | 0/0/256 |
| mz67/ALL_INVALID | HEAD_NEAR | 0/0/256 | 0/0/256 | 0/0/256 | 0/0/256 |
| mz67/ALL_INVALID | HEAD_FAR | 0/0/256 | 0/0/256 | 0/1/256 | 1/0/255 |
| mz67/DROP_CLOSE | BODY_NEAR | 186/0/70 | 182/0/74 | 185/0/71 | 216/0/40 |
| mz67/DROP_CLOSE | BODY_FAR | 156/37/100 | 156/37/100 | 156/37/100 | 156/37/100 |
| mz67/DROP_CLOSE | HEAD_NEAR | 213/15/43 | 205/15/51 | 221/15/35 | 223/15/33 |
| mz67/DROP_CLOSE | HEAD_FAR | 73/19/183 | 93/19/163 | 102/19/154 | 158/19/98 |
| mz61/ALL_INVALID | BODY_NEAR | 40/0/216 | 44/0/212 | 139/0/117 | 80/0/176 |
| mz61/ALL_INVALID | BODY_FAR | 0/0/256 | 9/0/247 | 3/0/253 | 3/0/253 |
| mz61/ALL_INVALID | HEAD_NEAR | 0/0/256 | 0/0/256 | 12/0/244 | 0/0/256 |
| mz61/ALL_INVALID | HEAD_FAR | 3/4/253 | 61/11/195 | 58/7/198 | 20/5/236 |
| mz61/DROP_CLOSE | BODY_NEAR | 226/0/30 | 215/0/41 | 235/0/21 | 229/0/27 |
| mz61/DROP_CLOSE | BODY_FAR | 222/10/34 | 223/11/33 | 222/10/34 | 225/10/31 |
| mz61/DROP_CLOSE | HEAD_NEAR | 240/5/16 | 239/5/17 | 243/5/13 | 239/5/17 |
| mz61/DROP_CLOSE | HEAD_FAR | 107/23/149 | 149/23/107 | 158/23/98 | 139/23/117 |

| HELD source/profile | Query | D vs C TP gain/loss | FP added/removed | Gained D native/wrong/UNKNOWN | Lost C native/wrong/UNKNOWN |
| --- | --- | --- | --- | --- | --- |
| mz67/ALL_INVALID | BODY_NEAR | 57/3 | 0/0 | 38/13/6 | 3/0/0 |
| mz67/ALL_INVALID | BODY_FAR | 0/0 | 0/0 | 0/0/0 | 0/0/0 |
| mz67/ALL_INVALID | HEAD_NEAR | 0/0 | 0/0 | 0/0/0 | 0/0/0 |
| mz67/ALL_INVALID | HEAD_FAR | 1/0 | 0/1 | 1/0/0 | 0/0/0 |
| mz67/DROP_CLOSE | BODY_NEAR | 31/0 | 0/0 | 29/1/1 | 0/0/0 |
| mz67/DROP_CLOSE | BODY_FAR | 0/0 | 0/0 | 0/0/0 | 0/0/0 |
| mz67/DROP_CLOSE | HEAD_NEAR | 6/4 | 0/0 | 4/0/2 | 3/0/1 |
| mz67/DROP_CLOSE | HEAD_FAR | 58/2 | 0/0 | 39/1/18 | 2/0/0 |
| mz61/ALL_INVALID | BODY_NEAR | 4/63 | 0/0 | 4/0/0 | 61/0/2 |
| mz61/ALL_INVALID | BODY_FAR | 1/1 | 0/0 | 1/0/0 | 1/0/0 |
| mz61/ALL_INVALID | HEAD_NEAR | 0/12 | 0/0 | 0/0/0 | 11/1/0 |
| mz61/ALL_INVALID | HEAD_FAR | 3/41 | 1/3 | 1/0/2 | 18/0/23 |
| mz61/DROP_CLOSE | BODY_NEAR | 1/7 | 0/0 | 1/0/0 | 6/1/0 |
| mz61/DROP_CLOSE | BODY_FAR | 3/0 | 0/0 | 1/0/2 | 0/0/0 |
| mz61/DROP_CLOSE | HEAD_NEAR | 0/4 | 0/0 | 0/0/0 | 0/1/3 |
| mz61/DROP_CLOSE | HEAD_FAR | 6/25 | 0/0 | 3/0/3 | 14/1/10 |

Native columns inspect saved winners on actual newly positive/lost decisions. They do not credit unchanged OLD_NEG detections to a new branch, recompute the dense-logit argmax, or establish causal use of native surfaces. Query UNKNOWN and local UNKNOWN remain separate.

| HELD source/profile | Query | CONTROL additions beyond OLD: native/wrong/UNKNOWN | DIVERSE additions beyond OLD: native/wrong/UNKNOWN |
| --- | --- | --- | --- |
| mz67/ALL_INVALID | BODY_NEAR | 3/0/0 | 38/13/6 |
| mz67/ALL_INVALID | BODY_FAR | 0/0/0 | 0/0/0 |
| mz67/ALL_INVALID | HEAD_NEAR | 0/0/0 | 0/0/0 |
| mz67/ALL_INVALID | HEAD_FAR | 0/0/0 | 1/0/0 |
| mz67/DROP_CLOSE | BODY_NEAR | 3/0/0 | 32/1/1 |
| mz67/DROP_CLOSE | BODY_FAR | 0/0/0 | 0/0/0 |
| mz67/DROP_CLOSE | HEAD_NEAR | 35/1/27 | 45/0/20 |
| mz67/DROP_CLOSE | HEAD_FAR | 24/0/12 | 57/1/34 |
| mz61/ALL_INVALID | BODY_NEAR | 131/2/4 | 73/5/0 |
| mz61/ALL_INVALID | BODY_FAR | 3/0/0 | 2/0/1 |
| mz61/ALL_INVALID | HEAD_NEAR | 11/1/0 | 0/0/0 |
| mz61/ALL_INVALID | HEAD_FAR | 25/0/30 | 6/0/11 |
| mz61/DROP_CLOSE | BODY_NEAR | 36/3/0 | 32/1/0 |
| mz61/DROP_CLOSE | BODY_FAR | 0/0/0 | 1/0/2 |
| mz61/DROP_CLOSE | HEAD_NEAR | 3/1/4 | 3/0/1 |
| mz61/DROP_CLOSE | HEAD_FAR | 44/1/26 | 25/0/27 |

Ranking is raw-score ROC AUC / grouped-tie AP on the same known, MZ37-negative, mutually supported opportunity within each comparison. No operating point is selected. Historical G/NULL eligibility and matched C/D eligibility are reported separately; no silent common-denominator assumption is made. Zero-positive or zero-negative cohorts are NOT_EVALUABLE. These global ranks do not guarantee a useful low-FP tail.

| Source/profile/role | Query | C/D positive/negative | G/NULL positive/negative | G AUC/AP | NULL AUC/AP | CONTROL AUC/AP | DIVERSE AUC/AP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mz67/ALL_INVALID/TRAIN | BODY_NEAR | 512/1536 | 512/1536 | 0.717796/0.595899 | 0.742404/0.637634 | 0.781305/0.673863 | 0.999627/0.998974 |
| mz67/ALL_INVALID/TRAIN | BODY_FAR | 512/1536 | 512/1536 | 0.802733/0.603064 | 0.807800/0.654019 | 0.871950/0.724988 | 0.995415/0.977474 |
| mz67/ALL_INVALID/TRAIN | HEAD_NEAR | 512/1536 | 512/1536 | 0.530059/0.275945 | 0.576191/0.319383 | 0.601610/0.381779 | 0.778984/0.550988 |
| mz67/ALL_INVALID/TRAIN | HEAD_FAR | 512/1536 | 512/1536 | 0.593268/0.351533 | 0.581262/0.371791 | 0.660867/0.427148 | 0.877263/0.656386 |
| mz67/ALL_INVALID/HELDOUT | BODY_NEAR | 256/768 | 256/768 | 0.732427/0.643820 | 0.752665/0.667637 | 0.794398/0.700503 | 0.999873/0.999624 |
| mz67/ALL_INVALID/HELDOUT | BODY_FAR | 256/768 | 256/768 | 0.801707/0.607160 | 0.802175/0.654330 | 0.856288/0.731663 | 0.993266/0.977516 |
| mz67/ALL_INVALID/HELDOUT | HEAD_NEAR | 256/768 | 256/768 | 0.545817/0.294350 | 0.582977/0.337211 | 0.606455/0.383314 | 0.782557/0.550310 |
| mz67/ALL_INVALID/HELDOUT | HEAD_FAR | 256/768 | 256/768 | 0.594264/0.359399 | 0.577754/0.377621 | 0.656560/0.413548 | 0.853800/0.612735 |
| mz67/DROP_CLOSE/TRAIN | BODY_NEAR | 355/1536 | 355/1536 | 0.877481/0.745043 | 0.878576/0.732619 | 0.899402/0.778326 | 0.998934/0.996022 |
| mz67/DROP_CLOSE/TRAIN | BODY_FAR | 191/1465 | 191/1465 | 0.850047/0.543490 | 0.833851/0.478341 | 0.936354/0.762837 | 0.995940/0.976066 |
| mz67/DROP_CLOSE/TRAIN | HEAD_NEAR | 193/1508 | 193/1508 | 0.971547/0.914322 | 0.968482/0.895991 | 0.977502/0.939813 | 0.985342/0.958478 |
| mz67/DROP_CLOSE/TRAIN | HEAD_FAR | 380/1502 | 380/1502 | 0.885502/0.775609 | 0.853136/0.736493 | 0.915686/0.835749 | 0.984666/0.966179 |
| mz67/DROP_CLOSE/HELDOUT | BODY_NEAR | 185/768 | 185/768 | 0.927759/0.844051 | 0.929540/0.848970 | 0.948888/0.887688 | 0.999796/0.999193 |
| mz67/DROP_CLOSE/HELDOUT | BODY_FAR | 111/732 | 111/732 | 0.823007/0.580832 | 0.823254/0.558930 | 0.923067/0.778201 | 0.994043/0.970713 |
| mz67/DROP_CLOSE/HELDOUT | HEAD_NEAR | 98/753 | 98/753 | 0.980175/0.900644 | 0.972545/0.868933 | 0.984159/0.935189 | 0.989606/0.954710 |
| mz67/DROP_CLOSE/HELDOUT | HEAD_FAR | 190/749 | 190/749 | 0.872089/0.739320 | 0.840665/0.688207 | 0.918558/0.831970 | 0.991329/0.975461 |
| mz61/ALL_INVALID/TRAIN | BODY_NEAR | 510/1536 | 510/1536 | 0.999990/0.999970 | 0.999999/0.999996 | 1.000000/1.000000 | 1.000000/1.000000 |
| mz61/ALL_INVALID/TRAIN | BODY_FAR | 512/1536 | 512/1536 | 0.981285/0.940615 | 0.985034/0.938254 | 0.999330/0.998298 | 0.993235/0.979959 |
| mz61/ALL_INVALID/TRAIN | HEAD_NEAR | 512/1536 | 512/1536 | 0.768567/0.595070 | 0.859918/0.738462 | 0.918935/0.846766 | 0.848637/0.719965 |
| mz61/ALL_INVALID/TRAIN | HEAD_FAR | 509/1530 | 509/1530 | 0.927927/0.844023 | 0.948521/0.889380 | 0.974543/0.928617 | 0.932526/0.837595 |
| mz61/ALL_INVALID/HELDOUT | BODY_NEAR | 254/768 | 254/768 | 0.999795/0.999428 | 0.999846/0.999578 | 1.000000/1.000000 | 1.000000/1.000000 |
| mz61/ALL_INVALID/HELDOUT | BODY_FAR | 256/768 | 256/768 | 0.969488/0.884675 | 0.981405/0.916734 | 0.997955/0.993701 | 0.988022/0.944749 |
| mz61/ALL_INVALID/HELDOUT | HEAD_NEAR | 256/768 | 256/768 | 0.778788/0.611537 | 0.835434/0.704470 | 0.904714/0.828696 | 0.827047/0.679395 |
| mz61/ALL_INVALID/HELDOUT | HEAD_FAR | 253/764 | 253/764 | 0.904280/0.765876 | 0.914016/0.802456 | 0.956041/0.880968 | 0.898925/0.747398 |
| mz61/DROP_CLOSE/TRAIN | BODY_NEAR | 294/1536 | 294/1536 | 0.999854/0.999263 | 0.999938/0.999675 | 0.999998/0.999988 | 0.999991/0.999954 |
| mz61/DROP_CLOSE/TRAIN | BODY_FAR | 88/1524 | 88/1524 | 0.979323/0.821907 | 0.982828/0.767191 | 0.999925/0.998840 | 0.996600/0.964973 |
| mz61/DROP_CLOSE/TRAIN | HEAD_NEAR | 38/1525 | 38/1525 | 0.988594/0.849830 | 0.984245/0.811931 | 0.994271/0.940471 | 0.995496/0.950572 |
| mz61/DROP_CLOSE/TRAIN | HEAD_FAR | 332/1490 | 332/1490 | 0.979506/0.933056 | 0.985510/0.951697 | 0.996733/0.987873 | 0.980296/0.948382 |
| mz61/DROP_CLOSE/HELDOUT | BODY_NEAR | 162/768 | 162/768 | 0.996528/0.993513 | 0.997058/0.994513 | 0.999960/0.999808 | 0.999421/0.997946 |
| mz61/DROP_CLOSE/HELDOUT | BODY_FAR | 37/758 | 37/758 | 0.981637/0.658247 | 0.981923/0.628199 | 0.999002/0.978506 | 0.982136/0.764008 |
| mz61/DROP_CLOSE/HELDOUT | HEAD_NEAR | 21/763 | 21/763 | 0.970730/0.616578 | 0.964551/0.600583 | 0.978656/0.873328 | 0.976034/0.777064 |
| mz61/DROP_CLOSE/HELDOUT | HEAD_FAR | 169/745 | 169/745 | 0.980056/0.935960 | 0.978817/0.938936 | 0.995028/0.983054 | 0.977038/0.949873 |

Non-MZ37 ranking exclusion or support differences: []. Every MZ37-positive exclusion and eligible index remains in the detailed result.

| Old profile/group | G TP/FP/FN | NULL TP/FP/FN | CONTROL TP/FP/FN | DIVERSE TP/FP/FN | D vs C TP gain/loss | FP added/removed |
| --- | --- | --- | --- | --- | --- | --- |
| IDEAL/legacy_noncal | 2899/66/49 | 2905/64/43 | 2905/70/43 | 2903/60/45 | 0/2 | 3/13 |
| IDEAL/mz48_nonfit | 752/45/240 | 771/43/221 | 783/39/209 | 792/36/200 | 18/9 | 0/3 |
| IDEAL/mz55_held640 | 581/98/135 | 580/97/136 | 579/94/137 | 592/91/124 | 13/0 | 0/3 |
| MERGE_CLOSE/legacy_noncal | 2886/51/62 | 2890/58/58 | 2889/50/59 | 2889/49/59 | 1/1 | 5/6 |
| MERGE_CLOSE/mz48_nonfit | 771/24/221 | 804/25/188 | 796/21/196 | 830/21/162 | 37/3 | 0/0 |
| MERGE_CLOSE/mz55_held640 | 579/85/137 | 585/87/131 | 578/82/138 | 597/82/119 | 20/1 | 0/0 |
| DROP_CLOSE/legacy_noncal | 2748/62/200 | 2749/67/199 | 2753/58/195 | 2747/51/201 | 6/12 | 2/9 |
| DROP_CLOSE/mz48_nonfit | 554/25/438 | 571/24/421 | 567/21/425 | 571/21/421 | 13/9 | 0/0 |
| DROP_CLOSE/mz55_held640 | 487/86/229 | 491/88/225 | 503/84/213 | 501/83/215 | 15/17 | 1/2 |

Consumed controlled sources. Native winner lookup does not recompute dense argmax or prove feature causality. AP groups exact ties; AUC gives half tie credit; no-positive/negative is NOT_EVALUABLE. Ranking uses matched common support with exclusions retained. No new cutoff, hardware, safety, promotion or trained-ceiling claim.

Complete metrics: [score-v1/result.json](../../../../artifacts.local/work/mz70-diverse-learning-20260911/score-v1/result.json). Exact changed IDs and both error directions: [score-v1/paired-events.json](../../../../artifacts.local/work/mz70-diverse-learning-20260911/score-v1/paired-events.json). The detailed result also preserves all four source profiles, all source roles/families/ranges/support contexts, support-pair changes, nine cohorts and every historical comparator, including MZ66 where inherited. No omitted profile is treated as passing.

[Execution and hashes](MZ70_EXECUTION_20260911.md) separate GPU fit/evaluation from CPU scoring. No hardware, natural-scene, deployment, safety or trained-ceiling claim is made.

A parallel old-model diagnostic, completed without reading MZ70 outcomes, separates HEAD_NEAR from HEAD_FAR-only negatives and from neither-head negatives. For frozen MZ68 NULL on MZ67 HELD ALL_INVALID, AUC is0.499619 against HF-only (256positive/256negative), and0.624657 against neither-head (256/512; BODY events are allowed). The same contrasts on MZ61 are0.753128 and0.876587. All original-cut TP and FP are0. Thus the earlier transfer failure involved both poor near/far separation and weaker head evidence; this is not an information-ceiling proof or a measurement of the newly trained MZ70 arms. Code inspection also found target height scales with front distance in the MZ67 generator, weakening angular-size cues; fixed height, lateral position and support/background still supply possible cues. [Diagnostic and exact evidence](../../../../artifacts.local/work/mz70-diverse-learning-20260911/representation-analysis-v1/report.md); receipt SHA `9ff7844622a2292a63a6c7c024f19081a00865f6b8cf5ddba66c2dea20f5f89d`.

Secondary preparation verified an idle worker and eight installed asset files on both hosts. The next4096-frame capture draft prioritizes slats, supported frames, ducts and head/body partial occlusion; bench HEAD geometry remains unapproved by source review. No next capture was launched in this experiment. The readiness check and volume/time estimates are retained in `secondary-capacity-v1`; they are not new collected data.
