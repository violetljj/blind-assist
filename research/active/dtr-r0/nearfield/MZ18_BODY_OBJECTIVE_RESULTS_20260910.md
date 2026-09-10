# MZ18: BODY attribution improves, pole alert coverage collapses

2026-09-10 · EXPLORE · consumed Development · one completed1200-step fit.

The BODY-specific witness objective improves actual BODY contributor ranking and reduces new BODY false alerts on placementDEV3000 from11 to3 compared with MZ16 HIGH_DETAIL. However BODY pole detections fall from25/25 clean and25/25 stressed to4/25 and1/25. The full candidate adds70 far TP and7 FP over MZ5 versus MZ16's80 and16. It fails the fixed error budget and pole retention; do not replace MZ5 or promote the candidate as an alert module.

This is useful positive evidence for the local attribution mechanism, with a failed task tradeoff. On the trained pole, all25 BODY maxima in each condition now belong to actual query contributors, versus13/25 and15/25 previously. Correct locations mostly remain below the frozen oldDEV-calibrated alert cutoff. Better location ranking is insufficient for reliable detection at that operating point.

## Intervention and scope

The user authorized this BODY-specific experiment after MZ17 exposed BODY-near/far conflict rates20.79/14.38%. MZ17's no-fit terminal and10% aggregate heuristic remain unchanged. This new experiment does not silently reopen that gate.

See [protocol](MZ18_BODY_OBJECTIVE_PROTOCOL_20260910.md). Only BODY query losses change: reward an eligible known actual witness and penalize the highest known incorrect candidate. Keep the local BCE,0.25 query coefficient, HEAD query terms and original global supervision denominator. HEAD candidate-logit gradients are identical at identical inputs, but shared trainable layers permit HEAD predictions to change. BODY receives up to two terms; effective gradient weighting changes with the target definition.

Both runs use the same native-detail224ROI features, frozen encoder,10740parameter readout, original initialization, normalization and saved1200x16 batches. There is no warm start from a final checkpoint.19200 draws cover7562 unique frames from the existing10200 pool, including the200 already trained diagnostic sequence frames. No new feature extraction, capture, EVAL or temporal module. Native contributor labels enter supervision/evaluation only; inference remains the unmasked maximum over geometric candidates. UNKNOWN is not trained as negative by the new objective.

## Task results

Order of vectors is BODY_NEAR,BODY_FAR,HEAD_NEAR,HEAD_FAR. Exact means four outputs correct per frame; TP/FP are output bits. MZ16 means its HIGH_DETAIL candidate composed with MZ5, and MZ18 also uses add-only MZ5 composition. All baseline positive judgments, including existing false alerts, remain. The model is not correcting premature HEAD_NEAR alarms.

| Cohort | Model | Exact frames | TP vector | FP vector | Added far TP | Added FP |
|---|---|---:|---|---|---:|---:|
| DEV | MZ5 | 912 | [187, 180, 183, 165] | [6, 11, 10, 7] | 0 | 0 |
| DEV | MZ16 | 943 | [196, 195, 193, 181] | [6, 11, 10, 7] | 31 | 0 |
| DEV | MZ18 | 935 | [188, 190, 192, 179] | [6, 11, 10, 7] | 24 | 0 |
| clean | MZ5 | 143 | [2, 6, 9, 0] | [5, 3, 19, 0] | 0 | 0 |
| clean | MZ16 | 176 | [9, 36, 9, 32] | [5, 3, 19, 0] | 62 | 0 |
| clean | MZ18 | 148 | [2, 12, 9, 37] | [5, 3, 19, 0] | 43 | 0 |
| stress | MZ5 | 143 | [2, 6, 9, 0] | [5, 3, 19, 0] | 0 | 0 |
| stress | MZ16 | 175 | [9, 36, 9, 31] | [5, 3, 19, 0] | 61 | 0 |
| stress | MZ18 | 144 | [2, 9, 9, 36] | [5, 3, 19, 0] | 39 | 0 |
| relation10000 | MZ5 | 1883 | [382, 357, 388, 375] | [2, 5, 26, 14] | 0 | 0 |
| relation10000 | MZ16 | 1914 | [389, 383, 400, 393] | [5, 6, 26, 19] | 44 | 9 |
| relation10000 | MZ18 | 1908 | [382, 378, 398, 390] | [2, 6, 26, 18] | 36 | 5 |
| distance5000 | MZ5 | 920 | [0, 0, 489, 446] | [0, 0, 25, 9] | 0 | 0 |
| distance5000 | MZ16 | 944 | [0, 0, 499, 482] | [7, 0, 25, 9] | 36 | 7 |
| distance5000 | MZ18 | 949 | [0, 0, 496, 480] | [0, 2, 25, 9] | 34 | 2 |

Sizes: oldDEV1000, clean200, stress200, relationDEV2000, distanceDEV1000. Dataset names do not give evaluated denominators. OldDEV supplies per-model zero-added-MZ5-FP thresholds; placementDEV does not calibrate them, but is already consumed Development. This is not independent confirmation or equal achieved test-set FPR.

Placement aggregate exact: MZ5 2803/3000, MZ16 2858/3000, MZ18 2857/3000. Relative to MZ16, MZ18 has8 fewer BODY FP but5 fewer BODY_FAR TP and7 fewer BODY_NEAR TP. HEAD has1 fewer FP,5 fewer far TP and5 fewer near TP. All7 remaining new placement false-alert winners still lack an actual source at their selected cell. In distanceDEV, false BODY_NEAR additions7 become0, while BODY_FAR additions0 become2: error reduction is not uniform across outputs.

## Pole: correct attribution is not sufficient alert strength

One trained25-frame configuration supplies25 BODY_FAR and24 HEAD_FAR opportunities. Do not count them as49 independent obstacles. HEAD pole detections remain24/24 clean and23/24 stress in both models. BODY behavior:

| Quantity | MZ16 clean | MZ18 clean | MZ16 stress | MZ18 stress |
|---|---:|---:|---:|---:|
| Actual eligible BODY contributor available | 25/25 | 25/25 | 25/25 | 25/25 |
| Highest score at actual BODY contributor | 13/25 | 25/25 | 15/25 | 25/25 |
| An actual BODY contributor exceeds alert cutoff | 25/25 | 4/25 | 25/25 | 1/25 |
| BODY detections | 25/25 | 4/25 | 25/25 | 1/25 |

BODY_FAR cutoffs are1.7779448032 for MZ16 and1.6905144453 for MZ18. The new numerical cutoff is lower; scores have changed scale/distribution too. Calling this merely a raised-threshold problem is incorrect. The saved score trace shows a margin failure at the fixed calibration rule, not proof that lowering a threshold would recover pole detections without new errors. No cutoff sweep was performed.

All49 opportunities still have actual eligible evidence under the simulator and these evaluator labels. Stress retains some useful evidence; neither this result nor that observation establishes a general guarantee about hardware returns. Candidate maxima use7x7 angular hypotheses; labels mean at least one contributing pixel in a cell, not an exact measured3D point.

## Local attribution

Known valid geometrically eligible return/cell/query candidates are evaluated against actual query-contributor presence. This conditional ranking metric is not whole-field segmentation or alert recall. Macro AP omits queries with no positive candidates. DistanceDEV has no BODY positive candidates, so its macro AP evaluates HEAD only and cannot confirm or refute BODY localization improvement there.

| Cohort | MZ16 macro AP | MZ18 macro AP |
|---|---:|---:|
| DEV | 0.816296 | 0.878794 |
| clean | 0.720636 | 0.810565 |
| stress | 0.727301 | 0.811725 |
| relation10000 | 0.806391 | 0.872868 |
| distance5000 | 0.942514 | 0.941656 |

On relationDEV, BODY_NEAR AP rises0.707776->0.837532 and BODY_FAR0.703209->0.828127. OldDEV BODY rises similarly, and clean/stress BODY ranking improves too. The generic all-cohort macro-AP-gain flag is false because distanceDEV HEAD macro AP falls slightly; it is a diagnostic flag, not the task acceptance rule or a BODY-specific mechanism verdict.

Local precision/recall/FPR below use separate cutoffs fitted to at most1% negative-cell FPR on oldDEV, then frozen. These do not select alert outputs. Actual transfer FPR is reported; cells and frames are correlated.

| Cohort | Model | Query | Positive/negative cells | AP | Precision % | Recall % | FPR % |
|---|---|---|---|---:|---:|---:|---:|
| DEV | MZ16 | BODY_NEAR | 31663/14738 | 0.717972 | 75.211 | 1.409 | 0.997 |
| DEV | MZ16 | BODY_FAR | 22016/10464 | 0.724190 | 78.601 | 1.735 | 0.994 |
| DEV | MZ16 | HEAD_NEAR | 41219/8036 | 0.930441 | 92.734 | 2.477 | 0.996 |
| DEV | MZ16 | HEAD_FAR | 12648/3456 | 0.892581 | 95.479 | 5.677 | 0.984 |
| DEV | MZ18 | BODY_NEAR | 31663/14738 | 0.835651 | 97.022 | 15.128 | 0.997 |
| DEV | MZ18 | BODY_FAR | 22016/10464 | 0.844002 | 97.904 | 22.066 | 0.994 |
| DEV | MZ18 | HEAD_NEAR | 41219/8036 | 0.940310 | 90.232 | 1.793 | 0.996 |
| DEV | MZ18 | HEAD_FAR | 12648/3456 | 0.895214 | 97.391 | 10.033 | 0.984 |
| clean | MZ16 | BODY_NEAR | 894/943 | 0.541824 | NA | 0.000 | 0.000 |
| clean | MZ16 | BODY_FAR | 1770/5568 | 0.522138 | 57.349 | 22.486 | 5.316 |
| clean | MZ16 | HEAD_NEAR | 1240/104 | 0.991060 | 100.000 | 0.806 | 0.000 |
| clean | MZ16 | HEAD_FAR | 1232/2410 | 0.827522 | 84.906 | 10.958 | 0.996 |
| clean | MZ18 | BODY_NEAR | 894/943 | 0.642402 | NA | 0.000 | 0.000 |
| clean | MZ18 | BODY_FAR | 1770/5568 | 0.774457 | 100.000 | 2.260 | 0.000 |
| clean | MZ18 | HEAD_NEAR | 1240/104 | 0.995625 | 100.000 | 4.274 | 0.000 |
| clean | MZ18 | HEAD_FAR | 1232/2410 | 0.829776 | 87.719 | 20.292 | 1.452 |
| stress | MZ16 | BODY_NEAR | 894/943 | 0.541824 | NA | 0.000 | 0.000 |
| stress | MZ16 | BODY_FAR | 1636/4656 | 0.536658 | 58.437 | 20.110 | 5.026 |
| stress | MZ16 | HEAD_NEAR | 1240/104 | 0.991060 | 100.000 | 0.806 | 0.000 |
| stress | MZ16 | HEAD_FAR | 1152/1965 | 0.839664 | 85.211 | 10.503 | 1.069 |
| stress | MZ18 | BODY_NEAR | 894/943 | 0.642402 | NA | 0.000 | 0.000 |
| stress | MZ18 | BODY_FAR | 1636/4656 | 0.768115 | 100.000 | 1.956 | 0.000 |
| stress | MZ18 | HEAD_NEAR | 1240/104 | 0.995625 | 100.000 | 4.274 | 0.000 |
| stress | MZ18 | HEAD_FAR | 1152/1965 | 0.840756 | 89.231 | 20.139 | 1.425 |
| relation10000 | MZ16 | BODY_NEAR | 66050/30178 | 0.707776 | 74.029 | 0.837 | 0.643 |
| relation10000 | MZ16 | BODY_FAR | 42742/20322 | 0.703209 | 76.252 | 2.637 | 1.727 |
| relation10000 | MZ16 | HEAD_NEAR | 86500/17618 | 0.920113 | 96.825 | 2.785 | 0.448 |
| relation10000 | MZ16 | HEAD_FAR | 24238/6256 | 0.894468 | 93.359 | 4.930 | 1.359 |
| relation10000 | MZ18 | BODY_NEAR | 66050/30178 | 0.837532 | 99.025 | 13.384 | 0.288 |
| relation10000 | MZ18 | BODY_FAR | 42742/20322 | 0.828127 | 96.502 | 20.589 | 1.570 |
| relation10000 | MZ18 | HEAD_NEAR | 86500/17618 | 0.929619 | 98.785 | 2.068 | 0.125 |
| relation10000 | MZ18 | HEAD_FAR | 24238/6256 | 0.896195 | 95.557 | 8.606 | 1.551 |
| distance5000 | MZ16 | BODY_NEAR | 0/34 | NA | NA | NA | 0.000 |
| distance5000 | MZ16 | BODY_FAR | 0/7 | NA | NA | NA | 0.000 |
| distance5000 | MZ16 | HEAD_NEAR | 122866/36945 | 0.921424 | 94.722 | 4.279 | 0.793 |
| distance5000 | MZ16 | HEAD_FAR | 34796/2488 | 0.963604 | 99.436 | 5.070 | 0.402 |
| distance5000 | MZ18 | BODY_NEAR | 0/34 | NA | NA | NA | 0.000 |
| distance5000 | MZ18 | BODY_FAR | 0/7 | NA | NA | NA | 0.000 |
| distance5000 | MZ18 | HEAD_NEAR | 122866/36945 | 0.922597 | 92.591 | 3.540 | 0.942 |
| distance5000 | MZ18 | HEAD_FAR | 34796/2488 | 0.960715 | 99.186 | 9.800 | 1.125 |

## Wrong correspondence and grouped results

Wrong-zone RGB removes all MZ18 added TP and FP on both placement sets and both sequence conditions. MZ16 placement wrong-zone also removes all far gains but retains6 addedFP. This establishes dependence on the supplied correspondence, not correct spatial ownership by itself.

| Cohort | Unit | MZ5 all-exact | MZ16 all-exact | MZ18 all-exact |
|---|---|---:|---:|---:|
| relation10000 | group | 325/400 | 341/400 | 340/400 |
| relation10000 | site | 47/100 | 58/100 | 55/100 |
| distance5000 | group | 424/500 | 445/500 | 450/500 |
| distance5000 | site | 57/100 | 65/100 | 67/100 |

Relation groups are400 obstacle configurations, distance groups are500 pairs; each spans100 sites and the datasets share sites. Do not add them as200 independent sites.

| Cohort | Family | MZ18 added TP vector | MZ18 added FP vector |
|---|---|---|---|
| relation10000 | crossbar | [0, 11, 7, 6] | [0, 0, 0, 0] |
| relation10000 | cabinet | [0, 1, 3, 2] | [0, 0, 0, 2] |
| relation10000 | oblique_rod | [0, 6, 0, 6] | [0, 0, 0, 2] |
| relation10000 | hanging_sign | [0, 3, 0, 1] | [0, 1, 0, 0] |
| distance5000 | crossbar | [0, 0, 4, 12] | [0, 0, 0, 0] |
| distance5000 | cabinet | [0, 0, 0, 8] | [0, 0, 0, 0] |
| distance5000 | oblique_rod | [0, 0, 3, 12] | [0, 0, 0, 0] |
| distance5000 | hanging_sign | [0, 0, 0, 2] | [0, 2, 0, 0] |

## Verification, cost and disposition

CUDA RTX5060 Laptop. One1200-step fit completes in18.47s; training/evaluation/artifact run total26.70s. No new encoder extraction or end-to-end latency claim. Three tests pass: BODY gradient direction, exact HEAD query-logit gradient preservation with mixed masks/ties, and BODY unknown/ineligible/multiple-witness handling. Initial checkpoint values and saved training batches match MZ16 exactly.

Independent audit recomputes31200 task output bits and662031 local candidate bits, matches AP to sklearn, checks cutoffs/receipts, and reconstructs family/group/site metrics. The pole follow-up only reads saved predictions/local scores with input hashes and verifies actual-above-cutoff implies detection. Its initial inline export hit a NumPy int64 JSON serialization error before producing an artifact; the durable script explicitly converts scalar types. No inference, threshold or fit was repeated.

Retain BODY attribution training as COMPONENT_OR_CHALLENGER in COMPONENT mode, with explicit pole alert-strength failure. The full fixed-budget augmentation is not retained: it still adds7 FP and falls below48/49 pole retention in both conditions (28/49 and24/49). Preserve positive evidence of real localization improvement without advertising an alert-performance upgrade or claiming the cause of all MZ16 errors is settled. MZ5 and the prior bounded MZ9 component remain unchanged. No training extension, coefficient/threshold search or successor run follows this experiment.

Artifacts: `artifacts.local/work/mz18-body-objective-20260910/run-v1/` contains start/receipt/result, model/initial checkpoint, batches, cutoffs, predictions, local samples and audit. Sibling `pole_trace.json` binds source artifacts and preserves actual-contributor score margins. Delivery receipts preserve scoped validation and resource release. All data are consumed controlled/synthetic Development, not independent hardware or product-safety evidence.
