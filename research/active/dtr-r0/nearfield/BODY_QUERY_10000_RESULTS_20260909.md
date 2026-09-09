# 10000-frame B comparison and distance diagnostic

2026-09-09 `EXPLORE`. Decision: `RETAIN_BODY_QUERY_10000_B_FOR_CONTROLLED_ALERT_SCOPE`; keep the expanded-B checkpoint as the historical comparator. The new arm improves the fixed alert task on the accepted shared-asset source, but its near/far readout remains unsuitable for a distance or deployment claim.

The frozen inputs are [the 10000-frame B protocol](BODY_QUERY_10000_B_PROTOCOL_20260909.md), [the distance protocol](BODY_QUERY_10000_DISTANCE_ANALYSIS_PROTOCOL_20260909.md), and the accepted [collection contract](BODY_QUERY_10000_COLLECTION_20260909.md). All source, model and result files are under `artifacts.local/work/body-query-10000-20260909/` and `artifacts.local/work/body-query-10000-b-20260909/`.

## Source admission

The finalizer output is `final-dataset-v2` with `status=PASS`: 10,000 frames, 2,000 complete groups, 500 unique XY sites, 10,000 unique RGB hashes, zero repeated RGB frames, and 6 m minimum within-region XY separation. Role counts are TRAIN_ONLY 5,000, DEV_ONLY 2,000 and EVAL_ONLY 3,000. All declared groups were labeled with zero intent mismatches and zero rejected groups.

The completed visual review covers 800 native 640x360 frames (80 per region): 260 `NO_VISUAL_DEFECT_IDENTIFIED` and 540 `BACKGROUND_VISUAL_LIMITATION_RETAINED`. Floating or changing background vehicles, and the dense02 roof-hole limitation, remain recorded limitations; they were not converted to `CLEAR`. The first finalizer attempt is preserved as `finalizer-v1-failure.json` because its five-region visual record did not satisfy the ten-region key-set gate; the v2 review then passed. The RGB-only adapter has `adapter-validation.json: status=PASS`, with source/label/hash, UNKNOWN pooling, split-isolation and QueryRGB readback checks; it performs no inference.

## Fixed fit and alert comparison

`OLD` is the accepted expanded-B checkpoint (`c7aef143...f94e776`). `NEW` starts from the unchanged G13/seed17 tensor (`0c141794...78d5c7b`) and uses the same BodyQuery B architecture, frozen BN buffers, AdamW settings and exactly 2,000 steps. The fit completed once on CUDA/NVIDIA GeForce RTX 5060 Laptop GPU in 208.05 s; the new checkpoint digest is `db39ccfc...ece0`. Thresholds were selected independently on the 10000-source DEV partition at the frozen empirical FPR rule before EVAL scoring: OLD BODY/HEAD `0.332480/0.708157`, NEW `0.039468/0.511381`.

### EVAL_ONLY (600 complete groups, 3,000 frames)

| Head | OLD TP / FP / FN / TN | NEW TP / FP / FN / TN | OLD AUC | NEW AUC |
| --- | ---: | ---: | ---: | ---: |
| BODY | 1142 / 69 / 58 / 1731 | 1187 / 69 / 13 / 1731 | 0.98458 | 0.99712 |
| HEAD | 1099 / 67 / 101 / 1733 | 1150 / 35 / 50 / 1765 | 0.97484 | 0.98695 |

Complete-group correctness is 439/600 for OLD and 508/600 for NEW. Paired EVAL correctness shows 103 groups correct only for NEW and 34 only for OLD; at the head level, NEW-only correctness contributes 122 cells versus 39 OLD-only, while 2,793 HEAD cells are correct for both.

The query-cell strata expose the remaining geometry gap. Values are TP divided by native-positive cells:

| Stratum | OLD | NEW |
| --- | ---: | ---: |
| BODY near | 1283/1752 (73.23%) | 1321/1752 (75.40%) |
| BODY far | 1270/1750 (72.57%) | 1495/1750 (85.43%) |
| HEAD near | 140/1788 (7.83%) | 115/1788 (6.43%) |
| HEAD far | 1509/1762 (85.64%) | 1615/1762 (91.66%) |

Control false alerts (BODY / HEAD) remain visible in the full `comparison.json`: `CLEAR` 4/17 to 7/12, `BODY_ONLY` 0/13 to 0/8, `HEAD_ONLY` 28/0 to 38/0, `LOW` 19/17 to 10/6, `ABOVE` 11/14 to 10/5, `LATERAL_OUT` 5/5 to 4/4, and `FAR_OUT` 2/1 to 0/0 (OLD to NEW). Thus the gain is broad on this controlled source, while BODY false alerts on HEAD_ONLY increase and the HEAD-near query signal does not improve.

## Frozen distance diagnostic

The separate finalized source contains 2,500 accepted near/far pairs (5,000 frames), with native truth used only for evaluator scoring. The analyzer made one CUDA RGB-only forward pass per arm, did no fitting (`fits=0`, `training_steps=0`), and passed all hash and preprocessing checks.

Primary EVAL_ONLY distance subset (750 pairs / 1,500 frames):

| Metric | OLD | NEW |
| --- | ---: | ---: |
| Far score higher | 557/750 (74.27%) | 574/750 (76.53%) |
| Near higher / ties | 193/750 / 0 | 175/750 / 1 |
| Both endpoints HEAD alert | 641/750 | 671/750 |
| HEAD hits, near / far | 725/750 / 648/750 | 730/750 / 672/750 |
| BODY false alerts, all frames | 54/1500 | 113/1500 |
| Mean / median `S_far(far)-S_far(near)` | -0.07393 / 0.00654 | -0.06815 / 0.00386 |
| HEAD-near nonempty-query recall @0.5 | 6/2250 (0.267%) | 1/2250 (0.044%) |

At native HEAD events and threshold 0.5, OLD versus NEW near/far TP are 61/665 versus 30/672, with near/far FN 689/85 versus 720/78. The all-pair Development view has far-higher direction 1736/2500 (69.44%) versus 1887/2500 (75.48%), BODY false alerts 359/5000 versus 363/5000, and HEAD-near query recall 71/7500 (0.947%) versus 14/7500 (0.187%).

The direction ranking moves slightly in the intended direction, but absolute near-frame far scores and near-event confusion remain poor, and the accepted pair source has no HEAD-negative endpoint denominator. These pairs are a correlated shared-site Development diagnostic; direction is not calibrated distance capability, and HEAD false-positive rate is not estimable here.

## Decision and limits

Keep NEW as a measured controlled-Development alert arm because EVAL BODY/HEAD AUC, complete-group correctness, BODY-far recall and HEAD-far recall improve at the same fixed architecture and budget. Keep OLD as the historical expanded-B comparator. Do not promote the new arm to a distance, natural-scene generalization, safety or deployment result: HEAD-near cell recall falls, BODY false alerts rise in the distance diagnostic, and near/far attribution is still not established. Stop here as declared: no extra seeds, losses, pooling, depth input, threshold rescue, capture or training follow-up was run.

## Reproduction and validation

Durable fit evidence is in `artifacts.local/work/body-query-10000-b-20260909/run-v1/` (`receipt.json`, `selection.json`, `comparison.json`, `validation.json`, checkpoints and predictions). Distance evidence is in `artifacts.local/work/body-query-10000-b-20260909/distance-analysis-v1/` (`receipt.json`, `validation.json`, `result.json`, `predictions.npz`, `evaluator_truth.npz` and frame metadata). The distance receipt records dataset index SHA `e63ce679...6110d`, pair SHA `40da3e7a...ebf394`, OLD checkpoint SHA `c7aef143...f94e776`, NEW checkpoint SHA `db39ccfc...ece0`, and `status=PASS`.

The task-owned Python sources compile successfully, and the focused distance math suite passes 4/4 tests. All raw source, labels, UNKNOWN masks, visual review, failed first-pass receipt and model outputs remain retained under the canonical `artifacts.local` junction.
