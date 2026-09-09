# Expanded-source B: same architecture and fixed training budget

2026-09-09 EXPLORE. Decision: `RETAIN_EXPANDED_B_WORKING_BASELINE`.

[Protocol](BODY_QUERY_EXPANDED_B_PROTOCOL_20260909.md), [training](body_query_expanded_train.py), [adapter](body_query_expanded_data.py), [independent analysis](body_query_expanded_analysis.py).

## Comparison

OLD uses unchanged final B weights. NEW starts from the exact original G13/seed17 tensor digest, with the same B architecture, losses, AdamW settings, frozen BN statistics and 2000-step batch32 budget. Only NEW is fitted; no continuation from final B. New TRAIN has 2500 frames, versus historical 240; expected presentations per frame are 25.6 versus 266.7. Thus this evaluates an expanded structured source package, not quantity independently of geometry/background changes.

New DEV independently selects each arm's BODY/HEAD cutoffs at empirical FPR<=10%. Both cutoffs were saved before EVAL inference. New EVAL has 1500 frames, 600 positives and 900 negatives per head, and 300 complete groups. All primary counts include the heuristic UNKNOWN cases.

| Split / head | OLD TP / FP | NEW TP / FP | OLD AUC | NEW AUC |
| --- | ---: | ---: | ---: | ---: |
| TRAIN BODY | 475 / 249 | 1000 / 2 | 0.7485 | 1.0000 |
| TRAIN HEAD | 370 / 285 | 997 / 0 | 0.6295 | 0.9994 |
| DEV BODY | 112 / 58 | 373 / 56 | 0.7097 | 0.9595 |
| DEV HEAD | 77 / 60 | 360 / 54 | 0.5964 | 0.9436 |
| EVAL BODY | 263 / 107 | 591 / 58 | 0.7822 | 0.9894 |
| EVAL HEAD | 177 / 139 | 571 / 34 | 0.6096 | 0.9729 |

| Complete groups | OLD | NEW |
| --- | ---: | ---: |
| TRAIN | 10/500 | 495/500 |
| DEV | 1/200 | 123/200 |
| EVAL | 1/300 | 228/300 |

## Equal low-FP diagnostics

These are descriptive EVAL curve points, not deployment thresholds. Tied scores move together.

| EVAL head / FP budget | OLD maximum TP | NEW maximum TP |
| --- | ---: | ---: |
| BODY / 45 | 155/600 | 589/600 |
| BODY / 90 | 235/600 | 593/600 |
| HEAD / 45 | 77/600 | 575/600 |
| HEAD / 90 | 123/600 | 581/600 |

## Spatial evidence

| Split / nonempty query stratum | OLD TP / positives | NEW TP / positives |
| --- | ---: | ---: |
| TRAIN BODY_near | 777/1455 | 1076/1455 |
| TRAIN BODY_far | 720/1460 | 1377/1460 |
| TRAIN HEAD_near | 161/1490 | 99/1490 |
| TRAIN HEAD_far | 307/1465 | 1453/1465 |
| DEV BODY_near | 211/582 | 376/582 |
| DEV BODY_far | 155/584 | 388/584 |
| DEV HEAD_near | 36/596 | 45/596 |
| DEV HEAD_far | 52/586 | 461/586 |
| EVAL BODY_near | 413/873 | 607/873 |
| EVAL BODY_far | 419/876 | 702/876 |
| EVAL HEAD_near | 62/894 | 79/894 |
| EVAL HEAD_far | 142/879 | 762/879 |

## EVAL error slices

| Condition | OLD HEAD TP / FP / FN | NEW HEAD TP / FP / FN |
| --- | ---: | ---: |
| CLEAR | 0 / 33 / 0 | 0 / 7 / 0 |
| BODY_ONLY | 0 / 86 / 0 | 0 / 9 / 0 |
| HEAD_ONLY | 82 / 0 / 218 | 282 / 0 / 18 |
| BOTH | 95 / 0 / 205 | 289 / 0 / 11 |
| LOW | 0 / 5 / 0 | 0 / 8 / 0 |
| ABOVE | 0 / 3 / 0 | 0 / 6 / 0 |
| LATERAL_OUT | 0 / 3 / 0 | 0 / 2 / 0 |
| FAR_OUT | 0 / 9 / 0 | 0 / 2 / 0 |

| Region | OLD HEAD TP / FP | NEW HEAD TP / FP |
| --- | ---: | ---: |
| big06 | 67 / 57 | 195 / 11 |
| dense02 | 58 / 53 | 192 / 0 |
| big07 | 52 / 29 | 184 / 23 |

Paired EVAL correctness (both heads and group accounting retained in comparison.json):

```json
{
  "BODY": {
    "old_only_correct": 45,
    "new_only_correct": 422,
    "both_correct": 1011,
    "both_wrong": 22
  },
  "HEAD": {
    "old_only_correct": 32,
    "new_only_correct": 531,
    "both_correct": 906,
    "both_wrong": 31
  },
  "groups": {
    "new_only": 228,
    "old_only": 1
  }
}
```

## Interpretation and evidence limits

Expanded structured training produces a large useful final-decision gain under
the same architecture and presentation budget: HEAD recall rises from29.5% to
95.2%, while FPR falls from15.4% to3.8%; BODY improves simultaneously. All three
EVAL regions improve. Retain NEW as the working final-alert baseline within
this controlled expanded-source scope; OLD remains the historical comparator.

Spatial attribution is still incomplete. NEW HEAD-near nonempty-cell recall is
only79/894 (8.8%) on EVAL, and99/1490 (6.6%) even on TRAIN. For the300 EVAL frames
whose native HEAD-near count sum is>=3, NEW detects289 (OLD81), but mean predicted
nonempty probability across the three near cells is0.240 versus0.635 across
the three far cells. The300 native HEAD-far positives yield282 hits (OLD96).
These cached-score diagnostics are retained in `run-v1/head-distance-diagnostic.json`;
the means include all three cells in each bin, not only GT-positive cells.
Strong final HEAD detection therefore does not establish correct depth-bin
attribution. Do not rename final-alert success as solved query geometry.
LOW and ABOVE HEAD false alerts also rise5 to8 and3 to6 despite the overall
FP reduction. The useful improvement and these residual failures coexist.

New-model selection is governed by the predeclared joint criterion, not by a favorable single metric. The original B historical results and BodyLift/R1 negative controls remain unchanged. No automatic extra steps, seeds or architecture experiments follow. These shared-asset, curated CitySample roles are Development; identical geometry designs recur across regions. No claim of natural-scene generalization, complete 3D understanding, protected confirmation or default-App promotion follows.

## Validation and resources

One final fit: 323.90 seconds, 2000 steps; full run 371.24 seconds on NVIDIA GeForce RTX 5060 Laptop GPU. Peak training allocation 2.00 GiB. Actual model/output CUDA devices recorded using research_backend.torch_observation.

Cache checks verify all 5000 bound RGB/label hashes, exact native count-to-near semantics, pooled UNKNOWN preservation and group/site/region role isolation. Independent analysis verifies all 12 head confusion/AUC results and frozen result/selection hashes. A tied-score fixture verifies inclusive cutoff and indivisible low-FP ties. Model initialization matches the original fit digest and fixed buffers remain unchanged.

Artifacts: `artifacts.local/work/body-query-expanded-b-20260909/`, including cache-v1, run-v1 checkpoints/predictions/selection/result/comparison/validation, preflight and release receipts. No remote training allocation. Durable evidence retained.
