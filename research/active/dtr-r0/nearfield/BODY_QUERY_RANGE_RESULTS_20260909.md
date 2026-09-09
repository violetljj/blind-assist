# Range R0: same model with four segment-event supervision targets

2026-09-09 EXPLORE. Decision: `RETAIN_EXPANDED_B_RANGE_R0_NEGATIVE_CONTROL`.

[Protocol](BODY_QUERY_RANGE_PROTOCOL_20260909.md), [operator](body_query_range.py), [training](body_query_range_train.py), [analysis](body_query_range_analysis.py).

BASE below means expanded-source B from20dd829c, not the earlier240-frame B. R0 starts from identical G13/seed17 tensors and uses the exact BASE sampling schedule for one2000-step fit. Only .25 mean range-event BCE is added to the original losses. No new parameters, RGB inputs, depth model or inference pooling; same data/calibration and frozen BN statistics.

## Diagnostic before training

Existing cached TRAIN/DEV counts were split into four3-cell event scores. HEAD-near within-split recall at FPR<=10% was62/500 and61/200; these are diagnostic cutoffs, including the TRAIN row, not deployed or DEV-selected TRAIN thresholds. At.5, HEAD-far fired on497/500 TRAIN near-only frames and182/200 DEV near-only frames. This justified a single optimization intervention rather than claiming only low confidence. The initial diagnostic was exploratory and did not inspect EVAL. Its script/results remain under artifacts.local/work/body-query-range-diagnostic-20260909.

## Final alerts using independently DEV-selected thresholds

| Split / head | BASE TP / FP | R0 TP / FP | BASE AUC | R0 AUC |
| --- | ---: | ---: | ---: | ---: |
| TRAIN BODY | 1000 / 2 | 1000 / 8 | 1.0000 | 1.0000 |
| TRAIN HEAD | 997 / 0 | 993 / 0 | 0.9994 | 0.9995 |
| DEV BODY | 373 / 56 | 362 / 59 | 0.9595 | 0.9528 |
| DEV HEAD | 360 / 54 | 356 / 58 | 0.9436 | 0.9316 |
| EVAL BODY | 591 / 58 | 585 / 63 | 0.9894 | 0.9851 |
| EVAL HEAD | 571 / 34 | 557 / 35 | 0.9729 | 0.9641 |

| Complete groups | BASE | R0 |
| --- | ---: | ---: |
| TRAIN | 495/500 | 485/500 |
| DEV | 123/200 | 98/200 |
| EVAL | 228/300 | 214/300 |

EVAL has600 positives and900 negatives per final head. Primary results include all1500 frames, including heuristic UNKNOWN. No EVAL alert threshold selection.

## Range evidence

| Split / range | BASE AUC | R0 AUC | BASE vs-other-range AUC | R0 vs-other-range AUC | BASE BCE | R0 BCE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TRAIN BODY_near | 0.9757 | 0.9950 | 0.9028 | 0.9800 | 0.2358 | 0.1076 |
| TRAIN BODY_far | 0.9058 | 0.9764 | 0.6232 | 0.9056 | 0.6216 | 0.2582 |
| TRAIN HEAD_near | 0.8589 | 0.8715 | 0.4355 | 0.4861 | 0.3905 | 0.3476 |
| TRAIN HEAD_far | 0.9794 | 0.9913 | 0.9210 | 0.9676 | 0.7414 | 0.4093 |
| DEV BODY_near | 0.9483 | 0.9667 | 0.8270 | 0.8967 | 0.3177 | 0.2005 |
| DEV BODY_far | 0.8285 | 0.8484 | 0.5308 | 0.6516 | 0.8780 | 0.5959 |
| DEV HEAD_near | 0.8423 | 0.8591 | 0.5215 | 0.5635 | 0.4124 | 0.3760 |
| DEV HEAD_far | 0.8668 | 0.8640 | 0.6763 | 0.7144 | 1.0068 | 0.7416 |
| EVAL BODY_near | 0.9623 | 0.9823 | 0.8576 | 0.9367 | 0.2856 | 0.1580 |
| EVAL BODY_far | 0.8869 | 0.9260 | 0.5967 | 0.7717 | 0.7212 | 0.3869 |
| EVAL HEAD_near | 0.8641 | 0.8772 | 0.5211 | 0.5690 | 0.3787 | 0.3546 |
| EVAL HEAD_far | 0.9122 | 0.9087 | 0.7329 | 0.7545 | 0.8574 | 0.6027 |

Versus-other-range AUC includes only frames with exactly one native range event positive for that head; whole-head empty negatives cannot inflate that comparison. Four range thresholds are calibrated on DEV after training for diagnostic reporting only, not final alert or checkpoint selection. Fixed.5 and low-FP curves also remain in range-result.json.

| Split HEAD-far on near-only positives at.5 | BASE FP / frames | R0 FP / frames |
| --- | ---: | ---: |
| TRAIN | 497/500 | 494/500 |
| DEV | 182/200 | 175/200 |
| EVAL | 286/300 | 276/300 |

| Split / nonempty query | BASE TP / positives | R0 TP / positives |
| --- | ---: | ---: |
| TRAIN BODY_near | 1076/1455 | 1025/1455 |
| TRAIN BODY_far | 1377/1460 | 1254/1460 |
| TRAIN HEAD_near | 99/1490 | 0/1490 |
| TRAIN HEAD_far | 1453/1465 | 1420/1465 |
| DEV BODY_near | 376/582 | 365/582 |
| DEV BODY_far | 388/584 | 314/584 |
| DEV HEAD_near | 45/596 | 1/596 |
| DEV HEAD_far | 461/586 | 412/586 |
| EVAL BODY_near | 607/873 | 574/873 |
| EVAL BODY_far | 702/876 | 583/876 |
| EVAL HEAD_near | 79/894 | 1/894 |
| EVAL HEAD_far | 762/879 | 679/879 |

## Errors and decision

| Condition | BASE HEAD TP / FP / FN | R0 HEAD TP / FP / FN |
| --- | ---: | ---: |
| CLEAR | 0 / 7 / 0 | 0 / 7 / 0 |
| BODY_ONLY | 0 / 9 / 0 | 0 / 10 / 0 |
| HEAD_ONLY | 282 / 0 / 18 | 274 / 0 / 26 |
| BOTH | 289 / 0 / 11 | 283 / 0 / 17 |
| LOW | 0 / 8 / 0 | 0 / 8 / 0 |
| ABOVE | 0 / 6 / 0 | 0 / 6 / 0 |
| LATERAL_OUT | 0 / 2 / 0 | 0 / 2 / 0 |
| FAR_OUT | 0 / 2 / 0 | 0 / 2 / 0 |

Predeclared attribution criterion met: False. Final-alert preservation criterion met: False. Replacement permitted: False.

Retain expanded B as the alert baseline and this specific R0 fit as a negative
control. BODY range sorting and several BCE values improve, so the additional
loss has an observable effect. However, HEAD-near versus HEAD-far separation
remains near chance on TRAIN (AUC0.4861). Its larger all-negative AUC0.8715
includes many empty-head cases and must not be mistaken for good distance
discrimination. The wrong far-range activation scarcely falls: EVAL286 to276
of300 native near-only frames. Final alerts and complete groups also regress.

The fixed.5 cell-recall collapse alone would be insufficient to reject a
representation; the complementary range discrimination and wrong-range errors
are what make calibration-only rescue unsupported here. This rules out the
tested .25-weight /2000-step recipe as a replacement, not every possible use
of range supervision or all RGB distance information. Do not infer that the
backbone is incapable from this joint optimization result. No weight sweep follows.

The per-cell truth was already present; this intervention reorganizes optimization rather than adding geometric information. Range-event improvement does not establish all lateral cells or continuous metric depth. No automatic extra weight, seed, step or new geometry capture follows. Expanded B and all historical results are preserved. Shared-asset repeated-geometry Development cannot establish natural-scene or protected confirmation performance.

## Validation and artifacts

One2000-step fit took 425.07s; full run 496.95s on NVIDIA GeForce RTX 5060 Laptop GPU. Baseline cached prediction maximum difference 0. Same initialization/schedule and fixed buffers verified;64-state enumerated event probability agrees to5.6e-16 with finite gradients. Independent rank AUC/confusion and source/checkpoint/cache hashes passed. Actual devices and process release are recorded.

Artifacts: artifacts.local/work/body-query-range-r0-20260909/run-v1 contains checkpoints, predictions, final-alert and range selections, result/comparison/range-result, baseline parity, protocol and validation receipts. Initial zero-training diagnostic is kept separately. No remote allocation; durable evidence retained.
