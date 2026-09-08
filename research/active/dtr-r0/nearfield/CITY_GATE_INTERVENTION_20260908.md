# City gate replacement diagnostic

Pre-outcome plan: zero training, existing seed17 and step200 weights only.
Use the unchanged 750 street TRAIN and 750 consumed plaza TEST frames. Keep
step200 backbone, deep projection, detail/support predictor and near parameters
fixed. Replace only the sigmoid support gate entering deep-feature pooling
with the cached original seed17 gate from the same image. Do not normalize,
threshold, rescale, shift or substitute a different image's gate.

Report both heads' TP/FP/FN/TN, recall and FPR at the original inclusive 0.5
threshold, alongside the cached original and step200 arms. Cache the hybrid
logits and sample IDs. Normal forward must match the explicit self-gate path;
zero gate must algebraically produce the near bias. The fresh step200 forward
must match prior cached normal probabilities within 1e-5 and have zero decision
flips, or report the failed parity rather than interpreting a mixed comparison.

Interpretation is joint: fewer FP with collapsed recall is attenuation, not
restored discrimination. Gate replacement is a computational intervention in a
fixed model, not identification of which training loss caused adaptation. A
hybrid combines components not trained together; failure does not exonerate the
gate or prove a backbone-only cause. Baseline-gate support metrics describe the
supplied gate, not a repaired step200 localization predictor.

Stop after this single intervention and parity/evidence checks. Do not run frozen
backbone training, source-diversity collection, counterfactual training, threshold
search or additional gate arms in this turn. The user explicitly asks for diagnosis
before training. Preserve original data and weights. This extends the existing
[fixed pilot](CITY_FINETUNE_PILOT_20260908.md) and
[zero-training FP diagnosis](CITY_FP_ATTRIBUTION_20260908.md), not fresh confirmation.

Execution placement: the secondary worker is occupied by another task's UE
collection. Read-only transfer of the immutable cache and step200 checkpoint
prepares the idle main GPU; do not interrupt or compete with that UE process.

## Execution and result

The main-machine Torch 2.11/cu130 attempt failed the declared numerical parity
check: max cached near difference 0.002754, support 0.005128, zero decision flips.
Self-gate and zero-gate algebra were exact. It is retained as a failed mechanical
attempt, not a scored arm; the tolerance was not loosened. The exact source was
then run after the worker's UE collection finished, using its original
Torch 2.9/cu128 environment on RTX 3060 Laptop. This passed in 7.540 s, zero
optimizer steps. TRAIN max cached near/support differences were 1.49e-7/1.22e-6;
TEST differences were exactly zero; self-gate, zero-gate and decision-flip checks
were zero in both partitions. Neither checkpoint changed and task processes exited.

| BODY (recall / FPR) | Street TRAIN750 | Plaza TEST750 |
| --- | ---: | ---: |
| Original seed17 | 0.74% / 0% | 2.96% / 0.49% |
| Step200 normal | 97.04% / 0.33% | 94.81% / 91.22% |
| Step200 + original gate | 19.26% / 0.16% | 10.37% / 1.14% |

| HEAD (recall / FPR) | Street TRAIN750 | Plaza TEST750 |
| --- | ---: | ---: |
| Original seed17 | 0% / 2.99% | 0% / 0.54% |
| Step200 normal | 73.33% / 0.27% | 60% / 47.35% |
| Step200 + original gate | 0% / 0% | 0% / 0.68% |

Hybrid confusion TP/FP/FN/TN: TRAIN BODY 26/1/109/614 and HEAD 0/0/15/735;
plaza BODY 14/7/121/608 and HEAD 0/5/15/730. Each region has 135 BODY positives
and 615 negatives, 15 HEAD positives and 735 negatives.

The original gate is much weaker on City: mean BODY gate over plaza is 0.001375
versus step200's 0.075835; HEAD is 0.002569 versus 0.021544. Replacement changes
both gate magnitude and spatial pattern. The observed FP reduction comes with
severe recall loss in **both** regions. This intervention is not a useful repair
and does not isolate spatial gating, the auxiliary loss or a backbone-only
failure. It demonstrates dependence of the fitted decision on the fitted gate.
The hybrid is diagnostic only; no model promotion or follow-on training occurred.

## Cached score separation, requested in the follow-up review

ROC/PR curves reuse the prior four prediction files via the verified per-frame
table. Ties are grouped; AP is noninterpolated average precision. Perfect,
reversed, tied and missing-class synthetic checks pass. All eight domain/head/arm
ROC-AUC and AP values match sklearn within 1e-12. No deployed threshold is chosen
or exported; low-FPR values below are label-derived descriptive curve envelopes
on consumed data, not independently calibrated performance.

| Plaza | Original | Step200 |
| --- | ---: | ---: |
| BODY ROC-AUC | 0.6821 | 0.8290 |
| BODY AP (positive prevalence 18%) | 0.4151 | 0.5681 |
| HEAD ROC-AUC | 0.1585 | 0.6242 |
| HEAD AP (positive prevalence 2%) | 0.0119 | 0.0882 |
| BODY recall envelope at FPR <=5% | 30.37% | 43.70% |
| BODY recall envelope at FPR <=10% | 45.19% | 62.22% |
| HEAD recall envelope at FPR <=5% | 0% | 33.33% |
| HEAD recall envelope at FPR <=10% | 0% | 60% |

Both Willow arms remain AUC/AP 1 on consumed VAL96. City score ordering improves,
despite the failed fixed 0.5 operating point. Thus the result does not support
"all ranking is broken" or "thresholds cannot help". Nor does it establish that
calibration alone achieves the desired high-recall/low-FPR task: for example,
the BODY curve reaches only 62.22% recall at <=10% FPR in this inspected cohort.
HEAD has just 15 positives, so each success changes recall by 6.67 points.

## Disposition

Retain G13-D architecture as a Development candidate and the original checkpoint
as this pilot's reference. Do not infer a unique cause from the gate swap. Before
a new fit, specify the training-data coverage and an independent Development
calibration/selection partition; plaza remains consumed. Then compare one change
at a time, including exact trainable parameter scope if freezing the backbone
(the projection/detail branches are separate from RepViT). A group-loss comparison
must use identical grouped batches in both arms. No larger backbone, new loss,
sampling change, data capture or threshold rescue was performed here.

Artifacts: `artifacts.local/nearfield/city-gate-intervention-20260908/` retains
the input transfer receipt, main `run-v1/failure.json`, successful
`worker-evidence/run-worker-v1/{result.json,receipt.json}`, two hybrid prediction
caches, and `score-separation-v1/{result.json,score-curves.png}`. The executable
sources are `city_gate_intervention.py` and `city_score_separation.py`; the worker
receipt records exact input and source hashes. This read-only follow-up inherits
the fixed pilot's diagnostic/no-promotion disposition.
