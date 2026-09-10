# MZ22: residual arbitration removes some errors but loses trusted far detections

2026-09-10, one completed1200-step fit on frozen MZ5/MZ20 evidence. Consumed
Development only. This residual arbiter is not retained as an upgrade: it adds
placement false positives and loses pole HEAD coverage, despite useful near
recovery and removal of three sequence BODY_FAR false positives.

The [protocol](MZ22_EVIDENCE_ARBITRATION_PROTOCOL_20260910.md) fixes18 observable
features per query: global/local scores, local score distribution and geometric
support/hypothesis coordinates. Four independent32-hidden residual MLPs start
at exact MZ5 identity. The1200 batches match MZ20; features are normalized only
on the7562 unique TRAIN frames. Both base models and the visual encoder stay
frozen. TRAIN base predictions are in-sample, a stacking/generalization limit.
No native contributor or known mask enters the arbiter. Unsupported queries
always preserve MZ5 exactly, so that branch also preserves MZ5 errors.

OldDEV chooses supported-query cutoffs to maximize TP at the original supported
MZ5 FP budget, with fewer FP and then higher cutoff as tie-breaks. This is a new
predeclared full-arbitration rule, not a retuned MZ20 result. Apply it unchanged
elsewhere. Report actual gains/losses; matched oldDEV budgets do not guarantee
placement FP preservation.

| Cohort | MZ5 exact | MZ20 exact | MZ22 exact | New FP / removed FP vs MZ5 |
|---|---:|---:|---:|---:|
| oldDEV1000 |912|942|953|0/0|
| clean200 |143|176|171|0/3|
| stress200 |143|175|165|0/3|
| relationDEV2000 |1883|1916|1916|5/0|
| distanceDEV1000 |920|946|948|2/0|

Across placementDEV3000, exact is2864/3000 versus MZ20's2862 and MZ5's2803.
It gains89 far true output bits but loses18 original MZ5 far true bits, net+71.
The7 new false bits are3 BODY_NEAR,1 BODY_FAR and3 HEAD_NEAR. Net near recall
does not decrease on the normal cohorts, but far-positive identity is no longer
preserved. It cannot be described as a baseline-preserving augmentation.

On the trained pole clip, BODY remains25/25 clean/stress, while HEAD becomes
16/24 clean and10/24 stress: total41/49 and35/49, below the frozen48/49 rule.
The three removed sequence false positives are BODY_FAR; the19 existing
HEAD_NEAR false positives remain. Under wrong-zone correspondence, relation
exact collapses1916->1372 and distance948->512; many original MZ5 true outputs
are erased. This is an observed robustness failure of arbitration, not merely
a desirable sensitivity check. MZ20 remains the stronger pole component and
MZ5 the retained baseline.

The next evidence target is support availability at angular locations: MZ21
shows all8 MZ20 additions lie outside actual observed query support. A learned
availability/source-support mechanism is a distinct hypothesis; evaluator
known masks must not become inference gates, and missing support is not CLEAR.
This experiment does not disprove all arbitration or justify training extension.

Two focused initialization/fallback/feature tests pass. Frozen local predictions
reproduce saved MZ20 within numeric tolerance on normal and wrong-zone cohorts;
all support bits agree. Saved-output audit checks calibration optimality,
per-query changes, unsupported identity, batches, hashes, exported model outputs
and family/group/site metrics:31200 output bits pass; CPU/CUDA export decisions
have zero differences (maximum logit difference3.81e-6). Raw TRAIN arbiter
features were not saved, so TRAIN-only normalization is verified by code/ID
routing rather than independent numeric re-extraction. Artifacts live in
`artifacts.local/work/mz22-evidence-arbitration-20260910/run-v1/`, including
start/receipt/result, checkpoints, normalization, cutoffs and predictions.
The2564-parameter arbiter fit takes3.63s on CUDA, excluding feature extraction;
this is not end-to-end inference latency. One process completed; no service,
capture or temporal state was created.

Retain this exact residual-arbitration recipe as NEGATIVE_CONTROL. No fresh
confirmation, default-App integration, physical hardware or safety claim.
