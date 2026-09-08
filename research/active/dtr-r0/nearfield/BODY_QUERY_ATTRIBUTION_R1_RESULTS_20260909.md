# Attribution R1: DEV improvement does not transfer; retain B

2026-09-09 EXPLORE. One new2000-step local GPU fit completed from the original
initialization, in171.74 seconds including native target preparation and saving.
No pooling or runtime architecture change. This challenger is not promoted.

## Matched intervention

[Protocol](BODY_QUERY_ATTRIBUTION_R1_PROTOCOL_20260909.md),
[training code](body_query_attribution_train.py).
The original B initial state digest and all2000 batch32 sample indices match.
Same backbone, AdamW lr1e-5, frozen BN statistics, near loss,0.25 support loss and
0.25 query-count loss. Added0.25 auxiliary local-attribution BCE through a
Linear32->1 training head on query-point features. Inference discards that head
and retains the original masked mean, count classes and near aggregation.

TRAIN native cell membership supplies soft local20x20-footprint targets through
the existing bilinear projection. Out-of-FOV or any footprint touching unknown
native depth is ignored. Of66,960 valid TRAIN point slots,43,590 are supervised,
including5,761 with positive target mass. These targets describe locally visible
membership, not exact ray ownership. This tests one auxiliary-supervision recipe,
not all attribution methods, gates or direct evidence bottlenecks.

## Final-checkpoint results

Each model uses its own original-policy DEV threshold; no EVAL cutoff tuning.
DEV/EVAL each contain16 positive and24 negative frames per head.

| Split / metric | B | R1 |
| --- | ---: | ---: |
| DEV BODY TP / FP |10 /2|12 /2|
| DEV HEAD TP / FP |5 /0|13 /2|
| DEV HEAD AUC |0.7982|0.8724|
| DEV complete groups |0 /8|3 /8|
| EVAL BODY TP / FP |7 /0|12 /4|
| EVAL HEAD TP / FP |8 /2|15 /17|
| EVAL BODY AUC |0.8958|0.8854|
| EVAL HEAD AUC |0.7865|0.7604|
| EVAL complete groups |2 /8|0 /8|

At a descriptive EVAL FP budget<=2, HEAD oracle TP falls8 to6 and BODY13 to10.
This comparison diagnoses ranking only; it does not select deployment cutoffs.
At FP0, HEAD oracle TP instead rises3 to5, so the curves are not uniformly ordered.
On DEV at FP<=2, HEAD rises5 to13, demonstrating region dependence.

Query nonempty TP/FN/FP: TRAIN559/17/21 to533/43/60;
DEV25/68/21 to18/75/12; EVAL29/68/23 to25/72/22. The intervention does not
improve nonempty query recall. TRAIN final near ranking AUC remains1.0 for both
heads, but DEV-selected R1 cutoffs introduce3 BODY false alerts on TRAIN;
complete groups fall48/48 to45/48. Do not claim perfect TRAIN decisions.

EVAL R1 HEAD false alerts include BODY_ONLY8/8, CLEAR3/8, LOW2/2,
LATERAL_OUT2/2, FAR_OUT2/2; ABOVE remains0/2. Thus higher selected HEAD recall
does not represent a usable resolution of the evidence-selectivity problem.

R1 DEV cutoffs are BODY0.0148587851 and HEAD0.0317568779. They were saved before
EVAL and verified unchanged. The probability scale changed substantially; the
fixed-cutoff failure is not the only evidence against this recipe, because
query recall and EVAL ranking at FP<=2 also decline.

## Decision and evidence

Retain B as the comparison baseline; preserve R1 as a negative control for this
specific auxiliary recipe. Do not continue its budget, sweep weights, or describe
DEV gains as cross-region success. Q3 geometric labels/tooling remain useful.
This result does not reject geometry supervision in general or establish which
component caused the remaining region shift. A different direct evidence
representation would require a separate scoped question, not an automatic fit.

Payload: `artifacts.local/work/body-query-q3-20260909/attribution-r1/` contains
TRAIN labels/masks, protocol, progress, history, final main/aux checkpoints,
selection, all-split cached predictions, result and terminal receipt.
Main checkpoint SHA256:
`454ec22db58a7548ecd0facfe40beabde4a5ed27280e76e905d013aa17d120d1`.
`comparison.json` and `training-validation.json` in the parent directory record
independently recomputed rank-based AUC, selected confusion, query confusion,
and source/checkpoint/result/selection hash checks. Syntax checks pass. The
training process completed exit0 and released its GPU allocation. Native arrays
and durable evidence remain; no worker job or temporary remote archive was made.

One seed, small previously consumed same-world regions: no fresh confirmation,
device deployment, or general geometry-reasoning claim.
