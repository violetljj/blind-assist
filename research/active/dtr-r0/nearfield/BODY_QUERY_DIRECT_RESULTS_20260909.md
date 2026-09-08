# Frozen direct readout: no useful final HEAD tradeoff

2026-09-09 EXPLORE. Zero training and zero model inference; cached CPU scoring
took0.69 seconds. B remains the final-count baseline. Stop these fixed direct
readout replacements; no k/threshold/fusion sweep or R1 restart.

## Contract

[Protocol](BODY_QUERY_DIRECT_PROTOCOL_20260909.md),
[implementation](body_query_direct_readout.py). All24 predeclared combinations
are reported. Point logits first become probabilities for max,mean,top3; count
uses frozen probe DEV point cutoffs and is divided by27 only for score scaling.
Only fixed projection validity masks are applied; native unknown/ownership/local
labels never select input evidence. Each frame head takes the maximum over all
six query scores, including near and far. These are uncalibrated direct scores,
not native pixel counts, and never enter near_from_counts.

Frame thresholds follow the original DEV FPR<=.10 rule. One whole method per
target/feature is selected by DEV minimum-head recall, macro recall, lower macro
FPR, then fixed method order. All cutoffs and choices are saved before EVAL
reporting. No EVAL choice or abstention filtering. Every frame remains counted.

## EVAL full comparison

BODY and HEAD each have16 positive and24 negative frames. Positive matched
delta counts are out of16 HEAD_ONLY-CLEAR and BOTH-BODY_ONLY pairs over8 groups.
Oracle TP at FP<=2 is diagnostic only, not a replacement cutoff.

| Readout | BODY TP/FP | HEAD TP/FP | HEAD oracle TP at FP<=2 | Positive pair deltas |
| --- | ---: | ---: | ---: | ---: |
| Original B |7/0|8/2|8|12/16|
| Original R1 |12/4|15/17|6|12/16|
| ray-B-max |2/0|14/19|5|10/16|
| ray-B-mean |8/0|12/5|5|12/16|
| ray-B-top3 |3/0|14/19|4|8/16|
| ray-B-count (DEV selected) |5/0|13/9|5|10/16|
| ray-R1-max |12/9|10/16|0|8/16|
| ray-R1-mean |11/12|12/13|5|11/16|
| ray-R1-top3 (DEV selected) |12/9|11/16|0|7/16|
| ray-R1-count |7/0|13/13|6|10/16|
| ray-XYZ-max (DEV selected) |0/0|0/0|0|0/16|
| ray-XYZ-mean |0/0|0/0|0|0/16|
| ray-XYZ-top3 |0/0|0/0|0|0/16|
| ray-XYZ-count |0/0|0/0|0|0/16|
| local-B-max |0/0|14/19|4|8/16|
| local-B-mean (DEV selected) |6/0|8/5|5|11/16|
| local-B-top3 |1/0|14/19|2|8/16|
| local-B-count |2/0|12/7|3|9/16|
| local-R1-max |12/6|10/12|1|7/16|
| local-R1-mean (DEV selected) |12/12|14/14|5|10/16|
| local-R1-top3 |12/7|10/14|1|7/16|
| local-R1-count |10/3|12/12|7|12/16|
| local-XYZ-max (DEV selected) |0/0|0/0|0|0/16|
| local-XYZ-mean |0/0|0/0|0|0/16|
| local-XYZ-top3 |0/0|0/0|0|0/16|
| local-XYZ-count |0/0|0/0|0|0/16|

No visual direct readout retains B's HEAD TP8 at frame FP<=2: the best posthoc
oracle across these rules is7, versus B8. This is a descriptive observation over
all tested rules, not a newly selected winner. The primary DEV-selected ray-R1
top3 gives EVAL HEAD TP11/FP16, compared with B TP8/FP2. Local-R1 selects mean,
giving TP14/FP14. Neither replaces B. Some rules trade BODY recall differently,
but do not produce the needed joint BODY/HEAD improvement.

## Counterfactual and query interpretation

XYZ differences are exactly zero for every fixed query position and frame pair
on every split, validating the control. Visual readouts do respond to obstacle
changes, but the response is not consistently useful: ray-R1 top3 has positive
frame deltas in7/16 EVAL pairs, mean11/16, count10/16, versus original B12/16.
Local-R1 count reaches12/16 with one tied pair, but still has HEAD TP12/FP12.
Positive delta alone is not correct classification or stable threshold transfer.
Per-group frame differences and all six HEAD query-position differences are
retained, including negative and tied differences; no positive-only filtering.

The near query signal persists without solving the whole head: EVAL ray-R1 count
query AUC is0.9079 for HEAD-near and0.5958 for HEAD-far; the frame HEAD AUC is
0.6732. Local-R1 count has near query AUC0.9291, far0.6286 and frame0.7318.
This demonstrates why the previously narrow near signal cannot simply be
promoted to all-range HEAD decisions. It does not prove a particular decoder
is the sole cause or that all possible evidence representations will fail.

## DEV-selected R1 EVAL condition errors

Numbers below are HEAD false alerts / negative frames. Positive-only conditions
are omitted from this negative-control table but remain in the complete result.

| Condition | Ray R1 top3 | Local R1 mean |
| --- | ---: | ---: |
|CLEAR|7/8|5/8|
|BODY_ONLY|6/8|7/8|
|LOW|0/2|0/2|
|ABOVE|1/2|0/2|
|LATERAL_OUT|2/2|2/2|
|FAR_OUT|0/2|0/2|

## Decision and evidence

Keep B. R1 final inference remains a negative control; its frozen features and
probes remain available for the previously documented narrow diagnostic role.
These four query aggregators followed by six-cell maximum do not turn the
available evidence into a better final alert tradeoff. Close this concrete
zero-fit replacement route rather than continue pooling-parameter searches.
This does not rule out every future direct readout or justify automatic new data
or representation work; another intervention needs a distinct scoped hypothesis.

`artifacts.local/work/body-query-direct-20260909/run-v1/` retains every query/frame
score, full TRAIN/DEV/EVAL conditions, pairs, selected cutoffs, result and receipt.
Parent `validation.json` independently reconstructs all72 method/split aggregates
with explicit point loops and verifies frame confusion, DEV FP budgets and
input/result/source hashes. Syntax passes. No GPU or remote worker allocation;
the scorer completed exit0. This is consumed same-world Development, not fresh
confirmation, a real-device result or solved cross-region geometry.
