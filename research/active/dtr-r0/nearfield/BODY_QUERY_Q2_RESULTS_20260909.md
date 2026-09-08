# Q2: sparse point responses exist, but are not selective evidence

2026-09-09, EXPLORE, consumed same-world Development. No model promotion.

## Decision

Frozen B contains both high-point/low-mean misses and all-weak-point misses.
Replacing the mean with strong-point selection recovers positives but creates
pervasive wrong-body-region and empty-cell responses, including on TRAIN.
This supports investigating query evidence formation and selectivity, but does
not isolate uniform pooling as the root cause or attribute weak responses to the
backbone. No V2 fit was launched. Balanced CE remains an unproven priority.

## Scope and reproduction

The [pre-outcome protocol](BODY_QUERY_Q2_PROTOCOL_20260909.md) froze B step2000,
checkpoint SHA256 `dd6fab0bddf436477e9076bb47a0df8914b4ddd21dcf0ea9897143643460726b`.
One pass over TRAIN240/DEV40/EVAL40 ran on the primary RTX5060 Laptop GPU,
torch2.11.0+cu130, batch32, in12.91 seconds including dumps/audit. Training steps:0.
No new capture, cutoffs, optimizer updates or remote worker allocation.

Implementation: [body_query_point_audit.py](body_query_point_audit.py).
Payload: `artifacts.local/work/body-query-q2-20260909/run-v1/` contains
`{train,dev,eval}-points.npz`, `result.json`, `receipt.json`; independent cached-array
verification is at `artifacts.local/work/body-query-q2-20260909/validation.json`.
Each dump retains point MLP input67, output32, linear count logits4, original
cell logits, near outputs, validity/projection grid and coarse support proxies.
The receipt hashes source, frozen inputs, result and every dump.

Masked mean point logits reproduce cell logits within5.73e-6. Saved near outputs
are bit-identical to the original B arrays on all splits. An independent explicit
per-cell loop reproduces all three methods' query confusion matrices; hashes and
Python syntax pass. There was no second model inference. The three BODY-near
cells have12 valid points each; the other nine cells have27. In particular, both
HEAD ranges have27: HEAD-near failure is not explained by fewer valid points.

## Point decomposition

The classifier is linear, so averaging valid point logits (including bias)
equals applying it to the averaged features. Averaging probabilities is not the
same operation. These are decomposed logits from a pooled-feature classifier,
not independently trained or calibrated point detectors. Diagnostic nonempty
threshold is0.5, with0.9 also reported. Best/top3 rank valid points by nonempty
log-odds; top3 averages their logits before softmax.

| All query cells | TRAIN | DEV | EVAL |
| --- | ---: | ---: | ---: |
| True nonempty cells |576|93|97|
| Original mean TP / FN / FP |559 /17 /21|25 /68 /21|29 /68 /23|
| Missed cells with best point >=0.5 |17 /17|23 /68|36 /68|
| Missed cells with best point >=0.9 |16 /17|12 /68|9 /68|
| Missed cells with every valid point <0.5 |0 /17|45 /68|32 /68|
| Best-point TP / FN / FP |576 /0 /736|48 /45 /91|65 /32 /130|
| Top3 TP / FN / FP |576 /0 /732|46 /47 /81|61 /36 /119|

Empty-cell denominators are2304/387/383. Available strong responses and missing
responses coexist on held-out regions; neither proposed branch describes all
failures. Recovering these responses alone is not enough: top3 TRAIN nonempty
recall reaches100%, but empty-cell false positives rise from21 to732.

## HEAD range breakdown

| Split / range | True nonempty | Mean TP | Misses with point >=0.5 | All-weak misses | Top3 TP / FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| TRAIN near |82|66|16|0|82 /348|
| DEV near |12|0|6|6|6 /33|
| EVAL near |6|0|6|0|5 /85|
| EVAL far |40|16|18|6|33 /22|

All6 EVAL HEAD-near positive cells have some point >=0.5 (5 also >=0.9),
despite none being mean-active. However, selecting strong points also activates
85 empty HEAD-near cells under top3. This is a response dilution observation,
not proof that correctly grounded obstacle evidence was diluted.

The strongest point in each of those6 EVAL HEAD-near misses has no positive
overlap in the bilinearly sampled cached HEAD support;4 have majority UNKNOWN
mass. The6 DEV HEAD-near strong misses also have no positive overlap, with no
majority UNKNOWN. This is only an18x32 head-level2D proxy: it is neither per-cell
3D truth nor a bound on the network's receptive field. Do not convert absence of
proxy overlap or UNKNOWN into proof that a point sees only background.

## Frozen-cutoff alert effects and negative controls

These are descriptive interventions using B's original DEV cutoffs and unchanged
count aggregation, not calibrated alternative models or deployment thresholds.

| Split | Original BODY TP / FP | Top3 BODY TP / FP | Original HEAD TP / FP | Top3 HEAD TP / FP |
| --- | ---: | ---: | ---: | ---: |
| TRAIN |96 /0|96 /5|96 /0|96 /48|
| DEV |10 /2|13 /5|5 /0|8 /7|
| EVAL |7 /0|11 /2|8 /2|15 /17|

TRAIN has96 positives and144 negatives per head; DEV/EVAL each16 and24.
All48 TRAIN BODY_ONLY frames become false HEAD alerts under both best-point and
top3, versus0 originally. This is a particularly direct counterexample to
interpreting every strong point as correctly assigned HEAD evidence.

EVAL top3 HEAD false alerts include CLEAR6/8, BODY_ONLY7/8, LOW1/2 and ABOVE1/2.
Best-point raises EVAL HEAD TP to15 but FP to18, including BODY_ONLY8/8.
Thus the apparent recovery of7 EVAL HEAD positives under top3 comes with15
additional false alerts. Mean pooling also suppresses misleading responses;
removing that suppression is not a demonstrated improvement.

## What this changes

The next research question is how to select evidence that is correct for the
specific body/query region, rather than merely strong under the current readout.
Q2 does not justify choosing a max/top-k replacement or automatically launching
gated MIL. Learned attention could behave differently and is not ruled out by
these zero-fit probes, but it would need its own matched test with wrong-height
and empty-cell controls. Likewise, weak point logits can arise from projection,
features, the point MLP or the decoder; they do not isolate backbone failure or
establish cross-attention as the remedy. No additional fit is part of Q2.

The cited [SA-FAS CVPR2023 paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Sun_Rethinking_Domain_Generalization_for_Face_Anti-Spoofing_Separability_and_Alignment_CVPR_2023_paper.pdf)
questions training-domain invariance under unseen-domain shift in face
anti-spoofing. It does not test our pooling or query architecture and supplies no
causal diagnosis for this experiment.

Retain B and Q2 tooling as diagnostic components under the previous source
boundaries. This is not fresh confirmation, complete3D occupancy, generalized
geometry reasoning, or real-device evidence. All task-owned inference completed;
durable arrays and receipts remain for reanalysis, with no GPU process reserved.
