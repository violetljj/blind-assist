# MZ30: branch selection removes63 baseline false bits but loses one true bit

2026-09-10 EXPLORE, consumed Development. **The fitted selector removes63/81
placement baseline false bits and preserves MZ28 additions, but loses one
relation BODY_NEAR true positive.** The predeclared no-baseline-TP-loss gate
fails, so `useful_effect=false`. This is a promising correction mechanism with
a measured retention failure, not an admitted baseline replacement.

[Protocol](MZ30_BRANCH_RESPONSIBILITY_PROTOCOL_20260910.md),
[preparation](mz30_prepare.py), [selector](mz30_select.py), [fit](mz30_train.py),
[independent audit](mz30_audit.py), [residual diagnostic](mz30_residual_diagnostic.py).

The preparation gate passed before fitting:80/81 placement false bits were
unsupported branch disagreements, versus the required41. TRAIN provided120
RGB-correct and48 ToF-correct eligible query examples, exceeding20/class.
Frozen MZ5 baseline reconstruction had maximum error3.8147e-6 and exact signs.
The RGB backbone was not rerun; existing full-image feature maps fed the frozen
projection/context decoder and two MZ5 readouts. Preparation opportunity is
separate from measured selector benefit.

One8641-parameter shared268->32->1 selector used seed123, Adam0.001 and the
original1200x16 batch stream. Normalization used7562 unique TRAIN frames;
inverse class weights were0.7/1.75. Only565 batches contained eligible queries,
with760 eligible query exposures in total. A target indicates which sensor
branch agrees with event truth, not whether the event itself is positive.
OldDEV selected fixed per-query confidence cutoffs under zero baseline TP loss.

| Normal cohort | MZ5 exact | MZ28 exact | MZ30 exact | FP removed vs MZ28 | TP lost |
|---|---:|---:|---:|---:|---:|
| oldDEV1000 |912|940|957|24|0|
| clean200 |143|176|182|20|0|
| stress200 |143|175|181|20|0|
| relationDEV2000 |1883|1916|1940|30|1|
| distanceDEV1000 |920|948|975|33|0|

Placement exact rises2864->2915/3000. Its remaining20 false bits are18 original
baseline errors plus2 inherited MZ28 additions. No new FP is introduced. All
MZ28 additions remain byte-identical, including72/75 retained placement far
additions; pole far coverage stays49/49 clean and48/49 stress. Agreement queries,
geometrically supported queries and original baseline-negative decisions are
unchanged, including the wrong-local-visual controls. Missing geometry never
serves as a negative event label or a CLEAR assertion.

The single lost TP is relation frame621/global11821, source index2021,
`big05_site_153`, group `bq10000-big05_site_153-crossbar`, BODY_NEAR.
RGB logit+3.212352 was correct; ToF logit-0.555607 was incorrect. MZ5 margin
1.328372 becomes the existing negative ToF score. Negative-branch confidence
0.810980 exceeds the frozen BODY_NEAR cutoff0.741196 by0.069784. It has no
geometric candidate and is an eligible disagreement; this is an actual
cross-site retention error, not a floating-point boundary mismatch.

TRAIN responsibility coverage is sharply query-dependent:

| Query | RGB-correct examples | ToF-correct examples | Raw responsibility errors |
|---|---:|---:|---:|
| BODY_NEAR |118|5|2|
| BODY_FAR |2|1|0|
| HEAD_NEAR |0|30|0|
| HEAD_FAR |0|12|0|

BODY_NEAR RGB-correct examples span117 explicit sites:48 old-source and70
relation examples. Its five ToF-correct examples comprise two old-source rows
at just `big02_site_019`, plus three MZ6 sequence rows without explicit site
identity. Both observed TRAIN responsibility errors are those two old crossbar
rows, global1824/1825, with negative-branch confidence0.348538/0.496708.
Across all168 eligible TRAIN queries the raw selector is correct166/168; the
fixed DEV cutoffs remove45 TRAIN false bits and zero true bits. These are
in-sample measurements. The aggregate120/48 admission count hides sparse
per-query/site coverage. Sufficient global counts for both classes do not
establish identifiable responsibility within each query: HEAD_NEAR/HEAD_FAR
TRAIN eligibility contains only the ToF-correct class.

The remaining20 placement FP identities are below. R/D denote relation/distance
DEV; HN/HF/BF denote HEAD_NEAR/HEAD_FAR/BODY_FAR. All site strings start with
`big05_site_`. The residual artifact stores full identity, both branch signs and
logits, confidence, cutoff and exact margins for every row.

| Cohort/frame | Global ID | Query | Site suffix | Family | Reason |
|---|---:|---|---|---|---|
| R92 |11292|HF|121|oblique_rod|Below fixed cutoff|
| R101 |11301|HN|133|crossbar|Below fixed cutoff|
| R277 |11477|HN|143|hanging_sign|Below fixed cutoff|
| R541 |11741|HN|160|crossbar|Below fixed cutoff|
| R871 |12071|HF|114|cabinet|Below fixed cutoff|
| R1049 |12249|HN|168|cabinet|Original geometry supported|
| R1171 |12371|HF|174|cabinet|Below fixed cutoff|
| R1201 |12401|HN|178|crossbar|Below fixed cutoff|
| R1437 |12637|HN|311|hanging_sign|Below fixed cutoff|
| R1488 |12688|HF|159|cabinet|Inherited MZ28 addition|
| R1564 |12764|HF|118|crossbar|Below fixed cutoff|
| R1569 |12769|HN|118|cabinet|Below fixed cutoff|
| R1584 |12784|HF|103|crossbar|Below fixed cutoff|
| R1644 |12844|HF|107|crossbar|Below fixed cutoff|
| R1645 |12845|HF|107|crossbar|Below fixed cutoff|
| R1677 |12877|BF|106|hanging_sign|Inherited MZ28 addition|
| R1781 |12981|HF|309|crossbar|Below fixed cutoff|
| R1784 |12984|HF|309|crossbar|Below fixed cutoff|
| R1793 |12993|HF|309|oblique_rod|Below fixed cutoff|
| D826 |14026|HF|107|oblique_rod|Below fixed cutoff|

Thus17 remain eligible but below the frozen cutoff, one is outside the
unsupported-query selector's remit, and two are additions deliberately preserved.
No remaining baseline FP is excluded by branch-sign agreement. HEAD_FAR's frozen
cutoff is0.993965; listing lower residual confidences is explanatory only and
does not propose posthoc calibration.

The next decision-changing check is to measure the wider original TRAIN
branch-disagreement population, including baseline-negative and geometrically
supported queries. Count both correct-branch targets and distinct explicit sites
per query, separately from the current destructive-action eligibility. This can
test whether those existing observations supply missing opposite HEAD classes
and more than one BODY_NEAR ToF-correct site before deciding any supervision
change. If coverage is added, broader responsibility supervision with the current
narrow inference modification domain is a distinct hypothesis to evaluate on
held-out TRAIN sites. If it is not, the same data do not justify that extension.
This finding does not justify simply disabling queries or changing thresholds.
Any separately declared successor must still remove at least41 baseline FP with
zero baseline TP loss and preserve additions. No expanded diagnostic, successor
fit, threshold or model change was run in this residual explanation.

Independent audit PASS covers31,200 task bits, exact masks/targets, initialization,
TRAIN normalization, class weights, exposure counts, calibration optimum/ties,
group metrics and preservation invariants. NumPy replay of47,048 selector logits
has maximum error3.93e-6. The residual pass separately asserts one lost TP and
20 remaining FP with complete identities, and binds input/code hashes.

Artifacts: `artifacts.local/work/mz30-branch-responsibility-20260910/run-v1/`
contains frozen preparation, fit and `audit.json`; `residual-v1/` contains the
zero-inference explanation and all168 TRAIN eligible rows/site counts. Both
receipts PASS and processes exited0. Residual work never edits run-v1. Frozen
base TRAIN predictions and selector measurements are in-sample; consumed DEV
calibration does not establish independent generalization. No new source,
protected EVAL, device, App/default-baseline, temporal or safety claim follows.
