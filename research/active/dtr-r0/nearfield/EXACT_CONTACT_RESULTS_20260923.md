# Exact contact supervision: training fit exists, selected transfer deteriorates

One user-authorized fit completed under the [fixed protocol](EXACT_CONTACT_PROTOCOL_20260923.md).
Adding exact train-only position/censoring supervision to the unchanged direct CDF
does NOT improve the selected model. Keep this exact joint-upgrade package as
NEGATIVE_CONTROL. Retain the narrower training-fit observation below; do not treat
it as a later checkpoint's validated transfer or replace the selected checkpoint.

## Controlled change

Same MetricContactModel, initial state tensors, frozen4864D public RGB/ToF features,
normalization, sampled72query locations per864train images,100epoch schedule,
batch32,AdamW1e-3/decay1e-4 and binary positive weight50168/12040.
The original direct-CDF binary-only model is the primary frozen control. Prior
sampled geometry is an additional stronger reference, not refit or relabeled.
The864/288/576 split is by24/8/16 layouts from the already consumed same generator.

The only scientific treatment is extra train supervision/objective: binary BCE plus
unit-weight negative log probability of a5cm interval around true first-contact Z,
or survival beyond3m. Exact targets use each sampled width/layer, irrespective of
sampled horizon:62208 labels,22808finite and39400right-censored,zeroleft-censored.
Repeated widths retain their original multiplicity. No-contact is not filled with
a fake distance. The position interval is a fixed optimization convention, not
measured sensor noise. Left-contact and exactly3m endpoints have focused tests.
Loss uses stable analytic CDF differences; fixed float32 q clamps can flatten fully
saturated gradients. Inference model and feature inputs are unchanged.

This deliberately adds privileged training information; it is not a pure head or
equal-information improvement. Mixed geometry JSON is read but targets are indexed
only from train rows. This is code-path separation, not evaluator-process isolation.

Same original72query devBCE selects epoch10(.2630412), then original devFPR<=5%
rule selects cutoff.8067280650. No metric dev labels select checkpoints. All10dev
logit arrays and predeclared train-curve diagnostics are retained. Predictions are
sealed before joining evaluation targets. Neither later epochs nor new cutoffs were
evaluated/selected as replacements after the primary outcome.

## Selected model at its dev-selected cutoff

| Layout-held metric | Binary-only CDF | Added exact supervision | Sampled geometry reference |
|---|---:|---:|---:|
| Z within5cm /352finite truths |117 (33.24%)|27 (7.67%)|168 (47.73%)|
| Width within5cm /528finite truths |57 (10.80%)|10 (1.89%)|134 (25.38%)|
| Missing Z crossings |94|165|47|
| False Z crossings /800right-censored truths |111|110|146|
| False width crossings /624right-censored truths |101|45|77|
| Query TP /FP /FN /TN |3247 /1399 /1235 /21767|2220 /1351 /2262 /21815|3518 /1196 /964 /21970|
| Query recall |72.45%|49.53%|78.49%|
| Query FPR |6.04%|5.83%|5.16%|
| Z conditional MAE |20.09cm|22.37cm|20.02cm|
| Width conditional MAE |22.93cm|32.35cm|16.82cm|

One fewer false Z crossing accompanies71 additional missing finite Z boundaries.
Recall loses22.91points, Z hits25.57points and width hits8.90points versus binary-only.
Every finite truth, including unresolved cases, remains in the hit denominator.
Conditional MAE excludes missing predictions. Continuous crossing diagnostic gives
35/352within5cm,187resolved,conditional MAE20.11cm; it does not replace grid reporting.
Both the predefined supervision-effect criterion and full component gate fail.

Selected training model also regresses: Z hits286/560 ->45/560,missing116 ->264;
width208/840 ->38/840. This alone must not be labeled immutable representation failure:
the fixed training trajectory contains much better position fits at another operating
point, as disclosed next. Query selection may matter, but its causal contribution is
not isolated, and better training fit does not establish held-layout transfer.

## Predeclared train trajectory at fixed0.5 (not replacement checkpoints)

| Epoch | Dev binary BCE | Train Z hits /560 | Train Z missing | False Z /1168 | Train width hits /840 |
|---|---:|---:|---:|---:|---:|
|10|.26304|237|10|288|56|
|20|.30165|496|0|302|56|
|30|.30852|548|1|254|63|
|40|.36926|491|5|207|113|
|50|.39116|490|28|123|269|
|60|.34195|494|55|117|177|
|70|.34388|375|14|92|247|
|80|.35377|430|38|131|116|
|90|.36204|483|31|32|428|
|100|.40654|465|14|17|415|

The97.86%train Z hit rate at epoch30 includes all560finite truths, but also254
false crossings. At the fixed final epoch100 it is83.04%,with17false crossings;
width remains49.40%. These are train-only diagnostic points, not held-layout results,
not unbiased best-epoch estimates and not matched-cutoff evidence of a lossless gain.
They demonstrate that the fixed model can fit many metric positions under this
supervision, while the unchanged classification selector favors epoch10.
Do not conclude that a metric selector would fix transfer without a separate test.

All selected width/horizon probability and binary reversals are zero. Intermediate
train width curves have probability reversals and epoch60 has5binary reversals;
width monotonicity is not structural. Full saved output retains all query sets,
families,relations,pair ranking, Brier and tails.487/576original sensor UNKNOWN
frames remain metadata; prediction everywhere is not validated abstention.
Whole-layout bootstrap16groups/1000resamples: recall difference95% interval
[-35.28,-12.84]points,FPR[-1.80,+1.39]points. One seed/generator limits generality.

## Execution and evidence

15new label/loss tests pass. Equal cloned optimizer-step probes selected
CPU_FASTER_MEASURED,CPU8.53ms versus CUDA10.29msmedian. Actual fit ran CPU2700updates,
31.00seconds including scheduled train diagnostics; scientific run33.52seconds.
No paid worker,capture process,server or background allocation was created.
Canonical evidence root `artifacts.local/evidence/ba-exact-contact-20260923` retains
inputs/source snapshots, exact train labels, initial tensors, backend/fit receipts,
selected weights, all dev logits, all10train diagnostic curves and selected outputs.
Independent audit PASS:62208exact labels recomputed in world coordinates, original
binary parity, initial tensor equality, ten dev selections and train curve summaries,
all selected metrics/gates and source/input/prediction hashes. All832896selected
probabilities replay exactly on CPU (maximum error0). No retraining, bootstrap,
optimizer-loss replay or intermediate-checkpoint replay is claimed. Receipt:
`artifacts.local/evidence/ba-exact-contact-20260923-audit/result.json`.

Stop after this one fit and evidence delivery. Preserve original binary-CDF,
sampling and dense-supervision outcomes; A/LOCAL/App and UNKNOWN authority unchanged.
Future metric-based selection would be a new declared comparison, not retrospective
promotion of a consumed epoch. Controlled cross-section geometry is not full-body,
hardware, natural-distribution, event-timing or safety evidence.
