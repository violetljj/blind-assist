# Frozen probes: some R1 HEAD-near information is readable; transfer remains mixed

2026-09-09 EXPLORE. Retain B as the final-count baseline and R1 as its recorded
negative control. This diagnostic preserves a narrower representation signal,
not a restored R1 model or automatic R2 fit.

## Execution

[Predeclared protocol](BODY_QUERY_PROBE_PROTOCOL_20260909.md),
[implementation](body_query_linear_probe.py). B32-D features came from hashed
Q2 dumps. R1 features were extracted once over320 RGB frames; all TRAIN/DEV/EVAL
near outputs exactly reproduce the saved R1 outputs. Neither main checkpoint
was optimized. Six tiny linear probes (33 parameters for B/R1,4 for XYZ) each
ran1000 full-batch TRAIN-only steps on the primary RTX5060 Laptop GPU. Total
extraction, target preparation, fitting and scoring took19.01 seconds.

Two targets are deliberately distinguished: exact nearest-ray OWN_CELL versus
known other visible surface; and majority local membership mass>=0.5 versus<0.5
using the Q3 footprint. The local target is binary for this probe; R1 originally
trained on soft membership mass. Local samples touching any native UNKNOWN are
ignored. No FREE class exists. The XYZ-only control uses the same labels and
TRAIN standardization. All probes share the same zero initialization, optimizer
and fixed budget; there was no sweep or DEV-based probe selection.

## All-point AUC

| Target / features | TRAIN | DEV | EVAL |
| --- | ---: | ---: | ---: |
| Exact ray / B |0.9224|0.7869|0.7757|
| Exact ray / R1 |0.9513|0.8254|0.7983|
| Exact ray / XYZ |0.6670|0.6384|0.7014|
| Local majority / B |0.9274|0.7660|0.8326|
| Local majority / R1 |0.9614|0.7428|0.7564|
| Local majority / XYZ |0.7215|0.7030|0.7381|

Ray positive/negative point denominators are3502/58091 TRAIN,530/9642 DEV,
727/9172 EVAL. Local denominators are4299/39291,667/6566,798/6128. Both visual
representations exceed the XYZ-only control in these pooled AUCs; this does not
prove that the excess is solely visual geometry rather than learned priors.
R1 improves exact-ray linear readability, but its all-point local readability
worsens on DEV/EVAL despite higher TRAIN AUC. There is no uniform R1 superiority.

## HEAD-near is a useful, narrowly supported exception

| HEAD-near target | DEV B | DEV R1 | EVAL B | EVAL R1 |
| --- | ---: | ---: | ---: | ---: |
| Exact-ray AUC |0.5293|0.9261|0.8079|0.9314|
| Local-majority AUC |0.6028|0.9236|0.8420|0.9377|

EVAL XYZ-only HEAD-near AUC is0.6999 ray and0.8195 local. A posthoc descriptive
low-FPR check within HEAD-near also shows available ranking signal: at point
FPR<=5%, EVAL ray oracle TP is B0/66, R1 38/66, XYZ9/66. Local oracle TP is
B3/67, R1 28/67, XYZ27/67. The latter shows that the apparent low-FPR advantage
over geometry priors is small for this local target. These oracle cutoffs are
not selected deployment or final-alert thresholds.

Critically, all EVAL HEAD-near positive points come from ONE counterfactual group,
`body-query-eval-s20260930-u03`, in its HEAD_ONLY/BOTH frames. DEV positives come
from two groups. Within the EVAL positive group, ray AUC is B0.8570/R1 0.9110;
local AUC is B0.9121/R1 0.9773. The within-group check supports a local signal,
but correlated points and one group do not establish broad transfer.

HEAD-far local AUC moves the other way: EVAL B0.7848 to R1 0.7480. Do not
generalize HEAD-near results to every head, range or scene.

## Frozen DEV operating points still shift across regions

Each target/model has one global point cutoff selected on DEV at FPR<=5%,
persisted before EVAL reporting. This controls point error, not final alert error.

| EVAL all-point decisions | B TP / FP | R1 TP / FP |
| --- | ---: | ---: |
| Exact ray |203 /673|422 /1348|
| Local majority |206 /207|413 /553|

Ray EVAL point FPR is7.34% for B and14.70% for R1; local FPR is3.38% and9.02%.
At fixed probability0.5, both ray probes detect0 EVAL positive points; local B
detects0 and R1 detects4. Ranking signal is not a stable operating threshold.

For EVAL BODY_ONLY HEAD ray points (0 positives,1013 known negatives), B/R1
false-positive counts under their global DEV cutoffs are127 and365;
negative-only strata correctly have null AUC. CLEAR, LOW, ABOVE, LATERAL_OUT,
FAR_OUT and every condition also retain point denominators and errors. These are
not frame-level false-alert counts and cannot be substituted for the original
near metrics.

## Decision and interpretation

There is evidence against the blanket statement that R1 features contain no
cross-region attribution signal: a new linear decoder reads stronger HEAD-near
signals under both label definitions. The deployed R1 count path did not convert
that narrow signal into improved final decisions. This makes a direct evidence
readout a defensible future candidate for that scope, not a demonstrated repair.

The outcome does not separate representation and aggregation into a binary
verdict. Linear classifiers optimize different targets from native counts;
R1 also loses local separability outside the strongest slice and remains poorly
calibrated across regions. TRAIN objectives improved but exact convergence was
not claimed; failure of this fixed probe does not prove information absent.

Do not resurrect R1's auxiliary recipe, sweep its loss weights, or sum27 sample
ownership probabilities as native pixel counts: samples overlap/correlate and
do not measure the native count units. Binary OWN/OTHER BCE already enforces a
complementary binary decision; renaming it competitive classification alone is
not a new mechanism. Existing data already contain matched height/position
counterfactual groups; any new data proposal should target a demonstrated missing
coverage slice rather than claim counterfactuals were entirely absent.

## Evidence

`artifacts.local/work/body-query-probe-20260909/run-v1/` retains R1 feature dumps,
labels/masks, six probe checkpoints, all-split logits, selection, history,
result and receipt. Parent `validation.json` independently recomputes18 AUCs
using sorted negative comparisons, verifies confusion matrices and DEV budgets,
and hashes source/results/inputs. `near-strata.json` records the posthoc
HEAD-near oracle and within-group diagnostic. Main model steps:0; no remote
worker allocation. The process completed and released its GPU resources.

This remains small, one-source, consumed Development evidence, not fresh
confirmation, complete3D occupancy, or device performance.
