# Contact-boundary exploration: relative ranking learned, metric boundary not established

2026-09-23, completed user-authorized EXPLORE under the
[fixed protocol](CONTACT_BOUNDARY_PROTOCOL_20260923.md).
**Neither cached-feature contact package passes the component criteria.**
The monotone geometry arm increases recall, but almost doubles false-positive
rate. It improves conditional boundary error without locating boundaries
accurately enough. Preserve these partial results and the exact package as a
NEGATIVE_CONTROL; A, LOCAL components, UNKNOWN and App behavior are unchanged.

## What was tested

Reuse all1728consumed query-occupancy images with original24/8/16 whole-layout
train/dev/evaluation split:864/288/576images. Same-image query changes and matched
INSIDE/BOUNDARY/OUTSIDE variants remain within their base-layout partition.
This is an internal layout-disjoint Development comparison on a previously used
source, not fresh confirmation, natural-distribution or hardware evidence.

Inputs are RGB and unchanged regional8x8single-return ToF simulation tokens.
Both arms share frozen ImageNet visual features, train-only feature normalization,
initial scene trunk, batches,100epochs and binary contact supervision. Direct
conditions an MLP on width, horizon and BODY/HEAD band. Geometry predicts a latent
folded-lateral/depth intensity grid and aggregates query-cell overlap, enforcing
monotonicity in both width and horizon. The grid has no geometric supervision
and is not validated physical occupancy. Stored carrier819633parameters is
identical; effective direct664833 versus geometry794288 is not equal capacity.

The query sweeps a virtual cross-section from axial Z=.3 to h, centred in camera
X, with fixed BODY/HEAD height bands. Travel is h-.3. It does not model a complete
human body or certify the near field. All declared rendered cubes contribute
labels; no native geometry, depth, baseline decision or metadata enters features.
All queries stay inside the old visible-surface audit domain. This does not
establish full hidden-scene knowledge. No paired appearance-only intervention
was available, and no appearance invariance claim is made.

Training and selection use four widths and nine horizons. Primary evaluation
uses three intermediate unseen widths and eight intermediate unseen horizons,
both height bands:27648queries on576images from16evaluation layouts. Queries,
pixels and adjacent poses are correlated; they are not27648independent trials.
The primary cohort has4482positive and23166negative queries.

## Measured effect

Each threshold is selected on seen-query dev layouts at FPR<=5%, then fixed.
Direct chooses epoch30/cutoff0.6229297518730164; geometry epoch10/cutoff
0.5281916856765747. Neither novel-query dev outcomes nor evaluation outcomes
select a checkpoint or threshold. These are equal dev-budget points, not equal
realized evaluation FPR points.

| Primary unseen-layout/unseen-query result | Direct | Geometry |
| --- | ---: | ---: |
| TP / FP / FN / TN |3346 /1162 /1136 /22004|3894 /2196 /588 /20970|
| Recall |74.65%|86.88%|
| Precision |74.22%|63.94%|
| FPR |5.016%|9.479%|
| Brier score |0.06456|0.08331|
| Correct ordering of truth-changing lateral pairs |2982/2988,99.80%|2964/2988,99.20%|
| Both decisions correct on those pairs |1405/2988,47.02%|1089/2988,36.45%|

Geometry gains548TP and1034FP. Its12.23point recall gain has a descriptive
whole-layout bootstrap95%interval[7.56,17.42]points; its4.46point FPR increase
has interval[3.27,5.76]points (1000paired resamples,16groups). It fails the
declared relative-cost criterion and neither arm meets the joint component gate.
Near-perfect pair ordering is therefore not an absolute decision boundary.

| Family: primary queries | Direct TP / FP / FN | Geometry TP / FP / FN |
| --- | ---: | ---: |
| BODY protruding plane |1002 /392 /384|1210 /727 /176|
| BODY suspended solid |976 /330 /446|1110 /578 /312|
| HEAD hanging plane |735 /289 /93|797 /516 /31|
| HEAD horizontal |633 /151 /213|777 /375 /69|

All four families gain recall and add FP. Full per-relation, width-only,
horizon-only, seen-query and extrapolation counts remain in result.json.
On evaluation layouts, seen-query FPR is4.866%/5.756%; combined interpolation
raises geometry FPR to9.479%. Width extrapolation gives89.69%/90.00%recall and
7.918%/8.011%FPR, not validated arbitrary-body generalization.

## Boundary accuracy and censoring

Boundary sweeps use .02m width spacing at h=3 and .05m horizon spacing at w=.6.
Both arms have zero observed probability monotonic violations at these sampled
queries. The unconstrained direct arm also learns monotonic answers here;
enforcing that property alone does not establish metric accuracy.

| Evaluation boundary measure | Direct | Geometry |
| --- | ---: | ---: |
| Width: within5cm / all528finite interior truths |37/528,7.01%|43/528,8.14%|
| Width: finite predicted-boundary coverage |483/528,91.48%|486/528,92.05%|
| Width: MAE on that resolved subset |22.12cm|18.09cm|
| Width: false crossings on624right-censored truths |108|87|
| Horizon: within5cm / all352finite interior truths |19/352,5.40%|61/352,17.33%|
| Horizon: finite predicted-boundary coverage |280/352,79.55%|294/352,83.52%|
| Horizon: MAE on that resolved subset |47.85cm|21.67cm|
| Horizon: false crossings on800right-censored truths |110|119|

Missing predicted boundaries remain failures in the within5cm denominator.
Conditional MAE excludes them and is not a complete error score. No left-censored
true boundaries occur in this cohort. At the h=.3 endpoint, the closed geometric
label could in general be positive while the intensity integral is zero; no
recorded true contact occupies that endpoint, and primary queries avoid it.

On training layouts with novel queries, recall/FPR is89.29%/4.38%direct and
98.98%/9.51%geometry. Even there, width within5cm rates are11.19%/14.76%, and
horizon rates6.07%/31.25%. The boundary limitation is not solely an unseen-layout
failure. This does not separate frozen feature limitations, sparse supervision
between query knots, optimization or output representation. No outcome-driven
additional fit was run to choose among those explanations.

## Evidence and interpretation

This check supports learning useful relative lateral ordering from public inputs
under this shared representation. It does not separately establish how much
RGB or ToF contributes, or prove that the signals identify every ambiguous scene.
The experiment retains a partial reduction in conditional first-contact error;
it does not justify a precise virtual-walking display or a system replacement.
Contact queries remain a viable research interface. This specific comparison
does not establish that direct contact prediction beats geometric aggregation.

Future work, if separately scoped, should distinguish boundary information in
the labels/features from poor absolute calibration before expanding a multiworld
or active-sensing system. Do not reinterpret this result as a general prohibition
on those different mechanisms, or retune this consumed run to rescue its gate.

Both fits used actual RTX5060Laptop CUDA:8.44s direct and9.38s geometry,2700updates
each. Frozen feature inference measured40.80msCPU versus3.34msCUDA per32images;
cloned direct training steps3.38msCPU versus2.46msCUDA. Complete scientific run
was26.31s, excluding governed wrapper overhead and later audit/plotting. These
are host timings, not endpoint or full-camera latency measurements.

22focused synthetic tests pass. They exposed and repaired two pre-fit atomic
threshold defects: unnecessary FP at zero recall, and float32 rounding of an
all-negative float64 cutoff.193536world-space query checks matched production
camera-space labels. All source/input and prediction seals remained unchanged.
Evaluation labels were computed after all predictions were sealed; full geometry
was loaded earlier for indexed train/dev labeling, so this is code-path separation,
not evaluator process isolation. Independent saved-output audit PASS reproduces
714816world-space labels, exact boundaries, all counts/pairs/bootstrap, dev-only
selection, normalization and byte seals. It confirms zero contact at h=.3 in
all1728frames and verifies the visible-domain subset. The receipt is retained in
independent-audit.json. Plot examples use the first lexicographic evaluation
HEAD-hanging group at frame5, selected by metadata rather than prediction quality.

Evidence root: `artifacts.local/evidence/ba-contact-boundary-20260923`.
It retains result.json, input/code/prediction seals, original source snapshot,
features, normalization, initialized/selected model weights, targets, full
predictions, independent audit, comparison.png/svg and plot-selection.json.
Run spec and console are adjacent `ba-contact-boundary-20260923-run.json` and
`ba-contact-boundary-20260923-console.txt`. All payloads use the canonical F:
artifact junction. The completed process started no UE, paid worker or service;
no continuing task-owned compute remains. Retained payload is durable evidence.

The documentation-link check passes. The repository-wide structure check reports
pre-existing oversized current pages, scripts/research and top-level documentation
count. These unrelated surfaces were preserved; this task adds no new route.

Terminal `terminal-contact-boundary-20260923` has explicit NEGATIVE_CONTROL
inheritance. Experiment `ba-contact-boundary-20260923` is archived through the
supported registry, anchored to source commit `ab271f65`; original ledger rows
are preserved. Initial short-SHA registration rejection and corrected full-SHA
success receipts remain beside the results.
