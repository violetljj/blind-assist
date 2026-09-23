# Fixed-representation dense boundary-slice supervision

2026-09-23 EXPLORE. User authorized continuing the boundary-supervision proposal
after the [error audit](BOUNDARY_ERROR_RESULTS_20260923.md).
Question: can finer, geometry-independent query supervision improve metric
boundary fitting and transfer with the existing features and geometry model?

## Fixed control and changed mechanism

Reuse the consumed1728-frame contact dataset and original864/288/576image split.
Baseline is the saved old geometry arm, including its selected epoch10 and
dev-selected threshold. It is a historical matched control, not a fresh rerun.
Preserve its NEGATIVE_CONTROL disposition and all original gates.

One new geometry-arm fit from the identical saved initial state, frozen4864D
public features and train-only normalization, original100image permutations,
batch32,100epochs,AdamW lr1e-3/weight_decay1e-4. Preserve the old aggregate
negative/positive class weight. No backbone/model/loss-parameter/cutoff sweep.
Only the training query schedule and corresponding binary labels change.

For each layer make three disjoint banks:

| Bank | Coordinates | Size | Samples/epoch/layer |
| --- | --- | ---: | ---: |
| A | Original4width by9horizon grid |36|12|
| W | Original2cm width curve at h=3, excluding A |47|12|
| H | Original5cm horizon curve at w=.6, excluding A |46|12|

Bank permutations use NumPy seed202609278, fixed before any outcomes. Each
epoch uses successive12-element cyclic windows from each bank, shared by every
image and mirrored across layers. There are72distinct queries/image/epoch,
258unique queries/image over100epochs, and7200exposures/image, equal to the old
total exposure budget. Exposure differs by at most one within each bank.
All schedule construction uses query coordinates alone, never image geometry,
labels or model predictions. Reuse the previously computed train curve labels;
verify the h=.3 labels are negative before fitting. Censored/outside examples
remain binary queries, never invented finite boundary targets.

This treatment changes query resolution, exposure allocation and unique label
count together. It does not isolate sparsity alone. Class prevalence may change
even though class weighting is fixed. Boundary slices become trained queries;
do not call their evaluation unseen-query generalization. Original combined
width/horizon interpolation queries remain untrained and are reported separately.

## Selection, evaluation and stopping

Use original seen-query dev BCE every10epochs, minimum with earlier ties, then
the unchanged atomic dev threshold rule maximizing recall at FPR<=5%.
No fine-query dev criterion selects weights or threshold. Keep all ten dev
checkpoint-logit arrays so selection losses can be audited. Seal all new train,
dev and evaluation predictions before loading evaluation targets or comparing
old outcomes. This is code-path separation on a consumed source, not protected
holdout isolation. CPU/GPU placement uses equivalent cloned initial-step probes;
actual training backend/device and timing are recorded through research_backend.

Report old/new query TP/FP/FN/TN,precision,recall,FPR,Brier, family/relation
subgroups, complete finite-interior within5cm rates, predicted coverage,
right-censored false crossings, curve monotonicity and train/evaluation results.
Native-support strata from the prior diagnostic are descriptive only.

Predeclared mechanism support requires >=10percentage-point improvement in
both width and horizon joint-within5cm accuracy on BOTH train and evaluation,
evaluation combined-query recall loss<=3points,FPR increase<=1point, and
right-censored false crossings increase<=5cases on each boundary axis.
Report fitting-only or accuracy-cost tradeoffs when the joint criterion fails.
The unchanged full component criterion remains recall>=.8,FPR<=.05 and both
evaluation boundary accuracies>=.8. No App or hardware promotion follows.

Stop after this one new100epoch fit and its audit, regardless of outcome.
No alternate seed/checkpoint/loss/bank budget or threshold rescue. Mechanical
pre-fit repairs keep receipts; an interrupted fit must not silently restart.
Finish focused schedule tests, independent saved-output audit, current/ledger
disposition and scoped delivery. Inputs, weights, failed packages and UNKNOWN
remain preserved; no new source capture, continuous-motion or safety claim.

Pre-fit mechanical receipt: initial launch reached cached-data feasibility but
failed to import tools.research_backend before probes or fit-start. Cached feature
reuse skipped the old extractor's sys.path setup. Version2 explicitly adds the
repository module root; inputs and scientific schedule are unchanged. Preserve
the original failed folder; the single fit budget remains unused at this repair.
