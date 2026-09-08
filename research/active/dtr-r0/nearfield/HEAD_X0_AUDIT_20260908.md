# HEAD-X0 cross-region shortcut audit

Pre-outcome EXPLORE audit, no training or new threshold selection. Full F2000
fits all1134 TRAIN near decisions and improves DEV, but its frozen DEV HEAD
threshold misses15/15 plaza positives and fires on171/735 negatives. This
establishes a transfer failure, not its cause or a universal capacity conclusion.

Question: does the error structure support score-order reversal, target-support
morphology mismatch, context sensitivity, or a combination? Preserve G13-D,
original seed17, frozen B2000, full F2000 and existing data identities. DEV and
plaza are consumed Development; no fresh confirmation or improvement claim.

Cached analysis covers HEAD scores for all1134 TRAIN,128 DEV and750 plaza
frames. Show six region-by-label distributions, tie-aware ROC-AUC and average
precision, and their denominators. Cached probabilities can yield reconstructed
clipped logits, not recovered exact raw logits. A low overall AUC alone does
not prove every operating point useless; retain descriptive low-FPR recall
envelopes without choosing or exporting plaza thresholds.

For all171 frozen-threshold FP and15 FN, save RGB, support/GT, peak and patch,
connected components, max/mean/activation area/centroid. Fixed seed17,k=4
peak-patch clustering supplies medoid examples, not inferred semantic labels.
Inspect all15 plaza positives. Compare GT support area/bbox/aspect/centroid,
orientation/thickness proxies and RGB contrast across partitions. These are
32x18 pooled body-query surface masks, not complete object silhouettes or
physical rod thickness; UNKNOWN is not background. Preserve source families
when available, never infer asset identity from the class label.

Bounded inference selection: plaza15positive+171FP, TRAIN15positive (old3,
three per each of four new families, distinct groups), DEV8positive (two per
family). Total209. Freeze exact identities before inference. Each model gets
original, retain ROI and erase ROI under mean-fill/blur, with equal-area
off-target sham ROI controls: nine variants per image. Positive ROI comes from
GT HEAD support; FP ROI is explicitly prediction-conditioned F support, with
its construction and empty-mask fallback recorded. These are surface-ROI
sensitivity probes, never complete foreground/background isolation. Compare
paired score changes and sham effects, retain all selected examples.

Fifteen fixed TRAIN/plaza positive pairs additionally receive bidirectional
donor-ROI copy-paste at unchanged donor pixel location/scale, with equal-area
donor sham-patch controls and unedited destination reference. Do not claim
geometrically valid labels, disentangled object identity or causal proof from
these distribution-shifting edits. Save masks, provenance and preview examples.

For feature drift, verify original-versus-B backbone identity first. If equal,
their backbone feature rows must agree; do not present them as separate drift.
Compare original and F features against complete TRAIN HEAD-positive and
negative references within each model, excluding same-group references. Report
positive and negative nearest cosine affinity plus the gap, not only absolute
similarity changes across representation spaces. Distinguish contextual deep
features pooled at an ROI from isolated patch/object embeddings. Include plaza
positives, FP and TN in feature distributions; selection metadata is evaluator
diagnosis and never training supervision in this audit.

Stop after cached analysis and this bounded inference. No augmentation training,
new loss, ensemble, step extension or threshold rescue. If responses track
both sham and target edits, report perturbation sensitivity without naming a
background cause. If target morphology differs, report support-domain coverage
evidence without claiming it explains all errors. Retain mixed/null findings.
Release task-owned worker resources; preserve input/source hashes and receipts.

## Source context

[Right Regions, Wrong Labels](https://arxiv.org/abs/2604.13326) studies semantic
identity flips under scene/category correlation shift; this is motivation to
separate localization from decisions, not evidence of a flip in our binary
HEAD model. The supplied [foreground/background pooling manuscript](https://openreview.net/pdf/dc40f55dd05b3cff0f7ea253fc9ae118339499ba.pdf)
is marked under review and concerns object detection with foreground masks.
Its complete-object-mask premise differs from our body-query support masks.
The attachment's project FPR numbers come from local experiment results, not
either paper. No causal conclusion or method adoption follows from those links.

## Cached results: reversed ranking and concentrated response

| HEAD partition | Positive / negative | Frozen B2000 AUC / AP | Full F2000 AUC / AP |
| --- | --- | --- | --- |
| TRAIN1134 | 207 / 927 | 0.89602 / 0.58229 | 1.00000 / 1.00000 |
| DEV128 | 64 / 64 | 0.66968 / 0.68997 | 0.90601 / 0.92033 |
| Plaza750 | 15 / 735 | 0.52472 / 0.02272 | 0.25751 / 0.01327 |

F old750 and new384 each separately have AUC1, not just a favorable mixture.
F positive/negative probability medians are0.999848/0.00007249 TRAIN,
0.996876/0.377893 DEV and0.489716/0.831268 plaza. Reconstructed-logit plots
include saturation counts; they are not raw logits. Root independently checked
plaza AUC0.2575056689342404/AP0.0132688615743568 with sklearn.
The descriptive plaza F recall envelope is zero at all three FPR ceilings
1%,5%,10%; B attains1/15 at10%. This directly rules out an ordinary increasing
scalar threshold attaining useful low-FPR recall on these consumed F scores.
No reverse-score classifier or plaza threshold is selected.

All171FP and15FN have saved RGB/support/GT/patch panels and scalar records.
The fixed k4 low-level patch clustering includes all186 errors; its medoids
are descriptive representatives, not discovered object categories. Among FP,
113 peaks fall at grid(x14,y8),35 at(x17,y8),19 at(x13,y8). Root viewed all12
medoid panels: peaks repeatedly point toward the distant building entrance
and adjacent tree/shadow region, rather than the injected bicycle/bench nearby.
This is a visual observation of representative panels, not a semantic count
of all171 errors. In all FP,136 peaks are known GT0 and35 UNKNOWN. Every FN
peak misses the positive mask:12 known0,3 UNKNOWN; UNKNOWN remains unknown.

## Morphology and effective coverage

All15 plaza positives use the same mesh
`/Game/Prop/Kit_Scaffolding_RR/Mesh/SM_Scaffolding_metal_N1`, with camera poses
00/01/03/07/09 and distances d0/d1/d2. There are15 formal condition groups but
only five pose settings for this one asset, not15 independent scenes. Visual
inspection of all15 confirms metal scaffold views; positives are not a novel
tree-branch family. Positive surface patches include diagonal intersections
and vertical members, so naming the full object does not determine the mask.

Median pooled positive area is6 cells TRAIN,8 DEV,6 plaza. Plaza aspect ratio
median1.0 and top offset6 cells differ from complete TRAIN medians1.333 and8,
but the old15 TRAIN scaffold cases have the same area/aspect/top distributions.
Root's exact paired check goes further: sample ids old+750 identify15 plaza
counterparts, and every paired positive HEAD mask is bit-identical at32x18.
Old F scores range0.9915 to0.999994, while plaza ranges0.000267 to0.959890.
Therefore absence of the pooled positive morphology/location is not a
sufficient explanation for these pairs. This does not prove full-resolution
geometry, illumination, occlusion, texture or complete object appearance equal.

The user's coverage hypothesis is plausible in this narrower form: available
TRAIN does contain the scaffold relation, but may not teach it across the
relevant appearances and backgrounds. Counts are frames, not independent worlds.
No sample-size scaling experiment or hard-negative training has been run here.
Adding consumed plaza examples to TRAIN would make subsequent scores on those
same examples training/adaptation results, not cross-region generalization.
Any such pilot needs group/source separation and an untrained evaluation set.
The relational384 already contributes48 HEAD positives per each of four
families and same-family negative variants. Thus absence of all relational
negatives is not established; the specific missed background/appearance
coverage should guide acquisition. Neither success nor failure of one small
data pilot can by itself prove architecture adequacy or inadequacy.

## Input interventions: some response specificity, substantial edit dependence

The accepted suite uses209 selected images and60 bidirectional paste variants,
1941 evaluated images per each original/B/F model. All209 shams preserve ROI
area and have zero overlap. FP ROI is F's top-K support cells, where K is the
selected-positive median area, not a ground-truth obstacle. Fifteen selected
TRAIN positives and eight DEV positives are bounded examples, not all positives.

Below is F's median paired probability difference, edited ROI minus edited
sham, in percentage points. Negative erase differences indicate a stronger
score reduction at the selected ROI; positive differences indicate the reverse.

| Selection | Erase by mean-fill | Erase by blur | Retain ROI, mean-fill outside | Retain ROI, blur outside |
| --- | --- | --- | --- | --- |
| TRAIN positive15 | -0.15 | -0.41 | -0.02 | -0.05 |
| DEV positive8 | +2.66 | -1.93 | -1.39 | approximately0 |
| Plaza positive15 | +3.70 | -0.16 | 0.00 | approximately0 |
| Plaza FP171 | +2.07 | -8.11 | 0.00 | +0.008 |

Probabilities saturate on TRAIN: its median erase-versus-sham raw-logit
differences are-5.607 mean-fill and-5.038 blur even though the probability
changes look small. Plaza FP corresponding raw-logit medians are+1.817 and
-1.477, confirming the sign depends on replacement. Blurring FP response
regions has a specific relative effect, but mean-fill does not corroborate a
simple removal story. Retain-only edits often drive both real and sham choices
toward a similar response. Their large absolute changes cannot be credited
solely to isolating the true foreground.

The15 TRAIN-to-plaza support-patch pastes have median probability change+10.72pp
relative to the unedited destination and+18.87pp relative to the sham paste;
the latter IQR is-9.56pp to+46.12pp. Plaza-to-TRAIN pastes largely retain the
already-saturated TRAIN probability (median change-0.003pp); their median
raw-logit change is-2.977. These do not cleanly isolate foreground or background:
only a few support cells are copied, original destination objects remain, and
global geometry is not recomputed. Root inspected original/retain/erase/sham
and bidirectional-paste example images. No edited input is used for training
or assigned a synthetic risk label. No intervention establishes pure background
causality or validates mask-outside augmentation as a remedy.

## Feature affinity: conditioned local evidence shifts

Original and B backbone tensors and actual selected features are exactly equal.
Only original/F are compared for backbone drift; original is the G13 seed17
initial checkpoint, not an independently evaluated fresh ImageNet model.
There are893 feature queries (all750 plaza,128 DEV,15 selected TRAIN), with
all1134 TRAIN references,207 HEAD-positive/927 negative. Same-group references
are excluded. Features are contextual native5x8 deep maps; ROI pooling uses
bilinear18x32 equivalents. Reference ROIs use the query's coordinates, rather
than each reference's own object mask.

Median nearest-positive minus nearest-negative cosine affinity:

| Plaza query | Original whole | F whole | Original ROI | F ROI |
| --- | --- | --- | --- | --- |
| Positive15 | -0.00048 | +0.01896 | +0.00118 | -0.05131 |
| FP171 | -0.00039 | +0.04143 | +0.00059 | +0.15039 |
| TN564 | -0.00048 | +0.02886 | -0.00063 | +0.02709 |

The local affinity contrast is consistent with F encoding FP response locations
as more HEAD-positive-like. It is not an independent causal attribution: FP/TN
use F-selected peak cells while positives use GT support; the feature receptive
field includes context, reference classes have unequal size, and cosine geometry
changes across models. Whole-image affinity also moves positive for true
positives and TN, so an unqualified selective-manifold-warp story is too strong.
Both mean and nearest-reference affinities and identities are retained, without
choosing an affinity threshold or changing the classifier.

## Disposition and next bounded step

Retain the audit as Development evidence. Score-order transfer failure,
concentrated non-target response and matched positive support geometry support
testing targeted appearance/context coverage next. The edit controls prevent
claiming that background causality is solved. G13-D stays unchanged; no new
weights, loss, ensemble, calibration rule or augmentation recipe was adopted.

A useful next data pilot is one small source-separated expansion, with positive
relations and visually similar negatives sharing each added background, then
evaluation on untrained groups/regions. The171 consumed FP can guide what to
reproduce; they must not be quietly added to TRAIN and then scored as held-out
success. Prefer one64-frame pilot over an immediate five-arm scaling sweep.
Whether to use a source-separated portion of plaza or freshly generated regions
must be made explicit before acquisition/fitting. That pilot was not run here.

## Execution, recovery and artifacts

Cached analysis took10.723s on CPU, with zero inference/training; a separate
cached-only supplement creates the six-distribution plot. It validates sample
identity and array hashes. Original cached score files without sample identities
are omitted from the full distribution comparison rather than guessed.

The initial probe stopped at cached parity: batch9 differed by0.00138479 in
near probability from the original batch32 path. Restoring deterministic flags
did not fix that discrepancy; original batch32 inference across all four
partitions instead had exact near/support equality. The accepted mechanical
recovery fixes edited forward batches at32 by padding and discarding filler
rows. Its preflight checks all209 originals with zero cached near/support
difference and exact custom-versus-model API equality. No tolerance was relaxed.
Two failed inference attempts and an undispatched launch-preflight error remain
recorded; none trained parameters. The final suite uses uniform backend settings
for all models and preserves the same samples, edits and thresholds.

Accepted worker run: Torch2.9.1+cu128, RTX3060 Laptop,62.36s script/66.97s job.
This duration excludes earlier failed attempts, transfer and code preparation.
Three models each1941 logical images are evaluated, with pad32 filler excluded
from reported sample counts. Raw feature arrays remain on the worker;88 thin
files/33,725,466bytes are transferred with SHA verification. Source and protocol
execution snapshots, masks, predictions, identities, parity and release receipts
are under `artifacts.local/work/head-x0-20260908/recovery-v3/`.
Executed probe source SHA:
`21c91865bee1d3023fc07cf93bb3231b8ed1ba0c73515a675c834c9b89e5ff55`.

Root independently checked plaza ranking, all15 paired positive masks, zero
sham overlap/equal areas, and summarized saved paired predictions and affinities.
Root viewed the six-distribution figure,12 medoids,all15 positive previews,
intervention/paste examples and the feature-affinity figure. Task-owned processes
and scheduled jobs are released. Worker durable outputs remain under
`G:/DevWorkspace/BlindAssist/artifacts/work/head-x0-20260908/`; transfer archives
are removed. Full per-frame evidence and plots are in `cached-v1/`, with
`recovery-v3/summary.json` and `feature-affinity.png` supplying final probe summaries.
