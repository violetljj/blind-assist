# Native-heatmap candidate quality learning pilot

2026-10-15 / user-authorized successor. Question: can native-heatmap supervision
learn to select the better functional point on project-unseen images, beyond
same-data, same-feature, matched-backbone-budget strong controls? No thesis choice.

## Budget and decision

Start15:44 HKT, finish17:44 HKT (120min). Download<=500MB, acquisition<=15min,
GPU inference process<=900s plus<=300s for observed repairs, CPU learning<=600s,
evaluation<=600s, independent review<=300s. No paid resource, new backbone or
backbone training. Existing local runtime; require3GB free VRAM before launch.
Payload artifacts.local/work/functional-parts-learned-ranking-20261015/.
Preserve source snapshots, failures, inputs and outputs; release owned processes.

Prior candidate oracle .5300 versus selected .3449 leaves selection headroom;
however fixed crops can be stronger cross-domain. Main question is ranking loss
versus identical pointwise model, and learned candidate system versus fixed-crop
learners and TTA. Report paired counts and2000 bootstrap intervals; a200-image
binary proxy changes0.5percentage points per image. Oracle alone is not success.
Stop after fixed comparisons, without fresh-evaluation parameter search.

## Data roles and leakage controls

AGD Seen official training archive lacks same-format native target heatmaps.
Use the old240 AGD native-test images explicitly as custom Development adaptation:
old eval160 becomes fit, old cal80 remains calibration. Do NOT call this official
training or an official benchmark split. Remove fit/cal near duplicates, keeping
calibration priority, and within-partition duplicates using deterministic source
order. Report resulting counts instead of replacing them using new images.

Select200 project-unseen AGD image groups from remaining489 paired groups after
excluding old240 plus four prior audit groups. One deterministic primary action
per group; favor action coverage using metadata only, never heatmap quality.
Use64bit pHash Hamming<=6, comparing original/horizontal-flip RGB against all old
images and within fresh selection, with SHA/source-group exclusion. Reject and
replace in fixed hashed order. This is a near-duplicate heuristic, not proof of
independent physical objects. If fewer than200 survive, preserve actual denominator.
RGB selection precedes GT download/decode; GT bytes may be checked for archive
CRC/hash and dimensions, but no fresh GT pixel content is used for selection.

Fit/cal/fresh evaluation manifests are separate; observation manifests only
sample identity, image path and action. Fresh GT stays unopened by learning and
prediction code. Freeze trained weights, feature schema, all calibration gates
and fresh predictions before fresh scoring. New images were not used in earlier
project experiments; we cannot guarantee foundation pretraining exposure absent.
Old IIT is not added to training; this pilot focuses on fresh native-heatmap data.

## Frozen candidates and93-dimensional observations

Reuse SelectiveCL checkpoint,ree branch and prior NMS/crop geometry. Per family,
three half-size crops each supply seed and refined local peak, in interleaved
seed/refined order, giving six point alternatives. Keep duplicates if coordinates
coincide; do not claim six independent examples. Candidate and fixed families
each cost base+3crops=4backbone calls. TTA retains the prior4call control and
2call base+flip reference. No GT determines proposals or features.

Per-point features, in this exact order:
1. x/W,y/H,is_refined,proposal_index/2 (4).
2. crop x0/W,y0/H,x1/W,y1/H (4).
3. seed-to-refined distance/image diagonal (1).
4. B(point),L(point),B(seed),L(seed),B(refined),L(refined) (6).
5. base and local spatial confidence1-H/log4096 at64x64 (2).
6. local36 raw class-map spatial means (36).
7. local36 raw class-map values at the point, bilinear sampled (36).
8. query-action GAP raw, point raw, GAP36softmax, point36softmax (4).

B/L follow inherited resize/minmax normalization. Pixel-center bilinear sampling
uses the actual ROI and14x14 grid transform. The point softmax is merely a feature,
not a claimed calibrated probability. No object label, image ID, split, GT area,
GT distance or GT-derived features. Fit-only standardization per family, std<1e-6
replaced by1. Calibration and fresh images do not refit transformations.

Target y is native heatmap value at point divided by its image maximum, using
inherited point metric. Fit/cal GT may supply targets; fresh GT may only score
sealed decisions. Targets remain continuous proxies, not physical contact labels.

## Fixed learners and fair comparisons

Each family uses the same fit images and six targets per image:
- Ridge quality regression alpha1, fit-intercept, clip predictions to[0,1].
- HistGradientBoostingRegressor:200iterations,learning_rate.05,max_leaf_nodes7,
  min_samples_leaf20,l2_regularization1,early_stoppingFalse,random_state20261015.
- Pointwise MLP93->32ReLU->16ReLU->1: MSE(sigmoid(z),y).
- Ranking MLP, identical architecture: above MSE plus.25 times image-mean
  (1/15)sum_i<j |yi-yj| softplus(-sign(yi-yj)*(zi-zj)); zero-gap term0.

MLPs: AdamW lr1e-3,weight_decay1e-3,batch32images sampled with replacement,
fixed600updates, CPU two threads, seeds20261015/20261016/20261017. Paired MLPs
share initialization and exact batch sequence. No early stopping, dropout or
hyperparameter search. Main score is arithmetic mean of three sigmoid outputs;
report each seed too. Ridge/HGB are practical strong controls, not equal-training-
FLOP controls. MLP parameter/step/feature/data/forward budgets match, additional
pairwise arithmetic and actual training/inference CPU time must be disclosed.

All learners select maximal predicted quality among six points, ties fixed point
order. Fixed-crop family receives the same learners, not a weaker heuristic.
Also score original argmax,2call flip,4call TTA, fixed/candidate uniform fusion,
old combined verifier and first-candidate point references. Evaluation of these
controls does not introduce training/model selection on fresh images.

## Evaluation and uncertainty

Primary: mean GT-relative point quality on fresh images. Secondary: local GT
mass and distance; y>=.5 hit proxy explicitly not binary native-mask accuracy.
Report paired rescue/harm, candidate/fixed six-point oracle and oracle-selected
gap, and seed/refined selected counts. Main ranking comparisons: same-family MSE,
same-family HGB and ridge, fixed-family ranking, fixed-family HGB and4call TTA.
Use2000 same-image bootstrap draws seed20261015 for mean differences; describe
selection history, sample dependence and multiple-comparison limitations.

Learner refusal confidence is selected predicted quality. Fit each method's
threshold on calibration only to retain>=ceil(.8N), with ties accepted, then
seal before fresh scoring. Fixed-control confidence follows prior entropy or
verifier convention. Report actual fresh coverage, retained quality, quality
sum, complete risk-coverage and matched rank counts50/70/80/90/100percent.
Eval confidence-ranked counts use noGT to choose threshold but are diagnostic,
not deployable calibration. Never compare unmatched coverage as gate superiority.
No new variants after fresh outcome inspection. Automatically select gain/loss/
oracle-gap cases only after scoring, for explanation rather than prevalence.
