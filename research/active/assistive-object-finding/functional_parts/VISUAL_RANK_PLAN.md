# Frozen dense visual features for functional point selection

2026-10-15, user-authorized Development successor. Does spatial visual information
select useful points already in the candidate pool? No thesis or novelty claim.

## Budget and decision

Start16:14 HKT, deadline18:14 HKT,120min. Local existing GPU/runtime/checkpoint;
no paid resources, new backbone download or backbone training. New data<=300MB,
acquisition wall<=15min from network start. GPU-process wall<=900s total plus
300s observed-failure repair; feature preparation CPU<=600s, learning CPU<=600s,
evaluation CPU<=600s, independent audit CPU<=300s. Record actual resource units
and failures; wall is not a multithread CPU upper bound. Require3GB free VRAM.
All payload under artifacts.local/work/functional-parts-visual-ranking-20261015/.
Stop after fixed comparisons, preserve evidence and release owned processes.

Previous fresh200 has candidate oracle144 heatmap hits but selected76; main
headroom is selection, not proposal count. At150 new images one hit is0.667pp.
Primary comparison unit is image, not candidate. Paired quality improvement
beyond matched global-feature and score-only learners, consistent across seeds,
would justify further development; oracle alone or isolated success cases would
not. Intervals crossing0 are inconclusive, not proof of no effect. No fresh tuning.

## Data

Reuse precisely previous fit158/cal79 manifests byte-for-byte. Do not add the
previous fresh200 to training. These are custom Development roles from native
AGD Seen test images, not official train/test benchmark roles.
Target150 new image groups from733 native paired groups, excluding old240,
four initial audit groups, and all216 RGB groups downloaded in the previous
ranking experiment (including its unselected extras). Preliminary remainder273.
Metadata-only action/object-balanced hashed ordering, seed20261015-fp-visual-fresh.
Apply original/flip64bit pHash Hamming<=6 and SHA exclusions against all prior
RGB images and accepted new images; no selection using GT pixels or quality.
If fewer survive, report actual denominator without relaxing screening.
Download GT only after RGB selection; header/CRC/SHA inspection is permitted,
pixel decoding for fresh scoring only after weights, gates and predictions seal.
Project-unseen does not establish foundation-pretraining or physical-object independence.

## Frozen observations and features

Reuse released SelectiveCL Seen checkpoint, ree output, NMS/crop geometry and
six seed/refined points per family. Candidate and fixed-crop families each cost
base+3crop=4 backbone calls per image. Preserve base/flip/four-view fusion and
candidate/fixed fusion references. Share identical views when caching actual work.
Rerun old fit/cal only to capture previously unsaved features; audit original
maps and candidate coordinates against inherited outputs before learning.

Passive hooks, no mutation:
- DINO keys: aff_proj input [1,196,384], already excludesCLS; reshape row-major
  [1,14,14,384] then permute to [1,384,14,14], preserve channel order.
- Head features: aff_classifier input [1,384,14,14], after aff_ego_proj.
- Retain all36 classifier maps and three output branches, verify hook consistency.

Every learner receives193 dimensions: inherited93 score/geometry features,
36-dimensional query-action onehot,64 visual dimensions. Five fixed variants:
scores (zeros64), dino_point, dino_gap, head_point, head_gap.
Point embedding is bilinear sampled at inherited ROI pixel-center coordinates
(point-origin)*14/ROI-size, with inherited-.5 grid offset and boundary clamping.
GAP is spatial mean of the same384-channel crop feature. For both point/GAP,
L2-normalize384 vector using norm floor1e-12, then multiply shared frozen
np.random.default_rng(20261015).standard_normal((384,64))/sqrt(64) matrix.
Persist matrix bytes/hash. No fit/PCA or labels determine projection. All variants
receive identical action encoding; zero-padding matches network size/arithmetic,
not effective information. Fit-only standardization per family/variant, std<1e-6=>1.

The DINO point-vs-GAP comparison tests spatially specific information; head
point-vs-GAP is secondary, testing whether task projection retained useful detail.
This is an architecture-motivated hypothesis, not assumed feature superiority.

## Matched learners

Each family/variant uses same158 independent training images and six native
heatmap point targets. Same sigmoid-MSE MLP193->32ReLU->16ReLU->1, seeds
20261015/16/17,600updates,32image batches with replacement,AdamW lr.001,
weight_decay.001,2CPUthreads. Identical initial parameters/batch arrays across
variants; no early stopping or tuning. Mean of3 sigmoid score vectors is primary;
report individual seeds. No pairwise loss in this feature-isolation experiment.
Also each variant receives Ridge alpha1 and HGB200iterations,7leaves,lr.05,
min_samples_leaf20,l2=1,no early stopping,random_state20261015.
These are strong practical controls, not equal training-FLOP algorithms. Main
MLP feature comparisons match model/updates/calls; report extraction overhead.
Fixed-crop learners have identical training and supervision to candidate learners.

## Sealing and scoring

Training cannot open fresh GT. Seal source/configuration, projection, scalers,
weights, calibration outputs and gates before fresh prediction/GT scoring.
Prediction accepts observation-only manifests; seal all outputs before scoring.
Calibration each method separately retains>=ceil(.8*79)=64 images, ties accepted.
Report actual fresh coverage and quality; confidence-ranked50/70/80/90/100%
is diagnostic, not a deployable gate. Include complete AURC, not a chosen slice.

Primary: continuous nativeGT point quality/GTmax. Secondary: y>=.5 heatmap-hit
proxy, candidate oracle, oracle-selected gap, rescues/harms, and selection indices.
Not physical contact accuracy or BLV user success. Main predeclared comparisons:
candidate dino_point MLP vs dino_gap MLP, scores MLP, dino_point ridge/HGB,
fixed dino_point MLP, candidate fusion and cheap original baseline. Secondary
head_point vs head_gap and dino_point, all seeds.2000 paired image bootstrap,
seed20261015; disclose small sample, same-source and multiple comparisons.
Automatically choose gain/loss/oracle-gap examples after scoring for explanation.
No additional variants or thresholds after seeing new outcomes.
