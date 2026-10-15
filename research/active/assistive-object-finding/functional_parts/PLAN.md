# Functional part localization pilot

2026-10-15 / EXPLORE. User-authorized 200–300-image automatic pilot to run a
strong available baseline, identify errors, and test headroom for local detail
refinement and reliable pointing. Do not choose a thesis or claim novelty from
this pilot alone. This is a new experiment; completed audio/finding stop rules
remain attached to their original runs.

## Resources and deliverable

Start 14:30 HKT, stop by 16:00 HKT (90 minutes). New downloads <=4 GB total,
including <=2 GB model/source and <=750 MB sample data; GPU process wall <=1800 s
(runner <=1500 s, remaining 300 s reserved for observed implementation repairs).
Data/evaluation CPU branches <=600 s each, source review <=300 s. Use local
existing runtime, no paid services or hardware. Respect competing GPU work;
less than 3 GB free VRAM means stop that launch and report, not seize resources.
All payload/cache/temp stays under artifacts.local/work/functional-parts-pilot-20261015/.
Preserve failures and receipts; release own processes/locks when complete.

Deliver runnable code, fixed manifests and source receipts, actual baseline
scores, paired comparisons, failure strata/examples and a thesis-direction
decision. Partial or not-evaluable work must retain explicit denominators.

## Fixed data and evidence roles

280 independent image files: 240 AGD20K Seen test images and 40 IIT-AFF test
images, treated as Development. Exclude the previously inspected audit samples.
Select without looking at predictions or GT pixel content, using the fixed seed
20261015-fp. AGD uses image IDs shared across actions and then image SHA to prevent
duplicate-image leakage; class-stratified quotas cover 36 actions at least three
times. Eighty image groups are calibration and 160 evaluation. Extra available
queries of the same image are diagnostic, not new independent images.

IIT selects four images for each of ten object categories using object-label
metadata, 16 calibration and 24 evaluation. Observation is the whole image and
the query hold; no GT bounding box enters inference. Evaluate the full-image
union of native grasp=5 and wrap-grasp=9 regions, and report both labels separately.
This is a cross-dataset diagnostic, not the native AGD benchmark. Preserve any
empty union as an explicit native-empty case; do not silently discard it.

AGD continuous GT is a heatmap, not a segmentation mask. The 35 unmatched Seen
images remain NOT_EVALUABLE; missing labels do not mean absence. IIT categorical
labels support image-plane functional-area evaluation, not safe grasping,
reachability or actual user task completion. No manually created labels.

Inference manifests contain sample identity, image path and action only.
Evaluation manifests own GT, role, object category and diagnostics. All queries
sharing an image/SHA stay on the same side; record any deduplication replacement.

## Strong baseline and branch selection

Official SelectiveCL source commit 5422ed50ba34c2fb5f368fddf4b46cf5c9b597ed and
author Seen checkpoint. Preserve DINO key descriptors, ClearCLIP pixel features,
author action order/prompts, RGB direct resize to 224×224, ImageNet normalization
and the original test_forward mathematics. Use float32. Constructor adaptation
may load the complete checkpoint without redundant pretrained downloads, but
must verify strict key/shape loading and disclose any compatibility patches.

The released code returns raw CAM, CLIP×CAM (ree), and CLIP×local-mean CAM (rem).
The paper's calibration description and these code branches differ; do not call
one branch an exact reproduction of the published table. Preserve all three.
Before any evaluation scoring, choose the branch minimizing author-native mean
KLD on the 80 AGD calibration images only; tie order raw, ree, rem. Seal the
selected branch and input hashes. IIT uses the same selected branch. This is a
calibration-selected released implementation, not selection on evaluation.

First execute baseline views, then seal the head, then execute crop/flip views.
Adaptive crop centers on the selected baseline heatmap's argmax after author
cv2 interpolation to original image size, using pixel-center coordinates.
Crop width/height are round(0.5W), round(0.5H), clipped into image bounds using
integer half-open boxes. A degenerate baseline falls back to the center crop
with an explicit flag. No GT-derived crop, point, component or confidence.

## Six fixed configurations

1. M1: baseline whole-image heatmap argmax.
2. M2: the same heatmap's probability-weighted centroid.
3. M3: distance-transform interior point in the >=0.5-relative-peak connected
   component containing the baseline peak (deterministic row-major ties).
4. M4: baseline plus horizontally flipped whole image, inverse mapped and fused.
5. M5: baseline plus fixed center crop at half width/height, fused.
6. M6: baseline plus peak-guided crop with exactly the same crop size, fused.

M4–M6 use two 224×224 views each. Compare M6 against both M5 and M4, reporting
actual inference time as well as call count. All maps use author resize and
min-max normalization, not ReLU of possibly negative logits. For original-image
pointing and fusion, L1-normalize each nonnegative map; a crop is mapped only
inside its true original-image box, with zero outside. Fuse equal probability
mass: 0.5 base + 0.5 second view, then normalize if total mass is positive.
M2/M3 keep M1's heatmap metrics; only the selected point changes.

Use original raw14→224 author normalization for M1–M3 native metrics, preserving
the exact baseline path. Fused variants map original-image fusion to 224 and
apply the same metric convention; document this different resampling path.
Save raw float arrays and transformations, not only colored images.

## Evaluation and reliable output

AGD: author-native KLD=KL(GT||prediction), SIM and NSS; the author's NSS GT>0.1
fixation convention is not a functional-area segmentation truth. Point proxies
are GT intensity divided by GT maximum, GT mass in radius 0.05 image diagonal,
and distance to nearest global-GT-maximum pixel divided by image diagonal.
IIT: native union-mask point accuracy, separate grasp/wrap-grasp hits, and
distance to functional-region boundary. Keep original image/query denominators.

Three fixed output gates, with no evaluation tuning:

- M1 with spatial confidence 1-H/log(4096), measured on a 64×64 nonnegative
  probability map to remove original-image-size dependence.
- M1 with confidence 1-distance(base point, inverse-flip point)/image diagonal;
  this retains the same base point, and costs an additional view.
- M6 with the same entropy confidence. This is a system comparison, not an
  isolated causal test of the gate.

For each dataset separately, choose the highest calibration threshold retaining
at least ceil(0.8N) images (AGD 64/80; IIT 13/16); accept ties. On evaluation,
report actual coverage and retained point quality/error. Calibration coverage
is not a guarantee of evaluation coverage. Peak margin is diagnostic only.

Constant min-max maps retain zero mass, not invented uniform evidence. Argmax
uses the normal row-major first pixel; zero-mass centroid falls back to image
center; empty DT region falls back to base point. Flag every fallback. Keep
point denominators; native KLD/SIM follow author behavior while undefined NSS
and entropy are NOT_EVALUABLE. Nonfinite confidence is rejected; if calibration
cannot reach the required count, report the target unreachable without retuning.

Analyze paired rescue/harm and errors by action, object category, functional
region effective area and elongation. Size/elongation cutoffs use calibration
GT medians solely for reporting strata; they never enter inference or gates.
One image is the paired unit. Report repeated queries separately. A simple
control matching or beating the candidate is evidence against that mechanism,
not permission to rename it or pick another threshold on evaluation.

The decision is whether real local evidence and selective pointing offer
useful headroom beyond strong/simple controls, with enough support to justify
further research. This pilot does not establish a novel method or BLV benefit.
