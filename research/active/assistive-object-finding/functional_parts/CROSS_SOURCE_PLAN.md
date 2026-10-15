# Frozen functional point selection on a separate native-mask source

2026-10-15 user-authorized Development successor. Test whether the local-visual
selection signal transfers from AGD heatmaps to IIT-AFF categorical grasp regions.
No thesis choice, target-domain training, parameter search or safety claim.

## Budget and decision check

Start16:38 HKT, deadline18:38 HKT (120min). Existing local GPU/checkpoint/runtime,
no new backbone/download/training or paid service. New data<=300MB, acquisition
wall<=15min from network start; GPU-process<=600s plus180s observed-failure repair.
Feature/prediction CPU<=600s, scoring CPU<=600s, independent audit CPU<=300s.
All payload: artifacts.local/work/functional-parts-cross-source-20261015/.
Preserve attempts and actual measured units, release task-owned processes/locks.

Old IIT40 images are consumed Development, not this evaluation. AGD fresh150
showed local-point benefit but strongest fixed-crop tree/fusion remained unresolved.
IIT native masks test a different label type and source, on unseen project images.
The denominator is image, not six candidates; at150 images one hit is0.667pp.
If candidates rarely hit, report proposal ceiling instead of blaming only selection.
Native-empty masks stay in the primary denominator; do not select for easy masks.
Consistent local-point benefits over GAP/scores and strong fixed/fusion controls
would motivate broader generalization work. No gain distinguishes a limited
same-source signal, without declaring all visual representation methods invalid.

## Data and target

Target150 unused IIT-AFF native test images. Reuse cached archive index/object
metadata to order class-balanced candidates deterministically with seed
20261015-fp-cross-source. Exclude all previously downloaded IIT RGB, including
initial audit and old pilot extras, by sourceID/SHA. Screen64bit original/flip
pHash Hamming<=6 against old IIT RGB and accepted new RGB. Record source prefixes
and class composition; heuristic dedup does not prove independent physical scenes.
RGB selection is sealed before native mask download. Never use pixel mask size,
presence, model scores or predictions for selection. If150 cannot survive within
budget, preserve actual nonzero count; never relax the screen to reach target.
GT downloads are raw text bytes, CRC/SHA only; no target mask parsing before
model outputs, points and thresholds are sealed. Evaluator checks native integer
IDs0..9 and dimensions, retains all native-empty images, records any invalid input
as NOT_EVALUABLE rather than treating corruption as background.

Pre-download clarification16:42 HKT: because both sources include web images,
apply the same SHA/pHash screen against all downloaded AGD RGB too (pilot240,
learned-ranking216, visual-ranking160 and initial audit RGB). No network or
newGT access preceded this clarification; original plan bytes are retained in
source_snapshots/CROSS_SOURCE_PLAN_before_cross_source_dedup.md. Keep candidate
order, thresholds and model decisions unchanged. All model outputs are pixel
centers n+.5; assert this so floor indexing equals inherited midpoint-tie rule.

All queries are the model's existing `hold` action. Evaluator target is native
grasp5 OR wrap-grasp9, full-image union over all instances, no object-box clipping.
This deliberately narrows the task to locating a graspable region, not resolving
a named instance, arbitrary part taxonomy, physical grasp success or BLV usability.
No target-domain images/GT join fit or calibration. Reuse exact prior AGD158/79
training, scalers, projection and gates. Unseen images do not certify absence from
foundation model pretraining; source was previously explored on other40 images.

## Frozen models and equal-compute controls

Use visual-ranking-20261015/training/training_seal.json and all its file hashes;
sealSHA62565df59aa967590cfc9d99c1b58bb12061809a6da2339b2e188eab8a353e43.
No retraining or calibration. Run unchanged run_visual_inputs.py on observation-only
manifest, retain raw DINO/head feature maps and inherited maps/candidate geometry.
Use the same five193D feature variants, projection, native pixel-center sampling,
and two6-point crop families. Every learned family and fusion costs4 backbone
calls/image; MLP variants have same6753parameters and3seed ensemble. Baseline
original1call and flip2call remain useful cheaper references. Fixed-crop HGB is
a strong practical comparator, not an equal-training-FLOP neural-network control.

Primary method: candidates_dino_point_mse, unchanged3seed score mean.
Seven predeclared primary paired comparisons: candidate dino_gap_mse, scores_mse,
dino_point_ridge, dino_point_hgb, fixed_dino_point_hgb, C0 candidate fusion, B0.
Report fixed_dino_point_mse, other predeclared variants, all individual seeds and
both candidate-family oracles too; no method selection on IIT. Gates for all68
methods inherited byte-for-byte from visual-ranking evaluation/sealed_gates.json
(derived solely from AGD79cal). Do not use old IIT16cal or new IIT to recalibrate.

## Sealing, metrics and interpretation

First verify source training hashes and feature/seed prediction consistency;
seal new scores, selected points, reference decisions and inherited gates. Only
then parse fresh native masks. Pixel hit uses floor(x),floor(y) under inherited
pixel-center convention, bounds asserted. Native IDs5/9 categorical membership
is primary; no bilinear thresholding or heatmap normalization of mask labels.
Report all-image hits/N and nonempty hits/N_nonempty; separate IDs5/9 and selected
wrong native labels. Empty target hit0, distance unavailable/null, not invented1.
Secondary: nearest target-region pixel distance/image diagonal on nonempty only,
candidate oracle hit, selected hits among oracle-positive, saved/harmed counts.
Report per-class diagnostic counts without picking a favorable subgroup.
Use2000 paired image bootstrap seed20261015 and two-sided exact binomial test on
discordant hit pairs (descriptive, unadjusted multiple comparisons). Also report
three individual-seed local-vs-GAP and local-vs-score differences. Do not treat
image pairing/pHash as physical-scene independence.

For inherited gates report retained count, accepted correct/wrong counts,
accepted error fraction among kept and all images, and native-empty accepted
count/empty denominator. Ranked50/70/80/90/100percent coverage and binary-risk
AURC are diagnostics; same-domain confidence is not assumed calibrated in IIT.
Auto-pick gain/loss/oracle-gap cases after scoring, display native mask overlay,
not manual labels. Stop after fixed analysis; no tuning from this outcome.
