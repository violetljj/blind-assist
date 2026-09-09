# MZ5: preserving RGB positions before regional ToF fusion

2026-09-10 EXPLORE. The fixed three-arm comparison and its permutation diagnostic
are complete. **The local representation adds only one exact row over its matched
pooled control and remains 23 rows below the fixed ensemble.** Preserve the
implementation as a mechanism control; use the [fixed ensemble](MZ5_FIXED_ENSEMBLE_RESULTS_20260910.md)
as the next controlled learned-readout baseline. This is not a rejection of all
spatial fusion methods or a default-App change.

## Frozen comparison

[Protocol](MZ5_SPATIAL_FUSION_PROTOCOL_20260910.md) was written after the fixed
ensemble diagnostic, before these three fits. All use the existing 5000 controlled
frames: 2500 TRAIN_ONLY, 1000 DEV_ONLY and 1500 already consumed EVAL_ONLY.
RGB encoder, normalization, query geometry, clean generic 8x8 radial ToF packet,
targets and original alerts are unchanged. Native geometry supplies labels and
the existing simulated packet; it does not enter the new RGB feature cache.

[Feature extraction](mz5_spatial_features.py) retains the original 12x27x64
sampled features before averaging. [The model](mz5_spatial_fusion.py) combines
each sample with fixed XYZ, its angular zone's two ranges/validity bits, valid
range residuals and ToF coverage. A point MLP 74->64->32 precedes masked query
pooling; the final 1412->88->4 MLP also receives the original visual/ToF features.
Regional returns remain possible surface evidence, not per-ray depth truth.

- POOLED_FUSION broadcasts the query mean RGB feature before the point MLP.
- LOCAL_FUSION retains the individual RGB features at their projected positions.
- LOCAL_RGB_ONLY retains positions but zeros observed ToF in both input paths.

All have 131580 parameters, the same seed53 initial parameters and the exact
MZ1 seed59 schedule: 300 batches of 128, AdamW lr0.001, weight decay0.0001,
mean BCE. Only TRAIN_ONLY targets participate in optimization. There was one
fit per arm, final checkpoints only, no threshold or checkpoint selection.

## Results

Exact means all four BODY/HEAD x near/far event flags match the labels. Wrong-far
counts far assertions on the 600 body/head near-only opportunities. Cross-body
counts BODY->HEAD plus HEAD->BODY confusion, with 300 opportunities each.
HEAD_ONLY near is a separate 150-frame subset; negative controls contain 600
frames. A zero prediction does not establish CLEAR.

| Readout | Exact /1500 | Wrong-far /600 | Cross-body /600 | HEAD_ONLY near exact /150 | Negative exact /600 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed ensemble | 1378 (91.87%) | 4 | 8 | 137 | 578 |
| LOCAL_FUSION | 1355 (90.33%) | 7 | 13 | 134 | 548 |
| POOLED_FUSION | 1354 (90.27%) | 9 | 16 | 131 | 546 |
| LOCAL_RGB_ONLY | 1298 (86.53%) | 15 | 21 | 125 | 543 |
| LOCAL_FUSION with fixed RGB permutation | 1353 (90.20%) | 7 | 13 | 134 | 548 |
| Frozen MZ1 FUSION | 1353 (90.20%) | 12 | 19 | 133 | 544 |
| MZ0 retained information component | 1310 (87.33%) | 7 | 15 | 131 | 589 |

LOCAL_FUSION gains16/loses15 exact rows against POOLED_FUSION, gains84/loses27
against LOCAL_RGB_ONLY, and gains29/loses52 against the ensemble. It gains19/
loses17 against MZ1 FUSION and gains145/loses100 against MZ0. All row identities,
event confusion tables and condition/family/region strata are retained in
[analysis.json](../../../../artifacts.local/work/mz5-spatial-fusion-20260910/run-v1/analysis.json).

The predeclared necessary criterion passes: exact is above both matched controls,
HEAD_ONLY near improves by3, and wrong-far/cross-body decrease against pooled.
Its magnitude is weak: exact improves by only1/1500, with 31 paired changes;
one region improves and two decline (local versus pooled: 473/478, 432/425,
450/451; each denominator500). This is not evidence of a reliable general gain.

The seed83 diagnostic permutes valid RGB samples within each query, preserving
their multiset and mean. It loses2 exact rows and gains0; HEAD_ONLY near and
the two named error totals are unchanged. The model depends weakly on this
specific spatial assignment test. The unpermuted global pooled RGB path remains
available, so this does not measure all spatial information in the system.

DEV exact is pooled864, local869, local-RGB810 out of1000. These are reported
without selection. Local's far-event true positives are BODY272 and HEAD265
out of300 each, versus ensemble257 and255: higher total exact does not imply
uniform superiority of the ensemble.

## Engineering and independent checks

The first recache, cache-v1, stopped before any fit: disabling cuDNN TF32 differed
from the original extractor's arithmetic and failed feature parity (maximum
error0.008072 on the first16 images). A diagnostic restored the original setting;
the final cache-v2 keeps cuDNN TF32 enabled and linear-head TF32 disabled. The
parity tolerance was not relaxed. The failed directory, partial payload and
source snapshot remain available; this was an engineering repair, not a second
scientific fit.

Final recache took52.57s on CUDA. Its averaged features agree with MZ1 within
2.39e-6 and all5000 original B alerts and JOINT flags agree. The cache contains
279 camera-valid query points, including230 within the generic ToF field of view.

The equivalent batch128 forward/BCE/backward placement probe measured CPU
28.39ms and CUDA1.73ms median, selecting CUDA on the local RTX5060 Laptop GPU.
The three fits took1.029/0.717/0.703s; the cached-head run took3.918s. These are
cached training timings, not camera-to-alert latency. Dense head MACs per frame
are 324*(74*64+64*32)+1412*88+88*4 =2322624, about17.58x MZ1's132096,
excluding feature extraction, indexing, masking and activations. Parameter count
alone would hide this extra per-point computation.

[Independent audit code](audit_mz5_spatial_fusion.py) checks geometry, zone axes,
all74 channels, pooled permutation invariance, complete RGB-only ToF ablation,
initialization, schedule and cached-feature parity. A separately expressed
direct-tensor forward reproduces all5000 rows of each arm plus1500 permuted
rows with maximum logit error0 and identical flags. Independent scalar scoring
matches9 split/arm groups,9 aggregate and297 stratified groups, plus paired
strata. Training itself was not independently repeated.

Evidence: [run receipt](../../../../artifacts.local/work/mz5-spatial-fusion-20260910/run-v1/receipt.json),
[cache receipt](../../../../artifacts.local/work/mz5-spatial-fusion-20260910/cache-v2/receipt.json),
[CPU invariants](../../../../artifacts.local/work/mz5-spatial-fusion-20260910/audit-self-v1/audit.json),
[full replay](../../../../artifacts.local/work/mz5-spatial-fusion-20260910/audit-run-v1/audit.json),
[analysis audit](../../../../artifacts.local/work/mz5-spatial-fusion-20260910/audit-analysis-v1/audit.json).
Source snapshots, checkpoints, schedule and hashes remain in the artifact tree.

## Decision boundary

Retain the local/pooled implementation and completed result as a controlled
mechanism comparison. Its marginal result does not justify making this local
architecture the practical baseline. Keep the compact fixed ensemble for the
next controlled comparison, with its far-recall tradeoff explicit. Further work
needs a distinct explanatory hypothesis; no continuation, extra seeds, temporal
fit, new source collection or threshold search belongs to this completed run.
All results use consumed synthetic/controlled Development and a generic two-return
observation model, not VL53L8CX firmware, natural walking or safety evidence.
