# NF-G12: pretrained representation and shallow detail

Completed 2026-09-08. EXPLORE, controlled same-Willow Development. G10 remains the incoming baseline; [G11](DETACHED_20260907.md) detach experiment is closed.

## Decision

Predeclared selected arm: **original_gate**. The full joint decision rule and every failed criterion are retained in `summary.json`; this is a descriptive Development decision, not statistical noninferiority, natural-scene or device certification.

**Representation replacement has a useful measured decision gain.** B improves
primary joint25/32 to30/32, bar recall28/32 to32/32, and total HEAD detection55/64
to64/64 without HEAD false alerts. Its paired primary seed counts are
26/29/28 versus A26/24/24; compatibility also improves22/32 to29/32. Retain B as
a strong decision challenger. It misses the complete replacement rule because
diverse joint stays14/16 and HEAD peak hit declines32/64 to29/64, despite improved
fixed-threshold IoU. HEAD VAL-mask IoU is also essentially unchanged/slightly
lower0.285 to0.281. This is not evidence that representation upgrading failed.

C improves both support IoUs and HEAD peak hit relative to B, but primary
ensemble joint falls30/32 to27/32 and diverse14/16 to12/16. That decline is
specific to the calibrated ensemble readout: C's three primary seed joint counts
27/31/28 are no worse than B26/29/28, and compatibility ensemble counts tie29/32.
Do not turn this into a working-point-independent claim that detail harms
classification. Retain C for the disclosed detail contrast; the current full
joint contract selects neither challenger to replace G10. No additional fit,
checkpoint choice, threshold sweep or consumed-cohort tuning follows this result.

## Fixed design

A `original_gate` reuses all three G10 expanded-region checkpoints with zero fits. B `repvit` uses official RepViT-M0.9 ImageNet 300-epoch distillation weights, explicitly loaded strictly before discarding classification heads. C `repvit_detail` differs from B only by an additional shallow-feature fusion branch. A→B measures the entire representation replacement, including initialization and input normalization; it cannot isolate the contribution of pretraining. B→C measures the added detail branch under the paired setup.

Official source revision `298f42075eda5d2e6102559fad260c970769d34e`; checkpoint SHA256 `857eb0e6a992591a16ed96b22a73135270032d2b97c2c1f74dd3d0050e0c83e8`. Source, license, README, transform code and weight hashes are retained in `pretrained/provenance.json`. [Official RepViT](https://github.com/THU-MIG/RepViT) supplies the backbone; the separation of context and detail is inspired by [BiSeNet V2](https://arxiv.org/abs/2004.02147), without its complete segmentation system.

Input remains full-scene RGB 144×256; B/C use ImageNet mean/std, with no center crop or new augmentation. Deep stride32 features (5×8) project to32 channels and bilinearly align to18×32. C adds the last stride4 feature (36×64), projected to16 channels then a3×3 stride2 convolution to32 channels. The support1×1 and unnormalized predicted-region-gated near head retain end-to-end gradients. Common B/C modules are identically initialized per seed before optional detail modules. The task grid and native support supervision remain18×32; no upsampled mask is treated as new fine truth.

Same800TRAIN images and G10 common96VAL, balanced batch32 with eight per geometric variant. B/C use seeds17/29/43, AdamW lr1e-4/weightdecay1e-4, nearBCE +0.25supportBCE. TRAIN-only throughput probes precede the frozen full-fit budget in `budget.json`; probe weights are discarded. Checkpoints use common VAL near-BCE first minimum, never TEST selection. No backbone search, detach, new loss, temporal module or post-outcome sweep.

## Evaluation contract

Primary calibration uses only G10 VAL: BODY overall positive recall>=95%; HEAD overall and bar-only positive recall>=95%. Among feasible inclusive thresholds in[0,1], minimize false positives then maximize threshold. Full near truth/prediction coverage is required. At48 positives/head this permits at most2 VAL misses; at24 bar-only positives at most1. Validation feasibility does not establish TEST recall. Secondary is the unchanged per-seed/ensemble G8 threshold compatibility readout; both policies and all seeds are reported.

Fresh TEST is32groupsg6000..g6031, seed12080908,16narrow16diverse, four geometric variants per group. Same G10TEST parameter ranges, same Willow map, no prior group IDs reused. Native intended-object visibility/query/floor verification passes all128frames. This batch is now consumed. All A/B/C see exactly the same inputs; G10 scores on a different batch are not a model improvement over earlier G10 results.

## Fresh results

### Primary ensemble

| Arm | Joint /32 | Narrow /16 | Diverse /16 | BODY TP/FP/FN | HEAD TP/FP/FN | Bar HEAD TP /32 | Box HEAD FP /32 |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| original_gate | 25 | 11 | 14 | 60/2/4 | 55/0/9 | 28 | 0 |
| repvit | 30 | 16 | 14 | 61/0/3 | 64/0/0 | 32 | 0 |
| repvit_detail | 27 | 15 | 12 | 62/0/2 | 61/0/3 | 32 | 0 |

Each head has64 positive and64 negative TEST samples. Joint correctness requires both heads correct across all four variants.

| Arm | Seed17 joint | Seed29 joint | Seed43 joint | Bar-removal success /32 | Box-removal success /32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| original_gate | 26 | 24 | 24 | 25 | 25 |
| repvit | 26 | 29 | 28 | 30 | 30 |
| repvit_detail | 27 | 31 | 28 | 27 | 28 |

### Secondary ensemble

| Arm | Joint /32 | Narrow /16 | Diverse /16 | BODY TP/FP/FN | HEAD TP/FP/FN | Bar HEAD TP /32 | Box HEAD FP /32 |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| original_gate | 22 | 13 | 9 | 64/9/0 | 59/3/5 | 29 | 3 |
| repvit | 29 | 16 | 13 | 64/6/0 | 64/0/0 | 32 | 0 |
| repvit_detail | 29 | 16 | 13 | 64/6/0 | 64/0/0 | 32 | 0 |

Each head has64 positive and64 negative TEST samples. Joint correctness requires both heads correct across all four variants.

| Arm | Seed17 joint | Seed29 joint | Seed43 joint | Bar-removal success /32 | Box-removal success /32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| original_gate | 24 | 22 | 22 | 27 | 24 |
| repvit | 29 | 28 | 27 | 32 | 29 |
| repvit_detail | 29 | 28 | 27 | 32 | 29 |

### Support localization

| Ensemble arm | BODY IoU@.5 | HEAD IoU@.5 | BODY peak /64 | HEAD peak /64 | BODY VAL-mask IoU | HEAD VAL-mask IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original_gate | 0.173 | 0.097 | 36 | 32 | 0.342 | 0.285 |
| repvit | 0.371 | 0.151 | 60 | 29 | 0.563 | 0.281 |
| repvit_detail | 0.401 | 0.173 | 61 | 45 | 0.597 | 0.367 |

VAL mask thresholds maximize positive-query mean IoU on0.05..0.95 step0.05 with smallest-threshold first tie. Fixed0.5, peak hits and negative activations remain available; these are body-query support regions, not full-object segmentations.

VAL-selected mask thresholds are BODY/HEAD0.90/0.95 for A and0.95/0.95 for both
B/C. At those thresholds, any known-negative-image activation counts are A64/64
BODY and61/64 HEAD, versus0/64 for both heads of B/C. This readout is tied to the
reported mask thresholds; it is not a threshold-free absence claim.

### Predeclared practical retention

Versus A, a candidate must preserve bar-only and total BODY/HEAD recall, increase each head FP by no more than2, improve joint by>=2/32 and diversejoint by>=1/16, improve both fixed IoUs by>=.03 without peak-hit loss, and have nondecreasing primary joint in at least2/3 paired seeds. If B qualifies, C also needs no loss vs B in barrecall/joint/diversejoint, both IoUs better with summed gain>=.04, and no peak-hit loss. Otherwise prefer B; if neither qualifies retain A.

- repvit: qualifies=False; failed criteria: diverse_gain_at_least1, HEAD_pointing_no_loss.
- repvit_detail: qualifies=False; failed criteria: diverse_gain_at_least1.
- detail_increment: qualifies=False; failed criteria: joint_no_loss, diverse_no_loss.

## Consumed regression

| Arm | Primary joint /32 | Compatibility joint /32 |
| --- | ---: | ---: |
| original_gate | 25 | 21 |
| repvit | 29 | 31 |
| repvit_detail | 30 | 31 |

The old G10 TEST is disclosed regression only. Original G10 reproduction and checkpoint/input hashes are recorded independently; its fresh model evaluations do not count as refits.

## Execution and boundaries

The secondary RTX3060 Laptop executed native GPU pair capture at full settling and burst cadence.128frames took33.032s acquisition,71.782s capture script,112.836s engine lifecycle; the capture/verification/package job took127.993s. Unchanged native verifier took1.148s. All11 tracked process identities exited and both scheduled job records were released.

The initial remote job stopped before UE launch because one source file had different line endings. Raw hashes and normalized-text equality were then checked; the corrected second job captured the single planned cohort. Original failed job/logs remain. No research outcomes drove this repair. Full raw depth/editor logs remain worker-owned under `worker artifacts/evidence/nf-g12-representation-20260908/capture`; primary received140 hash-verified RGB/label/mask/receipt files required for A/B/C inference/evaluation. Transfer archives were removed after verification.

Static closing remains UNKNOWN/unscored. The new representation has not been integrated with the old approach head; actual temporal replay is required before video integration. Workstation training/capture timing is not edge-device inference latency.

Evidence: `artifacts.local/nearfield/representation-20260908/`: protocol, budget, pretrained provenance, probe, main learned/evaluation, summary, worker source comparison/process audit and registration manifest. See the training receipt for actual backend, parameter counts, selected steps, per-fit timing and numerical reproducibility limitations.

## Measured training cost and checks

Six600-step fits completed3600 optimizer steps; summed fit time 730.06s, full training/inference pipeline 747.88s. Actual device NVIDIA GeForce RTX 5060 Laptop GPU, torch 2.11.0+cu130. Strict deterministic CUDA execution remained enabled. All three original G10 score arrays and saved TEST support arrays reproduce bit-exactly.

| Arm | Seed | Parameters | Selected step | Fit seconds |
| --- | ---: | ---: | ---: | ---: |
| repvit | 17 | 4,730,948 | 500 | 120.58 |
| repvit | 29 | 4,730,948 | 550 | 112.49 |
| repvit | 43 | 4,730,948 | 200 | 118.99 |
| repvit_detail | 17 | 4,736,372 | 600 | 124.45 |
| repvit_detail | 29 | 4,736,372 | 550 | 126.56 |
| repvit_detail | 43 | 4,736,372 | 200 | 127.00 |

B has4,730,948 parameters and C4,736,372, versus45,646 for the complete retained G10 model. The detail increment is5,424 parameters; the representation replacement itself is about104 times larger. Seven focused tests passed (two official-load/initialization/interface/gradient tests and five calibration/evaluator tests), plus fresh source allocation, actual TRAIN-only CUDA backward/throughput and native capture verification. B/C common initialization and sampled batch sequences are asserted equal per seed during the full run.

A scientific visualization of the fixed first narrow/diverse groups is retained as `fixed-group-support.png`. Examples are fixed by group order and do not select the reported metric.

Visual inspection of those fixed examples shows substantially reduced background
activation for B/C, while their HEAD responses can remain thicker than the native
thin-bar support. The figure illustrates the measured localization limits; it
does not prove a new causal mechanism or justify selecting other examples.
