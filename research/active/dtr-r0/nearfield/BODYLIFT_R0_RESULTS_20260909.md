# BodyLift R0: frozen-feature depth lifting does not improve the low-FP target

2026-09-09 EXPLORE. Retain original B. Neither heads-only CONTROL nor LIFT is
promoted. Two new matched2000-step fits completed on the primary RTX5060 Laptop
GPU; no remote job, depth-model download or backbone training.

## No-return audit

The current native exporter collapses nonfinite, nonpositive and>=10000cm red
values into0, otherwise converts centimetres to metres:
[BlindAssistCapturePair.cpp](../unreal/native_capture/BlindAssistCapture/Source/BlindAssistCapture/Private/BlindAssistCapturePair.cpp:159).
The alternate readback implementation does the same. Saved native arrays contain
only valid positive depth or0: TRAIN4,564,074 zeros and50,731,926 valid pixels;
DEV1,287,745/7,928,255; EVAL962,558/8,253,442. No saved nonfinite,negative or>=100m
values remain. Their original causes cannot be reconstructed from zero alone;
no FREE_RAY or NO_SURFACE label is warranted. Zeros remain ignored.

## Matched design

[Protocol](BODYLIFT_R0_PROTOCOL_20260909.md), [implementation](bodylift_r0.py).
Freeze final B's backbone, deep/detail projections and support branch. Warm-start
the query MLP/count readout from B and fit them in both arms with identical
seed17,2000 batch32 samples, AdamW lr1e-4 and near BCE+.25 count CE. CONTROL uses
the original appearance/projection/mean path. The frozen support loss is constant
with respect to trained parameters and is omitted.

LIFT adds a1x1 64->16 depth head and0.25 histogram CE. Fifteen0.3m bins cover
(0,4.5m), with valid[4.5,100m) in an overflow bin. A20x20 native-pixel histogram
supervises each18x32 feature tile; mixture targets preserve multiple surfaces,
invalid pixels contribute no target, and valid pixel fractions weight the CE.

At each image location, appearance is scaled by16 times the probability of the
query's depth bin BEFORE bilinear projection. Uniform initialization is exactly
identity; this is probability relative to the uniform prior and can amplify as
well as suppress features. It is not the previous bounded cell veto. Both paths
initially reproduce B near outputs exactly on all320 frames. XYZ, mean pooling,
native count classes and count-to-near aggregation remain unchanged. Inference
uses RGB-derived context and predicted depth only; no native runtime input.

## Final results

Each arm uses its own original-policy DEV cutoff, persisted before EVAL scoring.
DEV/EVAL each contain16 positive and24 negative frames per head.

| Metric | Original B | CONTROL | LIFT |
| --- | ---: | ---: | ---: |
| DEV BODY TP/FP |10/2|10/2|11/2|
| DEV HEAD TP/FP |5/0|7/2|8/2|
| EVAL BODY TP/FP |7/0|6/0|6/0|
| EVAL HEAD TP/FP |8/2|8/2|11/6|
| EVAL HEAD AUC |0.7865|0.7526|0.7721|
| EVAL HEAD oracle TP at FP<=2 |8|8|6|
| EVAL complete groups |2/8|2/8|2/8|
| EVAL query TP/FN/FP |29/68/23|36/61/34|41/56/30|

The oracle row is a descriptive low-FP diagnostic, not EVAL threshold selection.
LIFT improves query TP and reduces query FP relative to CONTROL, but does not
translate this into the required HEAD low-FP gain. EVAL BODY ranking also falls:
AUC B0.8958, CONTROL0.7721, LIFT0.7292. LIFT HEAD false alerts are CLEAR1/8,
BODY_ONLY4/8, LOW1/2; ABOVE/LATERAL_OUT/FAR_OUT each0/2. Neither primary nor
diagnostic low-FP result reaches the proposed >=10/16 target.

Both new arms classify all TRAIN near labels correctly at their DEV cutoffs,
with48/48 complete groups. Query TP/FN/FP is CONTROL573/3/3 and LIFT571/5/12
on576 positive cells. DEV query counts are CONTROL39/54/48 and LIFT30/63/30.
Do not equate perfect TRAIN near decisions with perfect spatial evidence.

## The learned depth is not yet a strong near-field information source

Uniform16-bin CE is ln(16)=2.7726. LIFT histogram CE:

| Scored tiles | TRAIN | DEV | EVAL |
| --- | ---: | ---: | ---: |
| All valid tiles, weighted by valid fraction |1.5234|2.3389|1.9111|
| Tiles with majority valid depth mass below4.5m |2.4041|3.3131|3.0223|

On the near-surface tile slice, held-out depth CE is worse than uniform. This
is a mixed-surface histogram metric, not HEAD-only depth accuracy. It limits the
interpretation: this frozen visual feature plus tiny depth head did not acquire
reliable near-field depth distributions. The experiment cannot establish that
good depth would fail, that pooling is the sole culprit, or that all depth-aware
representations should be abandoned. Conversely, its overall depth CE decrease
does not justify using this model when the near-field task does not improve.

## Decision and validation

Stop this R0 recipe at its fixed budget and carry it as a negative control.
Do not sweep bins, depth-loss weights or add attention automatically. Original
B remains the final comparison baseline. Future work needs a distinct question;
this result is not a general rejection of depth supervision or an endorsement
of a replacement pooling architecture.

CONTROL took20.89 seconds and LIFT42.25 seconds; final run including preparation
and scoring82.89 seconds. `run-v1` stopped before any training because copied
query heads inherited frozen flags. This setup bug was corrected by explicitly
enabling those heads; its source/failure record is retained. No interrupted fit
or hidden extra training occurred. `run-v2` is the completed result.

Artifacts: `artifacts.local/work/bodylift-r0-20260909/run-v2/` contains frozen
contexts, native depth histograms/weights, checkpoints, history, selection,
predictions, result and receipt. Parent `comparison.json`/`validation.json`
independently check AUC, frame confusion, histogram normalization/ignored zeros,
and source/checkpoint/result hashes. All native inputs were checked against the
existing source receipt. Cached and end-to-end RGB LIFT inference agree within
2.39e-7 on the checked frame. Processes completed; no GPU/worker allocation remains.

Primary sources motivate lifting and depth supervision in different tasks:
[LSS](https://arxiv.org/abs/2008.05711),
[BEVDepth](https://ojs.aaai.org/index.php/AAAI/article/view/25233),
[ADD](https://ojs.aaai.org/index.php/AAAI/article/view/25391).
They do not guarantee efficacy for this frozen-feature near-field prototype.
One seed, consumed same-world frames: no fresh confirmation or device claim.
