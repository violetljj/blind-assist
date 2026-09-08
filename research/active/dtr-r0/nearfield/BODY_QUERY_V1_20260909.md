# BODY-QUERY-V1: matched visible-count evidence bottleneck

2026-09-09. EXPLORE, authorized implementation of the single-RGB BODY/HEAD
query comparison. This is a new experiment, not another fit of a closed G13,
HEAD-P1, Dice or CITY-CROSSREGION protocol. Existing datasets and terminals stay
unchanged. The first implementation was started late on 2026-09-08; artifact
prefix `work/body-query-v1-20260908` remains stable across midnight.

## Question and mechanism

Does requiring near decisions to pass through spatially defined visible-evidence
counts improve held-out-condition transfer over the existing support-gated G13
readout, under equal supervision and training exposure? Existing support is
already geometry-derived. G13 is not an ungrounded whole-image classifier.

Both arms retain original RepViT/G13 multiscale features, dense BODY/HEAD support,
and the same new query branch. A keeps the existing support-gated near readout;
B derives near only from the twelve query predictions. No teacher mask, pose,
depth, group ID, world coordinates or test label enters either model.

The fixed BODY_BOXES and 3m horizon are partitioned into three lateral sections
and two forward ranges per head. These are subdivisions of the current query,
not three navigation routes. A 3x3x3 sample lattice per cell reads projected deep
and shallow features, plus fixed normalized XYZ; a shared MLP predicts native
visible-pixel count classes 0,1,2,3+. Fixed bilinear projection uses matrix
multiplication for deterministic GPU backward. Sampling along rays does not
solve monocular distance ambiguity; far geometry may share projected features.

Count capping preserves the historical >=3 native-pixel near target exactly:
`sum(min(count_cell,3)) >= 3`. A per-cell binary OR would not. B uses the
probability that six independently modeled capped counts sum to at least3.
Conditional independence is a modeling assumption, not a calibration claim.

The support auxiliary remains separate from this causal path: B's alert is
forced through count predictions, not through the displayed support mask.
Inspect query correctness AND support accuracy; do not claim mask faithfulness
or actual geometry reasoning from the bottleneck alone.

## Source and supervision

Existing raw City TRAIN/DEV sources have varying eye heights and pitches; they
cannot be silently used with fixed query projection. Acquire a small new source
with optical height1.70m, pitch/roll0, 640x360, HFOV100 degrees. Query projection
receives those system constants only. RGB is BOX-resized to144x256 as before.
Ordinary RGB inference is conditional on this fixed calibration, not arbitrary
phone installation or free head pitch.

Budget:320 admitted full-capture frames, TRAIN240 / DEV40 / EVAL40. Four
supported fixture families retain CLEAR/BODY_ONLY/HEAD_ONLY/BOTH. Crossbar
groups additionally include LOW, ABOVE, LATERAL_OUT and FAR_OUT, with existing
supports retained or rigidly moved. Other families need not unrealistically
realize every state. Complete parent groups stay in one split; heterogeneous
group sizes and actual condition coverage must be reported separately.

TRAIN uses known street-frontage positions, DEV the opposite/east frontage,
EVAL the plaza. The source generator and complete roles are frozen before
model outcomes. These are new fixed-camera conditions in a reused synthetic
world, with shared asset libraries and potentially shared visible backgrounds;
not a fresh-city, asset-disjoint or protected TEST claim. The unrelated sealed
336-frame cross-region plan is untouched. Canaries are engineering-only and
excluded from the320-frame training/evaluation cohort; retain failed attempts.

Native depth/visibility labels remain authoritative over placement intent.
The new partition must exactly match the existing native support mask and near
bits for every frame; floor/capture validation uses existing capture checks.
Zero observed visible support is not certified free space or hidden occupancy.
Retain UNKNOWN pixels, ignore them in support loss, and preserve mismatches.

## Frozen matched training

Two fits only, seed17, original G13 seed17 checkpoint SHA
`0c14179486102993508dd2385a3b6f9e17d51a6d53e657c0240325e5f78d5c7b`.
All shared initial tensors must be identical. Both update all active parameters;
BN running buffers stay frozen. AdamW lr1e-5, weight_decay1e-4, batch32,
2000 steps, identical saved uniform-with-replacement TRAIN index sequence.
Near BCE + .25 existing known-pixel balanced support BCE + .25 mean query-count
cross entropy. No augmentation, consistency, contrastive loss, Dice, GroupDRO,
backbone/resolution change, warm start, early selection or rescue fit.

Both final fits finish before DEV selection. Use the existing inclusive float64
selector: maximize recall at empirical FPR<=.10, then lower FPR, then higher
threshold. Predeclare minimum8 positives and8 negatives per head for this small
DEV cohort. This is a new sample-size criterion, not a change to old DEV protocols.
Freeze A/B thresholds before EVAL. The historical Coverage1198 checkpoint is
an optional zero-fit context reference, never a matched architecture comparator.

## Evaluation and disposition

Report complete TRAIN/DEV/EVAL confusion, AUC, selected and fixed.5 performance,
whole-group correctness (group sizes explicit), condition-specific LOW/ABOVE/
lateral/far false alerts, support IoU/recall/unknown/negative activation, query
count correctness, GPU forward latency, memory and active/inactive parameters.
The demo uses every EVAL group in fixed source order and independent frame
inference, never temporal state or selected good examples.

Operational UNKNOWN is a predeclared heuristic: within .5 logit of the selected
decision threshold. The same rule applies to both arms. This does not estimate
physical unobservability or establish calibrated abstention. Primary error and
group counts include ALL frames; secondary coverage never hides mistakes.

Retain B as a challenger only if EVAL HEAD recall and group correctness improve
without increasing HEAD false positives, losing BODY recall/FP control, or
degrading evidence accuracy; report mixed outcomes without forced promotion.
One seed and small same-world regions give exploratory evidence only. No
automatic repeat or extension follows. If A/B are similar, retain the simpler
effective implementation; both improving over history cannot isolate data from
new supervision. Mechanical failures preserve receipts and can resume evaluation
from completed checkpoints without repeating fits.

Deliver source, a fixed-calibration single-RGB inference command, all-case HTML
comparison, measured results and scoped Git delivery. Release task-owned UE,
training, sessions and temporary transfers; preserve raw source/weights/receipts.

## Result

Pending execution. No trained-model improvement is claimed here.
