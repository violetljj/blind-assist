# BODY-QUERY-V1 results: mixed effect, no baseline replacement

Completed report. The [original pre-outcome protocol](BODY_QUERY_V1_20260909.md)
is retained byte-for-byte for its registered input fingerprint; its pending-result
line describes registration time. This file owns completed outcomes and amendments.

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

**Completed: mixed result; do not replace the baseline with this fixed B recipe.**
Both primary fits completed2000 steps from identical initial state, and each fits
all240 TRAIN frames perfectly at its saved DEV thresholds. The count bottleneck
improves BODY ranking and reduces HEAD false alarms, but loses HEAD recall and
does not consistently improve evidence quality. This does not establish that a
spatial query structure forces geometry reasoning or repairs regional transfer.

Selected thresholds are A BODY/HEAD0.28673643/0.51943046 and B
0.37010410/0.84503776, fixed before EVAL. Each DEV/EVAL head has16 positives and
24 negatives. No UNKNOWN frame is removed from the primary counts.

| Split / arm | BODY TP / FP | HEAD TP / FP | BODY AUC | HEAD AUC | All-state groups correct |
| --- | --- | --- | --- | --- | --- |
| TRAIN A | 96 / 0 | 96 / 0 | 1.0000 | 1.0000 | 48/48 |
| TRAIN B | 96 / 0 | 96 / 0 | 1.0000 | 1.0000 | 48/48 |
| DEV A | 4 / 0 | 15 / 1 | .6406 | .9635 | 0/8 |
| DEV B | 10 / 2 | 5 / 0 | .8099 | .7982 | 0/8 |
| EVAL A | 4 / 0 | 11 / 8 | .7083 | .6797 | 0/8 |
| EVAL B | 7 / 0 | 8 / 2 | .8958 | .7865 | 2/8 |

EVAL HEAD recall is68.75% to50%, FPR33.33% to8.33%; BODY recall25% to43.75%
with zero FP in both. B's two completely correct groups are four-state groups
u06/u07; neither eight-state crossbar group passes. At the common0.5 cutoff,
A BODY TP3/FP0, HEAD TP11/FP9, groups0/8; B BODY TP6/FP0, HEAD TP8/FP5,
groups1/8. Thus reporting only selected FPR or AUC would omit a real recall loss.

| EVAL condition | Frames | A BODY/HEAD misses | B BODY/HEAD misses | A BODY/HEAD false alarms | B BODY/HEAD false alarms |
| --- | --- | --- | --- | --- | --- |
| CLEAR | 8 | 0 / 0 | 0 / 0 | 0 / 1 | 0 / 0 |
| BODY_ONLY | 8 | 5 / 0 | 4 / 0 | 0 / 5 | 0 / 2 |
| HEAD_ONLY | 8 | 0 / 3 | 0 / 4 | 0 / 0 | 0 / 0 |
| BOTH | 8 | 7 / 2 | 5 / 4 | 0 / 0 | 0 / 0 |
| LOW | 2 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| ABOVE | 2 | 0 / 0 | 0 / 0 | 0 / 2 | 0 / 0 |
| LATERAL_OUT | 2 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| FAR_OUT | 2 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

LOW is a useful retained negative control, but2 EVAL examples with zero alarms
do not establish general selectivity. The dataset change itself has no ablation.

| EVAL evidence metric | A | B |
| --- | --- | --- |
| BODY positive-frame support IoU | .1366 | .1474 |
| HEAD positive-frame support IoU | .2396 | .2582 |
| BODY positive-pixel support recall | 90/195 (46.15%) | 102/195 (52.31%) |
| HEAD positive-pixel support recall | 84/148 (56.76%) | 90/148 (60.81%) |
| BODY support peak hits | 2/16 | 4/16 |
| HEAD support peak hits | 5/16 | 3/16 |
| BODY negative known-pixel activations | 237/9886 | 80/9886 |
| HEAD negative known-pixel activations | 214/9823 | 112/9823 |
| Query count accuracy | 82.08% | 81.04% |
| Nonzero-query TP / FP / FN / TN | 12 / 1 / 85 / 382 | 29 / 23 / 68 / 360 |
| BODY heuristic UNKNOWN frames | 2/40 | 2/40 |
| HEAD heuristic UNKNOWN frames | 13/40 | 2/40 |

Support UNKNOWN pixels remain6574 BODY and6543 HEAD (28.53%/28.40%); they are
ignored, never negative truth. A/B HEAD peaks on UNKNOWN are8/16 and13/16.
Count accuracy is dominated by empty cells:383/480 EVAL cells are empty, an
always-zero predictor gets79.79%. TRAIN count classes1/2 occur only1/3 times;
all nonzero EVAL cells are3+. Exact count semantics are implemented, but this
source does not meaningfully test learning rare count classes or metric depth.

| Primary RTX5060 Laptop cost | A | B |
| --- | --- | --- |
| Fit seconds | 160.85 | 162.34 |
| Peak allocated GPU memory | 1,974,335,488 bytes | 1,971,073,536 bytes |
| Warm forward P50 / P95,30 samples | 11.20 / 17.22ms | 11.47 / 20.46ms |
| Stored parameters | 4,741,336 | 4,741,336 |
| Unused legacy near parameters | 0 | 770 |

Timing excludes preprocessing and IO, not an end-to-end device benchmark.
Runtime: torch2.11.0+cu130, timm1.0.28, numpy2.4.4, Pillow12.2.0.

Disposition: retain source, counterfactual controls, query-label generator,
checkpoints and viewer as diagnostic components. This fixed B recipe is not a
replacement challenger under the predeclared no-recall-loss criterion. Existing
route model authority stays unchanged; A is a matched comparator, not a newly
promoted production model. Further work requires a distinct scoped experiment,
not more steps or threshold tuning on these consumed40 EVAL frames.

### Validation and delivered behavior

- Three query semantic tests pass, including exhaustive4^6 discrete count cases.
- Independent NumPy audit reproduces confusion, AUC, DEV optimum, support IoU,
  all-state group counts, and input/checkpoint/source hashes. Independent full
  count convolution reproduces B near within1e-7 without using its model readout.
- Ordinary PNG CLI reproduces the single-frame evaluator's scores and maps
  within1e-6 using no truth input. For all40 frames in each arm, single-frame
  versus saved batch inference changes0/80 binary decisions. Floating-point
  batch differences remain: maximum near A/B .00406/.00693 and support
  .00762/.05487. Primary metrics/viewer retain the original batch predictions;
  no thresholds or model weights were changed to remove those differences.
- Browser verification confirms all40 cases, LOW filtering2 cases, next-frame
  switching, HEAD_ONLY filtering8 cases, and an explicitly displayed missed
  HEAD example (sample282). No browser console errors observed. This wrong,
  low-score example does not trigger heuristic UNKNOWN, demonstrating its limit.
- Offline `body-query-comparison.html` includes RGB, model maps,12 count
  distributions, fixed cutoffs and separate evaluator truth. No server is needed
  when opening the saved HTML. It preserves all errors and original case order.

Evidence lives under the artifact prefix: `model-run-local-v4/result.json`,
`receipt.json`, `selection.json`, final A/B weights and resume states;
`independent-audit.json`, `single-frame-audit.json`, `single-rgb-B-v3.json`;
`worker-attempts/worker-attempt-release.json`. The worker release receipt confirms
zero owned processes/tasks; concurrent unrelated CitySample work was untouched.

### Execution amendments before outcomes

All320 frames passed native-label and support equivalence checks. The accepted
cache has240/40/40 frames,48/8/8 groups, and96/16/16 positives for each head.
Both held-out splits contain six four-state groups and two eight-state crossbar
groups; use "all-state group correctness", not historical quartet accuracy.

Two worker starts failed before optimizer updates: missing task-local timm,
then CUDA deterministic-mode rejection of the 3D cross-entropy NLL kernel.
Reused the existing timm1.0.28 package. Replaced mean count CE by the identical
negative log-softmax/gather mean; checked numerical equivalence and deterministic
full-loss backward. No loss weights or scientific settings changed.

The third worker start was interrupted when the user permitted local training
and concurrent unrelated CitySample work left178MiB free VRAM. A had started
training but no completed fit/checkpoint or DEV/EVAL outcome existed. The exact
partial step count is not recoverable beyond the progress log (last recorded A100,
so at least100 updates). This is an extra
incomplete infrastructure attempt, not one of the two completed comparison fits;
disclose it rather than claiming only4000 optimizer steps were ever attempted.
Worker receipts are retained. Both comparison arms now run on the primary
RTX5060 Laptop under torch2.11.0+cu130, with the same frozen source/cache, seed,
initial tensors, schedule and2000-step budget per arm. No worker outcome selected
the local restart. Local runtime/output names are `model-runtime-local-v4` and
`model-run-local-v4` beneath the original artifact prefix.

### Reproduction and artifact entry points

The accepted local cache is `artifacts.local/work/body-query-v1-20260908/accepted/cache-v1`;
all20 transferred cache files passed SHA256 verification. Original RGB/native
payload remains on the worker under `work/body-query-v1-20260908/capture-all-v3`.
Capture sources are `body_query_capture_spec.py` and `body_query_capture_run.py`;
their task-specific worker runtime/config paths are explicit, not portable UE setup.
The frozen source/site inventory and canary contacts live beside the cache entry.

```powershell
$base = 'E:/linnan/linnan/artifacts.local/work/body-query-v1-20260908'
$code = 'E:/linnan/linnan/research/active/dtr-r0/nearfield'
# A fresh output is mandatory; never repeat a completed fit to recreate a viewer.
& E:/codex-tools/bin/blindassist-research-gpu.cmd -B "$code/body_query_train.py" `
  --cache "$base/accepted/cache-v1" --pretrained "$base/model-inputs/pretrained" `
  --initial "$base/model-inputs/checkpoint/decoupled_seed17.pt" --output "$base/NEW_RUN"
# Reuse completed outcomes for the offline all-case comparison.
& E:/codex-tools/bin/blindassist-research-gpu.cmd -B "$code/body_query_demo.py" `
  --cache "$base/accepted/cache-v1" --run "$base/model-run-local-v4" `
  --output "$base/body-query-comparison.html"
# IMAGE.png is an ordinary calibrated16:9 RGB frame; no native/evaluator input.
& E:/codex-tools/bin/blindassist-research-gpu.cmd -B "$code/body_query_infer.py" `
  --image IMAGE.png --arm B --checkpoint "$base/model-run-local-v4/B-step2000.pt" `
  --pretrained "$base/model-inputs/pretrained" --selection "$base/model-run-local-v4/selection.json" `
  --output "$base/NEW_SINGLE_FRAME.json" --device cuda
```
