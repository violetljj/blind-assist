# MZ9: source supervision improves regional readout; preserve coverage boundaries

2026-09-10 EXPLORE. **Keep the source-supervision implementation and bounded
readout as a research component; retain frozen MZ5 as the whole baseline.**
SOURCE_RGB gives198/200 exact with thin48/49 and25/25 pole-absent controls correct,
versus MZ8's177/200 and22 pole-absent errors. Wrong correspondence reduces thin
48->1. However old DEV BODY_NEAR recall169/200 remains below MZ5's187/200; all28
lost previously correct near bits have no eligible geometric candidate. This
readout cannot serve as a whole replacement regardless of more training steps.

## Actual-source supervision and matched new exposure

[Protocol](MZ9_SOURCE_SUPERVISION_PROTOCOL_20260910.md),
[contributor reconstruction](mz9_contributors.py), [label cache](mz9_labels.py),
[readout](mz9_source_readout.py), [training](mz9_train.py), [audit](mz9_audit.py).
Reconstruct the actual pixels in each selected first/last supported0.1m radial
bin. Group them by7x7 angular subcell and actual native body-query membership.
Keep outside-query contributions and multiple contributors. A candidate query
label indicates at least one source pixel in that subcell/query; global query
truth keeps its original>=3-pixel rule. Fully unknown cells and invalid returns
are masked. All22 old pole-absent winning candidates now have negative BODY_FAR
support labels, instead of the old positive distance-compatibility surrogate.

3700 cached packets reproduce validity exactly; maximum range difference is
1.1921e-7m (stored float32 comparison). Native data only makes evaluator/training
labels; inference sees frozen RGB features, observed ranges/validity and fixed
camera/query geometry. No contributing coordinates, identities, pose, history
or paired absent RGB enter inference. Shared regional MLP65->32->4 plus four
biases has2248 parameters; MZ8 max pooling and geometric eligibility are unchanged.

Three fits each use seed109,1200 Adam steps and the identical saved batch-index
array: eight old TRAIN and eight MZ6 per step. Old2500 TRAIN and all200 consumed
MZ6 samples train; old1000 DEV selects thresholds under frozen per-query FP budgets.
MZ5_ADAPT starts from retained weights and trains its independent heads; SOURCE
arms start from the same fresh state and differ only in RGB versus zero tokens.
MZ5 versus SOURCE is a practical equal-new-exposure comparison, not isolated
architecture causality: initialization and auxiliary supervision differ. The
MZ8-to-MZ9 comparison also changes the regional output interface and seed, so
the full metric change is not attributable to supervision alone.

## Results on targeted training/regression, not confirmation

| Clean MZ6 measure | Frozen MZ5 | MZ5_ADAPT | SOURCE_RGB | SOURCE_NO_RGB |
| --- | ---: | ---: | ---: | ---: |
| Four outputs exact /200 |143|199|198|166|
| TP BN/BF/HN/HF |2/6/9/0|9/36/9/38|9/35/9/38|0/36/5/30|
| FP BN/BF/HN/HF |5/3/19/0|0/0/0/1|0/1/0/0|0/4/0/9|
| Thin far TP /49 |0|49|48|49|
| Pole-absent exact /25 |25|25|25|25|

The matched training control is important: simple MZ5 adaptation also learns the
thin configuration. SOURCE_RGB's benefit cannot be described as the only way to
read the complete evidence. Its one FP is lateral-entry BODY_FAR; MZ5_ADAPT's one
FP is approaching-bar HEAD_FAR. All three preserve the100 paired empty frames.

SOURCE_RGB wrong-zone intervention gives154/200 exact, TP8/1/9/5 and zeroFP;
thin48/49 falls to1/49. Unlike MZ8, this fitted model's thin predictions depend on
correct visual placement. But SOURCE_NO_RGB independently achieves49/49 on the
same thin configuration: correspondence dependence does not establish RGB is
necessary there. RGB improves the broader matched comparison (198 versus166,
fewer regional errors); wider placement evidence is still missing.

Existing artificial return deletion changes no output bit in any arm. The
first-step complete-evidence result is encouraging, but this stress condition
does not expose a recoverable temporal information gap. Temporal stays closed.

## Old-data retention is the remaining gate

| DEV fitting cohort /1000 | Frozen MZ5 | MZ5_ADAPT | SOURCE_RGB | SOURCE_NO_RGB |
| --- | ---: | ---: | ---: | ---: |
| Exact |912|917|899|544|
| TP BN/BF/HN/HF, each /200 |187/180/183/165|180/180/185/173|169/189/196/170|58/62/88/40|
| FP BN/BF/HN/HF |6/11/10/7|5/10/7/6|5/9/10/7|6/10/10/6|

All DEV FP budgets hold by threshold fitting; DEV is not independent validation.
SOURCE_RGB loses28 formerly correct BODY_NEAR bits and gains10, net-18. **All28
losses lack any eligible geometric candidate.** The geometry/source-support
ceiling is170/200, and the model detects169. Near regression cannot be repaired
by training the same support-gated head harder. This establishes candidate absence,
not which upstream cause (angular coverage, selected returns or geometry) removed
each candidate; it must not be automatically translated into missing RGB evidence.

MZ5_ADAPT loses7 BODY_NEAR TPs with no gains there; it has no corresponding hard
geometric gate, so this is a separate adaptation-retention failure. SOURCE_RGB's
other DEV TP transitions are BF-5+14, HN-0+13, HF-8+13. Do not hide near loss
behind improved far/head metrics or training-regression totals.

No standalone arm satisfies every predeclared retention condition. No new
confirmation capture, aggregation change, further fit or threshold rescue was
run. The retained component is accurate contributor supervision and a readout
whose use must be bounded by available candidate support; an availability-aware
combination with the retained baseline is a separate untested next question.
It must not be silently promoted from these results. No App or hardware change.

## Checks and evidence

Under `artifacts.local/work/mz9-source-supervision-20260910/`, `labels-v1/` holds
actual source/query counts and packet parity. `run-v1/` holds all three checkpoints,
the common batch-index array, thresholds, raw support/logits, predictions, fit
history and receipts. Label reconstruction14.37s and three fits plus preparation/
scoring20.22s used RTX5060 Laptop CUDA; these are experiment times, not latency.

Six focused tests cover contributor conservation, original packet parity,
outside-boundary points, invalid cells, regional output/support shape and
NO_RGB input/gradient invariance. Pre-fit independent code review found no material
axis, masking, inference leakage or new-exposure mismatch. Scalar scoring passes81
arm/clip groups; source/adapted DEV replay matches8000 bits. Threshold-boundary
replay uses the original32-frame scoring batch size; changing it altered floating
roundoff at a selected DEV threshold, repaired without refitting or threshold changes.
Processes exited; no UE/worker/port was allocated. Unknown remains UNKNOWN, not
clearance; all200 sequence samples remain consumed training/regression.
