# NF-G9-B: ordinary model data coverage comparison

## Fixed question and design

Can full object-presence coverage improve the selective BODY/HEAD near response
that failed in [G9-A](GROUNDING_20260907.md), without changing the model?
This is EXPLORE, controlled Development on the same frozen Willow background.

64 fresh geometry groups g1000..g1063, parameter seed 100907, split 40 TRAIN /
12 VAL / 12 TEST. Each group has both objects, bar only, box only and neither,
with identical retained geometry/camera and rotated presentation order. One
current RGB per condition gives 256 images. Four separate preflight groups
g1064..g1067 never enter training, validation or scoring. Every present object
must pass independent native-depth visibility and BODY/HEAD query checks.
Neither means no task object, not all-scene free space.

| Arm | TRAIN variants | Unique TRAIN images | Fit budget per seed |
| --- | --- | --- | --- |
| frozen_ordinary | Original G8 weights | 0 new training images | No fit |
| single_data | Bar only, box only | 80 | 300 steps, batch 32 |
| factorial_data | Both, bar only, box only, neither | 160 | 300 steps, batch 32 |

Seeds 17/29/43 start from the same corresponding G8 ordinary checkpoint in both
fitted arms. Each batch contains equal counts of its allowed variants, giving
50% positives per BODY/HEAD head and 9,600 presentations per fit. There are six
fits and 1,800 optimizer steps in total. This controls steps, presentations,
geometry groups and per-head priors; it does not control unique image count.
The intervention combines joint-state coverage and more unique images, and
does not isolate pairing as a mechanism.

The 45,646-parameter ordinary architecture is unchanged. AdamW learning rate
1e-4, weight decay 1e-4; near BCE plus 0.25 times the existing class-balanced
support BCE. Only spatial/near/support modules train. No new pair loss or
mask-gated head. Common all-variant VAL selects minimum near BCE every 25 steps,
with first tie retained. This gives both arms the same validation exposure.
Motion/approach weights are frozen and checked; closing behavior is not claimed
unchanged because shared spatial features change. Closing stays UNKNOWN and
unscored in this static experiment.

Primary scoring reuses exact G8 ordinary per-seed/ensemble thresholds for every
arm. Secondary thresholds use only new VAL at <=5% known-negative FPR; actual
TEST FPR is reported separately. New TEST is scored once after all fits. Old
G9-A is disclosed consumed regression, never fitting or selection data. Its
original outputs remain intact. Frozen ordinary near probabilities are checked
against saved G9-A predictions with numerical tolerance and alert-flip counts.

## Efficiency and acquisition

The near heads consume only current RGB. Each condition therefore exports one
RGB/native pair instead of three, retaining 27 color-capture updates (51 for
the first condition), matching the previous last-of-three update count. Update
count equivalence is not a claim of bit-identical temporal rendering.

Preflight alternated the order of two conversions on each of 16 identical raw
UE depth arrays. Reading each pixel property once gave byte-identical output in
16/16 comparisons. Median conversion time fell from 187.69 to 142.38 ms
(24.1% lower time, 1.318x conversion throughput). Main uses this selected path.
These are call-boundary wall measurements, not GPU utilization or kernel timing.
The prelaunch estimate was 229.82 seconds of capture-script time, excluding
engine boot outside the script.

All current RGBs are decoded/resized once and cached on CUDA. Training computes
only the required spatial branches; UE exits before training begins. Large
native conversion/write overlap changes are deferred until this data comparison
shows whether more acquisition is useful.

## Results and decision

All 256 images passed native verification. Near truth/prediction UNKNOWN is zero;
closing remains 48 UNKNOWN per head in TEST and is not scored. Six fits completed
the planned 1,800 steps, observing all 80/160 eligible images respectively.
Best steps were 300/225/300 for single_data and 300/300/300 for factorial_data.
No test-guided checkpoint selection or additional fit occurred.

Primary frozen ensemble thresholds are BODY 0.1558650881 and HEAD 0.3789404432.
Each head has 24 positive and 24 negative TEST samples.

| Primary ensemble | BODY TP/FP/FN | HEAD TP/FP/FN | Both heads correct across all four variants |
| --- | --- | --- | --- |
| frozen_ordinary | 19/0/5 | 22/7/2 | 1/12 |
| single_data | 22/12/2 | 24/1/0 | 2/12 |
| factorial_data | 24/2/0 | 23/5/1 | 4/12 |

Full coverage improves BODY: the single-data ensemble falsely alerts on 9/12
neither scenes, whereas factorial_data alerts on 0/12. BODY false alerts on
bar-only scenes are 3/12 versus 2/12. However, HEAD on box-only scenes is worse
for factorial_data than single_data: 5/12 versus 1/12 false alerts. Bar-only
recall is 11/12 versus 12/12. These tradeoffs prevent an unconditional win claim.

| Primary ensemble | Initially both heads correct | Bar removal + BODY retained | Box removal + HEAD retained |
| --- | --- | --- | --- |
| frozen_ordinary | 8/12 | 2/12 overall; 2/8 conditional | 6/12 overall; 6/8 conditional |
| single_data | 10/12 | 9/12 overall; 9/10 conditional | 7/12 overall; 7/10 conditional |
| factorial_data | 12/12 | 7/12 overall; 7/12 conditional | 9/12 overall; 9/12 conditional |

| Individual seed | BODY TP/FP/FN | HEAD TP/FP/FN | Joint all-four correct |
| --- | --- | --- | --- |
| frozen 17 | 19/1/5 | 21/7/3 | 0/12 |
| frozen 29 | 9/0/15 | 20/2/4 | 1/12 |
| frozen 43 | 19/0/5 | 22/7/2 | 1/12 |
| single_data 17 | 19/7/5 | 24/1/0 | 5/12 |
| single_data 29 | 21/14/3 | 22/0/2 | 0/12 |
| single_data 43 | 22/3/2 | 24/2/0 | 7/12 |
| factorial_data 17 | 24/2/0 | 23/1/1 | 9/12 |
| factorial_data 29 | 24/3/0 | 22/2/2 | 7/12 |
| factorial_data 43 | 24/1/0 | 23/6/1 | 5/12 |

Seed variation is substantial; factorial_data does not beat single_data on
joint correctness for seed 43. The ensemble is not uniformly better than its
members under their separately frozen operating thresholds. No winning seed is
selected from TEST.

Secondary VAL-calibrated ensembles give frozen/single/factorial joint correctness
2/12, 4/12, 6/12. Their BODY TP/FP/FN are 15/0/9, 19/5/5, 24/0/0; HEAD are
19/2/5, 24/4/0, 23/5/1. Actual TEST FPRs are BODY 0%,20.83%,0% and HEAD
8.33%,16.67%,20.83%; VAL's <=5% constraint does not guarantee TEST FPR. The
factorial BODY threshold becomes 0.4881930848, HEAD 0.3588985503. This remains
secondary and does not replace the 4/12 primary headline.

Consumed G9-A regression, using original frozen thresholds:

| Ensemble | BODY TP/FP/FN | HEAD TP/FP/FN | Joint all-four correct |
| --- | --- | --- | --- |
| frozen_ordinary | 20/2/4 | 21/8/3 | 1/12 |
| single_data | 24/14/0 | 24/0/0 | 1/12 |
| factorial_data | 24/4/0 | 23/2/1 | 6/12 |

The frozen baseline reproduces all three seeds and ensemble near probabilities
exactly (maximum absolute difference 0; zero threshold-alert mismatches). Thus
the one-current-frame inference path itself does not change old G9 predictions.
The regression supports partial improvement but is not fresh confirmation.

Fresh TEST positive-support IoU BODY/HEAD is 0.213/0.325 frozen,
0.283/0.315 single_data and 0.284/0.366 factorial_data. Pointing is 8/24 and
16/24 frozen, 21/24 and 22/24 single_data, 22/24 and 21/24 factorial_data.
Factorial HEAD probability mass averages 15.30% on other-object-only pixels and
67.13% on background. This includes diffuse low scores; it is not causal
attribution or the same denominator as query coverage. The fixed first TEST
group g1052 visual in `first-test-quartet.png` includes a factorial bar-only miss;
it was not selected to show a success.

**Decision:** retain full joint-state coverage as a useful data ingredient and
the ordinary factorial model as a Development challenger. Data alone improves
BODY and some joint behavior but does not establish reliable part-selective
HEAD decisions. Retain single_data as the explicit comparator: its better HEAD
response is a relevant counterexample. Do not promote a TEST-selected seed,
increase the budget on this consumed split, replace the backbone, or claim
natural generalization. A future same-data head/separation comparison can reuse
these disclosed training inputs and require a separately versioned test; it is
not launched automatically here.

Actual main capture-script wall time was 199.047 s versus the prelaunch estimate
229.822 s; first frame arrived at 39.547 s. Steady median stage wall times were
preparation/settling 432.67 ms, PNG export 35.01 ms, native readback 2.76 ms,
conversion/write 110.20 ms, total 580.75 ms. Different preflight/main conditions
mean 142.38 versus 110.20 ms is not another controlled speedup result. The only
paired conversion comparison is the 16-array preflight above. Rendering updates
and synchronization are now the largest measured stage; no GPU-utilization
percentage was sampled.

All six fits, including their validation/checkpoint operations, totaled 12.596 s.
The measured training/inference pipeline was 17.235 s internally, 19.974 s as a
subprocess including startup; native verification subprocess 3.814 s and scoring
0.648 s. Actual backend CUDA, NVIDIA GeForce RTX 5060 Laptop GPU,
torch 2.11.0+cu130. Main RGB CUDA cache 113,246,208 bytes; regression cached
separately. These costs are for this small 300-step near-only probe, not a matched
speedup against G8's different training workload.

Ten focused tests passed (two source, three training, five metric checks), plus
actual capture verification, training, fresh evaluation and regression. Both
owned UE process trees are released and birth-identity process audit found no
survivors. The saved Willow map hash is unchanged. One launch invocation using
system Python failed before UE launch because psutil was absent; its log is
retained, and the already available GPU runtime completed the sole main capture.

## Reproduction and evidence boundary

Artifact entry: `artifacts.local/nearfield/factorial-20260907/`. The prelaunch
`protocol.md` and preflight `selection.json` retain the fixed budget and measured
choice. Entry points are `factorial_spec.py`, `factorial_launch.py`,
`factorial_verify.py`, `factorial_train.py`, and `factorial_evaluate.py`.
Acquisition uses `factorial_capture.py`; native verification reuses unchanged
`grounding_verify.py`. Training reads only sanitized RGB, TRAIN/VAL supervision
and a TRAIN-only variant map. Native depth, poses and TEST labels stay outside
model training inputs.

Source geometry, training balance/frozen weights/input rejection, and evaluation
threshold/split/removal/UNKNOWN checks have focused tests. The exact command and
runtime receipts retain authoritative hashes, devices, costs and model budgets.
One completed comparison is the experiment stop condition. No natural-scene,
cross-background, dynamic closing, phone latency or default-App claim follows.
