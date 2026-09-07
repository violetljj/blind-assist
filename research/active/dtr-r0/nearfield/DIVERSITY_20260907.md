# NF-G10: data expansion x region-dependent near head

## Question and fixed comparison

G9-B improved BODY and joint decisions, but the factorial ensemble still gave
HEAD false alerts on 5/12 box-only TEST scenes. Its predicted support was
auxiliary: the near classifier did not directly use it. G10 asks whether expanded
view/geometry coverage, direct predicted-region dependence, or both improve
part-selective near alerts. This is controlled EXPLORE Development, not a claim
that the earlier error's causal mechanism has been identified.

| Arm | TRAIN images | Near head |
| --- | --- | --- |
| existing_plain | G9-B 160 images, 40 complete quartets | Original |
| existing_region | Same 160 | Predicted-region feature gating |
| expanded_plain | Same 160 plus 640 new images | Original |
| expanded_region | Same 800 | Predicted-region feature gating |

All four arms initialize from corresponding original G8 ordinary weights for
seeds 17/29/43. No G9-B fitted checkpoint initializes a privileged arm. All run
600 optimizer steps, batch 32, eight examples per variant sampled uniformly
with replacement. Extra data changes both unique-image quantity and coverage;
this is not a pure diversity-at-fixed-image-count comparison. Each arm has the
same number of sample presentations and per-head positive prior.

Region heads multiply current spatial features by the predicted BODY/HEAD
sigmoid support map, then use the same coarse average pooling and copied
corresponding linear classifier row. Pooling is not normalized by mask mass,
so absence can reduce features. Gradients flow through the gate. It uses no
native depth or true masks at inference. Spatial/near/support modules train;
temporal/approach parameters remain frozen but dynamic behavior is not claimed
preserved. Static closing scores and labels remain UNKNOWN and unscored.

AdamW lr 1e-4, weight decay 1e-4; near BCE plus 0.25 existing support BCE in all
arms. Common new VAL has 24 groups, 12 narrow and 12 diverse; minimum near BCE
every 50 steps selects checkpoints with first ties retained. Old G9-B VAL/TEST
do not enter training or selection. New TEST has 32 groups, 16 per stratum,
scored once after all 12 fits. Old G9-B TEST is disclosed regression only.

Primary operating thresholds remain the exact G8 ordinary seed/ensemble values
for every arm. Secondary thresholds use common new VAL at <=5% known-negative
FPR, with actual TEST FPR reported separately. Confusion, UNKNOWN, box-only HEAD
FP, bar recall, selective removal, and all-four BODY/HEAD correctness are
reported overall and per stratum. The 2x2 differences and interaction are
descriptive; seed variation and recall/false-alert tradeoffs matter.

## Source and acquisition

The procedural source has 216 fresh complete groups g2000..g2215: 160 TRAIN,
24 VAL and 32 TEST, four conditions each and one current RGB per condition.
Parameter seed 10080907. Independent preflight g4000..g4007 has 32 images and is
excluded. Each quartet retains identical remaining object/camera geometry and
rotates presentation order. BODY box and HEAD bar stay vertically separate.

Diverse camera pitch ranges -12 to 3 degrees, yaw -7 to 7 degrees, lateral
position +/-1.1 m, and object lateral offsets +/-0.18 m; obstacle widths vary.
TRAIN diverse X anchors are 22/26/30 m, VAL 23/27/31, TEST 24/28/32, with +/-0.2 m
jitter. Narrow strata use the previous camera range around X=26 m, pitch -5,
yaw zero, with newly sampled groups. Optical height stays 1.70 m above the floor.
All share one Willow map: held-out groups and positions are not independent
background assets or natural scenes. Camera rotation varies independently of
the world-X forward body query; no unknown body pose is supplied to the model.

Every present object must pass native depth agreement/visibility and intended
BODY/HEAD label checks. Missing intended visibility fails the source rather than
becoming a negative. Neither means no task object, not all-scene free space.
Training supervision contains exactly TRAIN+VAL; variant metadata is TRAIN-only.

Existing native GPU-pair capture uses bounded asynchronous readback and background
encoding with burst scheduling. Each condition retains full 26 settling updates,
plus the actual capture (50 plus one for the first frame). There is one frame
per clip, so repeated-pose history reuse provides no assumed benefit. Actual
independent-state preflight cost determines whether the planned capture fits the
600-second engine budget. The prior repeated-workload 31 fps is not substituted
for a measurement of this workload.

## Results and disposition

All 864 main images passed native verification. Twelve fits completed all 7,200
planned steps, observing all 160/800 eligible images respectively. Every model
has 45,646 parameters and frozen temporal/approach weights remained unchanged.
Near truth and prediction UNKNOWN are zero; closing is 128 UNKNOWN per head in
new TEST and unscored. No extra fit or TEST-selected seed followed scoring.

Primary frozen ensemble thresholds: BODY 0.1558650881, HEAD 0.3789404432. Each
head has 64 positive and 64 negative new TEST examples. The new gate changes
the functional mapping even at identical initial weights; frozen thresholds do
not guarantee matched calibration. Secondary common-VAL calibration is retained
to expose this operating-point sensitivity.

| Primary ensemble | BODY TP/FP/FN | HEAD TP/FP/FN | Joint all-four correct | Narrow | Diverse |
| --- | --- | --- | --- | --- | --- |
| existing_plain | 64/12/0 | 64/22/0 | 14/32 | 14/16 | 0/16 |
| existing_region | 64/18/0 | 64/31/0 | 14/32 | 14/16 | 0/16 |
| expanded_plain | 64/20/0 | 62/9/2 | 13/32 | 10/16 | 3/16 |
| expanded_region | 64/9/0 | 63/4/1 | 21/32 | 15/16 | 6/16 |

The combination improves joint correctness from 43.75% to 65.625%, while neither
single change improves that primary ensemble endpoint. Data effect is -1 group
with the plain head and +7 with the region head; head effect is 0 with existing
data and +8 with expanded data. Difference-in-differences is +8/32 (25 percentage
points), descriptive for these fixed implementations, not a significance claim.

The combined ensemble reduces HEAD box-only false alerts from 14/32 to 4/32,
while bar-only recall changes from 32/32 to 31/32. Its BODY has 9/64 false alerts
versus 12/64 baseline, with 64/64 recall preserved. The harder diverse stratum
still has BODY 8/32 false alerts and HEAD 4/32, with HEAD 31/32 recall; 6/16
all-four correctness there is insufficient for a robust generalization claim.

Selective-removal joint correctness (remove only the corresponding alert while
retaining the other correct alert):

| Primary ensemble | Initially both heads correct | Bar removal + BODY retention | Box removal + HEAD retention |
| --- | --- | --- | --- |
| existing_plain | 32/32 | 18/32 | 23/32 |
| existing_region | 32/32 | 16/32 | 22/32 |
| expanded_plain | 32/32 | 23/32 | 15/32 |
| expanded_region | 32/32 | 28/32 | 25/32 |

All initially correct denominators are 32 here, so conditional and unconditional
rates coincide. Neither large score deltas nor missing predictions count as
successful selective decisions.

| Individual seed | BODY TP/FP/FN | HEAD TP/FP/FN | Joint /32 | Diverse /16 |
| --- | --- | --- | --- | --- |
| existing_plain 17 | 64/9/0 | 54/14/10 | 17 | 3 |
| existing_plain 29 | 63/10/1 | 62/21/2 | 15 | 2 |
| existing_plain 43 | 63/11/1 | 59/25/5 | 12 | 0 |
| existing_region 17 | 64/10/0 | 61/25/3 | 15 | 0 |
| existing_region 29 | 64/23/0 | 64/29/0 | 12 | 0 |
| existing_region 43 | 64/11/0 | 64/31/0 | 16 | 0 |
| expanded_plain 17 | 64/18/0 | 60/10/4 | 15 | 4 |
| expanded_plain 29 | 64/24/0 | 58/4/6 | 9 | 3 |
| expanded_plain 43 | 64/11/0 | 58/13/6 | 15 | 4 |
| expanded_region 17 | 64/6/0 | 61/3/3 | 24 | 8 |
| expanded_region 29 | 64/14/0 | 62/2/2 | 19 | 8 |
| expanded_region 43 | 64/3/0 | 61/3/3 | 25 | 10 |

The combined model improves joint correctness over corresponding existing_plain
and expanded_plain seeds in all three cases. It does not improve every metric:
BODY seed29 false alerts rise from 10 to 14 relative to existing_plain. Ensemble
thresholds are independently frozen from G8, so ensemble scores need not beat
each individually thresholded seed. No best-TEST-seed deployment is selected.

Secondary VAL-calibrated ensemble results, clearly separate from primary:

| Ensemble | BODY TP/FP/FN | HEAD TP/FP/FN | Joint /32 | Narrow /16 | Diverse /16 |
| --- | --- | --- | --- | --- | --- |
| existing_plain | 63/7/1 | 43/9/21 | 11 | 10 | 1 |
| existing_region | 64/10/0 | 48/7/16 | 14 | 10 | 4 |
| expanded_plain | 64/8/0 | 60/8/4 | 20 | 14 | 6 |
| expanded_region | 64/11/0 | 63/3/1 | 22 | 15 | 7 |

Data alone is beneficial at the recalibrated endpoint, so the primary -1 group
effect is not a universal claim that expanded data cannot help a plain model.
The combined secondary BODY FPR is 17.19%, HEAD 4.69%, versus plain expanded
12.5%/12.5%. VAL's <=5% rule does not guarantee TEST FPR. This 22/32 secondary
joint score does not replace the primary 21/32 headline.

Old G9-B TEST-only consumed regression, using primary thresholds:

| Ensemble | BODY TP/FP/FN | HEAD TP/FP/FN | Joint /12 |
| --- | --- | --- | --- |
| existing_plain | 24/2/0 | 23/1/1 | 8 |
| existing_region | 24/0/0 | 24/0/0 | 12 |
| expanded_plain | 24/3/0 | 22/4/2 | 5 |
| expanded_region | 24/0/0 | 23/0/1 | 11 |

This is a new fit protocol with 600 steps and new common VAL, not a reproduction
of the earlier 300-step G9-B checkpoint. Original G9-B artifacts remain intact.

**Localization limitation:** direct use of a predicted region does not make it
a faithful localization explanation. Positive support IoU BODY/HEAD is
0.327/0.343 existing_plain, 0.219/0.152 existing_region, 0.247/0.179 expanded_plain,
and 0.173/0.102 expanded_region. Pointing for existing_plain is 55/64 BODY and
49/64 HEAD, versus 32/64 and 47/64 for expanded_region. The improved decisions
coexist with worse binary support localization. Near-loss gradients through the
gate can change what the map represents; these observations alone do not isolate
why it changed. Do not present the learned gate as a true obstacle mask or a
certified causal explanation. First fixed narrow/diverse TEST quartet visuals
are retained as `g2184-quartet.png` and `g2185-quartet.png`; the inspected diverse
example includes a combined-model BODY false alert on a bar-only scene.

**Disposition:** retain expanded data plus predicted-region gating as the next
near-field Development candidate, with the plain model and same-data comparator
preserved. It gives useful decision improvement across three seeds and a second
operating-point policy, but has substantial diverse-scene residual errors and
degraded support-map fidelity. Neither architecture alone nor increased data
alone is an unconditional winner. Keep acquisition capability, fixed comparison
code and checkpoints ready for further separately scoped tests; no additional
experiment, cross-map claim, closing claim or default-App promotion follows.

## Measured cost and validation

Independent-state preflight: 32 images, all-frame acquisition 5.391 s (5.936 fps),
full engine 75.328 s. Predicted 864-image acquisition 145.557 s and engine
215.494 s. Main actual acquisition was 154.937 s (5.576 fps), capture-script wall
193.578 s and engine 223.141 s. Acquisition includes the first warmup and final
drain; engine additionally includes loading/startup/shutdown. It is not a
submission-only rate. All 864 pairs completed with zero failures or pending jobs,
and queue depth peaked at its bound of four. GPU copy/readback accumulated
0.742 s, encoding 23.402 s and writing 3.402 s; overlapped counters are not added
to infer wall time. Same-map storage and source hashes are unchanged.

Twelve actual fits including validation/checkpoint work totaled 47.895 s;
training plus prediction pipeline 59.261 s internally, 62.215 s including Python
startup. Peak CUDA allocation 1,283,291,136 bytes; actual RTX 5060 Laptop GPU,
torch 2.11.0+cu130. Native verification subprocess 6.951 s, evaluator 1.361 s.
No training transfer or duplicate capture was needed. This is not a matched
speedup against a different historical sample count or training schedule.

Twelve focused tests passed: two source allocation/geometry, five training and
gating/boundary, five evaluation/split/threshold/contrast tests. Actual native
verification, all fits, fresh scoring and consumed regression completed. Owned
UE process trees were released, with no matching birth-identity survivors.

## Evidence and lifecycle

Artifact entry: `artifacts.local/nearfield/diversity-20260907/`. `protocol.md`
records the pre-outcome design. Sources: `diversity_spec.py`,
`diversity_verify.py`, `diversity_model.py`, `diversity_train.py`, and
`diversity_evaluate.py`. Acquisition uses the existing `tools/ue_native_capture.py`
factorial route with the built native plugin and task-owned source snapshots.
Native verification reuses `grounding_verify.py`. Existing G8/G9 outputs remain
unchanged. Raw evidence, models and receipts stay in the ignored artifact tree.
One completed comparison ends experimental sampling; delivery and cleanup follow.
