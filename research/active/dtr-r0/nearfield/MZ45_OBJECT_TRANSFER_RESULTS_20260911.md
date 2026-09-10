# MZ45: richer objects expose a transfer failure hidden by the rod controls

2026-09-11 EXPLORE. [Fixed comparison](MZ45_OBJECT_TRANSFER_20260911.md).
The MZ43 restricted-training gain does not transfer to the four new forms in
this pilot. RGB, MZ5 and MZ43's equal-logit ensemble miss every one of their 32
positive event bits under all three observation conditions. Eight of the 32
new-form frames are true nonintruding negatives; predicting every event absent
therefore gives 8/32 exact frames, not useful obstacle detection.

All 40 near/far attempts and ten paired groups are included: 32 new-form frames
and eight retained-rod controls, with 160 known event bits and zero UNKNOWN.
The labels were independently reconstructed from the original native arrays.
This is a same-site controlled object-replacement result with small scaled and
floating forms; it does not isolate object category from size/support context.

## New forms, separated from the retained controls

Pipe, ladder, pouch and woody birch each contribute eight frames and eight
positive event bits. Below, entries are false positives / false negatives.

| Input condition | MZ5 | MZ43 ensemble | MZ28 and MZ37, each |
|---|---:|---:|---:|
| Ideal comparator | 0 / 32 | 0 / 32 | 0 / 19 |
| Close returns merged | 0 / 32 | 0 / 32 | 0 / 15 |
| Close returns missing | 0 / 32 | 0 / 32 | 0 / 26 |

Local-support MZ28/MZ37 recover 13, 17 and six true bits respectively with zero
false bits here. The merged proxy's higher count is a decoder/proxy outcome,
not evidence that actual unresolved returns help a sensor. MZ35 and MZ37 equal
MZ28 in this new-form block. The joint FUSION comparison also recovers none of
the 32 new-form positives; mixed joint FUSION adds two false bits on ideal input.

The retained rod tells a different story: MZ5 misses one bit under each input,
while MZ43 ensemble misses one on ideal input and zero under either restriction,
all with zero false bits. The small overall improvement comes entirely from
this familiar control. Both predeclared new-form transfer criteria fail.
Retain MZ43's earlier measured scope; do not promote it for unseen forms.

## What the failure does and does not explain

On ideal input, mixed ToF alone has 13 correct new-form positives; averaging
with RGB suppresses all 13. Their RGB logits range from -10.45 to -3.01, whereas
the positive ToF logits range from 0.014 to 2.47. The same suppression affects
13 true bits under merging and one under missing returns. This establishes the
arithmetic effect of the current fixed mean, not calibrated branch reliability.
Taking every positive ToF output would also add 19, nine and three false bits
respectively, so unconditional union is not a demonstrated solution.

Original angular geometry supplies candidate support on 28/32 true bits for
ideal/merged input and 19/32 for missing input; the fixed availability bank
preserves those support counts. Some usable candidate geometry exists even
when the global RGB/ensemble outputs are negative. Candidate support does not
identify which physical surface caused a sensor return or establish clearance.

The structural audit also finds a confound in the data: new forms use two target
mesh actors and no ancillary supports, whereas the rod uses one cross-member
plus 12 support actors. Supports lie outside the body/head corridors but remain
visible. Two clamps move with the rod's target state. New-form nonintruding
controls move laterally, while the rod's control raises its member. Consequently
shape, apparent extent, support context and negative-placement convention are
not independently controlled. Reliance on these cues is a hypothesis, not yet
proven by this comparison.

A useful next intervention keeps target meshes, dimensions, positions and
native event labels fixed while changing only ancillary support context,
including the reciprocal rod-without-support control. That can distinguish a
fixture-context dependency from the broader shape/size problem before another
model change. This intervention has not been rendered or scored in MZ45.

## Verification and cost

The 16-frame original replay matches all checked retained visual, packet,
support and model decision arrays exactly. Compact/dense mixed ToF decisions
match on every executed batch. Independent scalar scoring recounts 19,200
grouped event bits; independent NumPy geometry reproduces all 160 native event
counts/labels from the ZIPs. No fitting, calibration, source extraction or
permanent dense-feature cache was used. The two source packages remain unchanged.

Model initialization, parity and new inference took 10.82 seconds on CUDA.
The 40-frame full/crop visual computation took 0.227 seconds and all three
condition readouts 0.126 seconds; these are internal segments, not device or
end-to-end latency. Predictions and compact diagnostic features total 380,684
bytes. Core evidence is under `artifacts.local/work/mz45-object-transfer-20260911/`:
`prepared-v1/receipt.json`, `inference-v1/{receipt.json,parity.json,predictions.npz}`,
`score-v1/{receipt.json,result.json,failures.json,audit.json}`, `diagnostics.json`
and `structure-audit.json`. Retain this fixed new-form failure as a negative
comparison for the next mechanism, with all source and model baselines intact.
