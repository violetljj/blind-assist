# Native thin-positive sampling with old-scene rehearsal: result

Status: **RETAIN_ORIGINAL / COMBINED_RECIPE_FAILED**, 2026-09-08.

The one new seed17 replay-plus-thin fit preserves more Willow localization than
the cached native-only fit, but does not fix the route bollard alert miss. DEV
BODY recall reaches 53.6%; HEAD falls to 16.7%. Willow BODY/HEAD support IoU
retains only 52.2%/44.0% of the original, below the frozen 80% requirement.
The final step300 checkpoint is not promoted. No rescue fit or new acquisition
follows this result.

## Recipe and authority

Followed the [frozen protocol](CITY_NATIVE_REPLAY_PROTOCOL_20260908.md).
This is one combined sampling/replay intervention, not evidence isolating either
ingredient. Initialized from original G13-D seed17, never the prior fitted
checkpoint. Exactly 300 AdamW steps of batch16, head/backbone LR 1e-4/1e-6,
weight decay 1e-4, masked near BCE plus 0.25 globally class-balanced UNKNOWN-aware
support BCE. All 318 BN running buffers remained bitwise unchanged.

Each batch drew four verified native TRAIN bollard BODY positives, four uniform
native TRAIN frames (including possible thin positives), and eight original
Willow TRAIN frames. Native corpus: unchanged corrected TRAIN45; thin sample
indices 6,8,11,13. Replay corpus: first64 saved-order TRAIN samples, 16 groups
g2000..g2015. The complete seed17 schedule contains 1,200 dedicated thin draws,
1,200 uniform native draws and 2,400 Willow draws. Uniform native draws include
100 additional thin exposures, giving 1,300 thin-pool presentations in total.

Original dataset, RGB, training-label/support and model-source hashes were
verified; sample/group/frame exclusions and replay RGB separation from City
DEV/route and original Willow VAL16 were checked before fitting. Native visible
surface labels, not intended target names, determine losses. Corrected native
labels-v2 retain UNKNOWN; parking-meter independent collision verification
remains withdrawn. Same-instance early collision remains UNKNOWN rather than
being reclassified as occlusion. No VAL/TEST samples enter gradients.

DEV45, consumed route40 and Willow VAL16 are reused Development/regression data.
These sources already informed development; there is no fresh confirmation or
independent-world claim. The earlier native-only fit is a cached comparator,
not rerun. Original-arm result dictionaries match exactly across the two runs.

## Near alerts

Each cell is `TP/positive; FP/negative`. Inclusive historical cutoffs remain
BODY 0.982388 and HEAD 0.885662. DEV-selected policies apply the same frozen
FPR<=10%, maximum-recall/lower-FP/higher-threshold rule separately to each model;
they do not share one numerical cutoff. Thresholds were saved before route-label
access. Original/native-only/replay DEV cutoffs are respectively
BODY 0.493135/0.977686/0.929635 and HEAD all-negative/0.821759/0.761072.

| Source / policy | Head | Original | Cached native-only | Replay + thin |
| --- | --- | --- | --- | --- |
| DEV / historical | BODY | 0/28; 0/17 | 12/28; 0/17 | 3/28; 0/17 |
| DEV / historical | HEAD | 0/18; 0/27 | 7/18; 1/27 | 1/18; 1/27 |
| DEV / DEV-selected | BODY | 12/28; 1/17 | 13/28; 0/17 | **15/28; 0/17** |
| DEV / DEV-selected | HEAD | 0/18; 0/27 | 8/18; 2/27 | **3/18; 1/27** |
| Route / historical | BODY | 0/17; 0/22 | 2/17; 0/22 | 1/17; 0/22 |
| Route / historical | HEAD | 0/4; 0/35 | 1/4; 0/35 | 0/4; 0/35 |
| Route / DEV-selected | BODY | 1/17; 1/22 | 3/17; 0/22 | 2/17; 0/22 |
| Route / DEV-selected | HEAD | 0/4; 0/35 | 2/4; 0/35 | 0/4; 0/35 |

Replay DEV-selected BODY recall/FPR is 53.6%/0%; HEAD is 16.7%/3.7%.
Thus the joint two-head DEV criterion fails. All 45 DEV near labels are known.
Each route head keeps one UNKNOWN frame (39/40 near-evaluable); it is not a
negative or a miss. All three models retain 32/32 correct Willow historical
head decisions. Replay loses zero previously correct Willow decisions under
either policy; alert retention does not establish support retention.

## Target recovery and localization

Reliable opportunities below are two frames per row. Values are
`alert hits / support-overlap hits / joint hits`, each out of two. The displayed
target counts are identical under historical and DEV-selected policies.

| Verified target/head | Original | Cached native-only | Replay + thin |
| --- | --- | --- | --- |
| Bollard / BODY | 0 / 0 / 0 | 0 / 2 / 0 | **0 / 2 / 0** |
| Supported sign / BODY | 0 / 0 / 0 | 0 / 2 / 0 | 0 / 2 / 0 |
| Supported sign / HEAD | 0 / 0 / 0 | 1 / 2 / 1 | **0 / 2 / 0** |

The required bollard joint recovery >=1/2 fails despite support overlap 2/2.
Replay does not increase verified target misses versus original, but loses the
native-only sign HEAD recovery. Parking meter is UNKNOWN and excluded from
verified success/failure claims. Target masks cannot supply whole-query false
alert denominators; route scene labels above supply those.

Mean positive-frame support IoU, fixed support threshold 0.5 and UNKNOWN-aware
20x20 pooling (positive wins, otherwise UNKNOWN wins):

| Source/head | Original | Cached native-only | Replay + thin |
| --- | --- | --- | --- |
| DEV BODY | 0 | 0.1905 | 0.2492 |
| DEV HEAD | 0 | 0.1124 | 0.1346 |
| Route BODY | 0 | 0.1116 | 0.1020 |
| Route HEAD | 0 | 0.0968 | 0.1487 |
| Willow BODY | 0.4428 | 0.1345 | **0.2312** |
| Willow HEAD | 0.2037 | 0.0266 | **0.0897** |

Replay improves Willow localization relative to native-only while remaining
well below original: 52.2% BODY and 44.0% HEAD retained, versus required 80%.
Visible support recovery and low near-alert FPR also do not imply clean support
maps: replay produces at least one false-positive support pixel in 22/22
near-negative route BODY frames and 34/35 near-negative HEAD frames.
On DEV, negative-mask activation occurs in 17/17 BODY and 27/27 HEAD negative
frames. Positive IoU recovery does not establish a reliably localized detector.

## Execution and evidence

Actual backend: CUDA on NVIDIA GeForce RTX 5060 Laptop GPU, Torch 2.11.0+cu130.
Fit/checkpoint time 31.443 s; DEV evaluation 1.503 s; consumed route 2.810 s;
Willow16 0.238 s; total 41.404 s. Receipt PASS means execution completed, not
that retention passed. Final step300 only; no early selection or extra epochs.

Seven new replay contract tests and five existing native-adaptation tests passed
during implementation; they cover split/corpus/schedule invariants, actual-mask
thin admission, UNKNOWN loss/pooling, thresholds, and localization rejection.
Narrow whitespace validation passed. Post-fit reporting reads cached outputs
only and performs no additional inference.

Evidence root: `artifacts.local/nearfield/city-native-replay-20260908/fit-v1/`.
Cached comparator: `artifacts.local/nearfield/city-native-adapt-20260908/fit-v1/`.
Scientific comparison: `city-native-replay-20260908/delivery/replay-comparison.png`
under the same nearfield artifact entry.

| Artifact | SHA-256 |
| --- | --- |
| Replay final checkpoint | `32f391be00c67dafdd4a22403c19f40e9c522edf3ce9ded6c8d2f1bcdf47561b` |
| Training schedule | `b398ffd295321f61dd8f4f6ae781bcaac50afadf90fc2817fd9ddafb2e6fecfc` |
| Run protocol | `5d66dad93469e798e4998a7f134a3addcbdb0ed1f4bb3b93f3bb5d8d60ffba3c` |
| Locked DEV choice | `87e9ce789489d97560b7c9d4d1451c5e2160bb5321b6ac6b318dae431080f7d1` |
| Result | `5e3b2735d725c3ffa15f3f04d4c0386f96262632ee7871131311702ad88221b0` |
| Receipt | `b38eac8c9c38a68b898de12d76002dde5be7d026d6fb15c5788827c6b926f55a` |

Disposition: retain the original model. The tested mixed recipe is a negative
result for the combined retention objective; the partial localization benefit
does not authorize promotion, more training, or new universal reliability claims.
