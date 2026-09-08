# Small native City adaptation: response improves, retention not met

2026-09-08. Completed one seed17,300-step fit under the unchanged
[pre-outcome protocol](CITY_NATIVE_ADAPT_PROTOCOL_20260908.md). No extra steps,
second fit, outcome-driven source replacement or automatic expansion followed.
Artifacts: `artifacts.local/nearfield/city-native-adapt-20260908/`.

## Source and geometry

TRAIN45 original-mesh views cluster around x=-371m on the plaza sidewalk;
DEV45 use another street frontage around x[-233,-225], y=-16m, viewed facing
the opposite direction. The scout centers (-350,-240) are discovery locations,
not mandatory target centers. Unique instances were chosen before any model
output; duplicate colocated signs were excluded. Camera separation is139.93m
between splits; TRAIN/DEV minimum distances to the original40route are71.57m
and58.47m. Target identities and RGB hashes are disjoint, but map, asset types
and possibly distant backgrounds are shared. This is same-map Development.

Full nearby loading plus distant HLOD is retained. Source map/project hashes
remain unchanged, all90 new views are retained. TRAIN BODY31positive/14negative,
HEAD20/25; DEV BODY28/17, HEAD18/27. Both splits have zero missing-floor frames.
Training/evaluation supervision describes visible native rendered geometry
relative to measured floor, not certified collision/free-space truth.

**Label correction:** during new source admission, a ray hitting the same
original instance earlier than its rendered surface was incorrectly called
OCCLUDED. Such a hit is geometry disagreement and now stays UNKNOWN. Only a
different nearer instance may supply occlusion. The observed native lamp sample
differs by16.5cm, well beyond the3cm criterion. A focused regression test covers
this distinction. Raw captures and v1 label outputs are preserved. Cached rays
were reclassified and train/dev/route labels-v2 generated before this fit.

Under the corrected independent check, TRAIN bollard/sign and DEV sign have
reliable opportunities; meters and DEV lamp do not. In the original40route,
the two previously reported independently verified meter opportunities are
withdrawn: meter is NOT_EVALUABLE for collision-verified target metrics.
Bollard2 BODY, sign2 BODY and sign2 HEAD opportunities remain. The original
all-scene rendered support labels and alert denominators are unchanged.
This correction supersedes the old meter collision-verification claims; it
does not remove these images or invent a negative label. Independently suspended
bar coverage remains untested.

## Fixed comparison

Initial = original G13-D seed17 (not the previous three-seed ensemble). Adapted
= same initialization after300 steps, batch16, headLR1e-4/backboneLR1e-6,
masked near BCE +0.25 existing UNKNOWN-aware balanced support BCE. All BN running
buffers are unchanged. No original-route or DEV labels enter fitting.

Each model's operating points are independently chosen by the same fixed DEV
FPR<=10% rule and locked before regression labels are opened. Only finalstep300
is evaluated. The following table uses that DEV-selected policy:

| Domain/head | Initial TP/positive | Adapted TP/positive | Initial FP/negative | Adapted FP/negative |
| --- | ---: | ---: | ---: | ---: |
| DEV BODY | 12/28 | 13/28 | 1/17 | 0/17 |
| DEV HEAD | 0/18 | 8/18 | 0/27 | 2/27 |
| Consumed route BODY | 1/17 | 3/17 | 1/22 | 0/22 |
| Consumed route HEAD | 0/4 | 2/4 | 0/35 | 0/35 |

One original-route frame remains UNKNOWN in both heads. These are correlated
frames, not independent hazard counts. At unchanged historical thresholds,
DEV initial BODY/HEAD TP=0/0 becomes12/7 (FP0/1); route initial TP=0/0 becomes2/1
(FP0/0). Thus the observed response gain is not solely DEV threshold selection.
DEV support positive-frame mean IoU rises from0/0 to0.1905/0.1124 BODY/HEAD;
route support IoU rises from0/0 to0.1116/0.0968.

On independently checked route targets under the DEV-selected policy, bollard
BODY remains0/2, sign BODY remains0/2, and sign HEAD improves0/2 to1/2. Meter is
NOT_EVALUABLE under the corrected collision criterion and is not scored as a
miss. Partial all-scene alert improvement is not reliable target localization.

On the original first16 Willow VAL samples, unchanged historical thresholds
retain32/32 correct head decisions. However, BODY/HEAD support IoU falls from
0.4428/0.2037 to0.1345/0.0266: substantial localization forgetting is present.
The protocol's alert-regression criterion alone does not capture that loss.
At initial DEV-selected HEAD cutoff, the all-negative sentinel itself suppresses
Willow alerts; do not confuse that operating-point effect with model forgetting.

## Decision and verification

**RETAIN_ORIGINAL.** Adapted DEV recall46.43% BODY and44.44% HEAD both miss the
predeclared50% requirement. Do not promote or automatically continue fitting.
Retain the checkpoint/evidence as a diagnostic component: native source
adaptation revives some support and alerts, but thin-obstacle misses and
localization forgetting remain unresolved. A future comparison should address
those measured gaps explicitly rather than rescue this run by threshold or
budget changes. No fresh TEST/generalization claim is supported.

CUDA / RTX5060 Laptop, Torch2.11.0+cu130: fitting43.36s, whole model
run59.64s. Local execution avoids transferring these small freshly captured
arrays; no remote worker was occupied by this task. Final checkpoint SHA256
`4703d479c09c18b80a3537fb03d4a50848f542cc556a6eea24a800b1fe5f00cc`.
Five focused supervision/selection/ray regression tests pass. Capture, corrected
label generation and the single bounded fit/evaluation completed. Task-owned
UE and Python processes exited; receipts retain source hashes, source separation,
BN equality, sampled schedule, locked DEV choice, UNKNOWN and all predictions.

Main evidence: `source-admission.json`, `source-separation.json`,
`{train,dev,route}-labels-v2/label-validation.json`,
`fit-v1/{protocol,locked-dev-choice,result,receipt}.json`.
