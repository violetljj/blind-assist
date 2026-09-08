# Native city surroundings, target labels and fixed-method diagnostic

2026-09-08. User-authorized Development: complete the original route's street
surroundings, check native thin/protruding/raised surfaces, then run existing
methods once before considering more data. No training or threshold search.

## Fixed plan before model outcomes

Question: does the existing G10/G13 near-field signal transfer to visible
obstacles on one original City Sample sidewalk with reliable scene labels?
Hypothesis: geometry and source transfer may expose missed natural thin or
raised surfaces despite success on controlled Willow groups.

Source: unchanged official `/Game/Map/Small_City_LVL`, same original sidewalk
as [the first native route](CITY_NATIVE_ROUTE_20260908.md). Load full-detail
surroundings x[-400,-190], y[-100,100], z[-10,200] metres; permit distant HLOD
appearance only beyond the editor's 100 m setting, never as a near target.
Check actual start/middle/end views before final capture. Discover target
instances from the original map; do not add synthetic visible obstacles.

Keep one local route, at most 48 settled poses with natural approaches and
lateral clear views if needed to cover the existing targets. Route placement
may use native geometry and visual inspection, never model predictions. These
are curated Development observations, not a random or independent benchmark.
Do not claim real-time temporal behavior from spatially ordered editor views.

Baselines: fixed G10 expanded_region and G13 decoupled, each averaging seeds
17/29/43. Historical common-VAL recall-first thresholds:
G10 BODY/HEAD 0.7690758109092712/0.5946889917055765;
G13 0.9823878407478333/0.8856616814931234. Support threshold 0.5.
Keep checkpoint, source, input and threshold hashes. Cache predictions; no
refits, no City-selected threshold, no result-driven route replacement.

Evaluator-only labels: native scene axial depth, measured floor, original
instance identity and transform, isolated original-mesh depth, plus sampled
Visibility-channel complex collision rays against the original instance.
Rendered RGB is exported before evaluator-only clones exist; clone collision
and shadows are disabled, and clones are removed before the next view.
Clones are never the source of collision truth or model inputs.

All-scene support uses the existing BODY/HEAD boxes swept forward 3 m with
wearer yaw aligned to the camera's horizontal heading. Missing depth/floor
stays UNKNOWN. Negative means no visible in-query surface, not free space.
Target identity requires >=3 native-agree pixels within 3 cm and no unexplained
nearer-clone pixels. Independent ray status must pass (>=3 original-instance
matches within 3 cm, no unexplained sampled rays); occlusion is reported.
Failed or absent target geometry remains UNKNOWN/NOT_EVALUABLE, not a model
miss. Report thin pole, body protrusion and raised supported sign separately;
do not equate a supported sign with an independently hanging bar.

Primary diagnostic: visible-support alert misses/false positives and UNKNOWN
for fixed G10/G13; target-localization/support misses only on reliable positive
opportunities. Explain event/frame denominators and missing category coverage.
One source does not support method promotion or natural-world generalization.
If labels fail, keep the uncertainty and report only the supported diagnostic;
if models miss reliable targets, prioritize the observed transfer failure
before expanding data volume. Stop after this one fixed inference comparison.

## Evidence and result

Inputs, capture source snapshots, native arrays, target checks and predictions
are under `artifacts.local/nearfield/city-native-validation-20260908/`.
Surrounding-map three-view engineering capture `surround-v2` passes transport
checks and restores visible neighboring street blocks. Final `final-route-v1`
contains 40 settled poses along 16.95 m, maximum step 0.5 m. Start/end appearance
and target views were inspected: surrounding buildings are present; near native
meshes remain distinct. This loads the bounded full-detail region plus distant
HLOD appearance, not every full-detail actor in the whole city. Map and project
hashes are unchanged. Engine lifecycle was 97.39 s and process release passed.

`final-labels-v1` verifies three original instances: metal bollard (thin pole),
parking meter (body protrusion), and a supported parking sign (raised face).
Each has two reliable positive in-query frames: bollard 6/7, meter 14/15,
sign 28/30. Sign contributes both BODY and HEAD; the other two contribute BODY.
Rendered isolated/native agreement and original-instance collision-ray checks
gate these labels. Unsupported/occluded target opportunities are not negatives.
Frame 29's downward probe hits at camera height, so its floor and both heads
remain UNKNOWN. It is retained in all 40 predictions and excluded from known
label denominators. A truly independently suspended bar remains untested.

`fixed-models-v1` ran both frozen three-seed ensembles on CUDA once, with zero
optimizer steps. Model loading plus inference: G10 0.477 s, G13 7.914 s on
NVIDIA GeForce RTX 5060 Laptop GPU. These are batch diagnostic timings, not
device latency. RGB alone enters the models; predictions were cached before
opening evaluator labels. Original thresholds above were unchanged.

| Independently checked target | G10 alert misses / positive frames | G13 alert misses / positive frames |
| --- | ---: | ---: |
| Bollard BODY | 2/2 | 2/2 |
| Parking meter BODY | 2/2 | 2/2 |
| Supported sign BODY | 1/2 | 2/2 |
| Supported sign HEAD | 2/2 | 2/2 |

These are correlated frame opportunities, not independent obstacle counts.
An alert is query-level; support overlap is additionally checked for attribution.
G10 support overlaps all these target-positive frames but is diffuse (no target
peak hits); it is not equivalent to reliable localization. G13 support masks at
0.5 are empty. Do not interpret its zero false alerts as successful perception.

For the broader visible native-geometry reference (39 known frames, one UNKNOWN),
G10 BODY TP/FN/FP/TN=5/12/0/22 and HEAD=0/4/2/33; G13 BODY=0/17/0/22 and
HEAD=0/4/0/35. All-scene surfaces do not each have the independent identity check
of the three named targets. These results diagnose this fixed curated route,
not general city performance or walking safety.

Decision: retain the improved capture/label pipeline and this consumed
Development route; do not expand acquisition yet. Fixed G10/G13 transfer is
insufficient here. The next related model diagnostic should carry these misses
and UNKNOWN cases forward, examining source transfer and alert calibration
without calling post-hoc changes fresh confirmation. No training, threshold
search or additional route acquisition was performed after these outcomes.

Evidence entry points: `final-route-spec.json` (SHA256
`594cca5edcde96dc755b605ac7a38476678cf3a6e77d7bd2b341d6b6d73ed78e`),
`final-route-v1/{receipt,source-integrity,process-release}.json`,
`final-labels-v1/{label-validation,native-route-labels}.json`, and
`fixed-models-v1/{protocol,result,receipt}.json`. `route-preview.mp4` is a
40-frame, 13.33 s spatial slideshow at 3 fps, not a real-time walking video.
Target overlays mark BODY green and HEAD magenta.

Validation: nine focused label/evaluator tests passed; live 40-frame capture,
label generation and fixed inference completed. Failed engineering probes
`surround-v1` and `target-probe-v1` remain preserved with their error evidence.
