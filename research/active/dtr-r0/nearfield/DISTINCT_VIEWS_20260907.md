# NF-G3: unchanged near-field geometry on distinct static views

Phase EXPLORE, new controlled synthetic views in the same frozen Willow map.
Question: does the retained ground-relative mechanism preserve useful near-field
reference agreement under changes in object distance, background and pitch?
This is a source expansion, not independent natural/real-camera confirmation.

Freeze18 cases from `distinct_view_cases.py` before capture/model access:
ordinary ground; low block,4 cm pole,overhead bar,wall,L-shaped geometry each
at2/5 m center distance; low block at pitch-25/+25 degrees; low block with
background panels8/18 m away; a1 m wall that hides ground; flush2 mm marking;
low block centered3 m away as a retained transition case. Near surfaces have
finite thickness, so nominal center distance is not a scored depth threshold.
All cases use camera position(26,0,1.82)m, floor .12m, except stated pitch.
HFOV100 degrees,640x360, roll0. Temporary geometry is destroyed between views;
no map/material save, no art improvement, no actor data passed to inference.

Capture RGB and native forward depth once per case in one offscreen editor
session. Rendering warmup is allowed for image settling, not1:1 trajectory
playback. No motion episode or first-alert latency score: views are independent.
Verify frozen map hash before/after and receipt-owned process release. Preserve
failed captures and stop at18 views; repair mechanical defects without tuning
algorithm/source parameters from scores.

Same existing Hypersim Small metric model,input518,max20m,CUDA,one prediction
per image. Save predictions before evaluation. Compare raw,ground-only,and
ground-plus-scale with unchanged NF-G1 fitter and default depth support. Do not
retry rejected fits, hold stale scale, sweep thresholds or change weights.
Dependent arms return UNKNOWN for rejected fits. Report all162 cells including
UNKNOWN positives as missed alerts; retain transition cases and per-view rows.

Primary measures: full direction-height TP/FP/FN against the unchanged native
supported encoder, fit acceptance, UNKNOWN, per-condition outcomes. Report
center-cell vectors to inspect intended manipulations, but not object-instance
recall: the native reference also includes existing scene surfaces, and semantic
case names do not prove target visibility. Same-map novel views are not novel
categories. Ground-hidden and source geometry are test conditions, not runtime
labels. Native depth is evaluator-only; camera height/pitch remain privileged.

Only retain the generality claim if the frozen method shows a useful consistent
advantage with disclosed FP/UNKNOWN costs. Otherwise identify the limiting
condition and stop; do not repair failures by loosening the3 m or height limits.
Future model comparison or learning follows this outcome, not an automatic run.
Hash/code/model receipts and cache enable later evaluation without UE/inference.
Measure capture wall time,inference and postprocess separately; desktop CUDA is
not phone latency. NumPy/JSON metadata use CPU TASK_NOT_GPU_SUITABLE; model and
dense encoding use CUDA with research_backend device observations.

## Results

Completed one18-view capture,18 model calls,and one cache-only evaluation in
`artifacts.local/nearfield/distinct-views-20260907-v1`. No method/threshold changes.

| Arm | TP | FP | FN | Binary TN | UNKNOWN | UNKNOWN positives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw | 22 | 0 | 65 | 75 | 27 | 3 |
| Ground-relative | 37 | 0 | 50 | 75 | 52 | 19 |
| Ground-relative + scale | 37 | 0 | 50 | 75 | 32 | 18 |

Denominator:87 native-positive cells of162 observations. These are scene-wide
direction/height cells, not87 inserted objects. Binary TN can include UNKNOWN;
0 FP is reference agreement in this small source, not demonstrated absence of
false alerts in natural scenes. The three failed fits are included, not dropped.

## Distinct conditions change the decision

Center-direction evidence (height bands remain low/body/head):

| Case | Native center reference | Ground-relative outcome |
| --- | --- | --- |
| Near low block | low | recovered |
| Near4 cm pole | low/body/head | recovered |
| Near L shape | low/body/head | recovered |
| Near suspended bar | head | missed |
| Near wall | low/body/head | all UNKNOWN: insufficient ground coverage |
|1 m wall hides ground | body/head | all UNKNOWN: no plausible plane |
| Low block, pitch-25 | low | missed |
| Low block, pitch+25 | none; target outside view | UNKNOWN; not an obstacle-recall opportunity |
| Low block,8/18 m backgrounds | low in both | recovered in both |
|3 m-center transition block | low | missed; retained in denominator |
| Ordinary center floor / flat marking | none | no near alert |

All five far-object cases have no center near alerts. However, ground-only
low/pole/bar far cases have UNKNOWN center states, while wall/L far cases have
NO_NEAR_OBSERVED. Therefore this does not establish reliable FAR classification.
Ordinary-ground and marking frames still contain existing side obstacles:
their full-frame reference has positives, so they are not globally empty scenes.
Case labels are source conditions, not independent target-instance recognition.

The strongest new failure is the ground-fit gate: in the two wall cases the
raw branch already retains15 native-positive cells (9 and6); dependent ground
branches discard all15 into UNKNOWN. The third failed fit,upward pitch, has no
native-positive cells and correctly exposes absent ground observation. Aggregate
improvement22→37 therefore hides a substantial per-condition regression.

Keep ground-relative geometry as a useful component, not a mandatory entry
gate for all alerts. Next implementation should separate existing credible
near-obstacle/direction evidence from unavailable ground/height evidence. The
present experiment does not validate a raw fallback or evidence fusion; do not
count the15 counterfactual retained cells as a new algorithm score.

Suspended-bar and downward-pitch low-block failures remain after an accepted
fit, so ground availability alone is insufficient. Those cached views are the
next diagnostic targets for distance/height/local-detail loss. Do not sharpen
boundaries, relax thresholds or retrain based solely on this aggregate score.

## Source, cost and verification

The contact sheet was visually inspected: inserted low blocks,poles,bars,walls,
L geometry and background panels are visible in the intended views; the upward
view excludes the low block and the close wall fills the image. Materials are
simple controlled primitives in an existing street, not photorealistic obstacle
assets or natural-category coverage. No explicit isolated-shadow manipulation
was included; existing scene shadows are not an independent shadow test.

Acquisition checks passed for all18 pairs: RGB/native dimensions and float32
depth, camera calibration,distinct near/far native geometry,and clear-floor
optical height (maximum error0.00000573 m). Frozen Willow hash stayed
`cf35e5c9df54cd0f781f09ea8105fe8ef6078ed0822d4e594d64216e79a254fb`.
The copied evaluator spec and source spec hashes match. `process-release.json`
reports released=true,no survivors/errors; no UnrealEditor remained afterward.
A verifier import-path error was repaired before scoring; no recapture or
algorithm/source change followed it. All five new Python source files parsed.

Editor startup through exit took approximately77 s from process/log timestamps,
including map loading and render settling. This is one-time capture cost, not
temporal replay. All18 predictions took5.716 s internally including model load;
model-call P50 was91.21 ms (first-call startup retained,no extra warmup inference).
Subsequent cache-only evaluation took0.845 s internally,4.28 s process wall.
Fit P50 was2.02 ms; arm postprocess P50 approximately3.28–3.35 ms, with failed
fits included as short UNKNOWN returns. These are workstation measurements,
not phone or first-alert timing. Prediction and evaluation are separate CLI
actions, so future representation checks reuse the same cached depth.

Central registration again failed before mutation on the existing index line252
fingerprint mismatch; `registration.log` preserves it. Local evidence is complete,
but no new registered terminal,promotion or natural validation is claimed.

Prediction receipt SHA-256:
`fa29b6ba9e2fecf94caffd0ca05732c579f17fe43ac2ef3b35dfa96057e010ec`.
Evaluation receipt SHA-256:
`e332b251bdd0267897855ad41fbafdf5cb40f817037e9c08b00d4de8a8b8174a`.
Source spec SHA-256:
`830982d6c13a78c504f82ff548811c4e730b7b09218ff30c9966aab99b861865`.
Receipts preserve code/model/input identities; capture contains native arrays,
RGB,contact sheet,verification and process-release evidence. No task-owned
resource remains allocated.

Use existing CUDA Python; each output directory/file must be new:

```powershell
python research/active/dtr-r0/nearfield/distinct_view_cases.py --output artifacts.local/nearfield/new-views/spec.json
python research/active/dtr-r0/nearfield/launch_distinct_views.py --spec artifacts.local/nearfield/new-views/spec.json --output artifacts.local/nearfield/new-views/capture
python research/active/dtr-r0/nearfield/verify_distinct_views.py --capture artifacts.local/nearfield/new-views/capture
python research/active/dtr-r0/nearfield/run_distinct_views.py predict --capture artifacts.local/nearfield/new-views/capture --metric-source artifacts.local/downloads/depth-lab/src/Depth-Anything-V2-main/metric_depth --weights artifacts.local/models/depth-anything-v2-metric-hypersim-small/depth_anything_v2_metric_hypersim_vits.pth --output artifacts.local/nearfield/new-views/predictions
python research/active/dtr-r0/nearfield/run_distinct_views.py evaluate --predictions artifacts.local/nearfield/new-views/predictions --output artifacts.local/nearfield/new-views/evaluation
```
