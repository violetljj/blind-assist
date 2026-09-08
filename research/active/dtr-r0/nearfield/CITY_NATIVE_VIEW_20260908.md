# Same-instance native bollard view probe

2026-09-08. Follows the [pre-outcome protocol](CITY_NATIVE_VIEW_PROTOCOL_20260908.md)
and [fit/transfer audit](CITY_NATIVE_SCORE_AUDIT_20260908.md). One unchanged native
TRAIN bollard; eight views, no model fitting or threshold selection. This is
selected same-map Development, not independent-instance confirmation.

Controls0..3 recapture TRAIN6,8,11,13: yaw0, forward distances1.5/2.4m and lateral
offsets-0.08/+0.08m. Paired transverse views4..7 use yaw90 and the same distances,
camera height2.43m, pitch-5 and roll0. Camera-local offsets rotate with yaw;
target geometry and original map are not moved. Rendering includes surrounding
full-detail streets and distant HLOD. Exact source spec and capture receipts
are retained under `artifacts.local/nearfield/city-native-view-probe-20260908/`.

Interpretation requires both recapture control agreement and reliable target
visibility. Whole-scene query labels and isolated target labels have different
scope. UNKNOWN collision/render disagreement cannot become a target miss.
Changing viewpoint also changes background, occlusion and object appearance;
this experiment cannot assign a separate causal effect to any one of these.

## Measured result

**Same-instance view sensitivity reproduced; retain the original model.**
Replay/thin preserves all four control alerts at its existing DEV BODY cutoff
0.929635. All four transverse scores fall below it. Only the two near-distance
pairs (0->4, 1->5) have independently reliable target geometry on both sides:
their alert-plus-support-overlap hits fall **2/2 -> 0/2**. Far transverse6/7
remain target UNKNOWN, so their score changes are descriptive and are not
counted as target misses. All eight scene labels are BODY positive / HEAD
negative; scene label coverage does not repair unknown target identity evidence.

| Pair | Original control -> transverse | Native-only control -> transverse | Replay/thin control -> transverse | Target pair |
| --- | --- | --- | --- | --- |
| 0 -> 4 | 0.49205 -> 0.49276 | 0.85290 -> 0.68152 | **0.97307 -> 0.75760** | EVALUABLE |
| 1 -> 5 | 0.49184 -> 0.48558 | 0.85734 -> 0.23741 | **0.96092 -> 0.48924** | EVALUABLE |
| 2 -> 6 | 0.49262 -> 0.49250 | 0.97918 -> 0.42925 | 0.99090 -> 0.57959 | UNKNOWN treatment |
| 3 -> 7 | 0.49110 -> 0.49227 | 0.93033 -> 0.41627 | 0.97426 -> 0.55098 | UNKNOWN treatment |

On the two evaluable pairs, replay probability deltas are -0.21547 and -0.47168.
Both native-only and replay retain target support overlap2/2 on the control
and treatment sides at support cutoff0.5. Original overlap stays0/2 on both.
Thus the new view can retain target support activation while losing the global
near alert. This is a descriptive failure of the combined representation/head
on these views, not evidence isolating a causal internal mechanism.

At existing DEV cutoffs, original and native-only have0/2 control and0/2
treatment alerts on the evaluable pairs. At the unchanged historical BODY
cutoff0.982388, all three models have0/2 and0/2 on those pairs. No new threshold
was selected. Replay's four all-view score drops must not be reported as four
verified target misses: two transverse target labels are UNKNOWN.

## Recapture and scene support checks

Control poses exactly match original TRAIN6,8,11,13, but RGB is not byte-identical.
Full-resolution mean absolute RGB differences are 0.8752,0.8404,0.8690,0.8054;
after the exact256x144 BOX resize they are 0.4785,0.4462,0.4809,0.4223, on
the uint8[0,255] scale. Across four controls, maximum absolute BODY score
differences from the old TRAIN caches are 0.000157 original,0.024576 native-only
and0.010059 replay. Rendering and inference batch context are not separately
isolated by this check.

Control alert agreement with old cached TRAIN scores is original4/4,
native-only3/4 and replay4/4 under DEV policy; under historical policy it is
4/4,4/4,3/4. Native-only control2 newly crosses its DEV cutoff; replay control3
falls below the historical cutoff. These drift-sensitive decisions are disclosed.
The two near-distance replay controls retain their old DEV decisions and their
score drift (+0.00376,-0.00425) is much smaller than the paired transverse drops.

Pooled BODY positive-cell counts distinguish scene support from recorded target
support; UNKNOWN target portions are never silently assigned to another object.

| Frames | Whole-scene positive | Recorded target positive | Outside recorded target | Of outside: target UNKNOWN |
| --- | --- | --- | --- | --- |
| 0,1 (each) | 7 | 5 | 2 | 0 |
| 2,3 (each) | 2 | 2 | 0 | 0 |
| 4,5 (each) | 5 | 5 | 0 | 0 |
| 6,7 (each) | 2 | 0 | 2 | 2 |

The broad planter visible behind transverse4 does not add a separate positive
BODY support cell in this measured query: reliable transverse4/5 scene positives
coincide with target positives. Controls0/1 contain two untargeted positive
cells. Visual context still changes substantially. The near output is a
whole-query alert, not a target-specific classifier; joint alert and target
overlap are co-occurrence and cannot independently prove what caused the alert.

## Execution and evidence

`tools/evaluate_city_native_view_probe.py` froze its input manifest before one
CUDA inference pass per original/native-only/replay checkpoint. All eight new
RGB predictions were cached for all models before opening new evaluator labels.
Checkpoint, historical model-source, DEV-lock and old TRAIN prediction-cache
hashes were checked. No optimizer steps, extra capture or threshold changes.

Actual CUDA: RTX5060 Laptop GPU, Torch2.11.0+cu130. Model-load/inference times
5.997s original,0.380s native-only,0.517s replay; total10.138s. The first model
includes first-use loading, so these are not steady-state speed comparisons.
Process exited successfully. Narrow threshold-flip/inclusive-boundary,
UNKNOWN exclusion, overlap/IoU and scene/target partition checks passed;
syntax/import/help and scoped whitespace checks passed.

Evidence: `city-native-view-probe-20260908/evaluation-v1/` under the canonical
nearfield artifact tree. Input manifest SHA-256:
`5f50ffaf36976decea31389d71e94a1d6a4d9ea8f9b2baaf5a8128c21f64268e`.
Result SHA-256:
`188c5f5e78c4077d48021805b085393ff86235fdb528174860dcfe2a537a142d`.

The original-instance change from earlier TRAIN/route comparisons is no longer
necessary to reproduce the near-alert failure: it occurs on the same declared
TRAIN instance after a view change. Viewpoint, background, appearance and query
composition still vary together, and only two pairs have reliable paired target
geometry. Keep the conclusion at this small consumed-Development diagnostic
scope; no model promotion, universal robustness claim or rescue experiment.

Capture and label validation completed on eight views. All floor probes hit
approximately0.73m; scene near labels are BODYpositive/HEADnegative throughout.
Isolated/full rendered target agreement passes8/8. Corrected independent rays
verify0..5; frames6/7 each retain one unexplained collision hit near the bollard
base and are target UNKNOWN. Only the two near-distance transverse pairs can
support independently checked paired target claims. This supplies no HEAD
obstacle coverage. The source-quality receipt records unchanged map/project
hashes and process release with no survivors. All eight paired RGB images were
visually inspected in `view-pairs.png`; reproducible plot code is `view_pairs.py`.
