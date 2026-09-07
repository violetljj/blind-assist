# NF-G1: cached ground geometry factorial probe

Phase: EXPLORE; consumed synthetic Development only. Frozen before scoring.
Question: can predicted-only local ground geometry recover near-field alert
evidence lost before aggregation, and do predicted depth boundaries survive?

## Fixed comparison

Use all eleven cached predictions from `representation-20260907-v1`, bound by
its receipt SHA-256 `e23186a10e6f5021017c7336b8c9490617e6b86ebf7ce27b5225cec673f343c1`.
The four arms are raw, scale only, ground-relative only, and both. Same encoder,
3 m attention, height bands and spatial support. No tuning after scoring.
Fit once per frame from prediction plus calibrated camera height/pitch only.
Native depth and prior native decisions are evaluation inputs only; no native
mask, median depth ratio, labels, actors, plans or evaluator files enter fitting.

Fit gravity-frame camera-relative `Z=aX+bY+c`. Ground-relative height is vertical
`Z-aX-bY-c`; correction `k=known_camera_height/(-c)` multiplies depth and c.
This assumes a broad visible local ground surface and valid camera calibration.
Scale is anchored by camera height, not fitted to native obstacle outcomes.

Fixed fitter: image rows [62%,95%), columns [20%,80%), stride 8, at most 1000
deterministically spaced valid samples; finite raw depth (.08,12) m, downward
ray Z<-.12 and forward ray X>0. Seed 1707, 64 three-point RANSAC hypotheses,
vertical residual <=.08 m, negative intercept, slope <=20 degrees. Three
inlier least-squares refinements; >=30 inliers, >=50% support, >=6/9 ROI cells
with >=3 inliers each, and scale in [.5,2]. Failure makes all dependent-arm
cells UNKNOWN, never clear; no temporal hold or fallback. Geometric acceptance
does not identify ground: a broad raised surface can still fool this estimator.

Evaluation uses the unchanged native supported encoder reference (26 positive
direction/height observations out of99). Report TP/FP/FN/TN, UNKNOWN and fit
failures with fixed denominators, including unknown positives as missed alerts.
No object-instance recall or deployment safety claim follows.

Boundary diagnostic: native adjacent-pixel jump >=.05 m and absolute log ratio
>=log(1.04), adjacent to an eligible native near-field surface. Predicted edges
use only the log threshold with finite positive depths. Same orientation/sign,
Euclidean radius2 px matching; report existence, localization, contrast and
extra predicted edges in the native-conditioned neighborhood. Run once: positive
global scaling preserves these structural metrics. Ground-relative height does
not change the depth raster. Thin/shape-specific recall is NOT_EVALUABLE.

## Execution and stop rule

Validate receipt/source/prediction hashes once, preload arrays once, fit once
per frame, execute44 arm evaluations without timestamp sleeps, model loading,
inference, rendering, simulation or acquisition. Stop after all11 frames and
fixed metrics. No retuning, new cohort, or new model is part of this probe.
Retain failures and receipts. Small bounded three-variable fits use CPU
TASK_NOT_GPU_SUITABLE; dense encoding reuses the equivalent prior measured
CUDA placement, with actual tensor device recorded. Boundary diagnostics use
bounded NumPy/OpenCV CPU operations (GPU_BACKEND_UNAVAILABLE in this implementation).
Report load/hash, fit, encoding, boundary and total timings separately; cached
evaluation cost is not online perception latency. No claimed playback speedup:
the source spans only1 second, so startup can exceed its duration.

## Result

Completed one frozen run in `artifacts.local/nearfield/ground-anchor-20260907-v1`.
All35 receipt/source/sensor/prediction identities verified (including the parent
receipt); raw RGB and native alert outputs matched the saved baseline exactly.

| Arm | TP | FP | FN | Binary TN | UNKNOWN cells | UNKNOWN positives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw | 0 | 0 | 26 | 73 | 33 | 0 |
| Scale only | 1 | 0 | 25 | 73 | 3 | 0 |
| Ground-relative only | 21 | 0 | 5 | 73 | 34 | 1 |
| Scale and ground | 21 | 0 | 5 | 73 | 4 | 1 |

These are99 direction/height observations, not99 objects. Binary TN includes
reference-negative UNKNOWN observations; it does not mean observed clear space.
UNKNOWN positives remain in FN. No observations were excluded for fit failure.
All11 fits passed geometric gates, using720 samples,678–688 inliers and9/9
covered ROI cells. Estimated scales ranged0.8995–0.9596. This does not establish
ground identity or camera-height robustness outside this source.

Ground-relative height accounts for the observed recovery:21/26 (80.77%) versus
1/26 for scale alone. Adding scale to ground-relative height adds no positive
reference recovery here, although observation validity/UNKNOWN changes. Keep
both mechanisms distinguishable; do not interpret the old pixel median1.217 as
a scene-wide correction or claim this factorization proves general causation.

Boundary existence matches:284/969 (29.31%); low123/807 (15.24%), body161/162
(99.38%), head0/0 (NOT_EVALUABLE). There were1146 predicted edge locations in
the native-conditioned neighborhood,594 with no compatible native edge within
2 px. These are oriented adjacent-pixel counts, not object boundaries or alert
false positives. Per-frame localization, log-contrast ratios and unmatched
counts are retained in the receipt. Positive scaling cannot restore the absent
log-depth transitions. The low-boundary deficit remains despite alert recovery;
no thin-pole, unknown-shape or overhead-obstacle validation follows from this run.

## Efficiency, checks and disposition

One process completed the11 frames and44 arm evaluations. No UE, inference,
re-rendering or timestamp waits. Measured internal total1.962 s; tool-reported
process wall time5.58 s including interpreter/import startup. Source duration
is1.0 s: this is cache reuse, not a claimed faster-than-source-time replay.
Hash/read/decode took0.084 s. CPU fit P50/P95 was2.38/15.36 ms; per-arm encoder
P50 was4.07/3.34/3.84/3.51 ms in table order. Offline boundary evaluation P50
was103.97 ms and is outside the alert path. Timings exclude online acquisition,
model inference and feedback; no phone or end-to-end latency claim is made.

Twelve focused tests passed: prior five evidence contracts, four new ground
contracts (known scale, sloped vertical geometry, failed-fit UNKNOWN and default
parity), and three boundary tests (scale invariance, flat/invalid controls and
radius/sign matching). The executed runner also checked baseline equality for
every frame. No thresholds were changed after scoring.

Retain predicted-only ground-relative geometry as a Development component.
The next decision-changing check should target the remaining low-boundary loss
and independent small/overhead geometry coverage. This run does not trigger a
new capture or model-training campaign. Runtime App/motion policy is unchanged.
Future cached comparisons should reuse predictions and recompute only affected
stages; action-dependent closed-loop tests still require simulator execution.

Central registration retry failed before mutation on the existing
`experiments/index.jsonl:252` fingerprint mismatch. The first attempt's invalid
uppercase identifier and corrected attempt are preserved separately in
`registration.log` and `registration-retry.log`. No new structured terminal or
registered completion is claimed; local evidence is complete.

`result.json` SHA-256:
`deba17e8b67cf411ef8caebddd4b53a22e3cfc58f1783d907e5011d658c60214`.
`protocol.md` preserves the exact pre-score protocol; `started.json` records
executed code hashes. No task-owned process, model, port or lease remains.

Reproduction (fresh output required; uses existing CUDA Python environment):

```powershell
python -m unittest discover -s research/active/dtr-r0/nearfield -p 'test_*.py' -v
python research/active/dtr-r0/nearfield/run_ground_anchor.py --run artifacts.local/nearfield/representation-20260907-v1 --output artifacts.local/nearfield/ground-anchor-new-run
```
