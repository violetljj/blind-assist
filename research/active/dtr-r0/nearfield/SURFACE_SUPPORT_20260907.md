# NF-G2: miss attribution and fixed support comparison

EXPLORE, consumed synthetic Development. Protocol fixed before new scoring.
Keep NF-G1 ground-only as baseline, all eleven cached predictions, all99 cells,
no scale changes, distance/height/validity thresholds or fitter changes. Reuse
the receipt-bound plane per frame; no native data enters fitting or support.

First attribute the five FN in both NF-G1 ground branches. Report candidates in
each missed direction/height cell before/after 3 m cutoff and spatial support.
For native supported near pixels in each missed cell, report disjoint predicted
losses: invalid, over-range, height below/above range, changed height band,
same-band unsupported, same-band supported. Also report overlapping far/height
failures to avoid misreading first-failure order as unique physical causation.
Pixel correspondence is a synthetic reference diagnostic, not object identity.

Fixed challenger: three adjacent eligible pixels with elevation range <=.08 m.
Use the existing support rule's .08 m base tolerance as a single diagnostic
choice, without sweep or tuning. Compare depth, elevation, and their OR. OR
cannot remove depth-supported evidence, but can add coherent false surfaces.
Only one of the five misses has a near weak candidate, so support-only changes
can recover at most one cell while eligibility is fixed. Do not optimize to26/26.

Evaluate all eleven frames, not only misses. Retain all88 previous analytic
fixtures including thin/overhead shapes, invalid depth, isolated speckles and
2x2/3x3 correlated artifacts. Those fixtures use their declared calibrated
ground plane, not a fitted/validated RGB ground estimator. They are regression
controls, not new natural/RGB validation. No new acquisition or inference here.

Retain a challenger only for an observed useful gain with disclosed false-alert
cost; otherwise keep baseline and stop support work. Do not alter thresholds
to rescue failures. Next source selection follows attribution; no unbounded
model search, training, RGB-edge rescue or simulator campaign is authorized by
this protocol. Offline reports preserve UNKNOWN and fixed denominators.

CPU handles hashes/JSON only (TASK_NOT_GPU_SUITABLE); Torch dense masks and
encoding use CUDA, reusing prior equivalent placement measurements through
research_backend device observation. Arrays read/hashed once; no timestamp
waits. Record total and arm processing times separately from online latency.

## Results

Completed in `artifacts.local/nearfield/surface-support-20260907-v1`.

| Support | Willow TP/FP/FN | UNKNOWN | Analytic TP/FP/FN |
| --- | --- | ---: | --- |
| Depth baseline | 21/0/5 | 34 | 240/1/0 |
| Elevation | 21/0/5 | 34 | 240/1/0 |
| Depth OR elevation | 21/0/5 | 34 | 240/1/0 |

All99 Willow direction/height observations and all88 analytic frames retained.
The single analytic false alert remains the known3x3 correlated depth artifact.
Default depth output/state matched NF-G1 on every frame; analytic baseline
matched prior240 TP/1 FP. No observed gain justifies replacing depth support.
Challengers remain explicit opt-in research modes, not the App/default policy.

## Five-miss attribution

| Frame / left height | Native supported near pixels | Predicted forward P50 at those pixels | Diagnosis |
| --- | ---: | ---: | --- |
| 0 / body | 838 | 4.154 m | All beyond3 m;317 also change height band |
| 1 / body | 398 | 4.076 m | All beyond3 m;264 also change height band |
| 2 / body | 69 | 3.880 m | All beyond3 m;67 also change height band |
| 0 / head | 3 | 12.063 m | All far and shifted above the height range |
| 5 / low | 5298 | 3.097 m | 3002 first lost to range;2296 remaining below height floor |

Head evidence consists of only three native supported pixels, not a labeled
overhead-object instance. Do not infer broad head-height capability from it.
With NF-G1 scale correction the first four misses still have zero near
candidates; corresponding forward medians are3.737,3.772,3.566,10.851 m.
These are corresponding raster coordinates, not verified surface identities.
Far and height failures can coexist; the first-failure partition is not a
claim that range is the only underlying physical error.

For frame5 low,4742/5298 reference pixels also have the wrong height band;
predicted relative-height P50 is0.00469 m, consistent with loss of the local
elevation evidence under this estimator. This does not isolate model smoothing
from plane error or other local deformation. None of the native reference
pixels survives the fixed ground-only eligibility/range checks.

There are four near weak candidates elsewhere in that left-low cell. A separate
post-score diagnostic (`weak-diagnosis.json`) removed the consistency constraint
entirely while retaining three adjacent eligible pixels: zero near candidates
acquired support. Coordinates (v,u) are(326,121),(357,88),(358,88),(359,87),
relative heights0.0650–0.0738 m. They do not form an eligible triple. Therefore
the fixed ground-only branch cannot improve here by changing consistency alone.
This conclusion does not cover changing eligibility, neighborhood or scale.

The scaled branch has eight weak near candidates and seven same-band
unsupported reference pixels; it was attributed but is not a second tuned
support-comparison baseline. No challenger result is claimed for that branch.

## Execution, validation and next decision

The single comparison processed11 cached frames and88 analytic controls in
2.176 s internally (tool-reported process wall5.47 s). Input hash/read/decode
took0.096 s. No model loading/inference, simulator, timestamp sleeps or new
acquisition. CUDA executed dense geometry/support/encoding; CPU handled I/O and
metadata. Arm P50 was3.59/3.63/3.51 ms. The depth arm includes initial CUDA/kernel
startup and order was fixed: its high first-call/P95 cost is not evidence that
challengers are faster. Total includes controls and attribution, excludes
interpreter/import startup and final receipt writing; not online latency.

Eighteen focused tests passed, including six new tests for default parity,
horizontal-surface recovery, dual-support preservation, coherent-artifact
negative provenance, eligibility and invalid mode/plane rejection. The analytic
horizontal-surface unit fixture demonstrates the predicate can differ; it is
not an empirical improvement on the consumed Willow source.

Keep ground-relative geometry and default depth support. Stop this support
experiment with NO_INCREMENTAL_GAIN; do not sweep thresholds or chase26/26.
The next useful source should separate obstacle near/far distance, background
change, camera pitch/ground availability, low/thin/overhead geometry and ordinary
ground/texture/shadow. Acquire/infer once per distinct view, cache predictions,
and compare unchanged methods. Static views do not validate first-alert delay;
reserve short sequences for that question. This turn did not acquire those new
views, train a model, change the App, or validate generalization.

Central registration failed before mutation on the existing
`experiments/index.jsonl:252` fingerprint mismatch; `registration.log` retains
the failure. Local report and receipts are complete; no structured terminal
or registered completion is claimed. All task-owned short processes exited;
no simulator, worker, port or lease remains.

Receipts: `result.json` SHA-256
`e99e368c8b984adbf6c8e35ba926f50d924056fd20fb1b3eff6d4c8623264e00`;
`weak-diagnosis.json` SHA-256
`1d616672851606d84f5e3e9951b82d89ca9a7efdf77c62799f0bc2f11ed9a4ef`.
`protocol.md` preserves the pre-score protocol and `started.json` code hashes.

Reproduce with existing CUDA Python and fresh output:

```powershell
python research/active/dtr-r0/nearfield/run_surface_support.py --baseline artifacts.local/nearfield/ground-anchor-20260907-v1 --original artifacts.local/nearfield/representation-20260907-v1 --output artifacts.local/nearfield/surface-support-new-run
python research/active/dtr-r0/nearfield/diagnose_weak_support.py --baseline artifacts.local/nearfield/ground-anchor-20260907-v1 --output artifacts.local/nearfield/surface-support-new-run/weak-diagnosis.json
python -m unittest discover -s research/active/dtr-r0/nearfield -p 'test_*.py' -v
```
