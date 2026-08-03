# Depth observer clearance A0 consumed comparison

Date: 2026-08-03

Terminal: `METRIC3D_FP16_BALANCE_DAV2_DUAL_FREQUENCY_DIAGNOSTIC_FAIL`

## Decision

The objective is not for a light observer to exceed the heaviest model's
absolute accuracy. The selection is a quality/cost Pareto problem. Metric3D
FP16 is the current single-model balance point because it preserves all five
standalone task gates while reducing latency and CUDA allocation. Depth
Anything V2 Small Metric FP16 is the efficiency endpoint and remains a
dual-frequency candidate, but is not admitted as an unanchored standalone
metric-clearance source. MoGe-2 ViT-S Normal is not a useful balance point on
this screen.

No fresh data, threshold search, scale fitting, model search, or per-model
clearance change was used. All arms used the same four already consumed TUM
windows (000, 016, 020, and 024), 30 frames each, the same registered sensor
depth comparator, published intrinsics, depth-RANSAC ground recovery, bands,
horizons, and five continuation gates.

## Result

| Observer | Valid | Clearance MAE | Envelope agreement | False-clear | Temporal delta MAE | Steady CUDA median | CUDA peak | Role |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Metric3D ViT-S FP16 | 119/120 | 0.11832 m | 93.45% | 4.03% | 0.09204 m | 142.33 ms | 573 MiB | standalone balance, PASS 5/5 |
| MoGe-2 ViT-S Normal FP16 | 118/120 | 0.38722 m | 81.70% | 0.19% | 0.15927 m | 129.54 ms | 1,145 MiB | not preferred, FAIL 2/5 |
| DA V2 Small Metric Hypersim FP16 | 119/120 | 0.38014 m | 75.24% | 24.29% | 0.11261 m | 54.27 ms | 328 MiB | dual-frequency candidate, standalone FAIL 2/5 |

Metric3D FP32 exactly reproduced the earlier A0 task metrics. Its FP16 arm kept
all five gates while reducing steady runtime and activation allocation. MoGe-2 is conservative
on this screen: false-clear is low, but clearance and lateral envelope geometry
are not accurate enough, with right-band agreement only 70.09%. DA V2 is the
fastest and lightest arm: relative to Metric3D FP16 its steady median implies
2.62x throughput and its CUDA peak is 42.8% lower. Its 24.29% false-clear rate
still rules out unanchored standalone metric clearance. Its useful result is
the efficiency end of the Pareto frontier, not current metric authority.

## Frozen identities

- Metric3D source commit `eb5b6fac0dc155e4e52f576e304fbf11655ff339`;
  checkpoint SHA-256
  `B34B2A2BE9148054991CEF7E417930E1320602BA7BC503B0EE4E7888543728F6`.
- MoGe source commit `925b8ed835a7a9cdb7578ba15c658a0afc969030`;
  `utils3d` commit `3fab839f0be9931dac7c8488eb0e1600c236e183`;
  `Ruicheng/moge-2-vits-normal` checkpoint SHA-256
  `79A16621928C2BF0ED04659218C55C01075E950507F40BB3332FB4C873D3E1DC`.
- Depth Anything V2 source commit
  `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`; official indoor metric ViT-S
  checkpoint SHA-256
  `B782898D8A3E8BE1F639DE33837ED85E9B4B73E40F8F5E5CD99067588D722545`.

Ignored machine reports are under
`artifacts.local/evidence/hftf/depth-observer-clearance-a0/`.

## Claim boundary and next action

This consumed comparison ranks observers only. It does not validate the final
external camera, outdoor transfer, thin/hanging/transparent obstacle coverage,
Android/NPU execution, alert behavior, user benefit, or safety.

Do not tune the failed arms on these consumed outcomes. The next candidate is a
single frozen dual-frequency diagnostic: DA V2 supplies high-rate relative
structure while a lower-rate metric source supplies scale anchors. This is a
new system hypothesis, not a reinterpretation of DA as accurate metric depth.
Common external-camera calibration, disagreement detection, and `UNKNOWN` must
be independently validated before any clearance decision or deployment claim.

## Fixed dual-frequency diagnostic

A single causal consumed replay then ran DA V2 FP16 every frame and Metric3D
FP16 every fifth frame. At each anchor, fixed per-band metre offsets were
updated; missing band anchors remained `UNKNOWN`. The period, gates, bands, and
thresholds were not searched.

| Measure | Fixed period-5 result |
|---|---:|
| Paired valid | 118/120 (98.33%) |
| Clearance MAE | 0.16506 m |
| Envelope agreement | 88.62% |
| False-clear | 6.97% |
| Temporal delta MAE | 0.12216 m |
| Steady sequential mean | 79.41 ms |
| Sequential P95 | 198.59 ms |

The replay recovered much of Metric3D's geometry at an estimated 1.80x higher
average throughput than Metric3D FP16, but it failed the envelope-agreement and
false-clear gates. Synchronous anchors also created periodic latency spikes,
and co-resident memory was not measured. Terminal:
`DUAL_FREQUENCY_CONSUMED_DIAGNOSTIC_TASK_GATES_FAIL`.

This supports an asynchronous-anchor implementation or a small real metric
sensor as the next independent hypothesis. It does not authorize anchor-period
search on the consumed cohort or promotion of the current hybrid.
