# Depth observer clearance A0 consumed comparison

Date: 2026-08-03

Terminal: `METRIC3D_ONLY_TASK_GATE_PASS_MOGE2_AND_DAV2_NOT_ADMITTED`

## Decision

Metric3D remains the only admitted dense metric-depth observer for the current
candidate clearance-field side lane. It is a baseline/teacher, not a required
final deployment model. MoGe-2 ViT-S Normal is not admitted as its replacement.
Depth Anything V2 Small Metric is not admitted as a standalone metric-clearance
source; its latency supports only a future fast relative-depth arm that remains
anchored and gated by a separately validated metric source.

No fresh data, threshold search, scale fitting, model search, or per-model
clearance change was used. All arms used the same four already consumed TUM
windows (000, 016, 020, and 024), 30 frames each, the same registered sensor
depth comparator, published intrinsics, depth-RANSAC ground recovery, bands,
horizons, and five continuation gates.

## Result

| Observer | Valid | Clearance MAE | Envelope agreement | False-clear | Temporal delta MAE | CUDA mean | CUDA peak | Terminal |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Metric3D ViT-S | 119/120 | 0.11579 m | 93.35% | 4.03% | 0.09031 m | 170.40 ms | 709 MiB | PASS (5/5) |
| MoGe-2 ViT-S Normal | 118/120 | 0.38722 m | 81.70% | 0.19% | 0.15927 m | 139.96 ms | 1,145 MiB | FAIL (2/5) |
| DA V2 Small Metric Hypersim | 119/120 | 0.38296 m | 75.21% | 24.32% | 0.12418 m | 59.30 ms | 351 MiB | FAIL (2/5) |

Metric3D exactly reproduced the earlier A0 task metrics. MoGe-2 is conservative
on this screen: false-clear is low, but clearance and lateral envelope geometry
are not accurate enough, with right-band agreement only 70.09%. DA V2 is the
fastest and lightest arm, but its 24.32% false-clear rate rules out standalone
metric clearance use. Its useful signal is computational, not current metric
geometry authority.

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

Do not rescue the failed arms on these consumed outcomes. The next authorized
route is common external-camera calibration followed by controlled final-lens
capture. If a dual-frequency design is opened later, DA V2 may supply high-rate
relative structure only; low-rate metric anchoring, disagreement detection, and
`UNKNOWN` must be independently validated before any clearance decision.
