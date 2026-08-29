# CARLA V18 X23 RGB-D metric tracklet Development result

Status: `CARLA_DTR_X23_RGBD_TRACK_SYNTHETIC_DEVELOPMENT_GATE_MET`

X23 passed every preregistered gate on the single allowed V18 evaluator open.
It replaces the X13-gated motion-refresh route with a new observable
representation: current person segmentation mask x depth-backprojected points
becomes a metric wearer-route-frame tracklet; a causal 0.50 s Theil-Sen fit
estimates motion, a bounded 0.60 s hold bridges physical occlusion, and two
consecutive same-track route-tube entries authorize risk. No tracker, tube,
closing-speed, seed, or hold-window sweep was run.

## One-score result

| Metric | Same-source X21 | X23 | Change / gate |
|---|---:|---:|---:|
| Physical-occlusion gap coverage | 0/11 (0%) | **11/11 (100%)** | **+100 percentage points**, gate >= +50 pp |
| Occluded first-alert lead | no alert | **2.95 s** | gate >= 2.50 s |
| Clear-crossing first-alert lead | no alert | **2.90 s** | gate >= 2.50 s |
| New static risk frames | - | **0** | gate = 0 |
| New parallel-motion risk frames | - | **0** | gate = 0 |
| New pre-threat risk frames | - | **0** | gate = 0 |

The physical-loss denominator is samples 75-85. X23 covers all eleven. Its
truth-blind risk interval is samples 60-128 in the physically occluded crossing
(69 frames) and 61-128 in the clear crossing (68 frames). Static and
moving-parallel negative controls contain no risk frames. During samples 70-90
of the occluded crossing, the primary target is measured at 70-73 and 86-90 and
held at 74-85, with no missing track state; the hold reaches but never exceeds
its fixed 0.60 s cap.

## Frozen evidence

The formal run is
`artifacts.local/evidence/dtr-carla-v18-x23/v18-flow-onset-20260830-003320/x23-rgbd-one-score-v1`.

- input index SHA-256: `4C4DA862E4A6A13C146DD94E7209DE568D78E7DD3F10DD11870EDA7350FC39A9`
- 804-file detector-candidate aggregate SHA-256: `980F63FB0714F69BB8AE6641254088D38EC058FD516C9A1385224B57F636AD02`
- same-source sealed X21 baseline SHA-256: `8C585E03D0E483A701B363EE789D17FC1559ECEA6001B25624DD91FD9674A421`
- freeze SHA-256: `55FECCADD9FC82F0CBE424FACC8FB178069F22B5FCA8F9C876D13914D02113EF`
- truth-blind predictions SHA-256: `8E9BF6601DFAD5502C23030622108083D65A20B54E32A15E9F5435B6987EFCBB`
- exclusive score-attempt SHA-256: `5056802C07CA8535CDC7428B28F3E7BF882CB11DA81ADF07B36B1E652880BCC7`
- evaluator manifest SHA-256: `4AACFBE648A999C3E22FBDC349D92E63747B14B0D4D9D0FDD518696E1E94DC36`
- result SHA-256: `E303A786F687857E431765BE88FAAEEA95178E22E687D3A2DA1563CF7FC0B4B0`

`freeze` and `predict` opened only the V18 model root, frozen NPZ frames,
truth-blind YOLO candidates, and the sealed X21 baseline. `score` wrote its
exclusive receipt before the first evaluator-manifest or evaluator-ledger
open. The successful result consumes the only V18 score; no post-score tuning
is authorized.

## Innovation and claim boundary

The defensible project innovation is the information-path change, not a claim
that short-window tracking or occlusion hold is new by itself: an
affordance-aligned metric mask-depth tracklet is born without X13 motion-gated
refresh, transformed into the wearer's route frame, and accepted as risk only
after bounded causal same-track route consensus. This combination directly
turns a zero-refresh/zero-risk source into complete physical-gap coverage while
preserving all three false-risk controls.

Prior work already covers wearable RGB-D dynamic-obstacle tracking and scene
flow ([Ou et al., 2022](https://arxiv.org/abs/2204.01154)), optical-flow motion
decomposition ([Residual Flow, 2019](https://arxiv.org/abs/1909.06999)), and
occlusion-aware suppression of unstable motion updates
([OccluTrack](https://arxiv.org/abs/2309.10360)). Therefore this result does not
claim "the first", generic occlusion tracking novelty, or generic optical-flow
novelty.

This is strong **synthetic repeated-geometry Development evidence**. It is not
source-disjoint confirmation, real-world generalization, production readiness,
or a safety guarantee. CARLA depth and the fixed straight route establish
information reachability and causal effect; deployment still requires an
authorized real RGB-D source and a fresh, source-disjoint confirmation cohort.
