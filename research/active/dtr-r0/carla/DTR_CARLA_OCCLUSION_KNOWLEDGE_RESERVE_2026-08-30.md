# DTR CARLA occlusion knowledge reserve — 2026-08-30

## Decision

The next algorithmic representation after an evaluable CARLA source should be
`X32_CELLWISE_FLOW_MEMORY`: transport already-authorized RGB-D surface cells
with local scene flow, maintain a short recurrent flow history, and delete
transported cells contradicted by current free-space rays.  It is a structural
successor to X31's component-level branches, not a threshold or detector sweep.

Never-observed hazards must remain separate.  Transport cannot create evidence
for an actor that has never been seen.  Such risk should enter either as an
explicit probabilistic occlusion anchor or as a bounded `UNSEEN_REACHABLE_SET`,
never as an observed X31 lineage.

## Primary-source reserve

| Source | Source fact | DTR adoption inference |
| --- | --- | --- |
| [TemPCC, Eurographics 2025](https://cgvr.cs.uni-bremen.de/papers/eg2025/tempcc/TemPCC_EG2025s_Paper.pdf) and [code](https://github.com/muehlenb/TemPCC) | Temporal point set; each occluded point uses nearby visible flow plus a 30-step GRU history, visibility contradiction cleanup, and density control. The paper reports 30k points in under 30 ms on RTX 4090 and 2.9/15 cm to 7.7/34.6 cm error/travel after 30 frames on two synthetic validation scenes. The repository calls itself highly experimental. | First implementation candidate: replace whole-component translation with per-cell local-flow transport, while retaining X31 birth authority and free-space revocation. Do not inherit the full CUDA stack or treat synthetic multi-camera evidence as BlindAssist performance. |
| [Scene Informer v2](https://arxiv.org/html/2309.13893v2) | Anchor queries inside occluded regions jointly predict occupancy and seven trajectory modes. Under its simulated limited-observability WOMD setup it reports occupied/free accuracy `80.5/72.8%` and occluded-agent minADE/minFDE `1.00/1.63 m`; the model has 11.3M parameters. | Use a very small anchor set only where an RGB-D visibility shadow intersects the immutable route tube. Store results as `INFERRED_UNSEEN`; never merge them into observed track ancestry. |
| [APRO, 2026](https://arxiv.org/html/2606.15046) | Represents hidden sets and danger zones as AH-polyhedra and checks separation with linear programs. It reports 100% modeled safety in its evaluated cases and `0.040±0.013 s` per step on F1/10 hardware, under perfect affine dynamics, perception, and hidden-set assumptions. | Lightweight planner-side alternative for never-seen hazards: extrude RGB-D visibility shadows by class motion bounds and intersect them with the wearer braking zone. Its formal guarantee does not transfer through perception/model error. |
| [FlowScene, NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/file/08c6112ac1712333f2c319f234b45b12-Paper-Conference.pdf) and [code](https://github.com/willemeng/FlowScene) | Uses forward/backward flow inconsistency as an occlusion mask, warps historical features, and refines them with occlusion guidance. | Reuse only the causal flow-consistency occlusion certificate; the full semantic-scene-completion network is too large for the first X32 experiment. |
| [MotionPerceiver, 2023](https://arxiv.org/html/2306.08879v1) | Maintains a fixed recursive latent state and sparsely queries occupancy at requested space-time points; reported route-scale query latency is compatible with embedded experimentation. | Reserve for a bounded route-field successor if cellwise branch count becomes the bottleneck. It is not evidence for unseen-agent birth. |

## CARLA scene reserve

| Source | Source fact | DTR use |
| --- | --- | --- |
| [ScenarioRunner v0.9.16](https://github.com/carla-simulator/scenario_runner/releases/tag/v0.9.16) and [scenario list](https://scenario-runner.readthedocs.io/en/latest/CHANGELOG/) | Official version matched to CARLA 0.9.16. Available scenario families include parking-obscured pedestrian crossing, construction obstacle, emergency-vehicle yielding, vehicle-turning pedestrian, and pedestrian crossing. | Lowest-cost next gauntlet: left/right emergence behind car, van, and long truck; add one construction pinch and one emergency pass. Freeze cases before algorithm access. |
| [DriveOcclusionSim / DOS](https://github.com/opendilab/DOS) | Four families × 25 cases: pedestrian behind parked vehicle, hidden cause of sudden braking, truck-blocked left turn, and truck-blocked red-light violator. The repository requires CARLA 0.9.10.1 and bundled Leaderboard/ScenarioRunner replacements. | Reuse the four interaction structures, not its old runtime. Port explicit cases into the current 0.9.16 capture stack; do not replace current runner directories. |
| [SafeBench](https://github.com/trust-ai/SafeBench) | Provides NHTSA-style scenarios and Scenic-based multi-agent generation on an older CARLA stack. | Reuse mutation ideas only after a fixed base source is evaluable. Any adversarially selected cases are Development, not independent confirmation. |

## Ordered research route

1. Make the CARLA source raster-evaluable without weakening the zero-pixel
   authority.  Use a native opaque compound representation and reject it with
   an instance-only prelaunch probe if any episode misses the unchanged gate.
2. Run the frozen X24/X31 comparison only after source admission.
3. Implement one `X32_CELLWISE_FLOW_MEMORY` successor on a fresh Development
   cohort.  Preserve detector, route, scorer, and lifecycle thresholds.
4. Add `UNSEEN_REACHABLE_SET` as a separate risk provenance only after the
   observed-before-occlusion transport mechanism is measured.
5. Expand to the ScenarioRunner/DOS interaction gauntlet; do not let selected
   adversarial cases stand in for source-disjoint confirmation.

## Evidence boundary

The sources above were located and their selected pages were fetched through
Exa.  Their reported metrics belong to their own datasets, sensors, hardware,
and assumptions.  This file is a design reserve, not reproduction evidence and
not a BlindAssist effectiveness, generalization, product, deployment, or
safety result.
