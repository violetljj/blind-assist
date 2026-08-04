# Latest RGB model curated showcase result

Date: 2026-08-04
Status: `RECORDED_MODEL_REPLAY_SHOWCASE_RENDERED_DEVELOPMENT_ONLY`

## Outcome

Five 30-second public-video segments were selected for presentation clarity and
rendered with the current RGB-only showcase chain:

`RGB -> Depth Anything V2 Metric Hypersim ViT-S -> frozen Scale-Free Traversability R0`.

All 1,500 output frames were produced by per-frame GPU inference. The generator
did not read source depth. Each MP4 is 1280x720 at 10 FPS and includes the RGB
input, model-depth preview, measured latency, frozen three-band decision, and a
dimensionless relative top-down view whose left/right orientation matches the
image.

| Clip | Fixed source interval | Presentation purpose | Median / P95 inference |
|---|---:|---|---:|
| roadwork narrowing | 652-682 s | fenced walkway and changing side clearance | 76.69 / 79.73 ms |
| dynamic pedestrians | 45-75 s | pedestrians and bicycles crossing the forward view | 76.96 / 81.62 ms |
| narrow alley | 22-52 s | constrained walls and oncoming pedestrians | 75.30 / 80.03 ms |
| waterfront walk | 88-118 s | railing, narrow path, and pedestrians | 77.03 / 81.90 ms |
| plaza cones | 28-58 s | open plaza, cones, and lateral pedestrians | 76.89 / 82.92 ms |

The canonical local output index is:

`artifacts.local/evidence/hftf/latest-rgb-model-curated-showcase-20260804/showcase_index.json`.

The machine-readable companion binds exact source and rendered-video SHA-256
values, interval parameters, decision counts, and performance values. Generated
MP4/JSONL/PNG payloads remain under ignored `artifacts.local/`; Git retains the
reproduction contract and result receipt rather than large binary videos.

## Authority boundary

This batch was selected for visual demonstration, not formal model evaluation.
It does not establish metric distance, safe direction, navigability, algorithm
quality, outdoor-domain validity, device performance, or production readiness.
The relative top-down panel is a rank visualization of the frozen scale-free
three-band scores, not a metric BEV. All outputs remain `DEVELOPMENT_ONLY`.
