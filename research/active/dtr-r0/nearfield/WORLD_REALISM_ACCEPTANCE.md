# G14 realistic procedural worlds: acceptance and execution

User steering, 2026-09-08. This document specifies intended acceptance; it is
not a claim that the current generator passes it. G10 and G13 model status is
unchanged. G14-A prioritizes scene engineering before bulk data and training.

## Machine ownership

The primary machine owns source, interactive scene creation, material/lighting
iteration, small native previews, and visual/label acceptance. Transfer an
accepted, hashed source/asset/configuration snapshot to the secondary machine.
Prefer that worker for bulk capture and long training when memory and measured
throughput permit. Avoid competing UE rendering and training on the same GPU.
Record engine/plugin versions, asset dependencies and hashes; do not maintain
two independently edited scene sources. Failed previews remain evidence.

## Initial inspected preview

Primary-machine `artifacts.local/nearfield/worlds-20260908/local-preview4-capture`
contains two native 640x360 previews. Capture and CUDA geometry verification pass;
the capture script reports 45.906 seconds (not total launch-to-result latency).
Seven focused specification/verifier tests pass. The SceneCapture Python show
flag setter was corrected to the reflected editor property; the previous failed
attempt remains retained. This engineering fix does not certify image quality.

Visual acceptance is **PENDING / NOT PASSED**: the outdoor view is still sparse,
with an unfinished distant ground transition and insufficient paving/planting
variation; the corridor is dark with conspicuous grain. Controlled rectangular
targets remain visibly artificial. Continue primary-machine look development;
do not start bulk generation from this preview. No B/D inference or training was
run for this acceptance update. The preview UE process was released after export.

The subsequent [asset and scene iteration](WORLD_REALISM_20260908.md) delivers
repaired target-free previews at `local-realism7-capture`, including four newly
downloaded CC0 assets and native 1280x720 appearance exports. Capture/truth checks
pass, but the wider 30-image realism acceptance and sample-PCG integration
remain pending. Initial failures above remain historical engineering evidence.

## First deliverable

Finish a campus-style sidewalk and an indoor corridor before adding more world
families. Inspect native RGB without overlays, including ordinary diffuse light,
near-ground details and the actual 144x256 model input. No model score can waive
an obvious visual defect. After representative previews pass, inspect a seeded
30-image selection across layouts before bulk generation; record selected IDs
and observed defects. Human visual acceptance is subjective, not a realism score.

Acceptance covers:

- Continuous environment: no unintended void, exposed stage edge, floating
  architecture, hovering furniture, or tree roots outside their soil support.
- Dominant surfaces: credible material scale, normal/roughness response and
  restrained variation; avoid obvious uniform tiling and plastic-like surfaces.
- Scene layout: furniture near edges/walls, vegetation in planting zones,
  plausible proportions, clear circulation and intentional temporary occupancy.
- Contact and detail: coherent curb/ground/wall connections and spatially
  motivated dirt, leaves, wear and small clutter, rather than uniform scattering.
- Vegetation: varied scale/orientation and layered planting where appropriate.
- Lighting/camera: readable ordinary lighting, controlled exposure, wearable
  viewpoint, and no dependence on cinematic depth of field, flare or grading.
- Truth: native rendered geometry and BODY/HEAD support agree; newly introduced
  visible clutter participates in the query. Hidden space is not certified free.

Use existing scanned surfaces and native assets first. Build extra material
layers, decals or blending systems only where actual previews show a need.
Asset path/provenance and redistribution permission must remain distinct.

## Render and camera contracts

`RESEARCH_REALISTIC` is the intended efficient research appearance;
`DEMO_ULTRA` is an intended higher-quality presentation profile. These names
are targets, not evidence of implemented settings. Keep scene layout, task
geometry and camera pose identical in paired profile captures. Log effective
resolution, material/LOD settings, lighting, exposure and capture settling.
Measure startup, scene build, settling, export, total useful pairs/s and memory;
editor FPS alone is not acquisition throughput.

Higher-quality vegetation/LOD, displacement, occlusion or camera distortion can
change visible support even at the same actor transforms. Regenerate native
truth for each affected capture; do not copy labels merely because a seed is
unchanged. Distortion must transform RGB and labels consistently. Resolution
changes require the corresponding camera intrinsics and support projection.

The current fixed 1.70 m eye-height, HFOV100 camera is an engineering profile,
not measured glasses calibration. Noise, exposure dynamics, walking motion,
compression and distortion require explicit parameters/provenance. Add measured
device characteristics when real recordings are available; synthetic parameter
randomization must not be described as calibrated sensor realism.

## Scientific boundaries and next decisions

The existing 48-frame height-intervention pilot remains a disclosed Development
diagnostic with controlled cube targets. It does not satisfy the future natural
hazard demonstration (attached branches/signs and plausible BODY obstacles),
independent-map generalization, or a photorealism claim. Preserve these controls
while building separately identified natural-hazard scenes with physical support.

G14-B data expansion/training follows scene and truth acceptance. Ten thousand
images is a candidate scale, not a current launch instruction. Split by source
world/layout and relevant asset families, not random neighbouring screenshots.
Low/medium/high or mixed appearance experiments need geometry-matched controls,
equal training budgets and held-out visual conditions. A synthetic mixed-domain
gain supports that tested setting; real-world transfer still requires real data.
