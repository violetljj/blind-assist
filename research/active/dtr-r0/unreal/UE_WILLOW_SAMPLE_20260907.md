# Willow sample segment: native scene and measurement interface

The user's requested deliverable is a visually credible, measurable 24 m street
sample. `/Game/StreetLab/WillowSampleV1` is a separate map in the local
`artifacts.local/unreal/BlindAssistStreetLab` project. It preserves the existing
StreetLabV4 map and its consumed algorithm results. This is scene engineering,
not a new algorithm comparison or evidence of real-world transfer.

## Scene and assets

The sample occupies x=22..46 m, y=-6..6 m, with its central floor at z=.12 m.
The 4.8 m central walkway has a low-relief tactile strip and drainage edges.
Raised side bands contain tree islands, seating, planting and street lighting.
Existing native building assets provide the surrounding architecture.

Four Poly Haven assets are downloaded from official endpoints with provider MD5,
file size and SHA-256 checks (34 files, 415,026,478 bytes):

| Asset | Use | Resolution |
| --- | --- | --- |
| [Concrete Pavement](https://polyhaven.com/a/concrete_pavement) | Scanned paving, 1.8 m texture repeat | 4K |
| [Concrete Wall 007](https://polyhaven.com/a/concrete_wall_007) | Concrete curbs and tree island edges | 4K |
| [Modular Street Seating](https://polyhaven.com/a/modular_street_seating) | Wood, metal and assembled seating | 4K |
| [Tree Small 02](https://polyhaven.com/a/tree_small_02) | Bark, branching and masked foliage | 2K |

The assets are [CC0](https://polyhaven.com/license). Local provenance lives in
`artifacts.local/unreal/sample-materials-v2/manifest.json`. Prepared seating
removes four spare angular connectors, an extension seat and two suspended
supports from the exploded source FBX. Nine retained component meshes are
assembled by model translation: grounded legs, crossbar, seat, back supports,
backrest and armrests. Their UVs, normals and material geometry remain intact.
`tools/prepare_sample_seating.py` preserves and checks the raw FBX property
payloads and emits a preparation receipt. The tree contains one LOD0 model.

Material imports use sRGB for color and linear samples for data maps, with
normal-map compression and green-channel inversion for OpenGL model normals.
World-aligned surface textures use typed texture objects. Foliage is masked and
two-sided, with reduced subsurface transmission. These are shading normals,
not geometric displacement. New assets stay under `/Game/SampleMaterialsV2`.
The tree's branch material uses UV channel 1; leaves and trunk use channel 0,
as verified against the actual FBX polygon UV assignments. Fixed exposure is
EV100 10.5. The seating is offset from the planting boxes to leave a clear gap.

## Measurement scope

The exporter enumerates every loaded static mesh component and every individual
instanced mesh using native transformed bounds. The evaluator receives the full
roster and explicit floor, overhead and region exclusions. Model inputs are
separate: RGB, forward depth, calibration, ego pose and issued navigation only.
The verification tool builds causal plan receipts and loads the resulting
manifest with the existing UE replay contract.

There are four native 1920x1080 views and eleven synchronized 640x360 RGB/depth
pairs over one second at 10 Hz. Native capsule sweeps cover center, left bypass,
right bypass and a seeded contact control. Sweeps test the configured physics
collision; native AABBs remain conservative proxies, not triangle truth. Whole
tree bounds can report proxy contact in a physically clear side corridor; those
discrepancies are retained rather than hidden. Dynamic actors and non-static
geometry are outside this static sample's coverage. No new controller success
rate or dynamic-scene acceptance is claimed.

## Reproduction

Use the project Python runtime and installed UE engine. All output paths must
be fresh directories under the canonical `artifacts.local` junction.

```powershell
python tools/download_sample_assets.py
python tools/prepare_sample_seating.py --assembled --output artifacts.local/unreal/sample-materials-v2/modular_street_seating/modular_street_seating_assembled_4k.fbx
# Only on a project without WillowSampleV1:
python tools/run_sample_segment.py build --output artifacts.local/unreal/willow-new-build
python tools/run_sample_segment.py materials --output artifacts.local/unreal/willow-new-materials
python tools/verify_sample_segment.py --run artifacts.local/unreal/willow-new-materials
```

`materials` rebuilds only the sample materials and replaces task-owned furniture
without duplication. `inspect` captures the existing saved map. Each runner owns
and releases its UE process tree, with a `process-release.json` receipt.
Maps, textures, FBX, renders and evidence are local artifacts; Git contains the
reproduction and verification code, not the downloaded binary payload.

## Iteration record

Initial sample views exposed oversized seating and coplanar paving; both were
corrected. The first scanned-material import exposed wrong texture-object
samplers, causing UE default fallback. The next iteration corrected the samplers.
Visual inspection then exposed spare FBX connector parts and excessive brightness.
Those findings led to the prepared seating and exposure/foliage changes above.
A closer view then exposed the remaining exploded armrests and support bar;
the final preparation assembles the single bench instead of combining kit parts.
Failed and intermediate outputs remain labeled by their runner receipts; an
exporter's PASS means capture completed, not that visual acceptance passed.

## Initial verified material version

Material build and final views: `artifacts.local/unreal/willow-sample-v3-assembled`.
Its `verification.json` passes: zero sample material compile failures, eleven
RGB-D pairs accepted by the sanitized replay contract, three native clear
route witnesses and one positive contact control. The exported roster has
662 static mesh components and 1,886 instances, yielding 2,436 bounds rows.
Both side-corridor AABB witnesses retain `PROXY_CONTACT` despite clear native
sweeps; this remains an explicit evaluator limitation. Five focused swept
geometry tests pass. All task-owned editor process trees were released.

Saved map SHA-256:
`03aca3ad148f8f721e8e69e9b6b9a30f6ca606651b3146aacf4504c86b30b2d2`.
Original StreetLabV4 SHA-256 remains
`3bb1ea3b8ed16d300e7fa3178542313f48548b5efdc276fc7ca188188c226e28`.

Visual review confirms resolved default paving fallback, assembled seating
without stacked spare parts, a gap from the planting boxes, concrete surface
detail and more controlled exposure. The surrounding older scene still has
repeated storefront interiors and unfinished distant boundaries; tree foliage
also remains light in direct sun. The new material pass is a concrete improvement,
not a claim that the entire laboratory is photorealistic or visually approved
by the user.

## Visual freeze and first-person acquisition

On 2026-09-07 the user paused visual optimization and directed acquisition from
approximately 1.7 m above the walking surface. The frozen visual version is
`artifacts.local/unreal/willow-finish-4k`, map SHA-256
`cf35e5c9df54cd0f781f09ea8105fe8ef6078ed0822d4e594d64216e79a254fb`.
It has four native 3840x2160 views. Do not resume art changes without new user
direction. Showcase cameras are presentation views and are not policy input.

The finishing pass reduced the inherited skylight intensity from 350 to 140 and
the directional light from 7,500 to 4,500 lux. A BaseColor capture retained green
leaf albedo, isolating the excessive grey-white appearance to illumination.
Lumen GI and reflections are explicitly enabled in the saved post-process
settings and verified in the SceneCapture settings. Engine source
`Renderer/Private/SceneCaptureRendering.cpp` documents that captures otherwise
default these methods to None despite the project-level Lumen defaults.

A native gallery and its ground floor close the western horizon; red backdrop
benches are replaced by the assembled bench. The near southern ground-floor
facade was unpacked into 87 native mesh actors with preserved world transforms,
because packed construction restored its original material assignments on
reload. Near panes now use an opaque coated-glazing approximation, replacing
repeated bedroom impostors; they do not model transmissive interior spaces.
`geometry-preservation.json` finds identical multisets of 2,789 mesh bounds
before/after facade unpacking at 0.1 mm rounding. This is a bounds check, not
triangle-level collision proof. The four-view and eleven-frame engineering
verification passes, including native clear/contact controls. Conservative tree
AABB discrepancies remain as previously documented.

The old camera used world Z=1.72 m over a floor at Z=.12 m, hence its actual eye
height was 1.60 m. The corrected acquisition uses **eye height=1.70 m**, world
Z=1.82 m. Pitch remains -10 degrees (slightly downward walking gaze), horizontal
FOV 100 degrees, RGB/depth 640x360. This is explicit camera pose, not an inferred
user height. Model inputs retain the sanitized sensor/ego/issued-plan boundary.

```powershell
python tools/run_sample_segment.py sensors --output artifacts.local/unreal/willow-eye170-first-person
python tools/verify_sample_segment.py --run artifacts.local/unreal/willow-eye170-first-person
```

`sensors` loads the frozen map without saving or changing its visual assets and
skips showcase rendering. It captures eleven paired frames over a one-metre
controlled trajectory at logical 10 Hz. The verifier checks optical height
from both the recorded pose and a known clear floor depth patch (2 cm tolerance).
This short sequence checks acquisition geometry and replay integration; it is
not a dynamic pedestrian mechanism experiment or a new controller score.

The corrected `willow-eye170-first-person/verification.json` passes all eleven
frames. The floor-depth optical-height estimates range from 1.699971 to 1.699998 m;
map SHA-256 is unchanged from the visual freeze. Task-owned capture processes
were released with no surviving descendants.
