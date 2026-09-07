# G14 environment references and asset decisions

2026-09-08. These are design references, not reproduced environments or evidence
that our output reaches their quality. Their screenshots are not UE capture evidence.

| Inspected reference | Applied design decision |
| --- | --- |
| [Jane Dwares, Hanoi Intersection](https://gamesartist.co.uk/hanoi-intersection/) | Establish pedestrian-view composition and architecture spacing; close unintended gaps before adding small detail. |
| [Saerom Youn, Middle Temple garden breakdown](https://80.lv/articles/creating-massive-middle-temple-with-garden-in-substance-3d-unreal-engine) | Separate canopy, planting and ground layers; concentrate planting at edges and keep circulation readable. |
| [Dominique Buttiens, interior details](https://80.lv/articles/getting-interior-details-right-with-ue4) | Define one coherent interior corner, reuse a small material/prop palette and inspect roughness and light together. |

Reuse suitable existing native building assemblies, scanned timber seating,
chairs and trees. For the first iteration, add explicitly licensed small assets
instead of downloading a whole city. The reference authors' project assets are
not assumed redistributable merely because their articles are public.

## Download selection

[Poly Haven's asset license](https://polyhaven.com/license) specifies CC0 for
its assets. Keep original download URLs, provider checksums and local SHA256
in `artifacts.local/unreal/g14-realism-assets/sources.json` (or the downloader's
manifest). Asset files and generated UE packages belong under artifacts.local,
not the Git source tree. No purchase is part of this iteration.

- [Concrete Pavement 02](https://polyhaven.com/a/concrete_pavement_02), Charlotte
  Baglioni: pedestrian paving with physical texture width 1.8 m; use 2K color,
  DX normal and roughness. No displacement changes to geometry.
- [Leafy Grass](https://polyhaven.com/a/leafy_grass), Charlotte Baglioni: ground
  cover around planting, not a substitute for three-dimensional vegetation.
- [Shrub 01](https://polyhaven.com/a/shrub_01), Rico Cilliers: actual shrub
  geometry for understory, rather than using only miniaturized trees.
- [Terrazzo Tiles](https://polyhaven.com/a/terrazzo_tiles), Amal Kumar: indoor
  floor PBR added after the native preview exposed a uniformly flat floor;
  200cm repeat is an authored design assumption, not a measured asset dimension.

Downloads and UE import must succeed before any selected asset is reported as
used. Native preview receipts own actual asset paths/settings and source hashes.
The existing OvercastCourtyard HDRI is reused from the previously imported CC0
[Overcast Industrial Courtyard](https://polyhaven.com/a/overcast_industrial_courtyard).

## Visible checks

Compare the unchanged 640x360 camera against local-preview4, then inspect
additional native 1280x720 appearance exports for material/placement defects.
The extra resolution is visual evidence only: no higher-resolution support
labels or model evaluation are implied. Record exposure and lighting changes,
so this is a joint art/appearance revision rather than a single-factor experiment.

The target-free look-development spec deliberately removes the old diagnostic
bar/box from both case and clip metadata. The 48-frame counterfactual generator
is unchanged by that option. Use the dedicated preview for visual inspection,
not for algorithm comparisons against the earlier target-containing images.

## Reproduction

Run from the BlindAssist checkout with a Python runtime containing requests,
NumPy, Pillow and (for verification) CUDA Torch. Payload downloads need network;
UE asset construction takes place in the canonical artifact project only.

```powershell
python -B research/active/dtr-r0/nearfield/worlds_assets.py
python -B research/active/dtr-r0/nearfield/worlds_spec.py --scene-preview --output artifacts.local/nearfield/NEW-preview/spec.json
python -B tools/ue_native_capture.py capture --capture worlds --spec artifacts.local/nearfield/NEW-preview/spec.json --output artifacts.local/nearfield/NEW-preview/capture --plugin PATH_TO_BUILT_PLUGIN --startup-policy fixed --lean-init --depth-export native --pair-export native_async --settling-policy full --cadence burst
python -B research/active/dtr-r0/nearfield/worlds_verify.py --capture artifacts.local/nearfield/NEW-preview/capture
```

Choose a fresh output path for every attempt. The optional `appearance_preview`
spec field adds native 1280x720 images and real-tick settling; it increases
preview cost and is not enabled in the normal 48-frame data spec. The standard
640x360 RGB/depth contract remains unchanged. UE shutdown releases the preview
process tree; generated assets, downloaded originals and receipts are retained.

## Next candidate: reuse sample generators

User supplied a proposal during this iteration. Primary-source checks support
evaluating sample assemblies/PCG before further broad hand-authored world work:

- [Epic's August 27, 2026 City Sample update](https://www.unrealengine.com/learning/city-sample-gets-a-major-update-with-pcg-and-unreal-mcp-workflows)
  confirms UE5.8, a new full PCG city and simpler example, the
  CitySamplePCG_demo plugin, five Megaplant species and Unreal MCP workflows.
  Epic also notes Experimental/Beta dependencies. Prefer the simpler example
  for an initial integration check, not a full-city capture commitment.
- [Electric Dreams official overview](https://www.unrealengine.com/en-US/electric-dreams-environment)
  lists PCGCloseRange, PCGSplineExample and PCGForest alongside the 4km world.
  Candidate use is local planting/path assemblies, not importing the whole world
  solely to obtain a small pedestrian view. This page describes its UE5.2-era
  origin; don't treat its old experimental status as current status for all PCG.
- [City Sample Buildings](https://www.fab.com/listings/008fe959-5511-428e-93bd-f99b1179f6d5)
  lists over 2,000 modules and UE-only use. Existing project assemblies already
  reuse this family; that does not mean the new City Sample PCG is installed.
- [Hillside](https://www.fab.com/listings/3277687b-a06f-4ef7-a285-63b981768c4a)
  explicitly lists educational-only and UE-only terms. It is an architectural
  reference candidate, not an approved general training-data source.

No new large Epic sample was downloaded in this iteration. PCG metadata needs
an adapter to the existing camera, hazard controls and native truth; sample
reuse does not by itself establish geometry-label compatibility or throughput.
Keep acquisition-specific asset licenses. Fab's service terms distinguish NoAI
restrictions on generative-AI uses; neither blanket ML permission nor a blanket
ban follows from the proposal's table alone. No training-use legal conclusion
was made here. The current new downloads use explicit CC0 assets.
