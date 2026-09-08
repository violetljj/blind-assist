# G14 scene realism iteration, 2026-09-08

Engineering / look development, not a model experiment. Source creation and
native preview run on the primary machine; no bulk capture or training was
launched on either machine. G10/G13 model decisions remain unchanged.

## Implemented

- Reused native building assemblies, scanned timber seating and trees; added
  street-edge planting, opposite trees, courtyard enclosure, paving borders,
  bins and road markings. The corridor has lower wall panels, cornices,
  noticeboards and waiting furniture.
- Downloaded four CC0 Poly Haven assets: Concrete Pavement 02, Leafy Grass,
  Shrub 01 and Terrazzo Tiles. Fourteen payloads total 42,032,063 bytes, checked against provider MD5
  and local SHA256. No purchase. The original manifest is
  `artifacts.local/unreal/g14-realism-assets/manifest.json`, SHA256
  `f88c97997b7f69b9eba7a9bfef5f8b7939cda230cdcfb0e05b3e0e26db6c8dbf`.
  The prior three-asset manifest is preserved as `manifest-v1-three-assets.json`.
- Imported into the new `/Game/G14RealismV1` namespace. Pavement has 180cm
  world-aligned texture scale; leafy ground uses 200cm. Color, normal and
  roughness are separate inputs; shrub has masked two-sided foliage shading.
  No displacement. Native depth includes the actual shrub geometry.
- Reused OvercastCourtyard lighting cubemap, softened sunlight, adjusted indoor
  exposure and explicitly disabled film grain. These are appearance settings,
  not measured glasses-camera calibration. The background sky still uses the
  loaded scene's sky representation.
- Added reproducible `worlds_spec.py --scene-preview` for target-free scene
  inspection. It also exports native 1280x720 appearance images; normal research
  RGB/depth stays 640x360. The 48-frame controlled-target generator is preserved.

See [references and reproduction](WORLD_REALISM_REFERENCES.md) and
[acceptance boundaries](WORLD_REALISM_ACCEPTANCE.md).

## Observed failure and correction

`local-realism5-capture` produced the outdoor image but failed on the corridor:
a file named ScannedCafe_chair was a Material, not a StaticMesh. File existence
alone had been insufficient. Replaced it with already rendered seating and
added an explicit native asset-type check with the offending path in errors.

The first outdoor image also showed planting submerged in the forecourt and
a remaining distant gap. Raised the ground layer and shrub bases to 0.24m,
staggered larger shrub clumps, and extended the background enclosure. Failed
receipts and the first image remain under
`artifacts.local/nearfield/worlds-20260908/local-realism5-capture`.
The failure did not modify the saved Willow map and its process tree was released.

## Final inspected output and validation

`artifacts.local/nearfield/worlds-20260908/local-realism7-capture` passes two
native RGB/depth exports and CUDA all-visible-surface verification. Both
1280x720 appearance images were inspected. The sidewalk has a continuous
backdrop, textured paving and planted edge; the corridor now has a textured
terrazzo surface and readable waiting areas. This iteration repairs observed
engineering defects and improves visible detail; full realism acceptance
remains pending. Eight focused generator/verifier tests and five capture-batch
tests passed; after the final floor edit the three affected generator tests
and native capture were rechecked. Capture-script time is 49.890 seconds,
excluding process startup. This is not bulk acquisition throughput.

The frozen Willow map hash is unchanged. The final process-release receipt
confirms shutdown; downloads, generated UE assets, failed/successful captures
and receipts are retained. There were no model fits, inference runs, threshold
changes or bulk dataset launches. The current scope delivers scene assets,
generator and preview tooling; sample-PCG integration is a researched next
candidate, not an installed or completed capability.

## Remaining limits

The next iteration now prioritizes [mature scene and PCG rule reuse](SAMPLE_PCG_REUSE_20260908.md):
City Sample 5.8's simpler pedestrian slice first, then Electric Dreams path and
planting examples. The subsequent [first PCG plaza integration](CITY_PCG_PLAZA_20260908.md)
passes native capture/control/truth checks; complete street surroundings and
Electric Dreams integration remain pending.

These are two procedurally furnished views inside a frozen map container, not
independent-map generalization or a completed natural-world dataset. Room signs
and notices remain simplified; existing architecture/tree variety remains
limited. No real-device transfer, calibrated noise, natural branch hazard demo,
30-image visual acceptance or model accuracy improvement is claimed.
