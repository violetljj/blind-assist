# Mature scene and PCG reuse: first integration slice

User direction, 2026-09-08. Engineering / look development only.
Status: ACQUISITION_PENDING; no sample-PCG integration or new preview passed.

## Order and concrete scope

1. City Sample 5.8: start from its simpler PCG example and retain one short
   pedestrian segment, including its surrounding architecture and planting.
   Reuse the sample graphs, assemblies and layout rules before adding custom
   furnishing. Determine exact map and graph paths from the acquired package;
   they have not been inspected and are not guessed here.
2. Electric Dreams: after the city slice, inspect
   `ElectricDreams_PCGSplineExample`, `ElectricDreams_PCGForest`, and
   `ElectricDreams_PCGCloseRange` for path and layered planting rules. Running
   the entire 4 km x 4 km world is not an acceptance prerequisite. A small
   runtime scope does not imply the provider offers a partial download.
3. Titan, Valley and Hillside remain references, with no acquisition scheduled.

## Verified starting state

- HEAD at inspection was `eae882b3`. The G14 realism iteration remains the
  baseline; its two previews are engineering evidence, not full realism acceptance.
- `F:/epic/VaultCache` contains CitySampleBuildings and FabLibrary. FabLibrary
  contains the Buildings listing cache. The checked launcher manifests list
  UE 5.8, Fab UE Plugin and Quixel Bridge, not either complete sample.
  `artifacts.local/unreal` has the existing lab projects and G14 asset output,
  with no top-level City Sample or Electric Dreams project. This is a bounded
  inventory, not a search of every disk or an account-entitlement check.
- `artifacts.local` resolves to the required F:-backed junction. Acquire sample
  projects and their generated caches through `artifacts.local/unreal/`.
- `worlds_spec.py` imports the frozen Willow `MAP_SHA`, generates its own
  sidewalk/corridor layout, and uses existing lab materials. It is not an
  arbitrary sample-map capture adapter.

## First native acceptance slice

Acquire the City Sample 5.8 package through Epic's supported Fab, UE Home Panel
or Launcher flow. Record actual package version, source URL, applicable asset
terms, project/plugin dependencies and hashes. Preserve the downloaded source
and use a separate integration project; do not resave the frozen Willow map.
Inspect the simpler map and its `CitySamplePCG_demo` dependencies before enabling
only those systems it actually needs. MCP is optional for this capture adapter.

Choose a walkable segment after opening the sample. Record its map, PCG graph,
seed, generated instances and camera poses. Preserve the existing engineering
camera profile (1.70 m above local floor, HFOV100), with new map-specific poses
and identity. Export target-free native appearance and RGB/depth first, then
use the same segment with a controlled BODY/HEAD obstacle and its clear control.
The adapter must leave sample scenery intact and keep evaluator-only geometry
out of model inputs.

Verify camera intrinsics and local floor, controlled-obstacle placement, and
native depth/support coverage for generated instances and foliage. Missing
geometry evidence remains UNKNOWN; collision proxies alone do not certify
visible support. Regenerate truth for geometry/LOD changes. Record startup,
PCG generation/settling, export and total time separately.

Inspect the native previews alongside `local-realism7-capture` for continuous
surroundings, surface scale, grounded vegetation, clear circulation and ordinary
lighting. These are different scene sources, not geometry-matched algorithm
comparisons. Fix observed integration defects before extending the segment.
Release task-owned UE processes after capture and retain receipts and previews.

The first slice ends with a working preview/control/truth export or a specific
failure receipt. Only a passed slice warrants the next greenway integration;
broader seeded visual acceptance remains governed by
[the existing acceptance record](WORLD_REALISM_ACCEPTANCE.md). No bulk capture,
training, model promotion or percentage-reuse benefit is established here.

## Inspected primary sources

- [City Sample update](https://www.unrealengine.com/learning/city-sample-gets-a-major-update-with-pcg-and-unreal-mcp-workflows),
  published August 27, 2026, fetched September 8: confirms UE5.8, full and simpler
  PCG maps, CitySamplePCG_demo, assemblies and Experimental/Beta dependencies.
- [Electric Dreams overview](https://www.unrealengine.com/en-US/electric-dreams-environment),
  fetched September 8: names the small spline, forest and close-range levels.
  Its feature-status discussion reflects the UE5.2-era introduction.

Acquisition-specific terms still require inspection. No training-use permission
or prohibition is inferred solely from NoAI, and no new legal conclusion is made.

## Current delivery limit

This update establishes the integration scope and verifies local prerequisites.
No complete Epic sample was downloaded, no sample graph was inspected locally,
and no PCG scene was run during this update. The next actionable dependency is
obtaining the actual City Sample 5.8 package through the supported Epic flow.
