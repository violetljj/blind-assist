# CITY-CROSSREGION-V1 source feasibility

Status: **seven descriptor-backed BigCity candidates, NOT ADMITTED**. No experiment frames, model inference, or training were run. This engineering audit does not establish held-out background isolation.

A bounded owned NullRHI editor session exported both maps' on-disk actor descriptors using UE 5.8 `WorldPartitionBlueprintLibrary.get_actor_descs()`, `get_editor_world_bounds()` and `get_runtime_world_bounds()`. Map and project hashes remained unchanged; the process-release receipt confirms release. NullRHI supplies metadata, not visual evidence.

| Map | Descriptors | Runtime XY bounds, metres |
|---|---:|---|
| `/Game/Map/Small_City_LVL` | 12,102 | (-972.410, -713.111) to (843.517, 800.977) |
| `/Game/Map/Big_City_LVL` | 101,981 | (-2821.500, -4684.700) to (2931.500, 1232.000) |

These are descriptor bounds, including nonstreet/world actors. They are not paved-city extents. BigCity includes 4,146 `SIDEWALK` and 6,520 `GROUND` labels and offers substantially more separation for reconnaissance.

## Candidate boxes

Each box extends 120 m around the listed center. Centers come from actual SIDEWALK descriptor bounds, constrained to be within 90 m of a GROUND_C descriptor center, then selected by deterministic farthest-point sampling. The minimum pairwise box gap is **1531.124 m**. Split names are proposed roles, not an isolation certificate. No center is an admitted camera pose; floor, obstruction, and three-road-type coverage remain unverified.

| Candidate | Proposed split | Center X, Y (m) |
|---|---|---|
| big_candidate_01 | TRAIN | -165.252, -1589.902 |
| big_candidate_02 | TRAIN | 1311.192, 724.485 |
| big_candidate_03 | TRAIN | -840.374, -3923.693 |
| big_candidate_04 | TRAIN | -848.428, 458.325 |
| big_candidate_05 | DEV | 1584.764, -2811.582 |
| big_candidate_06 | unseen TEST | -2209.799, -2319.459 |
| big_candidate_07 | unseen TEST | 1806.263, -1040.458 |

Evidence is under `artifacts.local/nearfield/city-crossregion-v1/source-audit/`: `Small_City_LVL.json`, `Big_City_LVL.json`, `seven-candidates.json`, `candidate-overview.png`, `select_candidates.py`, `receipt.json`, `source-integrity.json`, and `process-release.json`. Candidate JSON retains exact actor/package paths, labels, bounds, and source inventory hash for both sidewalk and neighboring ground. Its GUID fields are invalid transient Python struct representations, not exact GUIDs: BigCity has only 20 distinct GUID strings for 101,981 unique actor packages. Use map plus actor-package identity for this descriptor snapshot; component/instance identity remains unverified. See the [background frustum audit](BACKGROUND_FRUSTUM_AUDIT_20260908.md). Original JSON evidence and the selection script are preserved unchanged.

## Why existing inventory does not prove isolation

`research/active/dtr-r0/nearfield/city_native_inspect.py` enumerates only currently loaded `StaticMeshComponent` instances, skips HLOD actors, and filters by instance-origin distance. `city_pcg_capture.py:381` fixes that radius at 35 m. A large building can extend into a view while its origin is outside that radius. This inventory has no renderer visibility information and excludes other primitive types.

The current capture enables distant HLOD appearance separately; those proxies may represent multiple source actors. Distinct proxy paths or disjoint loaded regions therefore do not prove distinct physical building instances. The current region loader explicitly permits dependencies outside the requested box. Descriptor bounds are streaming bounds, not dense visible-instance masks. Shared lights/sky are also distinct from physical building-instance ownership and should be explicitly scoped in the contract.

## Required source admission before 336 frames

1. Source-only overview and ground reconnaissance must establish actual sidewalk, intersection, and narrow-passage opportunities in each candidate box. Record exact per-camera floor and obstruction evidence; do not infer walkability from descriptor labels. If using controlled narrow passages, label that construction explicitly.
2. Freeze the physical identity key and visibility authority. At minimum retain original map/package actor identity, component identity and instance index/transform; map HLOD proxies back to their constituent physical source instances. Actor labels or mesh asset names alone are insufficient.
3. Build an exhaustive conservative visible-candidate set for every final camera, including far buildings, large bounds whose origins lie outside the box, non-static primitives, dependencies and HLOD source membership. A proven-disjoint conservative superset is sufficient; sparse collision rays or nearby inventory are not. Ambiguous visibility or unmapped HLOD membership remains UNVERIFIED.
4. Check physical source-instance intersections across TRAIN/DEV/TEST. Geometric box separation is only a scouting aid. Keep all seven candidates NOT_ADMITTED until this check and route admission are supported.

No claim is made that SmallCity is mathematically incapable of seven regions, or that BigCity's 1.5 km box gaps eliminate shared skyline buildings. BigCity is the better evidenced reconnaissance option; visibility isolation remains the decisive open source requirement.
