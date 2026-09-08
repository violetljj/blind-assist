# Original City Sample: first spatial route capture

2026-09-08. Engineering trial requested by the user: use an existing city map,
define a pedestrian route, and collect observations without rebuilding a city.
Result: **21-frame native-map acquisition PASS; strict floor audit REVIEW**.
No training, model comparison, dynamic simulation or route promotion.

## Actual source and route

The unchanged official `/Game/Map/Small_City_LVL` is used directly, not
`BAResearchSlice`. Map SHA256 is
`0b0bf55569e75241eb7ec16dfa243547e997e5bba4dfb62d64011c12ffbb9f7f`.
Discovery found 12,102 actor descriptors and the map's original PlayerStart.
World Partition loads existing actors around that start; no obstacle is inserted.

The 10 m straight sidewalk segment runs from
`(-298.98724609375, -1.0074424743652344, 2.43)` to
`(-288.98724609375, -1.0074424743652344, 2.43)` metres, with yaw 0 and pitch -5.
There are 21 poses at 0.5 m spacing. Native downward collision probes report
floor z approximately 0.73 m at every pose, giving a 1.70 m camera height.
The requested loaded region is x `[-324,-274]`, y `[-26,24]`, z `[-5,30]` m.
Actor dependencies can extend beyond this box; it is not a memory bound.

RGB is 640x360, HFOV100; evaluator-only axial depth is float32 metres with
zero meaning UNKNOWN. Appearance previews are 1280x720. Each pose settles
before export, with asset/shader/streaming readiness preserved. This is an
ordered spatial sequence of a static editor world, **not 10 Hz simulation**.
Per-frame pose, path distance and wall-clock capture elapsed time are recorded
in the evaluator receipt, separate from the RGB-only model manifest.

## Findings and retained failures

All evidence is under `artifacts.local/nearfield/city-native-route-20260908/`.

- `start-capture-v1`: 160x160 m region selected 369 descriptors and triggered
  thousands of first-use assets. Proactively stopped after 234.25 s of script
  work with zero frames; process lifecycle 275.032 s. Cache retained.
- `start-capture-v2`: 50x50 m region selected 65 descriptors. Four frames passed
  transport checks, but actual images contained solid HLOD tree/building proxies
  overlapping native geometry. These images are excluded as usable data.
  Script 405.672 s; editor lifecycle 448.797 s, including first-use compilation.
- `start-capture-v3`: simply hiding actors did not fix editor-managed HLOD
  rendering. `start-capture-v4` exposed a Python HiddenActors property setter
  restriction. Both failures remain retained.
- `start-capture-v5`: disabling `wp.Editor.HLOD.AllowShowingHLODsInEditor` and
  using the callable capture exclusion API removed the overlapping proxies.
  The two actual appearance images were inspected; both passed capture QA.
- `route10-capture-v1`: all 21 frames captured, 21 unique RGB files, native
  reference RGB pixel/depth byte identity checks PASS, source project/map
  unchanged, task process release PASS. Complete editor lifecycle 59.125 s.

The route's actual model RGB at indices 0, 8 and 20, and appearance at 0, 10
and 20 were inspected. Native trees, benches, bollards, signs, parked vehicles
and the nearby building are visible without the giant HLOD proxy overlap.
Far surroundings are incomplete because only a bounded region is loaded and
HLODs are disabled. This is a local sidewalk acquisition check, not visual
acceptance of a fully loaded city. Editor-state vehicles do not establish
runtime vehicle physics or dynamic traffic fidelity.

`route10-capture-v1/route-audit.json` records CUDA geometry diagnostics on
RTX 5060 Laptop. All 21 floor patches have 4,800 valid pixels, all within 5 cm
of the downward-probe floor elevation. The inherited stricter median-error
criterion (<=2 cm) passes **19/21**: indices 8 and 9 measure 2.211 cm and
2.098 cm. Audit status remains **REVIEW**, without threshold adjustment.
These compare a forward image patch to an under-camera collision elevation;
surface variation and collision/render differences have not been separated.
Do not claim exact floor labels or a fully passed geometry acceptance.
Whole-image valid depth fractions are about 0.65–0.72; remaining pixels remain
UNKNOWN, including sky and unloaded/out-of-range surroundings.

The viewable `route10-preview.mp4` contains all 21 frames at 2 fps (10.5 s),
verified by ffprobe and full decoding. Playback rate is a preview choice, not
measured or simulated walking time.

## Reuse and decision

`tools/make_city_native_route_spec.py` generates an original-map spec from
camera waypoints. The usual `tools/run_city_pcg_capture.py` launcher now supports
an optional native World Partition region, full-detail-only rendering, floor
probes after readiness and route receipts. The existing PCG capture path keeps
its defaults. Use `route10-spec.json` as the exact retained input for this
engineering sequence; the route uses the already-inspected 50x50 m region.
The task-specific `audit_route.py` and source snapshots are retained with data.

The trial supports continuing with original-map route acquisition instead of
reconstructing street layouts. Before increasing data volume, load appropriate
neighboring blocks for complete surroundings and resolve the intended floor
label definition. This result does not certify collision-free traversal,
dynamic synchronization, unseen-world generalization or model performance.
All task-owned editor processes were released; persistent project caches,
inputs, failures and resulting data are retained for reuse.
