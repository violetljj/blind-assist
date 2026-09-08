# City Sample collection field

**Delivered v1:** [108-frame field acceptance and 36-frame replay](FIELD_V1_20260908.md).
Nine routes / three regions, all four fixture families validated; explicit floor
review flags and UNKNOWN supervision retained.

This entry owns scene/route engineering and repeatable data acquisition. Model
training and a particular detector's recall do not determine field acceptance.
The original Small_City_LVL map is not saved or altered by the collector.

`native-field-v1.json` assigns regions before acquisition. Each region is intended
to contain sidewalk, native intersection and narrow-passage routes; constructed
passages are explicitly marked as controlled guardrail assemblies on native
pavement. A candidate is not admitted merely because its name suggests a type.
Route status, ground preview evidence and measured floor govern admission.

The fixture family in `tools/city_field_fixtures.py` provides clear, grounded thin
pole, side protrusion, head crossbar and a suspended sign with visible hangers.
All retain the same grounded portal supports. They are controlled synthetic
assemblies using the existing city material palette, not scanned native props.
The clear variant removes the target hazard, not all native scene obstacles.

## Batch entry

Use the project's existing research Python with NumPy, Pillow and CUDA PyTorch.
Machine-specific project, engine and plugin paths come from a verified local
capture launch receipt; they are command arguments, not portable plan values.

```powershell
python tools/run_city_field_collection.py `
  --plan experiments/city-field/native-field-v1.json `
  --output artifacts.local/nearfield/city-collection-field-20260908/compiled-example `
  --compile-only
```

For execution, use a fresh output path, omit `--compile-only`, and provide
`--project <CitySample.uproject> --engine <UE root> --plugin <capture.uplugin>`.
The tracked default `capture-template-v1.json` fixes map hash and acquisition
settings; its relative map path is resolved against the supplied project.
`--region <id>` limits execution to explicitly selected regions. The collector
refuses to start a region while another Unreal editor is running. Its existing
launcher owns process-tree cleanup on success/failure. Immutable completed
regions and failure receipts remain on disk; never overwrite them to retry.
Repeat the same plan into a new output directory to reproduce acquisition.
Static settled snapshots do not claim continuous video or real-time timestamps.

Each region is captured in one engine session. Each route supplies its ordered
waypoints and a matched five-variant fixture group at the middle observation.
Native scene depth and isolated controlled target depth are recorded after the
same configured scene state. Evaluator-only clones never appear in RGB and are
destroyed before the next view. Source map hashes must remain unchanged.

## Dataset and checks

Region outputs retain capture source, RGB, float32 axial depth, shared calibration,
pose, stable configured instance IDs, actual runtime component bindings,
relative target geometry, native floor probes and per-view readiness receipts.
`*-bundle.json` joins them by original sample index. Controlled actors are
recreated per view; stable identity means the configured physical assembly,
not an Unreal memory address. No simulated depth is silently treated as a
monocular model prediction.

`tools/city_field_contract.py` exposes `validate_plan`, `ingest_native_bundle`,
`validate_bundle` and `validate_bundles`. It checks payload presence/hash/shape,
RGB-depth export pairing, calibration/poses, requested asset readiness, label
consistency, frame assignment and cross-split leakage. It preserves UNKNOWN.
RGB/depth engine transforms are read back for each pair (1 mm / 0.01 degree
agreement); commanded-only fallback is explicitly unverified. Floor discrepancy
above 2 cm is REVIEW, while a camera inside a surface or a discrepancy of at
least 0.5 m is an anomaly. No check silently moves the camera or changes truth.
Inactive configured fixtures are NOT_PRESENT only for that fixture; this is not
an all-scene free-space label. Target masks are independently collision-checked
where that evidence is available; disagreement excludes the target from metrics.

PASS means the checked contract holds. REVIEW retains usable data with explicit
unknown labels or missing acceptance evidence. INCOMPLETE identifies missing
payloads/pairs. FAIL identifies contradictory data or separation violations.
Neither a loaded-actor count nor absence of runtime errors certifies that every
distant building is present; route preview review remains part of admission.

Split assignment is by region and route, never random adjacent frames. All views
and clear/hazard variants of an assembly stay together. The plan declares minimum
region/route gaps and rejects reused target instances, cross-split identical RGB
and nearby cross-split camera positions. Every split includes all three scene
types. All partitions still share the City Sample map and mesh library; this is
spatial separation, not an unseen-city or unseen-asset benchmark. Previously
consumed data remain Development evidence regardless of a later split name.

## Inspection and replay

`python tools/report_city_field_collection.py <collection>` produces per-route
contact sheets containing every frame, target uncertainty lists and a summary.
For replay, collect the same plan with `--region west` into a fresh directory,
then run `python tools/compare_city_field_replay.py --first <collection>
--second <replay> --output <fresh-comparison.json>`. This compares spec, frame
assignment, measured poses, near labels and target reliability, and reports RGB
and depth differences rather than assuming bit-exact rendering.
