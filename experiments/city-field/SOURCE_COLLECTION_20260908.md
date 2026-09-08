# City source collection and worker handoff, 2026-09-08

The primary host captured 39 successful source reconnaissance frames on the
original Big City map: 5 initial views, 33 views across seven candidates, and
one native HLOD-export probe. All successful sessions retained unchanged source
hashes and released their owned editor processes. A failed zero-frame helper
attempt remains in the evidence tree. These are source views, not the 336-frame
research cohort, and no model was run on them.

Evidence root: `artifacts.local/nearfield/city-crossregion-v1-20260908/`.
`source-delivery-summary.json` joins seven v4 capture receipts, release receipts,
and floor grids. Each grid has 961/961 native vertical first hits; these are
candidate surfaces, not semantic sidewalk or walkability labels. Contact sheets
are in `scouting/big_candidate_01-review-v4/` through `07-review-v4/`.

Visual review found that several farthest-distance candidates favor waterfront
plazas and shared skyline backgrounds. Seven regions remain `NOT_ADMITTED`.
Three additional TRAIN-only dense-city candidates were prepared using local
building density and geographic distance, without model outputs. The worker
packet `worker-dense-specs-v1/manifest.json` requests five source views per
candidate; image review must still identify actual sidewalk, intersection, and
narrow-passage routes. Density of actor descriptors alone does not establish them.

## Reusable collection commands

```powershell
python tools/city_region_scout.py prepare --candidates <candidate-json> --project <CitySample.uproject> --output <fresh-spec-directory>
python tools/city_region_scout.py run --manifest <spec-directory/manifest.json> --project <CitySample.uproject> --plugin <BlindAssistCapture.uplugin> --engine <UE58-directory> --output <fresh-capture-directory>
```

The runner verifies the frozen specification and host-local source-map hashes,
refuses to share a host with an existing Unreal process, preserves failed
receipts, and writes contact sheets plus batch progress/completion. Outputs must
resolve under that checkout's canonical `artifacts.local`. The launcher retains
RGB, depth, camera parameters, native probes and optional HLOD membership.

## Secondary worker

The worker uses an isolated City Sample project and explicit hashed source
snapshot, leaving its older project and checkout unchanged. The asset dependency
closure contains 128,070 files / 62,645,882,810 bytes, including all 101,981 Big
City external actor files. A separate 370-file engine-content hash comparison
passed. Transfer uses 124 resumable chunks with per-file verification; only
verified transport archives are removed. Transfer completed successfully: all
124 chunks and 128,070 files passed SHA checks; transport archives were removed.

The queued sequence requires the complete transfer receipt and source identity
checks, then runs one native-source smoke frame followed by the three TRAIN
scouts (15 frames). Smoke failure stops the chain. Raw data stay on the worker;
thin receipts and contact sheets return for review. Live receipts are under
`artifacts.local/work/city-crossregion-worker-20260908/`.

The first worker smoke failed with zero frames: after the unchanged 900-second
readiness deadline, 7,110 assets still needed first-use derivation. Shader jobs
were zero at the terminal. The full cold-start process lasted 1,871.29 seconds;
source hashes stayed unchanged and all owned processes were released. The
returned `cold-terminal-v1/` preserves this failure. It does not invalidate the
primary capture, but shows that copying source assets alone is insufficient for
this worker's first capture within the existing readiness limit.

The corrective preparation is a physically independent 946-file / 22,268,658,018-
byte snapshot of the primary's quiescent Zen cache/cas and exported state indexes.
Its file hashes were verified before transfer. Authentication, sessions and logs
are excluded. The original primary cache and worker cold cache remain intact.
`--ddc-path <canonical-cache-directory>` on both the capture launcher and scout
runner selects an explicit persistent cache and records its path in `launch.json`.
The new worker source snapshot records the changed launcher hashes; the map,
camera specification and 900-second readiness limit remain unchanged. Cache
transfer passed all 37 chunks / 946 file hashes. The temporary primary transfer
snapshot was removed after verification, releasing 22.27 GB; the original cache,
worker seed, manifests and receipts remain.

The seeded smoke captured one frame with unchanged source hashes and released
its processes. Its readiness check took 6.969 seconds and full lifecycle 272.46
seconds. The same-camera worker/primary RGB mean absolute difference is 0.380
on the 0-255 scale; 0.080% of pixels have mean channel difference above 10. These
are diagnostics, not a predeclared similarity threshold.

**The worker smoke remains REVIEW, not ready data.** Its log contains 19 Nanite
resource-invalid messages and 24 virtual-texture DDC fetch failures before
capture/shutdown; the primary counterpart has zero matching messages. Similar
RGB alone does not resolve missing geometry/texture pages. The first dense batch
was blocked by the host's existing-UE guard when an unrelated worker task ran;
it captured no frames and that other process was preserved. The remaining 15
source views subsequently ran in a fresh output after natural host release.

`tools/city_render_health.py` records these observed resource-error signatures.
The capture launcher rejects affected ordinary captures before ready-data
validation; source-only reconnaissance retains them with a review report. The
log cannot bind every missing page to specific pixels, so rejection applies to
the whole affected capture. No matching messages is not a general visibility
certificate. Three focused tests cover rejection, unrelated profiler warnings,
and preserving total counts when examples are capped. Formal cohort completion
and a clean worker acquisition field are not claimed.

The missing-page cause was subsequently traced to concurrent Zen replacement,
not established as an incomplete cache seed. At 23:56:26 the unrelated task
logged that port 8558 had a different data directory, then shut down this
capture's Zen PID 17168 at 23:56:27 and opened its own cache. Our first VT miss
followed at 23:56:33. `zen-collision-evidence.txt` preserves the sequence. UE's
`ZenServerInterface.cpp` explicitly replaces a service on the desired port when
its data directory differs; a startup-only UE check cannot protect a later
overlap. The affected frame remains REVIEW despite the identified cause.

The launcher now probes a separate available port in 20000-29999, supplies it
through `Zen.AutoLaunch.DesiredPort`, and records it with the cache path. Existing
listeners are skipped, never terminated by this probe. Two focused tests verify
the range and preservation of an occupied listener. Availability is a preflight
probe rather than an atomic lease; UE still reports any later bind failure.
The source-v3 scout completed **15/15 frames across three candidates**, each
with unchanged map/project hashes and released processes. All three complete
logs have zero matching Nanite-invalid or VT-fetch-failure messages. The returned
`dense-terminal-v3/` contains 132 hash-verified files: all RGB and native/reference
depth arrays, camera specifications, receipts, floor grids, HLOD exports and full
editor logs. Each floor grid has 961/961 hits. `root-scout-audit.json` joins these
checks; ground hits remain candidate surfaces rather than walkability labels.

Visual review of all three contact sheets finds a building-enclosed pedestrian
court with benches at dense01, street-side pedestrian space and intersection
context at dense02, and a courtyard with low walls/benches at dense07. These are
useful dense-city source candidates, not three established route types in every
region. Actual route placement and full split isolation remain pending.

A separate one-frame source-v4 validation passed **1/1**, using actual Zen port
25229 with the same seed/spec. Its report has zero matching Nanite/VT errors.
Owned processes, scheduled tasks and the port listener were released; no further
captures are queued. The worker retains the City project, validated source
snapshots, persistent cache and durable data for reuse. This completes the
worker source-acquisition handoff; formal routes, interventions and the 336-frame
cohort still require their own admission.

## Background identity limit

The native plugin built successfully and the rendered one-frame probe exported
925 loaded HLOD proxies with 3,342 source references and zero unresolved native
containers. These references point to another HLOD level; this is not a complete
original-actor identity graph. Full-map expansion remains incomplete. The
[background audit](BACKGROUND_FRUSTUM_AUDIT_20260908.md) records conservative
frustum overlap, shared visible global surfaces, and the remaining limits.

Target for the next collection remains useful complete routes with natural
obstacles and matched controls. Formal cross-region isolation is an additional
requirement for the research split; successful file transport or capture does
not certify it.
