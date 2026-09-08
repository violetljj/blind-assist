# City acquisition worker commissioning

Scoped engineering setup for the prepared Street200V7 scene and five complex
real props. The worker uses the independent CitySampleSliceV2 project, its
existing native capture plugin, and its own persistent DDC/Zen. Full City Sample
editing or city regeneration is not required for acquisition.

Acceptance sequence: synchronize hashed source/used-asset deltas, capture two
clear/center/lateral groups, verify native geometry and grouping, then capture
the same 50 groups / 150 frames as the primary pilot. The full-batch input is
the primary `trial-50-spec-v2.json`, with worker-local map path adaptation.
Keep cases, camera calibration, 32 settling ticks, zero forced settling delay,
and native asset/shader/streaming readiness unchanged. Do not infer worker
throughput from the primary's 185.219 s result.

Raw RGB, depth, masks, previews and logs stay under the worker's artifact volume.
Return thin source/input/output hashes, terminal/geometry/group reports and
selected previews to the primary checkout. Release only task-owned processes
and scheduled jobs after completion; retain caches and durable evidence.

The earlier V7 cold single-frame check timed out at 900 seconds with zero
frames, while processing shaders and assets. Its failure and retained caches
remain evidence; the new authorized commissioning must establish actual output,
not merely successful transfer or editor startup. No model training or data
collection beyond this bounded 50-group acceptance is authorized by this record.

## Deployment

Complex-asset delta: 26 new files / 436,566,694 bytes, SHA-verified transfer in
33.597 s. The source/config package has 19 files / 265,744 bytes. The primary
independently rehashed all 18 source files against its current files; all match.
The configuration stays in ignored artifacts and contains worker-local paths.

`tools/run_city_worker_batch.py` is the fixed entry. On the worker it is deployed
under `artifacts/work/city-pcg-20260908/trial50-runtime-v1/payload/source/tools/`.
Pass absolute `--spec` and fresh absolute `--output` paths in the worker artifact
tree. The adjacent payload `worker-config.json` supplies engine/project/plugin
paths; the runner remaps the map path while preserving the original input hash.
It runs capture, CUDA geometry verification and group summary sequentially,
and stops on any failure. The deployed manifest records the exact runner hash.
`--production` selects asynchronous canonical RGB/depth without duplicate
reference files or 720p appearance. Original input hashes and these overrides
are recorded in the adapted spec; readiness and settling remain unchanged.

Initial small acceptance selects original pilot frames 0/1/2 and 90/91/92
(street bicycle and plaza scaffold), remapping baseline indices for the six-frame
spec. Initial scene-resource preparation is timed separately from the subsequent
150-frame run. Do not extrapolate cold initialization to per-frame throughput.

## Actual worker results

The six-frame initial run completes after 732.285 s of editor lifecycle, mostly
first-time scene resource work. Native capture, CUDA geometry (0.698 s) and
group checks pass, but visual inspection finds unsettled block artifacts in
the first appearance frame. Later frames are normal. Retain this as a cold-start
visual defect; automated geometry acceptance alone does not accept its imagery.

The subsequent cached 50-group / 150-frame `full-v1` run passes automated
capture, native geometry and grouping: editor lifecycle 210.695 s, CUDA verification 3.703 s, summary
3.405 s, complete job 224.017 s. The primary independently compares the retained
worker input against the primary pilot: all 150 cases and key capture settings
match, including map hash. BODY/HEAD support and distance states match on all
150 frames. No near-uniform black/white or majority-unknown review flags occur.
The cached first appearance frame is visually normal. However, further primary
inspection finds that first-use table/bench frames 46/61 omit the objects in
both appearance and actual model RGB, despite positive native support. The
model PNG hashes match the verification report, so this is a real capture
defect, not a preview copy error. **Full-v1 is not accepted as training data.**

Full payload location on worker G: is
`artifacts/work/city-pcg-20260908/trial50-full-v1/`, owned by BlindAssist City
acquisition. Primary thin evidence is
`artifacts.local/nearfield/city-pcg-20260908/worker-transfer-trial50-source-v1/full-evidence/`.
`primary-comparison.json` records the controller comparison. The worker release
receipt has `released=true` and no survivors; durable data and caches remain.

The cold-start defect initially motivated an additional settled render pass after native
readiness for the first view, or when readiness waited over 0.5 s. A three-frame
warm first-view check passed in 31.981 s but did not cover new asset first use.
The observed table/bench defect extends this pass to every first-use mesh and
records readiness per view, rather than only the final observation. Previously,
settling renders could occur before assets were ready and export immediately
after readiness. The fix preserves the case and existing actors, records
`post_ready_settling`, and leaves normal later views at the fast cadence.
The correction retains caches. Five first-use asset groups are checked visually
before repeating the same 150-frame acceptance; it does not expand the dataset.

The corrected five-asset / 15-frame check passes in 42.988 s editor lifecycle.
Primary inspection confirms all five first-use assets in actual model RGB.
The corrected `full-v2` then completes all 50 groups / 150 frames: editor
lifecycle 184.904 s, script 162.640 s, CUDA verification 3.612 s, grouping
3.369 s, complete worker job 197.964 s. Capture, geometry and grouping pass;
primary inspection confirms the previously missing table and bench in frames
46/61. The task-owned UE processes and scheduled job are released, caches
retained, and the shared checkout is unchanged. This accepts the corrected
cached acquisition path; it is not a new empty-cache initialization benchmark.
Thin evidence is `worker-transfer-trial50-source-v1/full-v2-evidence/` under
the primary City artifact tree. Earlier failed imagery remains excluded.

For the subsequent throughput work and production profile, see
[capture throughput](CITY_CAPTURE_THROUGHPUT_20260908.md).
The optional fast profile completes the same 150 frames in 127.286 s complete
job and 287.70 MB total output. Task-state checks pass, but its background
RGB/depth differs more than the standard repeat. Keep standard mode as default;
use the explicitly recorded fast profile with its documented evidence scope.
