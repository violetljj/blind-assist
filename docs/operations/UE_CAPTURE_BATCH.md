# UE capture batches

Use `tools/ue_capture_batch.py` for multiple independent settled-pose grounding
or factorial blocks on the secondary worker. One bounded editor session loads
Willow once, drains each block, destroys its task actors and render targets,
then starts the next block. The editor exits when the queue finishes or fails.
This is a finite queue, not an indefinitely resident service.

## Inputs and execution

Prepare each block as a complete spec with its own cases, samples and labels.
Keep whole clips together; the tool does not silently split or renumber a
scientific dataset. The manifest contains absolute worker paths:

```json
{"jobs": [
  {"id": "block-001", "capture": "grounding", "spec": "ABSOLUTE_SPEC_1.json"},
  {"id": "block-002", "capture": "grounding", "spec": "ABSOLUTE_SPEC_2.json"}
]}
```

From the configured worker source in the `research` environment:

```powershell
python tools/ue_capture_batch.py --manifest "$env:BLINDASSIST_ARTIFACTS/evidence/JOB/manifest.json" --output "$env:BLINDASSIST_ARTIFACTS/evidence/JOB/batch" --engine $env:UE_ENGINE_ROOT --plugin $env:BLINDASSIST_UE_CAPTURE_PLUGIN
if ($LASTEXITCODE) { throw 'Batch failed; inspect retained session and block receipts' }
```

Dispatch long runs with the registered job runner described in
[worker handoff](WORKER_HANDOFF.md). Inspect competing jobs first. The timeout
bounds the entire editor session; size the queue to fit that bound.

## Readiness, reuse and caching

Batch startup defaults to `ready`, using the matching native plugin to prepare
the actual capture view and check compilation and resource requests. Pose
settling remains in place (grounding initial 32, factorial initial 50).
In ready mode those initial settling renders span actual editor callbacks so
GPU feedback and Nanite streaming can advance; later settled poses retain burst
capture. `startup_settling_cadence` records this distinction. Do not substitute
ordinary texture counters for a visual/Nanite acceptance check.
The readiness gate covers requested resources, not all possible future views
or a proof of temporal convergence. A missing/unsupported bridge or timeout
fails the block; it never silently accepts an unknown state.

Lean initialization skips the unrelated editor startup map and disables only
CLion/VS Code source access plugins. It does not modify project files. Use
`--standard-init` for the original initialization and `--startup-policy fixed`
for the original 8+30 second reference waits.

`--settling-policy auto` retains the existing known static-pose protocol rule.
For other explicitly settled static work, `--settling-policy reuse` skips
additional settling only when consecutive clip ID, camera and object state
are present and exactly equal. Changed states retain their settling count;
each block's first frame always receives full initial settling. Do not apply
this to moving-world or temporal experiments merely because a camera is fixed.

Keep the installed engine, project Content, native plugin and project DDC at
their configured worker locations. Batches reuse those files; they do not build
or retransmit them. The launcher retains `UE-LocalDataCachePath` and `-ddc=NoShared`.
DDC avoids repeating derived-data work; it does not remove process/module/RHI
initialization. Compiled shaders are keyed by their inputs in DDC, as described
in [Epic's shader caching documentation](https://dev.epicgames.com/documentation/unreal-engine/shader-development-in-unreal-engine).

## Incremental availability and recovery

One host CPU validator checks drained blocks while the GPU captures subsequent
blocks. It decodes every PNG, checks every float32 depth array and the frozen
spec/count, then hashes payloads. A block becomes usable immediately when its
entry in the atomically updated `completed.json` is `PASS`. Consumers should
use that entry's `output` path, not guess an attempt directory or consume
partially written frames.

Run the same command with `--resume` after failure. Completed payloads are
rehash-checked; only absent, failed or damaged blocks get new attempt directories.
Old receipts, logs and payloads remain for diagnosis. A changed manifest, spec,
script, engine version, project configuration, saved map or plugin fingerprint
requires a new batch output. Cached blocks are not promoted to fresh scientific
evidence by resuming them. Each spec remains responsible for its asset provenance.

Only one host may own a batch output. `owner.lock` identifies that owner; a
crashed host can leave it behind. Verify the recorded process and session have
ended before removing a stale lock. Do not steal a live lock. A block-level
`stop.request` or the active session's `stop.request` stops acquisition while
preserving partial evidence. Process-release receipts cover task-owned observed
processes, using the existing lifecycle helper.

## Worker engineering check, 2026-09-08

The retained `artifacts/evidence/ue-session-v5-20260908/performance.json` records:

- Same changing 100-view spec: editor lifecycle 70.36 s reference to 33.30 s,
  a 52.7% reduction. Engine initialization itself measured 15.62 s (previous
  reference 17.59 s); most savings come from removing fixed waits.
- Four blocks / 302 pairs in one process: 45.56 s including startup and exit.
  The second 100-view block took 8.36 s; unchanged 100-view reuse acquisition
  took 2.59 s. These different workloads are not a uniform throughput comparison.
- CPU validation overlapped subsequent GPU acquisition by 1.76 s and 1.88 s.
  All-cached resume launched no editor; processes and scheduled tasks released.
- 402 final-path pairs passed format/count checks; a two-pair native probe on
  the same plugin matched same-target reference PNG pixels and depth bytes.

These are short engineering runs, not long-duration or pixel-equivalence claims.
The initial readiness implementation was rejected after a timeout; a subsequent
burst-only first-frame version showed missing canopy and was corrected to tick
settling. Both receipts remain. Compared with fixed waiting, the final run's
depth changed at 0.104% of pixels averaged across frames (3.38% on the first
frame), and RGB mean absolute difference was 3.63/255. Warm-session versus fresh
ready-mode depth differences averaged 0.019%. Rendering residency/temporal
behavior differs: do not replace frozen captures or mix startup policies without
the owning experiment's acceptance check.
