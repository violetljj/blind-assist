# Worker handoff

This document can be linked or copied into the secondary worker's README.
Connection details and credentials belong only in ignored controller config.
See [Host research compute](../HOST_RESEARCH_COMPUTE.md) for execution policy.

## Placement and authority

Prefer the worker for authorized UE capture, preprocessing and compatible
research work. Use `BLINDASSIST_SOURCE` for the checkout and
`BLINDASSIST_ARTIFACTS` for the worker's G: artifact volume. Keep raw RGB/depth,
intermediates, engine/project payloads and full logs there. The primary checkout
owns integration and research decisions; return thin manifests, hashes,
metrics, selected previews and terminal/process-release receipts by default.
Copy full payloads only for a specific investigation or requested delivery.

The project entry `artifacts.local/unreal/BlindAssistStreetLab` is a worker-local
compatibility junction into `artifacts/work/ue`. Verify its resolved target,
`UE_ENGINE_ROOT` and `BLINDASSIST_UE_CAPTURE_PLUGIN` before UE work. Do not assume
the primary host's physical paths apply. Preserve retained evidence, with an
explicit owner and retrieval path, after releasing task-owned processes.

## Required job record

- Exact committed source revision and any separately authorized WIP input hashes;
  inspect dirty files and preserve unrelated worker changes.
- Frozen protocol/spec, scene/model/data paths and hashes, allowed split and
  evidence boundary. Provisioning does not authorize new experiments or reruns
  of consumed/frozen cohorts.
- Selected environment, current capacity, unique job ID and exclusive output
  directory below `BLINDASSIST_ARTIFACTS`.
- Concrete command, stop condition, acceptance check and expected thin outputs.
- Payload owner, retained raw-data location and process/temporary-resource cleanup.

## Dispatch

From the controller checkout:

```powershell
pwsh -NoProfile -File tools/remote_worker.ps1 -Action Inspect
pwsh -NoProfile -File tools/remote_worker.ps1 -Action Exec -Profile research -ScriptFile <local-job.ps1>
```

Select `l10` or `export` when appropriate. The uploaded script enters the selected
environment automatically. For a long job, its body can dispatch the installed
runner from the worker checkout:

```powershell
Set-Location -LiteralPath $env:BLINDASSIST_SOURCE
python "$env:BLINDASSIST_WORKER_ROOT/scripts/run_job.py" --job UNIQUE_JOB_ID -- python YOUR_SCRIPT.py
if ($LASTEXITCODE -ne 0) { throw 'Dispatch failed; inspect receipt before retry' }
```

Launch acknowledgement is not completion. Inspect the existing job's receipt
under `artifacts/evidence/jobs/UNIQUE_JOB_ID`; do not redispatch an uncertain job.
Require a terminal result, acceptance evidence and verified process release.
Cancellation must match the recorded task/process identity and preserve logs.

For an authorized grounding capture, the uploaded PowerShell script can use:

```powershell
python "$env:BLINDASSIST_WORKER_ROOT/scripts/run_job.py" --job ue-UNIQUE_JOB_ID -- python tools/ue_native_capture.py capture --engine $env:UE_ENGINE_ROOT --plugin $env:BLINDASSIST_UE_CAPTURE_PLUGIN --capture grounding --spec "$env:BLINDASSIST_ARTIFACTS/evidence/FROZEN_INPUT/spec.json" --output "$env:BLINDASSIST_ARTIFACTS/evidence/UNIQUE_OUTPUT"
if ($LASTEXITCODE -ne 0) { throw 'Dispatch failed; inspect receipt before retry' }
```

Replace the input and output placeholders with the owning task's frozen spec and
fresh output. Match exporter, cadence and settling options to that task's protocol;
inspect `launch.json` for the effective settings. Use `native_probe` pair export
only for an engineering comparison against same-target reference outputs.

## Verified capability and limits

For consecutive independent UE blocks, use the
[batch capture workflow](UE_CAPTURE_BATCH.md): one editor process, per-block
validation overlapping acquisition, exact-state settling reuse, and hash-checked
resume. Keep raw blocks and caches on the worker; return thin manifests and
receipts to the controller.

Prefer the controller for interactive scene design and arbitrary UE C++ builds;
sync changed assets/scripts to the worker for deployment, scripted expansion,
variant generation and capture. Script-only scene expansion can run directly on
the worker. Keep generated scenes/raw data on G and return only needed previews.

The 2026-09-07 provisioning record verified generated-data checks:

| Profile/surface | Verified scope |
| --- | --- |
| Research | Torch/CuPy/Numba CUDA kernels, torchvision NMS, tiny random BERT CUDA forward, ONNX CUDA operator, image and FFmpeg roundtrips |
| L10 | Python 3.11 legacy NumPy/Pandas environment, HDF5 roundtrip and graph operation |
| Export | TensorFlow/Keras conversion to TFLite and actual interpreter invocation, in a separate CPU environment |
| Android build | Offline Debug APK with native compilation using SDK 35, NDK 27 and CMake 3.22.1 on the recorded source revision |
| UE capture (2026-09-08) | UE 5.8.2 loaded Willow and its precompiled capture plugin; two 640x360 RGB/float32-depth engineering frames matched same-target reference outputs, saved map unchanged, owned processes released |

These checks do not establish trained-model quality, current capacity, attached
device behavior or scientific outcomes. Consult the retained provisioning receipt
for exact source revision, package locks, hashes and commands. UE acceptance is
retained in `artifacts/evidence/ue-worker-setup/acceptance.json`: first editor
startup plus the two-frame check took about 190 seconds; this is not sustained
capture throughput. The runtime distribution omits engine Source/Intermediate
and static libraries, so arbitrary UE C++ rebuilding is not provisioned.
Cross-network access and reboot recovery remain outside the recorded validation.

For a UE engineering check, use a fresh output and the scoped capture/calibration
entrypoint chosen by the owning task. Record actual engine/plugin versions and
input hashes; keep its result separate from research cohort scores.

The 2026-09-08 tuning retained AC maximum processor state 100% (previously 99%)
and the existing pair queue of 4. The identical 100-view spec, full settling and
native async export improved from 9.204 s (10.86 pairs/s) to 7.875 s (12.70 pairs/s).
An 8-slot trial at the original power setting took 10.969 s; a 2-slot trial at
AC 100% took 9.859 s. Both queue changes were rejected. All runs validated 100
RGB/depth pairs and released owned processes; each configuration was tested once.
See `artifacts/evidence/ue-tune-power100-20260908/tuning-decision.json` and
`power-change.json` for provenance and restoration. Inspect the current plan
before changing power settings; the receipt is not permission to overwrite later
user preferences. Long-duration thermals and throughput remain unmeasured.
