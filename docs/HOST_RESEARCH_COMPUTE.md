# Host research compute

This route covers local or remote training, offline evaluation, and other long
research jobs. It does not define Android/device execution and contains no
credentials or SSH endpoints; machine configuration remains in ignored files.

## Registered secondary worker

The secondary Windows laptop is authorized for ordinary scoped development and
research execution whenever it can reduce turnaround or free the primary machine.
Do not ask again merely to use this worker. Prefer it for independent tests,
preprocessing, builds and compatible small GPU jobs. Account for transfer cost,
available memory and competing jobs; tiny work can stay local. This does not
authorize new experiments, frozen-cohort reruns, paid resources or broader scope.

The machine inventory, address, SSH identity path, remote environment entrypoint,
initial source revision and acceptance limits live in ignored
`artifacts.local/work/remote-worker/worker.json`. Start from
[`config/remote-worker.example.json`](../config/remote-worker.example.json) on
another controller; never commit credentials or host-specific paths.

```powershell
pwsh -NoProfile -File tools/remote_worker.ps1 -Action Inspect
pwsh -NoProfile -File tools/remote_worker.ps1 -Action Exec -ScriptFile <local-worker-command.ps1>
pwsh -NoProfile -File tools/remote_worker.ps1 -Action Exec -Profile export -ScriptFile <local-export-command.ps1>
```

`Inspect` reads current revision, dirty files, free RAM/disk, GPU and worker jobs.
`Exec` uploads the supplied PowerShell file to a unique worker artifact temp path,
runs it after entering the selected environment, and removes it in `finally`.
This avoids Windows' command-line length limit. It uses strict SSH host-key
checking and a specific local key;
establish host trust out of band. It does not upload arbitrary dependencies or
silently synchronize source. Native commands in the supplied script must check
`$LASTEXITCODE` when failure should stop the job.

Provisioned capability (2026-09-07): Ryzen 7 5800H, 16 GB RAM, RTX 3060 Laptop
6 GB; PowerShell 7, Git, JDK 17, Node 24, ripgrep and FFmpeg. Android SDK 35,
NDK 27.0.12077973, CMake 3.22.1 and Gradle 8.10.2 completed an offline Debug APK
build with native compilation in 5m 7s on the recorded source revision.

| `-Profile` | Environment | Validated capability |
| --- | --- | --- |
| `research` (default) | Python 3.13; Torch 2.9.1/cu128, torchvision 0.24.1, CuPy 14.1.1, Numba 0.66, ORT GPU 1.26, Transformers, RapidOCR, Ultralytics, scientific/image/video libraries | Torch/CuPy/Numba CUDA kernels, torchvision NMS, tiny random BERT CUDA forward, ONNX operator on actual CUDA provider, image roundtrip and FFmpeg encoding |
| `l10` | Python 3.11; NumPy 1.26.4, Pandas 1.5.3, PyTables 3.11.1, NetworkX 3.6.1 | HDF5 read/write and graph operation; legacy SEVN compatibility |
| `export` | Python 3.11; CPU Torch, TensorFlow/tf_keras 2.19, ONNX tools, onnx2tf and LiteRT | Keras-to-TFLite conversion followed by actual interpreter invocation |

These are generated-data runtime checks, not model/task-quality results. GPU
research and CPU conversion intentionally use separate environments. ORT GPU
1.26 matches the provisioned CUDA 12 stack; newer default packages may require
a different CUDA major version (see the [official compatibility table](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)).
UE 5.8.2, Willow scene/DDC and the precompiled native capture plugin are now
provisioned (2026-09-08). A two-frame 640x360 RGB/depth engineering capture passed
same-target reference comparison and process release; first startup plus capture
took about 190 seconds. Receipts are under `artifacts/evidence/ue-worker-setup`.
The copied engine is a runtime subset, not a full UE C++ build installation.
CARLA, task-specific models/datasets, attached-device tests, cross-network
and reboot recovery still need task-specific setup/verification. The laptop's
screen timeout is one minute; plugged-in sleep/hibernate remain disabled.

Worker tuning (2026-09-08): raising the active plan's AC maximum processor state
from 99% to 100% improved a fixed 100-view, 640x360 RGB/depth short run from
10.86 to 12.70 output pairs/s. Keep the original pair queue of 4; tested queues
of 2 and 8 were slower. Full settling and output quality settings were unchanged.
The worker-local `artifacts/evidence/ue-tune-power100-20260908` receipts retain the exact
inputs, power-plan change and restore command. This is a single short run per
configuration, not sustained throughput; do not override a later user power plan.

Provisioning inputs are tracked in
[`worker-research.requirements.txt`](../config/worker-research.requirements.txt),
the route's SEVN requirements and
[`worker-export.constraints.txt`](../config/worker-export.constraints.txt) with
`requirements-export.txt`. Create separate virtual environments from compatible
base interpreters. Install Torch/torchvision from the cu128 index for research
and CPU index for export before applying these requirements. Reuse verified
portable SDK/runtime archives and immutable Gradle dependencies from the primary
machine. Keep exact installed locks and transfer hashes in local evidence.

Deploy [`worker_environment.ps1`](../tools/worker_environment.ps1) into the worker's
scripts directory, with a thin `Enter-Workspace.ps1` accepting `-Profile` and
dot-sourcing it with `-WorkerRoot`. Worker-local `config/environment.json` stores
`revision` and `javaHome`. Deploy [`worker_run_job.py`](../tools/worker_run_job.py)
as `scripts/run_job.py`; it preserves selected CUDA/cache variables and resolves
commands to absolute executable paths before launch, avoiding Windows' base
Python lookup bypassing the virtual environment. Native runtime validation is
available in [`probe_worker_environment.py`](../tools/probe_worker_environment.py)
with `--profile` and an explicit ignored `--output` directory. Do not run model
downloads or scientific cohorts merely to validate a provisioned environment.

Throughput is the priority: proactively dispatch useful independent work and run
jobs concurrently when that shortens completion time, without renewed approval.
There is no fixed one-job or four-thread cap. Inspect current capacity when making
placement decisions and size concurrency/thread counts to CPU, RAM, VRAM and I/O;
avoid nested thread oversubscription and memory exhaustion. Overlap transfer,
CPU preprocessing and GPU execution when useful; do not serialize merely by habit.
The worker environment leaves CPU-library thread counts unset by default; set
per-job limits when sharing capacity. Match a specific committed source revision
and protocol. Preserve worker edits; transfer primary WIP only as an explicitly
scoped, hashed task input. Prefer existing compatible tool packages and required
models/data from the primary machine; verify hashes. Do not copy whole virtual
environments, machine-bound caches or credentials. The observed 21.2 MB/s SCP
sample is one 19 MB LAN transfer, not a guaranteed sustained or off-site speed.

The worker's source `artifacts.local` junction points to its own artifact volume;
primary-host drive mappings do not apply there. Keep the same six categories:
downloads, evidence, models, presentations, work and tmp. Primary checkout owns
integration, research decisions and accepted evidence; worker output alone is
not promotion or device validation.

## UE capture placement and handoff

Prefer the secondary worker for authorized UE execution, capture and raw-data
processing. Its G: artifact volume owns the engine/project payloads, capture
frames, intermediate arrays and full logs; select paths through
`BLINDASSIST_ARTIFACTS`, not the primary machine's drive layout. The worker's
`artifacts.local/unreal/BlindAssistStreetLab` is a compatibility junction into
its `artifacts/work/ue` tree. Resolve and verify this mapping before capture.
`UE_ENGINE_ROOT` and `BLINDASSIST_UE_CAPTURE_PLUGIN` expose the worker-local
engine and capture-plugin paths; do not embed controller paths in a job.

Return thin results to the primary checkout: source/input/output hashes,
manifests, terminal and process-release receipts, compact metrics and selected
diagnostic previews. Keep full RGB/depth streams and intermediate payloads on
the worker unless a concrete debugging or delivery requirement needs them.
Thin results must identify the retained worker payload and its owner; they are
not permission to delete raw evidence or replace independent validation.

Use [Worker handoff](operations/WORKER_HANDOFF.md) as the reusable job contract and link or
copy it from the worker-local README. Record the committed source revision,
frozen inputs, output location, stop condition and acceptance scope before
dispatch. The recorded UE smoke establishes startup and two-frame capture for
that source, scene and plugin combination. New combinations still need a scoped
worker check before a large acquisition.

For long jobs, invoke the installed runner from an `Exec` script:

```powershell
python "$env:BLINDASSIST_WORKER_ROOT/scripts/run_job.py" --job UNIQUE_JOB_ID -- python YOUR_SCRIPT.py
if ($LASTEXITCODE -ne 0) { throw 'Dispatch failed; inspect receipt before retry' }
```

The worker-local runner uses a scheduled task under the worker account (S4U).
It persists request, progress, logs and terminal result below
`artifacts.local/evidence/jobs/UNIQUE_JOB_ID`, then unregisters its scheduled task.
It does not inherit interactive network credentials or implement reboot/checkpoint
resume. Launch acknowledgement is not completion. Poll the existing job receipt;
missing terminal means incomplete/unknown, never success. On cancellation, verify
the recorded task, process identity and child tree before stopping only that job,
record a cancelled terminal and remove its scheduled task. Return required outputs
and hashes to the primary machine before accepting evidence. Do not leave idle
workers running or reset an uncertain job by redispatching it.

The worker-local workspace `README.md`, `AGENTS.md`, environment script and runner
are durable operational files; setup logs and dependency lock are retained under
its `artifacts.local/evidence/workspace-setup` and
`artifacts.local/evidence/environment-expanded`. Reprovisioning another machine
must validate these capabilities rather than assume this laptop's paths exist.

## Current DTR-R0 entrypoint

```powershell
pwsh -NoProfile -File tools/ba.ps1 doctor research-dtr-r0
pwsh -NoProfile -File tools/ba.ps1 smoke research-dtr-r0
pwsh -NoProfile -File tools/ba.ps1 materialize research-dtr-r0 -CanaryManifest <manifest.json> -CanaryOutput <ignored-output-dir>
pwsh -NoProfile -File tools/ba.ps1 run research-dtr-r0 -EventInput <events.jsonl>
```

Resolve Python and output paths through the `research-dtr-r0` profile. Keep
event ledgers, videos, model outputs, logs, progress state, and results under
ignored `artifacts.local/`.

The research `run` command is lifecycle-governed: it records inputs before
dispatch and automatically finalizes a thin result, cache/hard-case references,
output receipts, catalog reports, and a terminal journal. `-ResultOutput` may
override the thin-result path, while `-AssetInput alias=<asset-selector>` and
`-CacheInput alias=<cache-key>` declare additional reusable inputs.

`materialize` is the real-input canary adapter. It writes only an observation
ledger and input-health report; it does not read evaluator truth or run the
scientific Development gate. `run` remains sealed to synthetic mechanics until
a separate controlled evaluator is frozen.

## Backend and throughput

- Use a short representative benchmark before choosing batch size, worker
  count, or backend.
- When scientific results are equivalent, use the measured faster path.
- Prefer GPU for suitable batched tensor workloads; do not move archive I/O,
  Python control flow, or tiny operations to GPU merely to use it.
- Runtime speed is execution evidence, never algorithmic uplift.
- Preserve scientific batch, seed, threshold, metric, and denominator while
  benchmarking execution alternatives.

## Long-job minimum

A job likely to outlive the current interactive window needs:

- an explicit output directory and one owning process;
- bounded startup validation on representative input;
- machine-readable progress when completed/total is observable;
- a checkpoint or an honest statement that partial work cannot resume;
- a terminal success or failure artifact;
- cleanup instructions for task-owned processes and temporary resources.

Progress should report completed/total, current stage, last activity, failures,
and an evidence-based ETA or `unknown`. Monitoring is read-only and must not
change inputs, checkpoints, outputs, budgets, or evaluator state.

## Recovery

Resume only after confirming input/config identity, output ownership, checkpoint
integrity, and that no second writer is active. Never reset a frozen budget or
silently repeat an `in_doubt` external call. If iteration-level resume is not
implemented, state the maximum lost work before launching the long job.

## Escalation

Ordinary reversible Development training remains `EXPLORE`; duration alone does
not create a formal protocol. Enter [formal research governance](formal/RESEARCH_GOVERNANCE.md)
only before protected final/blind outcome access or a claim-critical one-shot
run. External paid compute additionally requires explicit resource ownership,
budget, lifecycle, and cleanup boundaries.
