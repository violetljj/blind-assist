# Host research compute

This route covers local or remote training, offline evaluation, and other long
research jobs. It does not define Android/device execution and contains no
machine-specific paths or hardware assumptions.

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
```

`Inspect` reads current revision, dirty files, free RAM/disk, GPU and worker jobs.
`Exec` runs the supplied PowerShell file remotely after entering the configured
environment. It uses strict SSH host-key checking and a specific local key;
establish host trust out of band. It does not upload arbitrary dependencies or
silently synchronize source. Native commands in the supplied script must check
`$LASTEXITCODE` when failure should stop the job.

Initial provisioned capability (2026-09-07): Ryzen 7 5800H, 16 GB RAM, RTX 3060
Laptop 6 GB; PowerShell 7, Git, JDK 17, Python 3.13 research environment,
PyTorch 2.9.1/CUDA 12.8, Android SDK 35 and Gradle 8.10.2. CUDA arithmetic,
8 research-backend tests, package consistency, Android preflight and a scheduled
job surviving SSH disconnection passed. This is a dated setup record, not a
fresh capacity check or a full APK/native build. NDK/CMake, TensorFlow/export
environment, UE engine/scenes, datasets, cross-network and reboot recovery still
need task-specific verification. Consult the local acceptance receipt.

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
its `artifacts.local/evidence/workspace-setup`. Reprovisioning another machine
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
