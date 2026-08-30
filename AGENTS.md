# BlindAssist agent map

## Project

BlindAssist is an Android showcase research prototype and thesis project. Optimize for genuine technical effect, controlled metrics, demo stability, and a clear algorithmic contribution. It is not a certified mobility product or a substitute for human safety judgment.

Keep the module boundaries stable: `:app` owns the shell/assets, `:feature:assist` runtime coordination, `:core:assist` pure risk logic, `:core:vision` detection, `:core:device` Android adapters, and `:core:ui` UI state/rendering.

## Start here

1. Read [project state](docs/PROJECT_STATE.md).
2. Open only the one classification current that matches the task.
3. For algorithm, model, training, benchmark, or dataset work with a known route, run `python tools/knowledge.py context --route <route> --json`; add `--query` for a named mechanism or failure.
4. Read one directly affected route, code, test, or contract entry.
5. Check `git status --short` before editing and preserve all unrelated work.

Knowledge context is a compact reusable-mechanism/prior-result view; it does not replace route authority or reopen a retired, rejected, consumed, or closed experiment.

Do not scan archives, snapshots, complete logs, `artifacts.local/`, generated outputs, or unrelated routes unless history, reproduction, or audit requires it. Dynamic routes, metrics, terminals, and successors belong only in current documents linked by `PROJECT_STATE.md`.

## Default execution policy

Default research mode is `EXPLORE` for thesis, demo, algorithm, training, and benchmark work. Ordinary engineering uses the same loop: one question, credible baseline, meaningful change, observable metric/check, and stop condition.

Implement and run the smallest meaningful change before adding process. Use the smallest falsifying check; when none exists, inspect the scoped diff or output.

In `EXPLORE`:

- Development data and transparently curated controlled scenarios are allowed;
- a failed experiment normally needs one concise current/ledger update, not a new governance layer;
- add protocols, schemas, validators, gates, receipts, audits, handoffs, broad tests, or frameworks only when they change the next decision or cover a named material risk;
- missing device, safety, release, or production evidence limits the claim but does not block an honestly labeled reversible experiment;
- historical terminals stay true but do not forbid a versioned Development experiment that discloses reused evidence.

Default to end-to-end autonomous low-risk work without human queues or gates. Ordinary public data may enter isolated internal research with source/provenance; public availability does not grant redistribution, promotion, consent, or license rights.

Prefer code and observed results over process documents. Update the owning current only when status, claim, successor, forbidden action, or next decision changes; undecided ideas stay in `idea.md`.

## Escalation

Use `FINAL` only before opening protected final/blind outcomes, producing a claim-critical terminal, or placing a number in a final paper table. Follow the owning current and [research governance](docs/formal/RESEARCH_GOVERNANCE.md), freezing only what protects that claim.

Use `EXTERNAL` only for release, deployment, credentials, privacy, destructive external actions, default-App promotion, or real-user/product-safety claims; follow the routed document and focused checks for the named risk.

Formal/external rules constrain the affected claim or action, not nearby reversible research.

## Integrity and evidence

- Never fabricate measurements, provenance, labels, licenses, credentials, consent, user decisions, authorization, or objective truth.
- Keep public goal identity, private evaluator truth, proposal, selection, and handoff/persistence as separate authority layers.
- `UNKNOWN` and `NOT_EVALUABLE` are not negative evidence.
- Name synthetic, replay, pseudo-labeled, model-reviewed, device, and natural evidence accurately; curated Development is not universal real-world, product, or safety performance.
- Preserve failed/consumed terminals. Reuse may support diagnostics, regression, or disclosed Development, never fresh confirmation authority.
- Do not leak protected outcomes, change denominators silently, hide collapsed coverage, or read evaluator-only truth from observations.

## Task routing

After `PROJECT_STATE.md`, read only the route needed for the task:

| Task | Route |
| --- | --- |
| Algorithm, model, training, benchmark, or dataset exploration | One classification current, then its single owning route/code entry; full governance is not required for `EXPLORE` |
| Protected final/blind evaluation or claim-critical protocol | [research governance](docs/formal/RESEARCH_GOVERNANCE.md) and the owning current contract |
| Android, CameraX, UI, or module code | [code map](docs/CODE_MAP.md), affected implementation, and focused test |
| Device, ADB, streaming, latency, or stability | [device regression](docs/DEVICE_REGRESSION.md) and the affected device contract |
| Release, versioning, APK delivery, or archive | [release and verification](docs/RELEASE_AND_VERIFICATION.md) |
| Hardware, glasses, ESP32, Bluetooth, or network | [hardware route](docs/GLASSES_HARDWARE_ROUTE.md) |
| Documentation, index, project layout, or artifact path | [document governance](docs/DOCUMENT_GOVERNANCE.md) and the affected index |
| Long or remote compute | [host research compute](docs/HOST_RESEARCH_COMPUTE.md) |
| Open-ended algorithm search or candidate improvement with SkyDiscover | [SkyDiscover playbook](docs/SKYDISCOVER_PLAYBOOK.md), then the owning BlindAssist route and evidence contract |

## External search and research discovery

For external web search, literature discovery, and multi-source research, actively prefer and make liberal use of the `Exa` plugin when it can improve recall, coverage, or research depth. Fall back to other search tools when Exa is unavailable or not well suited to the task.

## SkyDiscover auxiliary-system contract

SkyDiscover is an optional reusable engine agents may invoke without separate permission. BlindAssist owns the question, evaluator, evidence, decision, and claim; candidates remain proposals until route validation. Use the playbook's isolated launcher/task output. Never mutate or clean SkyDiscover's checkout, environment, runs, or caches. It cannot supply missing information, reopen consumed evidence, access protected outcomes, replace fresh confirmation, or exceed its input/evaluator evidence.

## Commands and validation

- Android/Gradle: use `pwsh -NoProfile -File scripts/run_android_gradle.ps1 <tasks...>`; do not replace its toolchain.
- Readiness: use `pwsh -NoProfile -File tools/ba.ps1 doctor <profile>` with `base`, `research-dtr-r0`, `research-l10-r0`, `android`, `device`, or `export`. DTR/L10 GPU work must use its profile Python, whose doctor probes real Torch/CuPy/Numba CUDA kernels and critical imports.
- Keep machine/SDK/Python/CUDA paths, credentials, and endpoints in CLI arguments, ignored local config, environment variables, or the credential store.
- Validate the changed surface only: `git diff --check` for text, `scripts/check_project_structure.ps1` for layout, and `scripts/check_docs_index.ps1` for hot links. Broaden only for the changed risk or an explicit gate.
- GPU-helpful work is GPU-first. Before launch verify availability/backend/device; after launch record framework placement plus observed utilization/memory. Configuration or startup alone is not proof; report idle GPU, CPU execution, or fallback. Speed is not algorithmic evidence.
- For inference, batch tensors, or large point-cloud matching, probe equivalent CPU/GPU work and select the faster backend, preferring GPU on a tie. Downloads, archives, metadata, and small scalar scoring stay on CPU without a benchmark.
- CPU selection for GPU-first work requires `CPU_FASTER_MEASURED`, `TASK_NOT_GPU_SUITABLE`, `ACCELERATOR_UNAVAILABLE`, `GPU_BACKEND_UNAVAILABLE`, or `FROZEN_PROTOCOL_CPU_ONLY`; measured choice records both timings.
- GPU-capable results persist actual device/name/providers, selection reason, and timings. CPU tensors or only `CPUExecutionProvider` under a declared CUDA backend fail launch. Reuse `tools/research_backend.py`.

## Workspace ownership and delivery

- Treat pre-existing/concurrent changes as user-owned. Edit and stage only task-owned paths or hunks; never revert unrelated work.
- Keep payloads, datasets, checkpoints, logs, screenshots, APKs, caches, SDKs, venvs, and raw outputs out of tracked source; local artifacts belong under ignored `artifacts.local/`.
- Never rewrite history, force-push, delete branches, change remotes, or run destructive Git/file operations without explicit authorization.
- Deliver routine research directly to the default branch without PR/CI wait unless requested; verify remote parity and never absorb unrelated changes.

Completion means the outcome exists, its narrow falsification check passes or the gap is stated, the scoped diff is reviewed, and task-owned resources are released. Stop without speculative polish.
