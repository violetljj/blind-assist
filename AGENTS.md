# BlindAssist agent map

## Project

BlindAssist is an Android showcase research prototype and thesis project. Optimize
for genuine technical effect, controlled metrics, demo stability, and a clear
algorithmic contribution. It is not a certified mobility product and must not be
presented as a substitute for human safety judgment.

Keep the module boundaries stable: `:app` owns the shell and assets,
`:feature:assist` runtime coordination, `:core:assist` pure risk logic,
`:core:vision` detection, `:core:device` Android adapters, and `:core:ui` UI
state and rendering.

## Start here

1. Read [project state](docs/PROJECT_STATE.md).
2. Open only the one classification current that matches the task.
3. For algorithm, model, training, benchmark, or dataset work with a known route,
   run `python tools/knowledge.py context --route <route> --json`; add `--query`
   when the task already names a mechanism or failure.
4. Read one directly affected route, code, test, or contract entry.
5. Check `git status --short` before editing and preserve all unrelated work.

The knowledge context is a compact reusable-mechanism and prior-result view. It
does not replace the owning route/current as authority and does not reopen a
retired, rejected, consumed, or otherwise closed experiment.

Do not scan archives, snapshots, complete logs, `artifacts.local/`, generated
outputs, or unrelated research routes unless the task explicitly needs history,
reproduction, or audit. Dynamic route names, metrics, terminals, and successors
belong only in the current documents linked by `PROJECT_STATE.md`; never copy
them into this file.

## Default execution policy

Default research mode is `EXPLORE` for thesis, demo, algorithm, training, and
benchmark work. Ordinary engineering also uses the same small execution loop:

- one clear question or requested behavior;
- one credible baseline;
- one meaningful change;
- one observable primary metric or focused check;
- one stop condition.

Implement and run the smallest meaningful change before adding process. Run only
the smallest check that can directly falsify the implementation or result. When
no meaningful automated check exists, inspect the scoped diff or output and
continue.

In `EXPLORE`:

- Development data and transparently curated controlled scenarios are allowed;
- a failed experiment normally needs one concise current/ledger update, not a
  new governance layer;
- do not add a protocol, schema, validator, gate, receipt, audit package,
  handoff, broad test suite, or new framework unless it can change the next
  algorithm decision or covers a named material risk;
- missing device, safety, release, or production evidence limits the claim but
  does not block an honestly labeled reversible experiment;
- historical terminals remain historically true, but they do not forbid a new
  versioned Development experiment that discloses reused evidence.

Default to an end-to-end autonomous workflow.
Do not create, preserve, or wait on a human-required queue or gate for low-risk
work. Data available through an
ordinary public channel may enter isolated internal research with recorded
source and provenance; public availability does not grant redistribution,
promotion, consent, or license rights.

Prefer code and observed results over process documents. Update the owning
current only when a result changes status, claim, successor, forbidden action,
or the next decision. Undecided ideas stay in `idea.md`.

## Escalation

Use `FINAL` only before opening protected final/blind outcomes, producing a
claim-critical terminal, or placing a number in a final paper table. Before
outcome access, follow the owning current contract and
[research governance](docs/formal/RESEARCH_GOVERNANCE.md), freezing only the data,
implementation, metric, threshold, missing-data behavior, and retry semantics
needed to protect that claim.

Use `EXTERNAL` only for release, deployment, credentials, privacy, destructive
external actions, default-App promotion, or real-user/product-safety claims.
Follow the relevant routed document and use focused security, device, release,
rollback, or recovery checks for the named risk.

Formal and external rules constrain the affected claim or action; they do not
silently turn nearby reversible research into a formal evaluation.

## Integrity and evidence

- Never fabricate measurements, provenance, labels, licenses, credentials,
  consent, user decisions, external authorization, or objective ground truth.
- Keep public goal identity, private evaluator truth, proposal, selection, and
  handoff/persistence as separate authority layers.
- `UNKNOWN` and `NOT_EVALUABLE` are not negative evidence.
- Name synthetic, replay, pseudo-labeled, model-reviewed, device, and natural
  evidence accurately. Do not present curated Development evidence as universal
  real-world, product, or safety performance.
- Preserve failed and consumed terminals. Reuse may support diagnostics,
  regression, or disclosed Development work, but never restores fresh or
  independent confirmation authority.
- Do not leak protected outcomes, silently change denominators, hide collapsed
  coverage, or read evaluator-only truth from observations.

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
| Open-ended algorithm search or candidate improvement with SkyDiscover | Use the SkyDiscover auxiliary-system contract below, then the owning BlindAssist route and evidence contract |

## SkyDiscover auxiliary-system contract

SkyDiscover is an optional reusable discovery engine for BlindAssist. Agents may
invoke it without separate permission when algorithm search, candidate
generation, evaluator-driven optimization, or bounded scientific exploration is
likely to advance the active BlindAssist route. It is an auxiliary tool, not an
authority root: BlindAssist owns the question, inputs, evaluator, evidence,
decision, and claim boundary. A SkyDiscover score or selected candidate is a
proposal until the owning BlindAssist route validates it.

The current operator-local checkout is `E:\SkyDiscover`. Treat that checkout as
a shared, independently maintained tool installation, not as a writable
BlindAssist workspace:

- Inspect its current commit, worktree status, documented entrypoint, and active
  jobs before use. Preserve existing `.runs/`, outputs, branches, environments,
  and concurrent work; never reuse or clean them by assumption.
- Keep BlindAssist adapters, initial candidates, evaluator code, configurations,
  prompts, datasets, checkpoints, logs, and outputs in the owning BlindAssist
  route or under ignored
  `artifacts.local/skydiscover/<route>/<task-id>/`. Pass explicit absolute paths
  to SkyDiscover and set an explicit task-owned output root. Do not create these
  files inside `E:\SkyDiscover`.
- Run the tool from a pinned, recorded SkyDiscover commit using its documented
  locked entrypoint. Do not edit SkyDiscover source, lockfiles, shared `.venv`,
  `.runs`, global Python/Conda state, shell startup, or shared caches for a
  BlindAssist experiment. If extra dependencies or mutable caches are required,
  place a task-owned, fingerprinted environment and cache outside the
  SkyDiscover checkout so they can be verified and reused safely.
- If BlindAssist genuinely needs a new SkyDiscover capability, handle it as a
  separate SkyDiscover change on its own branch or isolated worktree, generalize
  it beyond the one experiment, validate and deliver it in that repository,
  then consume a pinned version from BlindAssist. Never patch the shared
  SkyDiscover checkout inline during a BlindAssist run.
- Record enough provenance to reproduce a useful result: SkyDiscover commit,
  BlindAssist commit or source hash, config/evaluator/input hashes, model and
  budget, seed when applicable, output root, and selected-candidate hash.
  Promote only the relevant candidate and concise evidence into BlindAssist;
  bulky raw search state remains ignored local evidence.
- Release only task-owned processes, ports, locks, temporary files, and remote
  workers at completion. Retain the minimal reusable environment, checkpoint,
  and provenance needed for an explicit resume or a later comparable run.

Use the smallest bounded search whose result can change the route decision.
SkyDiscover does not reopen consumed evidence, authorize access to protected
outcomes, replace fresh confirmation, or raise a claim above the evidence class
of its evaluator and inputs.

## Commands and validation

- For Android/Gradle tasks use
  `pwsh -NoProfile -File scripts/run_android_gradle.ps1 <tasks...>`; do not
  hand-compose a replacement toolchain.
- Use `pwsh -NoProfile -File tools/ba.ps1 doctor <profile>` for scoped
  workstation readiness. Profiles are `base`, `research-dtr-r0`,
  `research-l10-r0`, `android`, `device`, and `export`.
- DTR and L10 GPU-capable work must use the Python selected by their respective
  `research-dtr-r0` or `research-l10-r0` profile. Do not launch it from a
  dataset-local or experiment-local CPU venv. Each research doctor runs real
  Torch, CuPy, and Numba CUDA kernels plus both routes' critical import probe.
- Keep machine paths, SDK locations, Python/CUDA paths, credentials, and local
  endpoints out of tracked instructions. Pass them by CLI, ignored local config,
  environment variables, or the owning credential store.
- Run one targeted check for the changed surface. Use `git diff --check` for
  text edits; run `scripts/check_project_structure.ps1` for root/layout policy,
  and `scripts/check_docs_index.ps1` for top-level/current documentation links.
- Run broader builds, hygiene suites, device matrices, or formal validators only
  when the changed surface or an explicit delivery gate requires them.
- When GPU execution can materially improve result quality or reduce wall-clock
  time without violating correctness or a frozen protocol, GPU is the required
  first-choice backend. Before the real run, verify accelerator availability and
  the selected backend/device; after launch, confirm actual GPU execution with
  runtime evidence such as framework-reported device placement plus observed
  process utilization or memory use. Never infer GPU use from configuration,
  installed CUDA, or a successful start alone. Detect and explicitly report an
  unavailable accelerator, idle GPU, CPU-only execution, and partial or full CPU
  fallback; do not silently continue under a fallback while describing the run
  as GPU-accelerated. Speed is not algorithmic evidence.
- Classify each research workload before launch. Model inference, batch tensor
  work, and large point-cloud matching are GPU-first: run a short equivalent
  CPU/GPU probe on representative inputs and use the faster backend, preferring
  GPU on a tie. Download, decompression/archive work, JSON/metadata handling, and
  small scalar scoring stay on CPU and do not need a GPU benchmark.
- A GPU-first workload may select CPU only with a launch record that says either
  `CPU_FASTER_MEASURED`, `TASK_NOT_GPU_SUITABLE`, `ACCELERATOR_UNAVAILABLE`,
  `GPU_BACKEND_UNAVAILABLE`, or `FROZEN_PROTOCOL_CPU_ONLY`; measured selection
  must include both probe timings. Never silently choose CPU from
  `torch.cuda.is_available()` or provider fallback.
- GPU-capable experiments must persist the actual framework-reported device,
  device name, framework/provider list, selection reason, and probe timings in
  their result or adjacent launch record. A declared CUDA backend that observes
  CPU tensors or only `CPUExecutionProvider` is a failed launch. Reuse
  `tools/research_backend.py` for this contract rather than inventing a weaker
  per-script check.

## Workspace ownership and delivery

- Treat pre-existing and concurrent changes as user-owned. Edit and stage only
  task-owned paths or hunks; never revert unrelated work.
- Keep local payloads, datasets, checkpoints, logs, screenshots, APKs, caches,
  SDKs, virtual environments, and raw benchmark outputs out of tracked source.
  Project-local artifacts belong under ignored `artifacts.local/`.
- Never amend or rewrite history, force-push, delete branches, change remotes,
  or run destructive Git/file operations without explicit user authorization.
- Routine research delivery goes directly to the default branch; do not create a
  PR or wait for CI/review unless requested. Verify local/remote parity after push.
  Never absorb unrelated changes to make a delivery look clean.

Completion means the outcome exists, the narrow falsification check passes (or
its exact evidence gap is stated), the scoped diff is reviewed, and task-owned
resources are released. Stop without speculative polish or unrelated validation.
