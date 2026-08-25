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
3. Read one directly affected route, code, test, or contract entry.
4. Check `git status --short` before editing and preserve all unrelated work.

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
[research governance](docs/RESEARCH_GOVERNANCE.md), freezing only the data,
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
| Protected final/blind evaluation or claim-critical protocol | [research governance](docs/RESEARCH_GOVERNANCE.md) and the owning current contract |
| Android, CameraX, UI, or module code | [code map](docs/CODE_MAP.md), affected implementation, and focused test |
| Device, ADB, streaming, latency, or stability | [device regression](docs/DEVICE_REGRESSION.md) and the affected device contract |
| Release, versioning, APK delivery, or archive | [release and verification](docs/RELEASE_AND_VERIFICATION.md) |
| Hardware, glasses, ESP32, Bluetooth, or network | [hardware route](docs/GLASSES_HARDWARE_ROUTE.md) |
| Documentation, index, project layout, or artifact path | [document governance](docs/DOCUMENT_GOVERNANCE.md) and the affected index |
| Long or remote compute | [host research compute](docs/HOST_RESEARCH_COMPUTE.md) |

## Commands and validation

- For Android/Gradle tasks use
  `pwsh -NoProfile -File scripts/run_android_gradle.ps1 <tasks...>`; do not
  hand-compose a replacement toolchain.
- Use `pwsh -NoProfile -File scripts/project.ps1 doctor` only when combined
  workstation readiness is relevant.
- Keep machine paths, SDK locations, Python/CUDA paths, credentials, and local
  endpoints out of tracked instructions. Pass them by CLI, ignored local config,
  environment variables, or the owning credential store.
- Run one targeted check for the changed surface. Use `git diff --check` for
  text edits; run `scripts/check_project_structure.ps1` for root/layout policy,
  and `scripts/check_docs_index.ps1` for top-level/current documentation links.
- Run broader builds, hygiene suites, device matrices, or formal validators only
  when the changed surface or an explicit delivery gate requires them.

## Workspace ownership and delivery

- Treat pre-existing and concurrent changes as user-owned. Edit and stage only
  task-owned paths or hunks; never revert unrelated work.
- Keep local payloads, datasets, checkpoints, logs, screenshots, APKs, caches,
  SDKs, virtual environments, and raw benchmark outputs out of tracked source.
  Project-local artifacts belong under ignored `artifacts.local/`.
- Never amend or rewrite history, force-push, delete branches, change remotes,
  or run destructive Git/file operations without explicit user authorization.
- Tracked task changes should end in one focused commit and a normal push to the
  configured default branch when delivery is in scope. Never absorb unrelated
  changes to make a delivery look clean.

Completion means the requested outcome exists, the narrow falsification check
passes (or its exact evidence gap is stated), the scoped diff is reviewed, and
task-owned temporary processes or resources are released. Stop there; do not
add speculative polish or unrelated validation.
