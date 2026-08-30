# BlindAssist agent map

## Project

BlindAssist is an Android showcase research prototype and thesis project. Optimize for genuine
technical effect, controlled metrics, demo stability, and a clear algorithmic contribution; it is not a certified mobility or safety product.

Keep module ownership stable: `:app` shell/assets, `:feature:assist` runtime,
`:core:assist` risk, `:core:vision` detection, `:core:device` adapters, and `:core:ui` UI.

## Load order

1. Read [project state](docs/PROJECT_STATE.md).
2. Open only the affected route `CURRENT.md`.
3. For known-route algorithm/model/data work, run `python tools/knowledge.py
   context --route <obstacle-avoidance|ten-meter-copilot> --limit 4`; add
   `--query` for a named mechanism/failure and use `--json` only for automation.
4. Open the detailed route `README.md`, code, test, or contract only as needed.
5. Check `git status --short`; for several lines, run `pwsh -NoProfile -File scripts/show_worktree_scope.ps1`.

Knowledge context does not replace route authority or reopen retired, rejected,
consumed, or closed work. Avoid archives, full logs, generated outputs, and
unrelated routes unless audit/reproduction requires them. Currents own active
status; route ledgers/result files own detailed metrics and terminals.

## Execution policy

Default research mode is `EXPLORE`: one question, credible baseline, meaningful
change, observable check, and stop condition. Implement first, then run the
smallest falsifying check.

In `EXPLORE`:

- transparently curated Development data and controlled scenarios are allowed;
- a failed experiment needs one concise current/ledger update, not new governance;
- add process only for a named material risk or decision-changing evidence gap;
- missing deployment/safety evidence limits claims, not reversible experiments;
- reused evidence may support disclosed Development, never fresh confirmation.

Prefer code and observed results over process documents. Update the owning
current only when status, claim, successor, forbidden action, or next decision
changes; undecided ideas stay in `idea.md`. Ordinary public data may enter
isolated internal research with provenance, but public access grants no
redistribution, promotion, consent, or license rights.

Use `FINAL` only before protected blind/final access or a claim-critical paper
number; follow [research governance](docs/formal/RESEARCH_GOVERNANCE.md). Use
`EXTERNAL` only for release, deployment, credentials, privacy, destructive
external actions, default-App promotion, or real-user/product-safety claims.
These modes constrain the affected claim/action, not nearby reversible work.

## Integrity and evidence

- Never fabricate measurements, provenance, labels, licenses, credentials,
  consent, user decisions, authorization, or objective truth.
- Keep public goal identity, evaluator-only truth, proposal, selection, and
  handoff/persistence as separate authorities.
- `UNKNOWN` and `NOT_EVALUABLE` are not negative evidence.
- Name synthetic, replay, pseudo-labeled, model-reviewed, device, and natural
  evidence accurately; curated Development is not universal product or safety
  performance.
- Preserve failed/consumed terminals. Reuse permits diagnostics, regression, or
  disclosed Development, never fresh confirmation authority.
- Do not leak protected outcomes, silently change denominators, hide collapsed
  coverage, or read evaluator truth from observations.

## Task routing

| Task | Read next |
| --- | --- |
| Algorithm/model/data experiment | One route current, then its owning code/result |
| Protected blind/final claim | [Research governance](docs/formal/RESEARCH_GOVERNANCE.md) |
| Android/CameraX/UI/module code | [Code map](docs/CODE_MAP.md) |
| Device/ADB/latency/stability | [Device regression](docs/DEVICE_REGRESSION.md) |
| Release/APK/archive | [Release and verification](docs/RELEASE_AND_VERIFICATION.md) |
| Hardware/glasses/ESP32/network | [Hardware route](docs/GLASSES_HARDWARE_ROUTE.md) |
| Documentation/layout/artifacts | [Document governance](docs/DOCUMENT_GOVERNANCE.md) |
| Long/remote compute | [Host research compute](docs/HOST_RESEARCH_COMPUTE.md) |
| SkyDiscover search | [SkyDiscover playbook](docs/SKYDISCOVER_PLAYBOOK.md) plus the owning route |

## Tools and compute

- Prefer Exa for external search, literature discovery, and multi-source research when available.
- SkyDiscover is isolated and optional. BlindAssist owns its question, evaluator, evidence, decision, and claim; never mutate/clean SkyDiscover or use it to replace missing or fresh evidence.
- Run Android/Gradle through `pwsh -NoProfile -File scripts/run_android_gradle.ps1 <tasks...>`.
- Use `pwsh -NoProfile -File tools/ba.ps1 doctor <profile>` for `base`, both research routes, `android`, `device`, or `export`.
- Keep machine paths, credentials, and endpoints in CLI arguments, ignored local config, environment variables, or the credential store.
- Validate only the changed surface with `git diff --check`, structure for layout, and docs index for hot links; broaden only for the named risk.
- GPU-helpful work is GPU-first. Record actual backend/device/providers and timings; benchmark equivalent CPU/GPU batch or point-cloud work. CPU requires `CPU_FASTER_MEASURED`, `TASK_NOT_GPU_SUITABLE`, `ACCELERATOR_UNAVAILABLE`, `GPU_BACKEND_UNAVAILABLE`, or `FROZEN_PROTOCOL_CPU_ONLY`. Small scalar/metadata work stays on CPU. Reuse `tools/research_backend.py`; never claim CUDA from CPU execution.

## Ownership and delivery

- Pre-existing/concurrent changes are user-owned; edit/stage only task-owned paths or hunks and never revert/reclassify unrelated work.
- Uncommitted files and untracked candidates are WIP, not route authority.
- Payloads and generated outputs stay under ignored `artifacts.local/`; on managed Windows it remains the canonical junction in [local artifacts](docs/LOCAL_ARTIFACTS.md). Never create a physical workspace-drive bypass; run hygiene after storage-path changes.
- Never rewrite history, force-push, delete branches, change remotes, or perform destructive actions without explicit authorization.
- Deliver routine research directly to the default branch unless requested otherwise; verify remote parity and never absorb unrelated changes.

Completion means the outcome exists, its narrow check passes or the gap is stated, the scoped diff is reviewed, task-owned resources are released, and no speculative polish remains.
