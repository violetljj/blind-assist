# BlindAssist

BlindAssist is an Android showcase research prototype for goal-driven visual
assistance. It combines camera perception, risk logic, and concise guidance to
demonstrate measurable effects in clearly stated controlled conditions. It is
not a certified mobility or safety product.

## Start here

- Current project state: [docs/PROJECT_STATE.md](docs/PROJECT_STATE.md)
- Current research decision: [docs/CURRENT_DECISION.md](docs/CURRENT_DECISION.md)
- Active experiment: [research/active/grail-r1cl/README.md](research/active/grail-r1cl/README.md)
- Code ownership: [docs/CODE_MAP.md](docs/CODE_MAP.md)
- Documentation map: [docs/README.md](docs/README.md)
- Historical lookup: [docs/history-index.md](docs/history-index.md)

Only one research route is operational in the current tree. Closed experiments,
their runners, frozen contracts, and long reports are preserved by the remote
tag `archive/pre-agent-surface-2026-08-26` and summarized in
`experiments/index.jsonl`.

## Workstation profiles

Copy the local template once and edit only machine-owned values:

```powershell
Copy-Item config/local.example.toml config/local.toml
pwsh -NoProfile -File tools/ba.ps1 setup base
pwsh -NoProfile -File tools/ba.ps1 doctor base
```

Profiles are independent: `base`, `research-r1cl`, `android`, `device`, and
`export`. Research setup does not install or probe Android tooling.

```powershell
pwsh -NoProfile -File tools/ba.ps1 setup research-r1cl
pwsh -NoProfile -File tools/ba.ps1 doctor research-r1cl
pwsh -NoProfile -File tools/ba.ps1 smoke research-r1cl
```

The Codex desktop environment exposes the same setup and common actions through
`.codex/environments/environment.toml`. A clean isolated workspace can be
created with `tools/open-grail-workspace.ps1`.

## Android

Run Gradle only through the repository wrapper:

```powershell
pwsh -NoProfile -File scripts/run_android_gradle.ps1 :app:assembleDebug
```

Module boundaries are stable: `:app` owns the shell and assets,
`:feature:assist` runtime coordination, `:core:assist` pure risk logic,
`:core:vision` detection, `:core:device` Android adapters, and `:core:ui` state
and rendering.

## Evidence boundary

Synthetic, replay, curated Development, device, and natural evidence are named
separately. `UNKNOWN` and `NOT_EVALUABLE` are not negative evidence. A focused
demo result is never presented as universal real-world or safety performance.
Protected final evaluations use [formal research governance](docs/formal/RESEARCH_GOVERNANCE.md).

## License

See [CONTRIBUTING.md](CONTRIBUTING.md), [GOVERNANCE.md](GOVERNANCE.md),
[the model card](docs/MODEL_CARD.md), [maintainer automation](docs/operations/CODEX_MAINTAINER_AUTOMATION.md),
[the threat model](docs/THREAT_MODEL.md), [LICENSE](LICENSE),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and [SECURITY.md](SECURITY.md).
