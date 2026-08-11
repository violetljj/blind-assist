# Contributing to BlindAssist

Thank you for helping improve accessible, inspectable on-device perception. Contributions are welcome in Chinese or English.

## Good contribution areas

- Android accessibility, TalkBack semantics, touch targets and localization.
- Kotlin tests, deterministic risk-policy tests and lifecycle reliability.
- Build portability, documentation, CI and reproducible evaluation tooling.
- Evidence-bounded research utilities that preserve `UNKNOWN`, provenance and negative results.

Please do not submit raw camera footage, private user data, credentials, restricted datasets, proprietary SDK binaries, generated model payloads or files from `artifacts.local/`.

## Before opening work

1. Search existing [issues](https://github.com/violetljj/blind-assist/issues) and documentation.
2. Open an issue before a broad architecture, model, permission, feedback-policy or research-contract change.
3. Keep one pull request focused on one reviewable outcome.
4. State whether the change affects the default App. Research results do not receive product authority automatically.

Security and privacy reports must follow [SECURITY.md](SECURITY.md), not a public issue.

## Development setup

The maintained local entrypoint currently targets Windows 11 and PowerShell 7 with JDK 17 and Android SDK Platform 35:

```powershell
pwsh -NoProfile -File scripts/run_android_gradle.ps1 -PreflightOnly
pwsh -NoProfile -File scripts/run_android_gradle.ps1 :app:testDebugUnitTest :app:lintDebug :app:assembleDebug
```

GitHub Actions provides the Linux validation path. Machine-local tools, SDKs, datasets and generated evidence belong outside Git or under ignored `artifacts.local/` paths.

## Repository boundaries

- `app/`: default Android application and packaged assets.
- `feature/assist/`: runtime coordination.
- `core/assist/`: deterministic risk, event and feedback policy.
- `core/vision/`: detection and image processing.
- `core/device/` and `core/ui/`: Android adapters and UI state.
- `apps/`: isolated benchmark, canary, demo and candidate apps.
- `scripts/` and `docs/`: stable tooling, governance and reproducible evidence contracts.

See [docs/CODE_MAP.md](docs/CODE_MAP.md) before cross-module changes.

## Evidence and safety rules

- Never interpret `UNKNOWN` as a negative or safe result.
- Label synthetic, pseudo-labeled and model-reviewed evidence explicitly.
- Do not describe a build, package, benchmark or research result as deployment, user-outcome or safety proof.
- Preserve failed and consumed outcomes; do not lower gates after observing results.
- Do not expose private paths, access tokens, device identifiers or non-redistributable payloads.

The current research policy is documented in [docs/RESEARCH_GOVERNANCE.md](docs/RESEARCH_GOVERNANCE.md).

## Verification

Run the narrowest checks that cover your change. All pull requests should include `git diff --check` and the applicable commands:

```powershell
pwsh -NoProfile -File scripts/check_repo_hygiene.ps1 -IncludeStructure
pwsh -NoProfile -File scripts/check_docs_index.ps1
python scripts/run_research_contract_tests.py
```

Android code, resources, build configuration or shared interfaces require the affected tests/lint/build through `scripts/run_android_gradle.ps1`. Device behavior claims require the relevant documented device regression; absence of a device must be reported as an evidence gap.

## Pull request checklist

- Explain what changed, why, and the affected users or maintainers.
- Link the issue when one exists.
- List exact verification commands and results.
- State remaining risks, unavailable checks and default-App impact.
- Keep unrelated formatting, generated files and local artifacts out of the diff.

By submitting a contribution, you agree that your contribution is licensed under the repository's [AGPL-3.0-only license](LICENSE), unless a file clearly states another license.
