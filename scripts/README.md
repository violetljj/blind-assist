# Stable scripts

Only cross-project entrypoints remain here.

- `run_android_gradle.ps1`: repository Android/Gradle wrapper
- `project.ps1`: compatibility shim to `tools/ba.ps1`
- `check_project_structure.ps1`: active-surface and repository-layout gate
- `check_docs_index.ps1`: hot-document link gate
- `check_repo_hygiene.ps1`: generated file, secret, and dependency hygiene
- `check_open_source_readiness.ps1`: public release metadata checks
- `generate_release_manifest.ps1`: release manifest generator
- `run_device_regression.ps1`: explicit device matrix entrypoint
- `archive_apk.ps1`: milestone APK archive helper

Research code lives only under `research/active/`. Closed scripts are available
from the archive tag documented in `docs/history-index.md`.
