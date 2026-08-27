# Code map

## Android modules

| Module | Ownership |
| --- | --- |
| `app/` | Application shell, manifest, packaged assets, and composition |
| `feature/assist/` | Assist-session coordination and feature UI |
| `core/assist/` | Pure risk and guidance logic |
| `core/vision/` | Detection and vision-facing interfaces |
| `core/device/` | Camera, sensor, and Android platform adapters |
| `core/ui/` | Shared UI state and rendering |
| `apps/benchmarks/` | Explicit benchmark applications; never default app authority |

## Research

The only current route is
[`research/active/grail-r1cg/`](../research/active/grail-r1cg/README.md).
Closed modules and their exact historical paths are searchable in
`experiments/index.jsonl` and preserved at the archive tag documented in
[history-index.md](history-index.md).

## Stable tooling

- `tools/ba.ps1`: profile-aware setup, doctor, smoke, run, and cleanup
- `scripts/run_android_gradle.ps1`: Android/Gradle execution
- `tools/data/generate_dataset_ledger.py`: regenerate the externalized dataset
  ledger and compact summaries
- `scripts/check_project_structure.ps1`: current layout policy

Local datasets, models, evidence, APKs, and caches belong in ignored
`artifacts.local/`.
