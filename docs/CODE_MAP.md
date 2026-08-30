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

The two current, independent routes start at
[L10 current](../research/active/l10-r0/CURRENT.md) for ten-meter goal completion
and [DTR current](../research/active/dtr-r0/CURRENT.md) for dynamic obstacle/risk
guidance. Open the route [L10 ledger](../research/active/l10-r0/README.md) or
[DTR ledger](../research/active/dtr-r0/README.md) only when detailed history,
metrics, or terminals are needed.
`l10-r0` and `dtr-r0` are stable historical directory names, not one shared
project version; each `CURRENT.md` owns the route's actual present status.
Closed modules and their exact historical paths are searchable in
`experiments/index.jsonl` and preserved at the archive tag documented in
[history-index.md](history-index.md).

## Stable tooling

- `tools/ba.ps1`: profile-aware setup, doctor, smoke, real-input materialize,
  synthetic run, and cleanup
- `scripts/run_android_gradle.ps1`: Android/Gradle execution
- `tools/data/generate_dataset_ledger.py`: regenerate the externalized dataset
  ledger and compact summaries
- `scripts/check_project_structure.ps1`: current layout policy

Local datasets, models, evidence, APKs, and caches belong in ignored
`artifacts.local/`.
