# Code map

[Documentation map](README.md) · [Current priorities](PROJECT_STATE.md) · [Script index](../scripts/README.md)

## Repository layout

| Location | What belongs here |
| --- | --- |
| `app/`, `core/`, `feature/` | Main Android application and shared modules; ownership below |
| [apps/](../apps/README.md) | Isolated benchmark, canary, demo and candidate applications |
| `research/active/` | Route-owned source, protocols and result records; enter through the route current below |
| [research/knowledge/](../research/knowledge/README.md) | Reusable sources, mechanisms and their route-specific use |
| [scripts/](../scripts/README.md), [tools/](../tools/README.md) | Build, maintenance and research command entry points; [shared research helpers](../scripts/research/README.md) have their own index |
| [config/](../config/README.md) | Environment examples, dependency specifications and the public asset manifest; local values use ignored overrides |
| `data/` | Asset policies, profiles and compact dataset-ledger metadata; raw datasets live under `artifacts.local/` |
| [docs/](README.md) | Shared operating guides and project navigation |
| `experiments/` | Historical lookup index and tracked acquisition specifications such as [City collection](../experiments/city-field/README.md); see [history](history-index.md) for archived source |
| `artifacts.local/` | Ignored datasets, models, evidence, APKs, caches and temporary outputs; see [storage routing](LOCAL_ARTIFACTS.md) |

## Android modules

| Module | Ownership |
| --- | --- |
| `app/` | Application shell, manifest, packaged assets, phone capture and mobile fusion entry points |
| `feature/assist/` | Assist-session coordination and feature UI |
| `core/assist/` | Pure risk and guidance logic |
| `core/vision/` | Detection and vision-facing interfaces |
| `core/device/` | Camera, sensor, and Android platform adapters |
| `core/ui/` | Shared UI state and rendering |
| `core/ustrf/` | USTRF research geometry, risk-field and trace logic |
| `apps/` | Explicit benchmark, canary, demo and candidate modules; results do not automatically change the main app |

[settings.gradle.kts](../settings.gradle.kts) owns Gradle module names and their
physical paths. Build variants and app dependencies live in
[app/build.gradle.kts](../app/build.gradle.kts).

## Find the implementation

| Change | Start in |
| --- | --- |
| Home navigation and app composition | [MainActivity](../app/src/main/java/com/linnan/blindassist/MainActivity.kt) |
| Phone capture screen and session | [PhoneCaptureActivity](../app/src/main/java/com/linnan/blindassist/PhoneCaptureActivity.kt), [PhoneCaptureController](../core/device/src/main/java/com/linnan/blindassist/capture/PhoneCaptureController.kt) |
| Capture files and background ZIP export | [PhoneCaptureStore](../core/device/src/main/java/com/linnan/blindassist/capture/PhoneCaptureStore.kt), [CaptureExportService](../app/src/main/java/com/linnan/blindassist/CaptureExportService.kt) |
| Mobile fusion experiment | [MobileFusionSession](../app/src/main/java/com/linnan/blindassist/MobileFusionSession.kt), [MobileFusionActivity](../app/src/main/java/com/linnan/blindassist/MobileFusionActivity.kt) |
| RGB preview pixels and depth runtime | [AtomPreviewPixels](../app/src/main/java/com/linnan/blindassist/vision/AtomPreviewPixels.kt), [Dav2DepthRuntime](../app/src/main/java/com/linnan/blindassist/vision/Dav2DepthRuntime.kt) |
| Assist lifecycle and effects | [AssistRuntimeController](../feature/assist/src/main/java/com/linnan/blindassist/runtime/AssistRuntimeController.kt) |
| Risk decisions and feedback policy | [RiskAnalyzer](../core/assist/src/main/java/com/linnan/blindassist/risk/RiskAnalyzer.kt), [FeedbackPlanner](../core/assist/src/main/java/com/linnan/blindassist/feedback/FeedbackPlanner.kt) |
| H3 connection and local hardware runtime | [H3RangeAdapter](../core/device/src/main/java/com/linnan/blindassist/device/glasses/H3RangeAdapter.kt), [HardwareLocalRuntime](../core/device/src/main/java/com/linnan/blindassist/device/glasses/HardwareLocalRuntime.kt) |

Tests live in each owning module's `src/test/` and, where present,
`src/androidTest/`. Run the affected module through
[run_android_gradle.ps1](../scripts/run_android_gradle.ps1); use
[device regression](DEVICE_REGRESSION.md) for device-dependent behavior.

## Research

| Working area | Start here |
| --- | --- |
| Sensor capture, calibration, firmware and mobile-fusion evidence | [Hardware current](../research/active/hardware-bringup/CURRENT.md) → [hardware ledger](../research/active/hardware-bringup/README.md) |
| Forward obstacle perception and retained DTR work | [DTR current](../research/active/dtr-r0/CURRENT.md) → [DTR ledger](../research/active/dtr-r0/README.md) |
| Paused ten-meter goal-completion work | [L10 current](../research/active/l10-r0/CURRENT.md) → [L10 ledger](../research/active/l10-r0/README.md) |

These are stable directory names, not a shared project version or a declaration
that all routes are advancing. Each `CURRENT.md` owns its present status;
the [cross-route decision](CURRENT_DECISION.md) sets priorities.
Closed modules and their exact historical paths are searchable in
[experiments/index.jsonl](../experiments/index.jsonl) and preserved at the archive tag documented in
[history-index.md](history-index.md).

## Stable tooling

- [tools/ba.ps1](../tools/ba.ps1): profile-aware setup, doctor, smoke, real-input materialize,
  synthetic run, and cleanup
- [scripts/run_android_gradle.ps1](../scripts/run_android_gradle.ps1): Android/Gradle execution
- [tools/knowledge.py](../tools/knowledge.py): retrieve route context and reusable research knowledge
- [tools/data/generate_dataset_ledger.py](../tools/data/generate_dataset_ledger.py): regenerate the externalized dataset
  ledger and compact summaries
- [scripts/check_project_structure.ps1](../scripts/check_project_structure.ps1): repository layout check
- [scripts/check_docs_index.ps1](../scripts/check_docs_index.ps1): hot-document navigation check

Environment examples start at [config/local.example.toml](../config/local.example.toml).
For local paths and payload placement, use [Local artifacts](LOCAL_ARTIFACTS.md).
