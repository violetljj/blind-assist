# Resource fabric

The project-wide authority for discovery, resolution, consumers, derivations,
and lifecycle is the [BlindAssist asset management system](asset-management/README.md).
This document covers its content-addressed object/cache layer.

BlindAssist uses one content identity for every immutable raw dataset object or
model, then reuses that identity across normalization, feature extraction,
failure mining, regression, and experiments. A Development cohort becoming
consumed ends its fresh-evidence authority; it does not end its engineering
value.

## Data flow

```text
unique raw data/model store
        -> shared normalized and feature cache
        -> reusable failure, hard-case, and evidence-gap library
        -> thin experiment: manifest + parameters + result + evidence boundary
```

The implementation is `tools/data/resource_fabric.py`. It uses only the Python
standard library and writes generated state below ignored `artifacts.local/`.

## Physical layout

The catalog is logically unified while payloads retain their correct storage
and evidence role:

```text
artifacts.local/
  downloads/resource-store/       # reproducible raw data/archive objects
  models/resource-store/          # immutable model objects
  evidence/resource-store/        # irreplaceable or sealed source objects
  work/resource-cache/
    normalized/                    # deterministic normalized observations
    features/                      # reusable model/geometry features
  evidence/resource-fabric/
    catalog/                       # object, registration, cache, lifecycle records
    hard-cases/                    # slices referencing objects/caches, no media copies
    experiments/                   # thin experiment directories
    reports/current/               # generated live utilization report
```

The resource id is `sha256:<digest>` for one file and
`tree-sha256:<digest>` for a directory. Different paths containing identical
bytes resolve to the same object. Registrations and lifecycle events are
append-only, so one object can retain multiple provenance and consumer records
without duplicating its payload.

## Lifecycle and legal reuse

Evidence authority and storage lifecycle are separate axes:

| Evidence status | Meaning |
| --- | --- |
| `reserved` / `fresh` | Outcome has not been opened within the declared evidence unit |
| `development_consumed` | Outcome has been seen; it can never become fresh again |
| `sealed_final` | Frozen final/blind evidence |
| `diagnostic` / `unknown` | Diagnostic-only or not yet classified |

| Storage status | Meaning |
| --- | --- |
| `active` / `shared` | Current consumer or reusable across consumers |
| `sealed_cold` | Preserved evidence or option asset, excluded from routine compute |
| `rebuildable` | Reconstructible payload with retained provenance |
| `unknown` | Ownership or recovery path still needs classification |

The default `development_consumed` transition explicitly allows diagnostics,
training, shared feature caching, hard-case mining, regression, and Development
replay. It explicitly forbids fresh confirmation, generalization claims, and
safety claims. `UNKNOWN` and `NOT_EVALUABLE` remain distinct from a negative
case.

## Commands

Run commands from the repository root with the supported project Python:

```powershell
E:\codex-tools\bin\python.cmd tools\data\resource_fabric.py `
  ingest artifacts.local\tmp\my-download\payload.zip `
  --name my-dataset-v1 --kind data --storage-class download `
  --route l10-r0 --consumer my-evaluator --evidence-role raw-observation `
  --owner l10-r0 --retention-reason "open source gap" `
  --evidence-status reserved --storage-status active `
  --reason "new source for the open gap" `
  --source-uri https://example.invalid/payload.zip --license-id dataset-license
```

`copy` is the safe default. For a completed immutable staging payload already
on the F: artifact volume, `--mode hardlink` creates the canonical name without
duplicating data blocks. Only remove the staging name after `verify --deep` and
after confirming no active caller owns that exact path.

Create a deterministic shared cache:

```powershell
E:\codex-tools\bin\python.cmd tools\data\resource_fabric.py `
  cache-put --layer features --source-id sha256:<digest> `
  --transform dino-crop --transform-version rev-123 `
  --parameters-json '{"size":518}' --producer research/active/l10-r0/producer.py `
  --payload artifacts.local\tmp\my-task\features.npz
```

Consumers resolve the cache through an auditable hit instead of opening the
producer experiment directory directly:

```powershell
E:\codex-tools\bin\python.cmd tools\data\resource_fabric.py `
  cache-use <cache-key> --event-id l10-evaluator-feature-hit `
  --consumer l10-evaluator-v2 --purpose feature-input `
  --experiment-id l10-evaluator-v2
```

The returned `resolved_payload` is the shared payload path. Replaying the same
event id is idempotent; a different consumer or run uses a new event id.

Register a consumed failure slice without copying its frames:

```powershell
E:\codex-tools\bin\python.cmd tools\data\resource_fabric.py `
  hard-case --id wrong-door-reacquisition-v1 --route l10-r0 `
  --case-kind failure --failure-layer exact-instance-binding `
  --evidence-split development --source-id sha256:<digest> `
  --selector-json '{"episode_ids":["case-17"]}' `
  --truth-authority frozen-private-evaluator `
  --selected-by frozen-failure-attribution-v1 `
  --observed-outcome WRONG_INSTANCE `
  --claim-ceiling "Consumed Development hard case only"
```

Create and finish a thin experiment:

```powershell
E:\codex-tools\bin\python.cmd tools\data\resource_fabric.py `
  experiment-create --id l10-example-v1 --route l10-r0 `
  --question "Does the new observation resolve the named gap?" `
  --evaluator research/active/l10-r0/evaluate.py `
  --source-id sha256:<digest> --cache-key <cache-digest> `
  --parameters-json '{"frozen":true}' `
  --boundary "Development mechanics only; not fresh confirmation or safety evidence."

E:\codex-tools\bin\python.cmd tools\data\resource_fabric.py `
  experiment-finalize --id l10-example-v1 --route l10-r0 `
  --result-json artifacts.local\tmp\my-task\result.json --status GATE_NOT_MET
```

The experiment directory may contain only `manifest.json`, `parameters.json`,
`result.json`, and `evidence-boundary.md`. Heavy predictions, tensors, frames,
and checkpoints must be resources or caches referenced by id.

After an outcome, preserve its future value and close its authority in one
append-only transition:

```powershell
E:\codex-tools\bin\python.cmd tools\data\resource_fabric.py `
  transition sha256:<digest> --event-id l10-example-consumed `
  --evidence-status development_consumed --storage-status shared `
  --experiment-id l10-example-v1 `
  --reason "Development result opened"
```

Generate and verify the live ledger:

```powershell
E:\codex-tools\bin\python.cmd tools\data\resource_fabric.py verify --deep
E:\codex-tools\bin\python.cmd tools\data\resource_fabric.py `
  report --inventory-root artifacts.local
```

Omit `--inventory-root` for the fast daily report. The full logical inventory
walks every physical file below the artifact target and is intended for
migration milestones; files that disappear during an active build are counted
as vanished entries instead of aborting the report. The fast report also shows
`cache_access_events`, `multi_consumer_caches`, and
`avoided_recompute_bytes`; these measure recorded reuse, not merely registered
cache capacity.

## Existing assets

Migration is incremental and non-destructive:

1. Register an exact immutable source and its current consumer.
2. Move repeated normalization and features behind deterministic cache keys.
3. Turn consumed failures and gaps into reference-only hard cases.
4. Point the next experiment at resource, cache, and case ids.
5. Only after the old caller is retired, classify its legacy payload as sealed,
   cold, or rebuildable. Deletion remains a separate manifest-owned operation.

Do not bulk-move current route data, rewrite sealed paths, or infer that a file
is disposable from age, extension, or a failed result. This fabric changes
reuse and identity first; storage reclamation follows only from verified
lineage and explicit target-level authorization.

The first non-destructive adoption was the existing L10 SEVN metadata/action
adapter. `research/active/l10-r0/l10_sevn_panolab_source_v1.json` resolves to
`sha256:a6478c29610f986fc50dd101de378f187f7ba9e69c201cc31d3b12b6cf7912da`;
its canonical JSON cache key is
`ca8ed54969814e315c6b279a8943dd62e502b3139744b18d74e201a950451d76`.
The full SEVN source is now available as `datasets/sevn`: its six files are
same-volume hardlinks to the legacy `F:\ba-data\SEVN` names, so both locators
address the same `29,684,140,135` logical bytes with zero duplicate payload
bytes. The content identity is
`tree-sha256:2b2c9f42ec4daf217af9287b70ef04312498429c6918905f82b47588c9d427b3`.
The earlier missing-image case remains a historical gap observation; it is not
the current availability state. The cohort remains `development_consumed`;
this adoption adds no fresh-confirmation, generalization, or safety authority.

SEVN and CARLA composite assets also expose component-level selectors through
the master catalog. For example, consumers resolve
`datasets/sevn#high-resolution-panoramas`,
`datasets/sevn#navigation-graph`, or
`runtime/carla-asset-library#dtr-carla-c4-multimap`. A selector records the
actual component role and boundary in the usage event while retaining one
zero-copy physical parent. The tracked semantic profiles are
`data/asset-profiles/sevn-1.0.json` and
`data/asset-profiles/carla-0.9.16-local.json`.
