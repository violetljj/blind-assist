# HFTF D7 public-real dataset

状态：`development / THESIS_DEVELOPMENT / source-intake-and-adjudication`

## 研究问题与版本

`HFTF_D7_PUBLIC_REAL_R1` asks whether legally accessible public real-world
recordings can supply session-disjoint parent events for a YOLO-HFTF relation
selector. This module only builds auditable source/identity/candidate/review
artifacts. It does not change YOLO, HFTF v2, thresholds, confirmation length,
backbones, Android, the default app, or safety authority.

Selection from detector/HFTF/segmentation output is discovery-only. Confirmation
selection must use source-native metadata/geometry or a model-blind random/coverage
rule. Missing source evidence, unresolved license/access, or reviewer disagreement
is preserved as `NOT_EVALUABLE`/`ACCESS_BLOCKED`; it is not converted to a negative.

## Stable interface

Run through the repository adapter:

```powershell
$py = 'E:\codex-tools\bin\blindassist-python.cmd'
& $py scripts/run_research_tool.py hftf-d7-public-real build_registry.py `
  --ledger DATASET_MASTER_LEDGER.csv `
  --candidate-report E:\linnan\linnan\artifacts.local\evidence\candidate-event-mining\<run>\candidate_report.json `
  --source-catalog scripts/research/hftf_d7_public_real/source_catalog.json `
  --output-root F:\ba-data\hftf-d7-public-real `
  --run-id d7-r1-intake
```

The command creates the required directory layout and deterministic JSON/JSONL
registries under the external F: root. Raw media are never copied into Git.

For the public extracted EgoWalk trajectory source, the reproducible sequence
is:

```powershell
& $py scripts/run_research_tool.py hftf-d7-public-real download_egowalk_metadata.py `
  --output-root F:\ba-data\hftf-d7-public-real --run-id d7-r1-egowalk-meta
& $py scripts/run_research_tool.py hftf-d7-public-real ingest_egowalk_metadata.py `
  --output-root F:\ba-data\hftf-d7-public-real --run-id d7-r1-egowalk-coverage
& $py scripts/run_research_tool.py hftf-d7-public-real merge_egowalk_intake.py `
  --output-root F:\ba-data\hftf-d7-public-real --run-id d7-r1-egowalk-merge
& $py scripts/run_research_tool.py hftf-d7-public-real download_egowalk_rgb.py `
  --output-root F:\ba-data\hftf-d7-public-real --run-id d7-r1-egowalk-rgb
```

The RGB download only consumes the public MIT-licensed extracted repository;
the separate raw-recordings repository remains `ACCESS_BLOCKED`. The RGB
receipt does not grant event truth.

Public source probes are similarly receipt-bound and do not promote labels:

```powershell
& $py scripts/run_research_tool.py hftf-d7-public-real inventory_sanpo_public.py `
  --output-root F:\ba-data\hftf-d7-public-real --run-id d7-r1-sanpo-inventory
& $py scripts/run_research_tool.py hftf-d7-public-real download_thor_public.py `
  --output-root F:\ba-data\hftf-d7-public-real --record-id 3382145 `
  --run-id d7-r1-thor-open-tracks --max-bytes 1200000000
```

The SANPO command inventories official GCS object metadata only. The THOR
command is limited to open tracks/LiDAR files; restricted or differently
synchronized video is not silently paired with them.

## Outputs and data roles

The canonical schema is frozen in `contract.json`. Required rows are keyed by
`dataset_id`, `source_session_id`, `ancestry_group`, `frame_id`, and source hashes.
Existing consumed/burned evidence remains Development/diagnostic only. A source
session is the unit of split isolation; frames, windows, cameras, and stereo
views never create independent parent events.

`source_receipts.jsonl` carries a hash for every catalog row. For inaccessible or
metadata-only sources, `source_hash_kind=CATALOG_ACCESS_SNAPSHOT` identifies the
receipt snapshot rather than pretending that source media or event annotations
were materialized; downloaded intake receipts use
`MATERIALIZED_INTAKE_RECEIPT`.

The target counts in the objective are targets, not permission to synthesize or
force class balance. Reports must state the actual accessible count and each
source/category gap.

## Review boundary

The eventual review bundle contains RGB Reviewer A/B/C, Geometry Evidence Reviewer,
Counterexample Reviewer, and Final Adjudicator inputs. RGB reviewers do not see
model outputs or trigger names. Geometry reads only source-native geometry. All
2/3 RGB agreement, geometry conflict, head-positive, and boundary-positive cases
escalate. No event is admitted without pre/alertable/passed phases for positives
or a continuous negative interval for negatives.

`materialize_pending_package.py` creates assignment-only review rows and explicit
`NOT_EVALUABLE` terminals; it must not be read as completed review. Use
`validate_d7_package.py` after every review/adjudication update. The validator
returns `NOT_COMPLETE` until the admitted-event and phase/session gates pass.
Run `audit_role_isolation.py` before any split materialization. An ancestry group
crossing prior roles is a `HOLD_ROLE_REVIEW`, even when source-session identities
are unique.

## Stop conditions

Fail closed for missing identities, non-monotone timestamps, duplicate source
hashes, unresolved session/ancestry overlap, inaccessible/gated sources, invalid
phase intervals, or reviewer output visibility drift. A shortfall below 50,000
candidate windows or 10,000 parent events is reported honestly with the measured
denominator and a source-specific next action.

## Failure asset reuse

Candidate-only and consumed evidence may be reused as Development discovery,
regression, contact-sheet, or counterexample material. It cannot be relabeled as
fresh Confirmation by copying, renaming, or rehashing paths.
