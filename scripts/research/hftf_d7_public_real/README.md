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

Once a specific SANPO session is selected, a bounded public-media canary may be
downloaded with provider MD5 verification. Nominal-time derivation is opt-in:
the command below records `nominal_time_ns = frame_index / 15 FPS` together
with the official source contract. It does not create an authoritative
`timestamp_ns`, because the public pose CSV has no timestamp or proven
pose-row/frame binding. It still does not infer event labels; without the
explicit `--fps` argument, the material remains untimed intake evidence:

```powershell
& $py scripts/run_research_tool.py hftf-d7-public-real download_sanpo_public.py `
  --output-root F:\ba-data\hftf-d7-public-real `
  --run-id d7-r1-sanpo-media-canary `
  --session-id=-5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG `
  --camera chest --view left --start-frame 0 --frame-count 60 --max-bytes 600000000 `
  --fps 15
```

THÖR-MAGNI's current public Zenodo record is a 22+ GB ZIP.  Inspect its
central directory with bounded HTTP ranges first, then extract only named
members.  The extractor verifies the local ZIP header, CRC, compressed size,
and local SHA-256; it never materializes the full archive:

```powershell
& $py scripts/run_research_tool.py hftf-d7-public-real inspect_thor_magni_archive.py `
  --output-root F:\ba-data\hftf-d7-public-real `
  --record-id 13865754 --run-id d7-r1-thor-magni-range-metadata
& $py scripts/run_research_tool.py hftf-d7-public-real materialize_thor_magni_members.py `
  --output-root F:\ba-data\hftf-d7-public-real `
  --manifest F:\ba-data\hftf-d7-public-real\manifests\<range-manifest>.jsonl `
  --run-id d7-r1-thor-magni-selected-members `
  --member THOR_MAGNI/CSVs_Scenarios/Scenario_1/<run>.csv `
  --member THOR_MAGNI/MP4_Videos/Files/<scene>.mp4
```

For an extracted THÖR-MAGNI run, `materialize_thor_magni_windows.py` uses QTM
`Frame`/`Time` at 100 Hz for non-overlapping four-second windows, then records
the actual `SceneFNr` values, missing frames, duplicate QTM rows, and centroid
coverage.  It emits a separate source manifest and does not merge or label the
D7 top-level event registry:

```powershell
& $py scripts/run_research_tool.py hftf-d7-public-real materialize_thor_magni_windows.py `
  --output-root F:\ba-data\hftf-d7-public-real `
  --scenario-csv F:\ba-data\hftf-d7-public-real\raw\<scenario>.csv `
  --rgb-path F:\ba-data\hftf-d7-public-real\raw\<scene>.mp4 `
  --device PPL --qtm-fps 100 --run-id d7-r1-thor-magni-window-intake
```

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

For THOR-MAGNI, use the receipt-bound `merge_thor_magni_candidate_intake.py`
adapter after `materialize_thor_magni_windows.py`. Pass exactly one canonical
window artifact per source-session; alternate reruns of the same session are
metadata alternatives, not independent events. The adapter binds the window
manifest to its window receipt, selected-member receipt, archive checksum,
source license, candidate/frame hashes, and local member hashes. It allows
same-session QTM-to-SceneFNr frame reference reuse, but rejects overlapping
source-time windows, candidate/frame/session/ancestry/event collisions, and
partial post-merge surfaces. Run it once with `--dry-run` before the append.
All resulting THOR rows remain assignment-only `NOT_EVALUABLE` until the same
five-role review and final-adjudication chain below completes.

For a lawful RGB/geometry pilot, first materialize a fresh, role-isolated input
bundle. The extractor is currently limited to EgoWalk's public extracted RGB
and source-native pose pilot; it fails closed on missing local media and never
writes a label:

```powershell
& $py scripts/run_research_tool.py hftf-d7-public-real materialize_review_bundle.py `
  --output-root F:\ba-data\hftf-d7-public-real `
  --run-id d7-r1-review-bundle-pilot `
  --batch-id d7-r1-egowalk-review-pilot `
  --dataset-id EgoWalk --offset 0 --count 5 `
  --roles RGB_REVIEWER_A,RGB_REVIEWER_B,RGB_REVIEWER_C,GEOMETRY_EVIDENCE_REVIEWER,COUNTEREXAMPLE_REVIEWER
```

Each reviewer must write one completed record into its own bundle directory.
Only after all selected role outputs pass the bundle firewall may they be
ingested into the primary review files. The ingest command backs up the prior
assignment-only files and still writes no adjudicated event:

```powershell
& $py scripts/run_research_tool.py hftf-d7-public-real ingest_review_outputs.py `
  --output-root F:\ba-data\hftf-d7-public-real `
  --run-id d7-r1-review-ingest-pilot `
  --batch-id d7-r1-egowalk-review-pilot
```

For the extracted EgoWalk videos, the pose parquet is the physical timeline
and each pose row binds to one video ordinal. The container advertises a 100 Hz
playback rate while the pose timeline is 5 Hz, so review contact sheets use
pose-row ordinals converted to container seconds; they must not seek by the
pose timestamps directly. `materialize_review_bundle.py` records the selected
ordinal frame indices in every RGB assignment.

If a separately frozen, hash-bound EgoWalk depth root is explicitly in scope
for Development review, augment an untouched bundle before any role writes an
output:

```powershell
& $py scripts/run_research_tool.py hftf-d7-public-real augment_egowalk_depth_evidence.py `
  --output-root F:\ba-data\hftf-d7-public-real `
  --batch-id <untouched-egowalk-batch> `
  --run-id <depth-evidence-run> `
  --media-root <hash-bound-egowalk-media-root> `
  --ffmpeg-path E:\codex-tools\ffmpeg-8.1.2-full_build-shared\ffmpeg-8.1.2-full_build-shared\bin\ffmpeg.exe
```

The augmentation is source-native and model-blind, but a media root marked
consumed/burned by another frozen protocol is Development-only and cannot
receive fresh Confirmation credit. Depth previews are descriptive aids; they
do not create segmentation, event truth, or an admission.

After all five independent roles are ingested, materialize a final-adjudicator
bundle and ingest only its terminal outputs. The adjudicator bundle exposes all
raw review records and copied evidence, but still has no model discovery fields:

```powershell
& $py scripts/run_research_tool.py hftf-d7-public-real materialize_adjudication_bundle.py `
  --output-root F:\ba-data\hftf-d7-public-real `
  --run-id d7-r1-adjudication-bundle-pilot `
  --batch-id d7-r1-egowalk-review-pilot
& $py scripts/run_research_tool.py hftf-d7-public-real ingest_adjudications.py `
  --output-root F:\ba-data\hftf-d7-public-real `
  --run-id d7-r1-adjudication-ingest-pilot `
  --batch-id d7-r1-egowalk-review-pilot
```

Legacy reviewer files may be canonicalized only with an immutable role
manifest. A negative support interval recorded in frame/time-from-source-start
fields may be bound to the manifest's full candidate window; incomplete positive
phases are downgraded to `NOT_EVALUABLE` rather than interpolated:

```powershell
& $py scripts/run_research_tool.py hftf-d7-public-real normalize_completed_review_fields.py `
  --path <role>\completed_review.jsonl --expected-count <N> `
  --role <role> --manifest-path <bundle>\manifests\<role>.jsonl `
  --canonicalize-completed-review --bind-support-intervals-from-manifest `
  --downgrade-incomplete-support
```

When all five roles are ingested and the source-native geometry role is
uniformly `NOT_EVALUABLE`,
`materialize_conservative_adjudication.py` can produce only fail-closed
`NOT_EVALUABLE`/`NOT_ADMITTED` terminals. It cannot create a positive or
negative event, and it must be followed by `ingest_adjudications.py`.

Generate the required twelve-item status report from the current machine-readable
artifacts, then run the validator once more so its artifact hashes bind the final
report:

```powershell
& $py scripts/run_research_tool.py hftf-d7-public-real materialize_final_report.py `
  --output-root F:\ba-data\hftf-d7-public-real --run-id d7-r1-final-report
& $py scripts/run_research_tool.py hftf-d7-public-real validate_d7_package.py `
  --output-root F:\ba-data\hftf-d7-public-real --run-id d7-r1-validation-final
```

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
