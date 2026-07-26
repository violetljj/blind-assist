# Egomotion-compensated looming research

状态：`FROZEN_RCLE_PRECURSOR + RCLE_MINIMAL_PHASE_A_R0_REVISE_VALID`

This module is an offline, research-only boundary for
`EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R0`.

The original R0/R1 program is stopped. The module is retained as reproducible
precursor evidence for the current RCLE mainline; it does not authorize an
R1 successor or count as RCLE-Minimal Phase A. New Phase A work must use a
separate `rcle_minimal` submodule and its own synthetic truth and gates.

RCLE-Minimal Phase A now exists only in the isolated `rcle_minimal/` submodule
and `run_synthetic_signal_audit_r0.py`. Its result is `REVISE / VALID`: clean
rotation suppression and closing retention passed, while preregistered
per-condition coverage failed. This does not unfreeze or import the precursor
R0/R1 program.

## RCLE-Minimal Phase A

The machine protocol was locked before the formal run with SHA-256
`d20e77f3ea5f7ac55376006f1d14feb0ffb5daffd10a42792912fb89cdb1b502`.
The formal `2520`-trial run and independent receipt validation are under:

```text
artifacts.local/evidence/rcle_minimal_r0/formal_run_r0/
artifacts.local/datasets/rcle_minimal_r0/formal_run_r0/
```

Validate without rerunning candidates:

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts\research\egomotion_compensated_looming\run_synthetic_signal_audit_r0.py `
  --validate-existing `
  --output-root artifacts.local\evidence\rcle_minimal_r0\formal_run_r0 `
  --dataset-root artifacts.local\datasets\rcle_minimal_r0\formal_run_r0
```

Receipt SHA-256:
`14ed23e38bacc913207aaa56903a7b2cd3bebe52631338c4760f02dc5c2041ca`.
Phase B and Replay Demo remain closed. The only next research boundary is a
versioned implementation-side Phase A coverage revision with the R0 protocol,
trials, thresholds, and negative coverage evidence unchanged.

## 冻结环境

从仓库根目录使用 `E:\codex-tools\bin\blindassist-python.cmd`。本次完整验证实际使用 Python 3.11.9、NumPy 2.1.3、OpenCV 4.13.0.92 和 Pillow 12.2.0；第三方包冻结在 `requirements-frozen.txt`。

默认 Python 3.14 环境可能缺少 OpenCV、NumPy 或 Pillow，并让部分测试跳过但仍返回成功，不能作为完整验证。完整测试必须显示 `Ran 43 tests` 和 `OK`。

## 稳定 Interface

Frozen reproducibility scope:

- validate the metadata-only source-authority inventory;
- validate the old-window clean-room denylist receipt;
- record the already-materialized Bonn base/oracle traces and the quarantined
  non-authoritative truth join while keeping claim authority, role promotion,
  threshold selection, App wiring, route fields, and lifecycle logic closed;
- reproduce the AV2/CODa metadata-only source-boundary receipts and the
  non-terminal priority-source summary.

The stable entry point is:
`validate_source_authority_inventory_r0.py`. Inputs and generated research
receipts live under
`artifacts.local/evidence/ustrf/egomotion_compensated_looming_r0/`.

AV2 audits only lidar-filename to camera-filename timing; annotation-to-camera
coverage remains `NOT_EVALUATED`. CODa keeps unbound TACC continuity separate
from checksum-bound TDR identity and verifies every HTTP Range response.

## 输出

The validator emits one compact JSON status line. It does not create or mutate
source admission, role splits, candidate results, or product artifacts.

The frozen claim-scoped R1 source-program validator is:

```powershell
python scripts/research/egomotion_compensated_looming/validate_r1_claim_scoped_source_program_r0.py
```

The controlled-capture protocol and canonical source program can be checked
together with:

```powershell
python scripts/research/egomotion_compensated_looming/validate_claim_scoped_r1_freeze.py
```

R1-A freezes three future discovery clusters and 84 rigid-target trials. It
does not capture validation/holdout or authorize a deployable estimator.
Separately, a concurrent Bonn execution decoded 598 discovery RGB members,
froze 596 base-flow pair traces, and then computed 594 oracle-rotation and
full-6DoF traces using pose and source depth. The source-program receipt records
those actual executions and the later diagnostic join as
`truth_join_or_scoring_run=true`, while enforcing
`authoritative_algorithm_result_available=false`.

The frozen Bonn R1 discovery path is metadata/role freeze, archive integrity,
then pose-mechanics ledger. Validation and holdout archives must remain absent:

```powershell
python scripts/research/egomotion_compensated_looming/freeze_bonn_claim_scoped_inventory_r0.py `
  --official-page artifacts.local/datasets/egomotion_compensated_looming_r1/bonn_metadata_r0/official_page.html `
  --output artifacts.local/evidence/ustrf/egomotion_compensated_looming_r1/bonn_claim_scoped_inventory_and_role_freeze_r0.json

python scripts/research/egomotion_compensated_looming/audit_bonn_discovery_archives_r0.py `
  --freeze artifacts.local/evidence/ustrf/egomotion_compensated_looming_r1/bonn_claim_scoped_inventory_and_role_freeze_r0.json `
  --archive-dir artifacts.local/datasets/egomotion_compensated_looming_r1/bonn_discovery_r0 `
  --output artifacts.local/evidence/ustrf/egomotion_compensated_looming_r1/bonn_discovery_archive_audit_r0.json

python scripts/research/egomotion_compensated_looming/build_bonn_discovery_pose_cell_ledger_r0.py `
  --acquisition artifacts.local/evidence/ustrf/egomotion_compensated_looming_r1/bonn_discovery_archive_audit_r0.json `
  --archive-dir artifacts.local/datasets/egomotion_compensated_looming_r1/bonn_discovery_r0 `
  --output artifacts.local/evidence/ustrf/egomotion_compensated_looming_r1/bonn_discovery_pose_cell_ledger_r0.json
```

The static-map truth audit is a separate, preregistered endpoint. It reads all
PLY point records once, uses a deterministic coordinate-hash sample, and may
decode only the six depth members frozen by
`freeze_bonn_transform_validation_samples_r0.py`. It never reads RGB or runs a
candidate signal:

```powershell
python scripts/research/egomotion_compensated_looming/build_bonn_static_map_geometry_r0.py `
  --acquisition artifacts.local/evidence/ustrf/egomotion_compensated_looming_r1/bonn_static_map_acquisition_r0.json `
  --archive artifacts.local/datasets/egomotion_compensated_looming_r1/bonn_discovery_r0/rgbd_bonn_groundtruth_1mm_section.zip `
  --official-page artifacts.local/datasets/egomotion_compensated_looming_r1/bonn_metadata_r0/official_page.html `
  --official-transform-script artifacts.local/datasets/egomotion_compensated_looming_r1/bonn_metadata_r0/compute_global_transformation.py `
  --output-points artifacts.local/datasets/egomotion_compensated_looming_r1/bonn_discovery_r0/rgbd_bonn_groundtruth_hash64_r0.npz `
  --receipt artifacts.local/evidence/ustrf/egomotion_compensated_looming_r1/bonn_static_map_geometry_r0.json

python scripts/research/egomotion_compensated_looming/validate_bonn_static_surface_truth_ledger_r0.py `
  --receipt artifacts.local/evidence/ustrf/egomotion_compensated_looming_r1/bonn_static_surface_truth_ledger_recheck.json `
  --points artifacts.local/datasets/egomotion_compensated_looming_r1/bonn_discovery_r0/rgbd_bonn_groundtruth_hash64_r0.npz
```

The frozen result is
`BONN_C2_STATIC_SURFACE_TRANSFORM_CANARY_FAILED / VALID`: three usable depth
frames agree numerically, but the four-frame quorum did not close. Do not
lower the quorum or resample in the same round.

A concurrent central-ROI/full-frame ledger is retained at
`bonn_static_surface_truth_ledger_r0.json`. It does not implement the
preregistered grid/500ms unit contract and self-reports grade A for a derived
map projection. The source-program authority review therefore keeps it
diagnostic-only; it cannot authorize R1-A signal execution or Bonn C2
confirmation.

The existing base and oracle trace receipts are retained. A concurrent process
subsequently joined them to the quarantined central-ROI truth proxy and emitted
a self-reported stop. That evaluation is itself quarantined: it compares a
global-image q90 signal with a central-ROI q05 static-depth rate, so the signal
and truth spatial units are not aligned. Its only allowed claim is that the
current global summary has weak and session-unstable association with that
exploratory proxy. It cannot stop R1-A, oracle rotation, or local expansion and
cannot count toward Bonn C2 confirmation.

## 冻结脚本清单

以下脚本保留是为了重放已经发生的 R0/R1 过程，不构成新执行队列：

| 脚本 | 冻结角色 |
| --- | --- |
| `audit_av2_join_and_cell_mechanics_r0.py` | AV2 metadata/timestamp join 边界审计 |
| `audit_coda_tiny_continuity_r0.py` | CODa TACC/TDR identity 与 continuity 边界审计 |
| `freeze_bonn_r1a_rgb_pair_manifest_r0.py` | 冻结已执行 Bonn discovery pair manifest |
| `freeze_r1a_oracle_signal_contract_r0.py` | 冻结 base/oracle/full-6DoF trace 合同 |
| `produce_bonn_r1a_base_flow_traces_r0.py` | 重放已保留的 base Farneback traces |
| `produce_bonn_r1a_oracle_flow_traces_r0.py` | 重放已保留的 oracle/full-6DoF traces |
| `build_bonn_static_surface_truth_ledger_r0.py` | 重放已隔离的 central-ROI truth proxy |
| `evaluate_bonn_r1a_continuous_signal_r0.py` | 只重放已隔离的空间不对齐 diagnostic，不得生成 RCLE 结论 |

完整重放需要 `artifacts.local/datasets/egomotion_compensated_looming_*` 和 `artifacts.local/evidence/ustrf/egomotion_compensated_looming_*` 中的 ignored 输入。单元测试和 validator 通过不代表 fresh clone 已具备这些输入。

## 安全边界

The old 15 positive / 15 negative LILocBench windows and the later canonical
CrowdBot/LILocBench sequence inventory are exclusion inputs only. Their frames,
outcomes, candidate traces, ranks, and thresholds must not be made available to
source admission or signal producers.

No file in this module may wire into App, Kotlin, YOLO, route, lifecycle,
feedback, shadow, human-safety, or production paths.

## 停止条件

The following conditions apply only when reproducing the stopped R0/R1 source
program. They do not govern the new RCLE-Minimal Phase A.

Stop before signal work whenever any required source/session/cell receipt is
missing, the clean-room firewall is invalid, or fewer than three real source
families can satisfy the frozen per-role denominator. Synthetic sources cannot
repair that denominator.

Run the focused validator from the repository root:

```powershell
python scripts/research/egomotion_compensated_looming/validate_source_authority_inventory_r0.py
```

Build the non-terminal three-source boundary summary:

```powershell
python scripts/research/egomotion_compensated_looming/build_priority_public_source_summary_r0.py `
  --audit-dir artifacts.local/evidence/ustrf/egomotion_compensated_looming_r0/source_audit `
  --output artifacts.local/evidence/ustrf/egomotion_compensated_looming_r0/source_audit/priority_public_source_summary_r0.json
```

The AV2 metadata-only probe lists S3 prefixes and server object metadata without
fetching object payloads:

```powershell
python scripts/research/egomotion_compensated_looming/probe_av2_official_inventory_r0.py `
  --output artifacts.local/evidence/ustrf/egomotion_compensated_looming_r0/source_audit/av2_official_inventory_r0.json
```

The completed HOT3D metadata-only audit verifies whether public five-second
clips can form same-sequence ten-second units. It reads no clip tar or image
bytes. HOT3D is not selected by the frozen R1 source program, so this command is
reproduction-only and must not be followed by payload acquisition:

```powershell
python scripts/research/egomotion_compensated_looming/audit_hot3d_clips_continuity_r0.py `
  --definitions artifacts.local/datasets/egomotion_compensated_looming_r0/hot3d_clips_metadata_r0/clip_definitions.json `
  --splits artifacts.local/datasets/egomotion_compensated_looming_r0/hot3d_clips_metadata_r0/clip_splits.json `
  --output artifacts.local/evidence/ustrf/egomotion_compensated_looming_r0/source_audit/hot3d_clips_continuity_and_authority_terminal_r0.json
```

The ADT prescreen freeze selects only `main_groundtruth` members from official
metadata. Activity strata are search proxies, not counterfactual-cell truth:

```powershell
python scripts/research/egomotion_compensated_looming/freeze_adt_groundtruth_prescreen_r0.py `
  --per-stratum 4 `
  --output artifacts.local/evidence/ustrf/egomotion_compensated_looming_r0/source_audit/adt_groundtruth_prescreen_freeze_r0.json
```

Only after that freeze, acquire the selected ground-truth ZIPs under the
800 MiB hard cap:

```powershell
python scripts/research/egomotion_compensated_looming/acquire_adt_groundtruth_prescreen_r0.py `
  --freeze artifacts.local/evidence/ustrf/egomotion_compensated_looming_r0/source_audit/adt_groundtruth_prescreen_freeze_r0.json `
  --output-dir artifacts.local/datasets/egomotion_compensated_looming_r0/adt_groundtruth_prescreen_r0 `
  --receipt artifacts.local/evidence/ustrf/egomotion_compensated_looming_r0/source_audit/adt_groundtruth_prescreen_acquisition_r0.json
```

After the geometry preregistration has passed independent review, create
groundtruth-only proposals:

```powershell
python scripts/research/egomotion_compensated_looming/run_adt_geometry_cell_prescreen_r0.py `
  --freeze artifacts.local/evidence/ustrf/egomotion_compensated_looming_r0/source_audit/adt_groundtruth_prescreen_freeze_r0.json `
  --acquisition artifacts.local/evidence/ustrf/egomotion_compensated_looming_r0/source_audit/adt_groundtruth_prescreen_acquisition_r0.json `
  --output artifacts.local/evidence/ustrf/egomotion_compensated_looming_r0/source_audit/adt_geometry_cell_proposals_r0.json
```
