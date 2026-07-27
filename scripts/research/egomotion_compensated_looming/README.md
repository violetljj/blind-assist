# Egomotion-compensated looming research

状态：`R1_FROZEN / EXTERNAL_COHORT_NOT_EVALUABLE`

This module is an offline, research-only boundary for
`EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R0`.

`rcle_unseen_external_confirmation_r0/` is the pure data-layer contract for
the next cross-source confirmation. It derives the unchanged old trigger and
causal three-pair R1 trigger from one ordered pair ledger, validates the exact
two-source/four-window cohort, and evaluates every local gate without pooled
rescue. Its dedicated 17-test suite passes. It does not select windows, decode
RGB, execute the estimator, or write a formal claim. Source discovery is
closed as `EXTERNAL_COHORT_NOT_EVALUABLE`, so no external payload or RGB
outcome is authorized in that evidence version.

The `rgb_algorithm_cid_sims_floor3_2_cross_sequence_holdout_r0/` module is a
separate 8-worker geometry-first development holdout for the different
official `floor3_2` run. Its frozen selection found 17 positive, zero
below-reference and one ambiguous window, so no RGB member bytes were read and
the unchanged RGB algorithm was not run. The formal validator terminal remains
INVALID because of an exact Decimal/float median representation check; the
bounded post-hoc R1 audit validates the immutable evidence but does not rewrite
that terminal.

The original R0/R1 program is stopped. The module is retained as reproducible
precursor evidence for the current RCLE mainline; it does not authorize an
R1 successor or count as RCLE-Minimal Phase A. New Phase A work must use a
separate `rcle_minimal` submodule and its own synthetic truth and gates.

RCLE-Minimal Phase A is version-isolated in `rcle_minimal/` (R0) and
`rcle_minimal_r1/` (the single permitted coverage revision). R1 retained the
R0 protocol and complete trial inventory, but partial-occlusion pitch worst-cell
coverage remained `0.60 < 0.70`. The machine result is `REVISE / VALID`; the
frozen one-revision stop semantics make the research terminal
`STOP_CURRENT_IMPLEMENTATION / VALID`.

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

The single R1 revision is under:

```text
artifacts.local/evidence/rcle_minimal_r1/formal_run_r1/
artifacts.local/datasets/rcle_minimal_r1/formal_run_r1/
```

Validate it without rerunning:

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts\research\egomotion_compensated_looming\run_synthetic_signal_audit_r1.py `
  --validate-existing `
  --output-root artifacts.local\evidence\rcle_minimal_r1\formal_run_r1 `
  --dataset-root artifacts.local\datasets\rcle_minimal_r1\formal_run_r1
```

R1 receipt SHA-256:
`d5edb9528abfa6d79b973bddfed5f4234795262fb303258c9e1a9e2628ca2b15`.
该 R1 本身不开放后继，且禁止第二次 coverage revision。后续 Observable Support
Recovery sealed validation 已另行 PASS；B0/B1 R5 历史边界见下文。当前只开放
新的 Phase B Progressive Discovery，RGB algorithm metric 与 Replay 仍关闭。

## 冻结环境

从仓库根目录使用 `E:\codex-tools\bin\blindassist-python.cmd`。本次完整验证实际使用 Python 3.11.9、NumPy 2.1.3、OpenCV 4.13.0.92 和 Pillow 12.2.0；第三方包冻结在 `requirements-frozen.txt`。

默认 Python 3.14 环境可能缺少 OpenCV、NumPy 或 Pillow，并让部分测试跳过但仍返回成功，不能作为完整验证。本轮完整 Module 测试必须显示 `Ran 67 tests` 和 `OK`。

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

## RCLE Phase B Bonn metadata 入口

`run_phase_b_bonn_metadata_gate_r0.py` 是锁定的 metadata-only authority /
cohort admission gate。它只读取 hash-bound 官方 metadata HTML，保留全部 26 条
分母，排除 9 条历史 identity，并确定性选择 6 条。R0 receipt 内容可复算，
但 runner override 未强制 one-run/canonical-output，已按
`EXECUTION_CONTRACT_FAIL` 关闭；不得重跑或用于 archive 下载、payload decode、
window inventory、Phase B 指标、Replay、Android、人体、安全或生产工作。

R1/R2 是 preclaim-order diagnostic failures。Canonical R3 使用 hash-bound
minimal bootstrap runner，在任何 project/control/metadata read 前先
exclusive-create 并 fsync 唯一 claim。正式 R3 receipt
`05a283b84f62bee000447bb567eadd63b424afaa9d81f5f0d83d36a9ed02489b`
已独立复算 `VALID`。原 Phase B B0 R0 因 claim 前六次无 body HEAD 的执行合同
违规而关闭；没有 formal run 或 payload body。版本化 B0 R1 保持固定 cohort、
URL、分母和门；design/implementation review PASS 后，唯一 canonical run 与
independent validator 均通过。B0 receipt `dc0ffe9a…1f86`，6/6 sequence 共固定
10 个 10 秒窗口。随后冻结的 B1 R5 与 B1A implementation 已完成一次性执行，
但 independent replay 在 24 个 abstaining pair 的 blank-grid 序列化上发现
producer/validator key-set mismatch，终态为 `INVALID_EXECUTION_CLOSE_B1`。
原 artifacts、锁和源必须保留，不得 patch 或重跑；B1B 与其依赖的正式 metrics
均关闭。随后项目已采用渐进式研究治理：B1 R5 evidence version 关闭但 RCLE
科学问题开放，当前只允许新的 Phase B Discovery 做来源画像、约束质疑、失败资产
回归复用和 hypothesis-driven source-native geometry；算法 canary/confirmation
仍关闭。

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

## PB-H1 role proxy Discovery

当前 Progressive Discovery 的首个真正几何实验位于
`pb_h1_role_proxy/`。它比较 raw speed、pose+depth translation-induced radial
expansion 与 time-normalized parallax，只使用受控 fixture 和 deterministic
burned Bonn 首窗，不读取 RCLE RGB algorithm outcome。

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts\research\egomotion_compensated_looming\run_pb_h1_role_proxy_r0.py

E:\codex-tools\bin\blindassist-python.cmd `
  scripts\research\egomotion_compensated_looming\run_pb_h1_role_proxy_r0.py `
  --validate-existing
```

结果为 `SUPPORT / VALID`；这只支持下一次 TUM `fr2/rpy`
metadata/pose/depth geometry audit，不开放 algorithm canary 或 confirmation。

## Real-data geometry canary R0

`real_data_geometry_canary_r0/` 已完成版本隔离 producer、独立 validator、严格
output schema、fixture/mutation tests 与 one-shot runner。实现锁
`0d833b835d242468fe8c466414882044c3717e8f0b16d6d79a6b5f112e1e2387`
通过 exact hash review；唯一正式执行与独立复算已
`VALID_IMPLEMENTATION_DEBUGGED_GEOMETRY_INTERFACE_ONLY`。该结果只证明 interface
readiness，不是 RGB algorithm outcome 或有效性证据。

## RGB algorithm canary R0 design-only tooling

`rgb_algorithm_canary_r0/` 只提供 F1 设计包的 outcome firewall 和 synthetic
mutation validator；`tests_rgb_algorithm_canary_r0/` 覆盖角色重叠、outcome
泄漏、source/identity/access/pair-ledger 漂移、缺字段、顺序、数值、summary、
cache，以及 progress timestamp/phase/status/PID/ETA/freshness/hash 恶意反例。
第三轮独立只读设计审查已 `PASS`。目录内没有
producer、formal validator、runner、cache materializer 或 implementation lock。

当前设计冻结 TUM windows `0/3/6` 的 raw-flow versus rotation-compensated
local-expansion paired comparator，window `4` 只作 abstention/interface stress。
真实 positive approach role 仍缺失，因此状态保持
`HOLD_ALGORITHM_CANARY_APPROACH_ROLE_INCOMPLETE / VALID /
EXECUTION_NOT_AUTHORIZED`。

## Real positive approach role admission R0

`real_positive_approach_role_admission_r0/` 只处理 geometry-only 数据角色准入。
它在任何 EVIMO2 source access 前 exclusive-create/fsync claim，并把唯一来源
冻结为 `EVIMO2 v2 / Flea3 / sanity_ll`。正式审计只读取
`dataset_info.npz + dataset_depth.npz`；classical RGB、events、mask 与 RCLE
algorithm outcome 均未读取。

13 条 sequence 的首个固定非重叠 10 秒窗共 `3895` pair；独立全量 replay
mismatch 为 `0`，但没有窗口同时通过冻结门，终态为
`HOLD_ALGORITHM_CANARY_APPROACH_ROLE_INCOMPLETE / VALID`。该模块不得改成来源
搜索器，也不得据此启动 algorithm implementation 或 execution。

## Real positive approach role admission R1

`real_positive_approach_role_admission_r1/` 将候选冻结为唯一 ETH3D `sofa_3`
RGB-D official URL，并在 claim 前绑定 burned manifest、source authority、
contract、implementation lock 与 host preflight。正式 run 只发生一次 GET，
无 HEAD/retry/mirror/replacement；source-access validator 独立复核 response
artifact hash、claim-before-request 与零替换计数。

终态为 `HOLD_ALGORITHM_CANARY_APPROACH_ROLE_INCOMPLETE / VALID`，原因是
`R1_OFFICIAL_CONTAINER_IDENTITY_INCOMPLETE`。geometry producer 未执行，
RGB pixels 与 algorithm outcome 未读取，performance qualification 不创建。
本模块与所有 `sofa_3` artifact 不得重跑、续传、换源或升级为 confirmation。
