# Local Artifact Cleanup Record — 2026-08-01

## Result

- Scope: approved priority batch and second batch.
- Deleted files: 89,734.
- Deleted logical bytes: 52,354,461,801.
- Deleted logical size: 48.759 GiB.
- E-drive free space: 10.730 GiB before, 59.641 GiB after.
- Execution failures: 0.
- Deletion was permanent and did not use the Recycle Bin.

The cleanup removed heavyweight, re-downloadable, or rebuildable payloads while
retaining manifests, receipts, hashes, source URLs/DOIs, licenses, timestamp
indexes, labels, truth/pair/frame ledgers, protocols, QA records, closeouts,
failure receipts, results, and scripts.

## Priority batch

The priority batch covered:

- `artifacts.local/evidence/datasets/revel-dynamic-bag-v1-20260720/dynamic.bag`
- `artifacts.local/datasets/cid_sims_v6/office_building/floor3/floor3_1.zip`
- `artifacts.local/datasets/cid_sims_v6/office_building/floor3/floor3_2.zip`
- `artifacts.local/datasets/rcle_tum_fr2_rpy_source_native_r0/rgbd_dataset_freiburg2_rpy.tgz`
- `artifacts.local/downloads/rcle_phase_b_real_positive_approach_role_admission_r0/npz_flea3_7_sanity_ll.tar.gz`
- archive payloads under `artifacts.local/datasets/rcle_phase_b_bonn_b0_r1/archives`
- archive payloads under `artifacts.local/downloads/rcle_phase_b_tum_prescreen/archives`
- ZIP payloads under `artifacts.local/datasets/egomotion_compensated_looming_r1`
- `artifacts.local/evidence/datasets/tartanair-preprocessed-japanesealley-20260720/japanesealley.tar.gz`
- `artifacts.local/evidence/datasets/carla-stage2-ped-town01-shard000000-20260720/carla-stage2-000000.tar`
- `artifacts.local/evidence/datasets/ub-visiogeoloc-cc0-seq10-20260715/10.7z`
- `artifacts.local/evidence/datasets/bonn-rgbd-moving-obstructing-box-v1-20260720/rgbd_bonn_moving_obstructing_box.zip`
- `E:/codex-tools/downloads/pytorch-cu128`
- obsolete `E:/codex-tools/projects/blindassist/toolchain/venv-export`
- legacy `.gradle-local`
- rebuildable `app/build` and `apps/benchmarks/device-benchmark/build`

## Second batch

The second batch removed selected payload extensions rather than whole evidence
directories:

- image payloads from the five materialized views under
  `artifacts.local/evidence/datasets/ub-visiogeoloc-cc0-seq10-20260715`
- expanded TUM PNG frames while retaining the source text indexes
- generated geometry `.npz` payloads while retaining formal receipts
- CARLA slice `.npy` and `.png` payloads while retaining JSON QA
- REVeL images and archives while retaining labels and QA
- periodic counterfactual `.npz` payloads while retaining JSONL ledgers
- failed or superseded segmentation images and model binaries while retaining
  freeze, closeout, failure, and JSON records

## Protected scope

The whitelist excluded current RISKSEG-ACT, HFTF outputs and datasets, SANPO
canonical data, current RISKSEG reconstruction inputs, active Python
environments, Android SDKs, and JDKs.

## Local audit records

The complete local, ignored audit trail is retained at:

`artifacts.local/cleanup-records/20260801-priority-and-second-batch/`

It contains:

- `CLEANUP_RECORD.md`
- `cleanup-targets.json`
- `deleted-payload-inventory.jsonl`
- `cleanup-result.json`
- `execute-cleanup.ps1`

The JSONL inventory has one entry per deleted file with the original absolute
path, byte size, last-write timestamp, batch, and cleanup reason. It remains
local because `artifacts.local/` is intentionally excluded from Git.
