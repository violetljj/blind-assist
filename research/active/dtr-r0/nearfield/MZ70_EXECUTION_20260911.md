# MZ70 execution

Two matched fits: 4096 steps per arm, 8192 total. Initial checkpoint `fb318f03a598d00b89fc011eb5b1a368873f9ac801d1dc0710618ba119f4d4aa`; 11020 parameters per arm. No historical comparator is retrained.

Actual run receipt time 540.455721s; launcher 545.990748s. Feature build 102.979179s; fit CONTROL 112.657173s, DIVERSE 117.469542s; evaluation 178.507697s. CPU score receipt 29.000558s; score launcher 29.533165s. Timing scopes are distinct.

Backend: CUDA frozen RGB encoder and Adam head; CPU source I/O; device: NVIDIA GeForce RTX 5060 Laptop GPU. Shared feature accounting: `{"training_extracted_frames":8724,"full_extracted_frames":21269,"full_cache_eval_hits":5191,"evaluation_extracted_frames":12545,"encoder_seconds":104.09827099950053,"build_seconds":102.95446459999948,"evaluation_seconds":154.89246889976494}`. RGB loads/encoded bytes read: 21269/8700879590. This is actual accounting, not a separately measured speedup.

Training exposure audit: `{"status":"PASS","steps_per_arm":4096,"profile_steps":[1024,1024,1024,1024],"CONTROL_mz61_each_id_each_profile":2,"DIVERSE_each_source_id_each_profile":1,"geometry_presentations_per_arm":16384,"mz48_presentations_per_arm":16384,"old_presentations_per_arm":32768,"shared_replay_exact":true,"seed":170}`. Geometry TRAIN coverage is complete at the declared per-profile counts; the report does not imply complete per-ID coverage for every other replay stream. Initialization checks use 32 TRAIN frames and 16 arm/profile/cohort comparisons, not zero neural probes.

Independent scoring audit: `{"status":"PASS","baseline_arrays_byte_exact":2954,"prior_metric_rows_exact":24138,"scalar_candidate_union_values":982400,"native_lookups":385024,"original_calibration_rows":1256,"cutoff_vectors_rebuilt":2,"mz36_UNKNOWN_bits":80,"local_UNKNOWN_cells":{"mz48":7562077,"mz55":7417741,"mz61":12188903,"mz67":12370308},"dense_argmax_recomputed":false,"new_fits":0,"new_inference":0,"new_cutoff_searches":0}`. Two four-query cutoff vectors are reconstructed by the original scalar rule on DEV1000 plus original MZ48 CAL256 under DROP. MZ55/MZ61/MZ67 CAL rows do not set them. ALL_INVALID means input ranges exactly zero and all statuses invalid; it is a stress condition, not a real-device failure probability.

Both model and scorer launch receipts record exit0 with no remaining processes. Task handle release: `{"status":"PASS","feature_handles_closed":true,"compact_handles_closed":true,"scratch_retained_until_score":true,"process_exit_required":true}`.

Task-owned temporary feature cache released: 8040042496 allocated bytes; 8040038528 logical bytes. The exact target is absent, no owning process remains, and all30 durable files/metadata/source bindings retain SHA, size and modification time. Full cleanup proof is linked below; raw sources and checkpoints remain.

| Bound artifact under mz70-diverse-learning-20260911 | SHA256 |
| --- | --- |
| [run-v1/receipt.json](../../../../artifacts.local/work/mz70-diverse-learning-20260911/run-v1/receipt.json) | `a3ddf2e8e6cca4742e9970eb7e3e1f9f2e1adf70f3284883882f15869ceaa1e8` |
| [score-v1/receipt.json](../../../../artifacts.local/work/mz70-diverse-learning-20260911/score-v1/receipt.json) | `2fe35ce61155ba80c7197400d4331f60410159cc550270dca623153e2f23f84a` |
| [score-v1/result.json](../../../../artifacts.local/work/mz70-diverse-learning-20260911/score-v1/result.json) | `1df5fdaa88bd9b7087ef84478ced94c31dee71a5d2d7f4c42c49e486f5f0bbbf` |
| [score-v1/audit.json](../../../../artifacts.local/work/mz70-diverse-learning-20260911/score-v1/audit.json) | `bc5269881efdf8587912db79fbcde335a65aeb7a16329ee2b9118559764a7540` |
| [score-v1/paired-events.json](../../../../artifacts.local/work/mz70-diverse-learning-20260911/score-v1/paired-events.json) | `d9d123d5111c943318fc4dff730417f9794e683cc5e492274b5935e7db95b321` |
| [score-v1/report.md](../../../../artifacts.local/work/mz70-diverse-learning-20260911/score-v1/report.md) | `fbfa5cfc0f14411fcd7d4c3b35bc4facfc141ce64f003b6032272a063afd9ee2` |
| [execution-v1.json](../../../../artifacts.local/work/mz70-diverse-learning-20260911/execution-v1.json) | `21d4d16ae052063718e3391cd9f8d92ad5628bc1551d6c02fe3d9fae43653539` |
| [score-execution-v1.json](../../../../artifacts.local/work/mz70-diverse-learning-20260911/score-execution-v1.json) | `ceaa00e6cfb3926fd5d3cbb28c26610dbdbe6c9cd06adb43c4ec4a3a78ac2bb4` |
| [handle-release.json](../../../../artifacts.local/work/mz70-diverse-learning-20260911/handle-release.json) | `cadd94e8985b244cfebb4e40f54c459428154f3d1ed5385965d1b04dd0553cbc` |
| [cache-cleanup-v1/receipt.json](../../../../artifacts.local/work/mz70-diverse-learning-20260911/cache-cleanup-v1/receipt.json) | `27e9f56568bf1b5636bbdf0007ecc6803f19980d343e08dbcc12233bf5f24abc` |

| Producer-recorded retained output | SHA256 |
| --- | --- |
| run-v1/predictions.npz | `79b1ab43b274a48b272d0500003f89fd91ed0a8916d89a3c01dc1e936eeca466` |
| run-v1/CONTROL.pt | `68b46ff8696c43cc21a0ffd5553a2356faf74b7f99ae269539ade44ed40dd3a9` |
| run-v1/DIVERSE.pt | `4d4599c457659b31185cad0c6dec2ebf3dcedd0e01470b4d786318171583afc2` |
| run-v1/CONTROL-cutoff.npy | `68f05149c773fdc38fd4662fe0ef42ad4bcfbe5eb20900d951f6fb3349b48e92` |
| run-v1/DIVERSE-cutoff.npy | `ee4d637b3f4bc12ccd5abcd312566c04c69106bd5037912af4a4918b0b1f9377` |
| run-v1/groups.json | `e971c5ff68b2a9c87d65564a7054f2ea6cf92a7079bf9f8b0985fda3e13210a7` |
| run-v1/schedule.npz | `3eecc19e52769bfaa6ad7f874640cf08e9b41d82779b108400d3cfc3f9a1c67b` |

The independent scorer already checks run-output bindings. This renderer rehashes receipt/report evidence, not large model inputs or caches. It changes only ignored draft files. Raw native/RGB data is not decoded; no model fit, inference, cutoff search or recomputed dense argmax occurs here. Original scores and failures remain unchanged.
