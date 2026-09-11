# MZ71 execution

Frozen MZ70 CONTROL/DIVERSE weights, 0 training steps, 2 state-cut vectors and 0 threshold sweeps. New ALL_INVALID inference covers 9544 admitted rows in seven old cohorts. Initialization parity uses 32 TRAIN frames; source61/67 scientific cohort reinference is 0. Existing source ALL_INVALID predictions remain byte-identical.

Run receipt 155.015561s; run wrapper 158.759510s. Independent CPU score receipt 37.108828s; score wrapper 37.697586s. Distinct timing scopes are retained. Components: `{"rgb_decode":56.633449600150925,"visual_views":55.15841389972775,"readouts":12.378294199763332}`.

Actual RGB loads 9576; encoded RGB bytes read 3937922987; shared view frame counts `{"global256x144_BOX":9576,"crop224x224":9576,"full640x360":9576}`. Loads include the explicit parity frames. No measured speedup is inferred from shared-view reuse.

Backend: CUDA frozen inference; CPU compact I/O and two fixed calibration rules; device: NVIDIA GeForce RTX 5060 Laptop GPU. No permanent dense cache; no old dense cache or native-depth read. Checkpoint bindings: `{"CONTROL":{"path":"F:\\ba-data\\blindassist-artifacts-20260805\\work\\mz70-diverse-learning-20260911\\run-v1\\CONTROL.pt","sha256":"68b46ff8696c43cc21a0ffd5553a2356faf74b7f99ae269539ade44ed40dd3a9","strict_state_tensors":19},"DIVERSE":{"path":"F:\\ba-data\\blindassist-artifacts-20260805\\work\\mz70-diverse-learning-20260911\\run-v1\\DIVERSE.pt","sha256":"4d4599c457659b31185cad0c6dec2ebf3dcedd0e01470b4d786318171583afc2","strict_state_tensors":19}}`.

Scoring audit: `{"status":"PASS","baseline_arrays_byte_exact":3363,"prior_metric_rows_exact":37530,"scalar_candidate_union_values":2412096,"native_winner_lookups":425984,"nonmissing_copied_scalar_values":814400,"old_all_invalid_inference_frames":9544,"new_calibration_vectors":2,"calibration_ids":1256,"observed_packet_prefixes_bound":36,"source_packet_profiles_recomputed":true,"initial_parity_winner_checks":[{"cohort":"mz61","key":"OLD_NEG/OPEN/winner","winner_equal":true},{"cohort":"mz61","key":"OLD_NEG/GATED/winner","winner_equal":true},{"cohort":"mz61","key":"MZ70/CONTROL/winner","winner_equal":true},{"cohort":"mz61","key":"MZ70/DIVERSE/winner","winner_equal":true},{"cohort":"mz67","key":"OLD_NEG/OPEN/winner","winner_equal":true},{"cohort":"mz67","key":"OLD_NEG/GATED/winner","winner_equal":true},{"cohort":"mz67","key":"MZ70/CONTROL/winner","winner_equal":true},{"cohort":"mz67","key":"MZ70/DIVERSE/winner","winner_equal":true}],"mz36_attempted_frames":400,"mz36_admitted_frames":380,"mz36_UNKNOWN_query_bits":80,"local_UNKNOWN_cells":{"mz48":7562077,"mz55":7417741,"mz61":12188903,"mz67":12370308},"new_fit":0,"new_model_inference":0,"new_cutoff_search":0,"native_argmax_recomputed":false,"ranking_recomputed":false}`. The36 packet/profile sets are bound to pre-inference source preparation; transformations and actual missing masks are independently checked. Both original and new cutoff arithmetic is rebuilt. Saved winner ties in the parity check are disclosed by the audit; retained scientific source winners are not replaced.

Both launch receipts report exit0 and no remaining processes. Handle release: `{"status":"PASS","compact_handles_closed":true,"permanent_dense_cache":false,"process_exit_required":true}`. No cache deletion or released-byte amount is invented; this run declares no permanent dense cache.

| Bound evidence under mz71-missing-state-calibration-20260911 | SHA256 |
| --- | --- |
| [run-v1/receipt.json](../../../../artifacts.local/work/mz71-missing-state-calibration-20260911/run-v1/receipt.json) | `49b3c67ebe513e35e01af49cd1fa28e6de1b9ec4750fe25bec8ebf2d9a476f34` |
| [score-v1/receipt.json](../../../../artifacts.local/work/mz71-missing-state-calibration-20260911/score-v1/receipt.json) | `4b18be6ff0348b738781f4695af7b4b6e0b29e47928bf2690c57512038a1adc9` |
| [score-v1/result.json](../../../../artifacts.local/work/mz71-missing-state-calibration-20260911/score-v1/result.json) | `e7988880e5e226a5b7454d87bb589a803c0946b5f5150734fa6002f518747bb7` |
| [score-v1/audit.json](../../../../artifacts.local/work/mz71-missing-state-calibration-20260911/score-v1/audit.json) | `724c64695ed34c3265292ed547dbb814d25fe31fd34021aea639ec6e1d2c5d09` |
| [score-v1/paired-events.json](../../../../artifacts.local/work/mz71-missing-state-calibration-20260911/score-v1/paired-events.json) | `1246437e1d8a02a7389fd41fb6179ec65c5e893d798da53e35bb3b7147827c9a` |
| [score-v1/report.md](../../../../artifacts.local/work/mz71-missing-state-calibration-20260911/score-v1/report.md) | `f8888df077b5c8a447f8b3a5b842413813eacd3c60530f706899d3daa196c624` |
| [execution-v1.json](../../../../artifacts.local/work/mz71-missing-state-calibration-20260911/execution-v1.json) | `35405987a8701b1049d6ec442b6bd7d615b44e3f240819e7015bc4b1881861e5` |
| [score-execution-v1.json](../../../../artifacts.local/work/mz71-missing-state-calibration-20260911/score-execution-v1.json) | `ea8a20e82af5d8055a60fd6effb6a5606b48059fed8f559cb5ca823762398de7` |
| [handle-release.json](../../../../artifacts.local/work/mz71-missing-state-calibration-20260911/handle-release.json) | `f6cccbf6aa3dccc7e3cad63e50e94ac0bbd6a53e55894f804f70a2a407f3524b` |

| Producer-recorded run output | SHA256 |
| --- | --- |
| run-v1/predictions.npz | `c643e805265897a1bbf57de252b759deb4e6c2c559c318b205c746e3c3eef569` |
| run-v1/groups.json | `e971c5ff68b2a9c87d65564a7054f2ea6cf92a7079bf9f8b0985fda3e13210a7` |
| run-v1/initial-parity.json | `c83479a636e20d4a83807c288acff0448bd9535da3c9d33abcb6fb8bc956ec43` |
| run-v1/CONTROL-cutoff.npy | `ec541e4a44e008e4b9536c75c138637cc167766c3c7ffe4c63b5d8c155ffc06b` |
| run-v1/DIVERSE-cutoff.npy | `ce2eaf14a3b8c32f4208bb1f481c6e13b93d62c0743419a1348e1a30ccdbff48` |
| run-v1/MZ70-CONTROL-cutoff.npy | `68f05149c773fdc38fd4662fe0ef42ad4bcfbe5eb20900d951f6fb3349b48e92` |
| run-v1/MZ70-DIVERSE-cutoff.npy | `ee4d637b3f4bc12ccd5abcd312566c04c69106bd5037912af4a4918b0b1f9377` |

The independent scorer checks run-output hashes. This reporting pass binds the sealed receipts and score evidence without decoding predictions, native arrays or RGB, or recomputing models/cuts. Drafts are ignored artifacts pending root review. Controlled consumed Development supplies no natural-scene, hardware, Android/default-app or safe-clearance claim.
