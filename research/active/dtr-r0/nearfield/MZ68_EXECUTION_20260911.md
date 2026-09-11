# MZ68 execution and evidence

One actual NULL_COVERAGE run and one independent CPU score completed. Gate FAIL 6/11; inheritance is limited to NEGATIVE_CONTROL. The sealed MZ64 comparator and MZ66 negative control were reused as outputs, without refitting them.

Actual run receipt: 322.964221s; feature build 97.462963s; fit 45.391370s; evaluation 157.455192s. CPU score receipt: 12.175956s. These are distinct timer scopes, not interchangeable with launcher wall time.

Execution wrappers (unaltered fields) and their exact exit/process checks:

```json
{
  "learner": {
    "status": "PASS",
    "exit_code": 0,
    "seconds": 328.1835456,
    "remaining_processes": []
  },
  "scorer": {
    "status": "PASS",
    "exit_code": 0,
    "seconds": 12.6164823,
    "owning_python_survivors": []
  }
}
```

Actual trained parameters 11020; total updates 1536; original calibration 1256 rows; new cutoff vectors 1; baseline arrays preserved 2664. All-missing training steps 384 and combined native/OLD presentations 6144. The runner checks zero anchors on every such step.

The original frozen encoder, mean/std, model geometry and native labels remain unchanged. Shared cached RGB features are reused across profiles. Actual cache/extraction costs are retained below; the MZ68 task owns its scratch, without reusing a deleted old cache.

```json
{
  "source_plan": {
    "training_unique": {
      "mz48": 1095,
      "old": 3533,
      "mz55": 0,
      "mz61": 2048
    },
    "training_unique_total": 6676,
    "shape": [
      6676,
      64,
      45,
      80
    ],
    "dtype": "float32",
    "bytes_per_frame": 921600,
    "data_bytes": 6152601600,
    "npy_header_bytes": 128,
    "expected_file_bytes": 6152601728,
    "fixed_encoder_batch": 16,
    "evaluation_frames": {
      "DEV": 1000,
      "relation10000": 2000,
      "distance5000": 1000,
      "rich": 44,
      "mz36": 380,
      "mz48": 2560,
      "mz55": 2560,
      "mz61": 4096
    },
    "evaluation_cache_hits": {
      "DEV": 0,
      "relation10000": 0,
      "distance5000": 0,
      "rich": 0,
      "mz36": 0,
      "mz48": 1095,
      "mz55": 0,
      "mz61": 2048
    },
    "uncached_evaluation_frames": 10497,
    "planned_full_rgb_encodes": 17173,
    "normalization": "Original encoder values; runner applies frozen mean/std",
    "scratch_owner": "mz68-missing-input-coverage-20260911",
    "prior_full_cache_reused": false,
    "calibration_or_heldout_in_training": false,
    "baseline_replay_frames": 0,
    "extraction_unit": "One per unique registered source/frame RGB reference",
    "evaluation_call_contract": "One batch outside arm/profile loops; reuse returned values for every arm/profile",
    "fit_arm": "NULL_COVERAGE",
    "target_source": "mz61",
    "reference_only_mz55_training_cache": false
  },
  "actual_feature_counts": {
    "training_extracted_frames": 6676,
    "full_extracted_frames": 17173,
    "full_cache_eval_hits": 3143,
    "evaluation_extracted_frames": 10497,
    "encoder_seconds": 80.72852900024736,
    "build_seconds": 97.44298030000937,
    "evaluation_seconds": 144.7148628999421
  },
  "rgb_loads": 17173,
  "rgb_bytes_read": 7002792656
}
```

Independent audit, including original scalar cutoff reconstruction, exact inherited arrays/metrics, known decision checks and native winner lookups:

```json
{
  "status": "PASS",
  "baseline_arrays_byte_exact": 2664,
  "prior_metric_rows_exact": 21162,
  "scalar_known_decisions": 360128,
  "native_winner_lookups": 126976,
  "schedule_groups_byte_identical": true,
  "original_calibration_rows": 1256,
  "cutoff_vectors_recomputed": 1,
  "mz36_UNKNOWN_bits": 80,
  "local_UNKNOWN_cells": {
    "mz48": 7562077,
    "mz55": 7417741,
    "mz61": 12188903
  },
  "new_fits": 0,
  "new_inference": 0,
  "new_threshold_searches": 0,
  "RGB_reads": 0,
  "native_depth_reads": 0,
  "dense_argmax_recomputed": false
}
```

Initialization uses32 unique TRAIN frames (16 MZ48×3profiles;16 MZ61×4profiles), comparing saved MZ62 CONTROL raw/support at atol2e-5/rtol1e-6. These are disclosed initialization probes; no full old-baseline inference is claimed. Dense per-cell argmax was not independently re-encoded. Truth/known/native counts remain loss/evaluator-only; missing status is not a negative label.

Task-owned handles/processes are released. Cache cleanup records 6152605696 allocated bytes released and the target absent; durable outputs/checkpoints remain. No unrelated process or source is released by this result renderer.

Evidence hashes:

| Receipt | SHA256 |
| --- | --- |
| [run-v1/receipt.json](../../../../artifacts.local/work/mz68-missing-input-coverage-20260911/run-v1/receipt.json) | 2565c40a93862c6513962484fa45ebecb4e9aabec503f299c931c45e61f813eb |
| [score-v1/receipt.json](../../../../artifacts.local/work/mz68-missing-input-coverage-20260911/score-v1/receipt.json) | 7fdbde94f6ee0cd713d47be07422a334b1a789ad9dedc41cb4834fb75df745ed |
| [execution-v1.json](../../../../artifacts.local/work/mz68-missing-input-coverage-20260911/execution-v1.json) | 57be4be4fc7bfd95c246ff2f859c723e8ae5d5f5a78610887c57da6ccb67e46f |
| [score-execution-v1.json](../../../../artifacts.local/work/mz68-missing-input-coverage-20260911/score-execution-v1.json) | 8cda54684c9af617e320ad24af57a9660dc8e6c66c2c7e05e833c80535a01d23 |
| [handle-release.json](../../../../artifacts.local/work/mz68-missing-input-coverage-20260911/handle-release.json) | cadd94e8985b244cfebb4e40f54c459428154f3d1ed5385965d1b04dd0553cbc |
| [cache-cleanup-v1/receipt.json](../../../../artifacts.local/work/mz68-missing-input-coverage-20260911/cache-cleanup-v1/receipt.json) | 87080135bb5c5e82e2475a23168e014c158380c0253eb1123d55b55c77a797cd |

No additional fit, cutoff search, source capture or scientific rerun is authorized by report generation. Numerical teacher-score preservation remains an independent candidate mechanism.
