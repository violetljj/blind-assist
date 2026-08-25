#!/usr/bin/env python3
"""Freeze GRAIL M1 V2 after rejecting the target-centered V1 Development generator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from freeze_grail_m1 import M0_CONSUMED_TEST, TEST_SHA256, VAL_SHA256, ranked_houses
from grail_procthor_native_m0 import sha256_file


V1_TEST = (38, 41, 144, 184, 187, 278, 310, 344, 439, 516, 652, 791)


def freeze(val: Path, test: Path) -> dict:
    if sha256_file(val) != VAL_SHA256 or sha256_file(test) != TEST_SHA256:
        raise ValueError("ProcTHOR split hash mismatch")
    train_dev = ranked_houses(val, "BLINDASSIST_GRAIL_M1_VAL_V2", {0})
    excluded_test = set(M0_CONSUMED_TEST) | set(V1_TEST)
    test_rows = ranked_houses(test, "BLINDASSIST_GRAIL_M1_TEST_V2", excluded_test)
    return {
        "schema": "blindassist_grail_m1_manifest_v2",
        "frozen_before_v2_visual_or_test_outcome": True,
        "source": {"dataset_revision": "439193522244720b86d8c81cde2e51e3a4d150cf", "val_sha256": VAL_SHA256,
                   "test_sha256": TEST_SHA256, "runtime_image_id": "sha256:36bc6640b8ecebd35b748712a44411455e09f7d3b984c9bb6d9c82dd2f4b9211"},
        "selection": {"val_salt": "BLINDASSIST_GRAIL_M1_VAL_V2", "test_salt": "BLINDASSIST_GRAIL_M1_TEST_V2",
                      "val_excluded": [0], "test_excluded": sorted(excluded_test), "reads_visual_or_teacher_outcome": False},
        "rosters": {"train": train_dev[:24], "dev": train_dev[24:30], "test": test_rows[:12]},
        "query_generation": {"distance_m": [1.75, 4.0], "position_order": "sample-hash",
                             "yaw_candidates_deg_around_target": [-60, -30, 0, 30, 60],
                             "yaw_order": "sample-hash; first rendered-visible target; target centering is not selected"},
        "frozen_encoders": {
            "visual": {"id": "facebook/dinov2-small local snapshot", "weights_sha256": "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"},
            "depth_baseline_only": {"id": "Depth-Anything-V2-Small-hf local snapshot", "weights_sha256": "3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70"},
        },
        "comparators": ["B0_POOLER_FIXED_DISTANCE", "B1_POOLER_RELATIVE_DEPTH", "B2_DIRECT_SINGLE_POSE", "GRAIL_LOCAL_TOKEN_FACTORIZED_K_SET"],
        "formal_gates": {"minimum_test_positive": 96, "minimum_wrong_target_cases": 24, "minimum_absence_cases": 96,
                         "grail_pose_success_absolute_uplift_over_best_b0_b1_b2": 0.10,
                         "maximum_wrong_target_rate_increase_over_best_candidate_baseline": 0.02,
                         "maximum_absence_false_commit_increase_over_best_pose_baseline": 0.0,
                         "permutation_consistency": 1.0},
        "one_shot": {"dev_selects_thresholds": True, "test_replay": False, "on_fail": "STOP_BEFORE_M2"},
        "claim_ceiling": "synthetic ProcTHOR RGB with oracle candidate masks and simulator-native pose truth; reference-goal mode only; no natural-scene, proposal, text-goal, Android, user, product, or safety claim",
    }


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--val",type=Path,required=True); parser.add_argument("--test",type=Path,required=True); parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); manifest=freeze(args.val,args.test); args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({role:[row["house_index"] for row in rows] for role,rows in manifest["rosters"].items()},indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
