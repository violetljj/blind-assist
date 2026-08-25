#!/usr/bin/env python3
"""Freeze source-disjoint ProcTHOR rosters for GRAIL M1 before visual outcomes."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from grail_procthor_native_m0 import canonical_sha256, sha256_file


VAL_SHA256 = "d808540514e26b6726cd2790490e669b572eeb94febb5188a2f403591dd21721"
TEST_SHA256 = "9a9fa6f134e76fe87f3fd92c00883651cf9fadf4e9ad4072d6d73be229f001dc"
M0_CONSUMED_TEST = (0, 87, 145, 161, 188, 285, 298, 325, 343, 366, 394, 421, 482, 498, 500, 518, 591, 605, 631, 676, 753, 795, 856, 906, 945)


def ranked_houses(dataset: Path, salt: str, excluded: set[int]) -> list[dict[str, Any]]:
    dataset_hash = sha256_file(dataset)
    rows = []
    with gzip.open(dataset, "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index in excluded:
                continue
            house = json.loads(line)
            house_hash = canonical_sha256(house)
            rank = hashlib.sha256(
                f"{salt}:{dataset_hash}:{index}:{house_hash}".encode("ascii")
            ).hexdigest()
            rows.append({"rank": rank, "house_index": index, "house_sha256": house_hash})
    return sorted(rows, key=lambda row: row["rank"])


def freeze(val: Path, test: Path) -> dict[str, Any]:
    if sha256_file(val) != VAL_SHA256 or sha256_file(test) != TEST_SHA256:
        raise ValueError("ProcTHOR split hash mismatch")
    train_dev = ranked_houses(val, "BLINDASSIST_GRAIL_M1_VAL_V1", {0})
    test_rows = ranked_houses(test, "BLINDASSIST_GRAIL_M1_TEST_V1", set(M0_CONSUMED_TEST))
    train, dev = train_dev[:24], train_dev[24:30]
    test_roster = test_rows[:12]
    return {
        "schema": "blindassist_grail_m1_manifest_v1",
        "frozen_before_visual_or_test_outcome": True,
        "source": {
            "dataset_revision": "439193522244720b86d8c81cde2e51e3a4d150cf",
            "val_sha256": VAL_SHA256,
            "test_sha256": TEST_SHA256,
            "runtime_image_id": "sha256:36bc6640b8ecebd35b748712a44411455e09f7d3b984c9bb6d9c82dd2f4b9211",
        },
        "selection": {
            "val_salt": "BLINDASSIST_GRAIL_M1_VAL_V1",
            "test_salt": "BLINDASSIST_GRAIL_M1_TEST_V1",
            "val_excluded": [0],
            "test_excluded": list(M0_CONSUMED_TEST),
            "reads_visual_or_teacher_outcome": False,
        },
        "rosters": {"train": train, "dev": dev, "test": test_roster},
        "frozen_encoders": {
            "visual": {
                "id": "facebook/dinov2-small local snapshot",
                "weights_sha256": "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1",
                "config_sha256": "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d",
            },
            "depth_baseline_only": {
                "id": "depth-anything/Depth-Anything-V2-Small-hf local snapshot",
                "weights_sha256": "3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70",
                "config_sha256": "c56698d3643dde1f83ea2212759e6b31a22b8f827246a36dd007ee8a22b3ff75",
            },
        },
        "data_contract": {
            "image": "320x240 egocentric RGB",
            "goal": "same-instance reference crop from a distinct native interaction pose",
            "proposals": "simulator instance masks for all visible stationary actionable candidates",
            "truth": "all native standing horizon-0 interaction poses transformed into the query camera frame",
            "absence": "reference paired with a different-house query; exact target instance absent",
            "instance_generalization": "house-disjoint object instances",
        },
        "formal_gates": {
            "minimum_test_positive": 96,
            "minimum_wrong_target_cases": 24,
            "minimum_absence_cases": 96,
            "grail_pose_success_absolute_uplift_over_best_simple": 0.10,
            "maximum_wrong_target_rate_increase": 0.02,
            "maximum_absence_false_commit_increase": 0.0,
            "permutation_consistency": 1.0,
        },
        "one_shot": {"dev_selects_thresholds": True, "test_replay": False, "on_fail": "STOP_BEFORE_M2"},
        "claim_ceiling": "synthetic ProcTHOR RGB with oracle candidate masks and simulator-native pose truth; reference-goal mode only; no natural-scene, proposal, text-goal, Android, user, product, or safety claim",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze(args.val, args.test)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({role: [r["house_index"] for r in rows] for role, rows in manifest["rosters"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
