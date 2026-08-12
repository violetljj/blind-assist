#!/usr/bin/env python3
"""Admit the Attempt-09 expansion for depth training while preserving UNKNOWN factors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt09-fit-expansion-labels-r0/result.json"
OUTPUT = RESULT.parent / "depth_training_admission_r1.json"

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_ag_r2_f1_factor_learnability_attempt03 import require, sha256_file  # noqa: E402


def main() -> int:
    require(RESULT.is_file(), "expansion result missing")
    require(not OUTPUT.exists(), "admission output exists")
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    require(result["frame_count"] == 108 and result["parent_count"] == 9, "expansion identity drift")
    exact = True
    depth_parents = set()
    joint_parents = set(result["joint_factor_parents"])
    support_valid_frames: dict[str, int] = {}
    parent_frames: dict[str, int] = {}
    unknown_frame_count = 0
    for row in result["frames"]:
        path = Path(row["output"])
        parent_frames[row["parent_id"]] = parent_frames.get(row["parent_id"], 0) + 1
        exact &= path.is_file() and sha256_file(path) == row["output_sha256"]
        with np.load(path, allow_pickle=False) as payload:
            if int(np.asarray(payload["metric_depth_valid_hw"], dtype=np.bool_).sum()) > 0:
                depth_parents.add(row["parent_id"])
            if not bool(np.asarray(payload["support_plane_valid"]).item()):
                support = np.asarray(payload["support_truth_valid_hw"], dtype=np.bool_)
                evidence = np.asarray(payload["evidence_truth_valid_hw"], dtype=np.bool_)
                exact &= not bool(support.any()) and not bool(evidence.any())
                unknown_frame_count += 1
            else:
                support_valid_frames[row["parent_id"]] = support_valid_frames.get(row["parent_id"], 0) + 1
    fully_unknown_parents = {
        parent for parent in parent_frames if support_valid_frames.get(parent, 0) == 0
    }
    gates = {
        "all_108_npz_hash_exact": exact,
        "metric_depth_parent_coverage_9_of_9": len(depth_parents) == 9,
        "joint_factor_parent_coverage_at_least_8": len(joint_parents) >= 8,
        "unsupported_support_remains_unknown": fully_unknown_parents == {"rgbd_dataset_freiburg3_long_office_household"},
        "held_metrics_unopened": result["preserved_canary_metrics_opened"] is False,
        "source_schema_roundtrip_task_firewall_pass": all(
            result["gates"][key]
            for key in ("roundtrip_exact", "schema_complete", "task_firewall", "unknown_fail_closed")
        ),
    }
    passed = all(gates.values())
    admission = {
        "schema": "blindassist_ag_r2_f1_attempt09_fit_expansion_depth_training_admission_v1",
        "status": "ATTEMPT09_DEPTH_EXPANSION_ADMITTED_SUPPORT_UNKNOWN_PRESERVED" if passed else "ATTEMPT09_DEPTH_EXPANSION_NOT_ADMITTED",
        "passed": passed,
        "result": {"path": str(RESULT.resolve()), "sha256": sha256_file(RESULT)},
        "depth_parent_count": len(depth_parents),
        "joint_factor_parent_count": len(joint_parents),
        "fully_factor_unknown_parents": sorted(fully_unknown_parents),
        "factor_unknown_frame_count": unknown_frame_count,
        "gates": gates,
        "admitted_use": "metric depth, validity and residual-depth uncertainty training; other factors only where their native validity is true",
    }
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(admission, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(admission, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
