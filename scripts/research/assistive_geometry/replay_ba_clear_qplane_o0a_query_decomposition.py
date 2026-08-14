#!/usr/bin/env python3
"""Replay frozen Q-Plane candidates to add the required per-query report."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from scripts.research.assistive_geometry import (
    run_ba_clear_qplane_o0a_representation_headroom as qplane,
)


DEFAULT_EXPERIMENT_DIR = (
    qplane.REPO_ROOT / "artifacts.local/experiments/ba-clear-qplane-o0a-headroom-r0"
)
EXPECTED_RESULT_SHA256 = (
    "8C8487D05C2F831EA51019EB680E1107AF78BD59EE9C4A6E920365BB86745B04"
)
EXPECTED_CANDIDATE_SHA256 = (
    "BA8D9845630C65F4976F73050F8E97D8598C9AF2B0D66A288447A09FF0B4BAF6"
)


def summarize_per_query(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[qplane.query_key(str(row["band"]), float(row["horizon_m"]))].append(
            row
        )
    return {
        key: {
            arm: qplane.summarize_records(rows, arm)
            for arm in qplane.ARMS
        }
        for key, rows in sorted(grouped.items())
    }


def run(experiment_dir: Path, output_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    result_path = experiment_dir / "result.json"
    candidate_path = experiment_dir / "candidate-plan.json"
    qplane.require(not output_path.exists(), f"output exists: {output_path}")
    qplane.require(
        qplane.sha256_file(result_path) == EXPECTED_RESULT_SHA256,
        "source result SHA drift",
    )
    qplane.require(
        qplane.sha256_file(candidate_path) == EXPECTED_CANDIDATE_SHA256,
        "candidate plan SHA drift",
    )
    source_result = qplane.read_json(result_path)
    qplane.require(
        source_result.get("status")
        == "BA_CLEAR_QPLANE_O0A_REPRESENTATION_HEADROOM_FAIL_CLOSE_NO_TRAINING",
        "source result terminal drift",
    )
    protocol = qplane.read_json(qplane.DEFAULT_PROTOCOL)
    data = protocol["data"]
    roster_path = (qplane.REPO_ROOT / data["roster"]).resolve()
    source_root = (qplane.REPO_ROOT / data["source_root"]).resolve()
    cache_path = (qplane.REPO_ROOT / data["depthart_cache"]).resolve()
    qplane.verify_inputs(
        qplane.DEFAULT_PROTOCOL,
        protocol,
        roster_path,
        source_root,
        cache_path,
    )
    roster = qplane.read_json(roster_path)
    candidate = qplane.read_json(candidate_path)
    qplane.require(
        candidate.get("status")
        == "CANDIDATE_PARAMETERS_FROZEN_BEFORE_TASK_OUTCOMES",
        "candidate status drift",
    )
    base_cache = np.load(cache_path, mmap_mode="r")
    poses: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    records: list[dict[str, Any]] = []
    maximum_pose_delta = float(data["pose_maximum_association_delta_seconds"])
    for index, row in enumerate(roster["rows"]):
        source_depth = qplane.load_source_depth(source_root, row)
        up_camera, pose_delta = qplane.pose_for_row(
            source_root, row, poses, maximum_pose_delta
        )
        qplane.require(
            abs(
                pose_delta
                - float(candidate["frames"][index]["pose_association_delta_seconds"])
            )
            <= 1e-12,
            "pose association drift from frozen candidate",
        )
        records.extend(
            qplane.evaluate_frame(
                row,
                candidate["frames"][index],
                np.asarray(base_cache[index], dtype=np.float64),
                source_depth,
                up_camera,
                protocol["representation"],
                protocol["negative_controls"],
            )
        )
    per_query = summarize_per_query(records)
    qplane.require(len(per_query) == 9, "frozen query count drift")
    output = {
        "schema": "blindassist_ba_clear_qplane_o0a_query_decomposition_replay_v1",
        "status": "REPORTING_REPLAY_OF_FROZEN_FAIL_TERMINAL",
        "source_result_sha256": EXPECTED_RESULT_SHA256,
        "candidate_plan_sha256": EXPECTED_CANDIDATE_SHA256,
        "query_count": len(per_query),
        "per_query": per_query,
        "runtime_diagnostic": {
            "device": "CPU",
            "seconds": time.perf_counter() - started,
            "claim_ceiling": "HOST_CPU_REPORTING_REPLAY_ONLY",
        },
        "authority": {
            "candidate_refit": False,
            "gate_or_parameter_change": False,
            "training_authorized": False,
            "fresh_outcome_read": False,
            "android_qnn_htp_authorized": False,
        },
    }
    qplane.write_json(output_path, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "query-decomposition.json",
    )
    args = parser.parse_args()
    output = run(args.experiment_dir.resolve(), args.output.resolve())
    print(
        json.dumps(
            {
                "status": output["status"],
                "query_count": output["query_count"],
                "runtime_diagnostic": output["runtime_diagnostic"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
