"""Build a fail-closed OpenLORIS trajectory candidate ledger for R3.

This tool only checks official source metadata and independent trajectory motion.
It cannot grant three-source credit or replace complete-sequence model review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from contract import read_json, sha256, write_json


SCHEMA = "blindassist_ustrf_sensor_replay_r3_openloris_prescreen_report_v1"


def trajectory_stats(path: Path) -> dict[str, float | int]:
    rows = np.loadtxt(path, comments="#", dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] < 2 or rows.shape[1] != 8:
        raise ValueError(f"invalid OpenLORIS ground-truth table: {path}")
    timestamps = rows[:, 0]
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"non-monotonic OpenLORIS ground-truth time: {path}")
    xyz = rows[:, 1:4]
    steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    duration_s = float(timestamps[-1] - timestamps[0])
    return {
        "ground_truth_sample_count": int(rows.shape[0]),
        "duration_s": duration_s,
        "path_length_m": float(steps.sum()),
        "net_displacement_m": float(np.linalg.norm(xyz[-1] - xyz[0])),
        "median_speed_mps": float(np.median(steps / np.diff(timestamps))),
    }


def build_report(
    repo: Path,
    config_path: Path,
    groundtruth_root: Path,
    groundtruth_archive: Path,
) -> dict[str, Any]:
    config = read_json(config_path)
    prereg = config["frozen_r3_prereg"]
    prereg_path = (repo / prereg["path"]).resolve()
    actual_prereg_sha256 = sha256(prereg_path)
    if actual_prereg_sha256 != prereg["sha256"]:
        raise ValueError("frozen R3 prereg hash mismatch; source replacement must not tune the round")
    review = config["complete_sequence_two_model_review"]
    if int(review["anchor_tolerance_frames"]) != 15:
        raise ValueError("R3 complete-sequence review tolerance changed")
    if int(review["minimum_admitted_trajectories_before_source_count_credit"]) != 3:
        raise ValueError("R3 minimum admitted source count changed")
    actual_groundtruth_archive_sha256 = sha256(groundtruth_archive)
    expected_groundtruth_archive_sha256 = config["dataset"]["groundtruth_archive"]["lfs_sha256"]
    if actual_groundtruth_archive_sha256 != expected_groundtruth_archive_sha256:
        raise ValueError("OpenLORIS ground-truth archive hash mismatch")

    archive_by_scene = {row["scene"]: row for row in config["archives"]}
    candidates = []
    seen: set[str] = set()
    for row in config["trajectory_candidates"]:
        trajectory_id = row["trajectory_id"]
        if trajectory_id in seen:
            raise ValueError(f"duplicate OpenLORIS trajectory identity: {trajectory_id}")
        seen.add(trajectory_id)
        truth_path = (groundtruth_root / trajectory_id / "groundtruth.txt").resolve()
        if groundtruth_root.resolve() not in truth_path.parents or not truth_path.is_file():
            raise ValueError(f"missing OpenLORIS ground truth: {trajectory_id}")
        archive = archive_by_scene[row["scene"]]
        candidates.append(
            {
                "trajectory_id": trajectory_id,
                "scene": row["scene"],
                "priority": int(row["priority"]),
                "trajectory_authority": archive["trajectory_authority"],
                "ground_truth_sha256": sha256(truth_path),
                **trajectory_stats(truth_path),
                "rgbd_archive_path": archive["path"],
                "rgbd_archive_lfs_sha256": archive["lfs_sha256"],
                "metadata_prescreen_passed": True,
                "visual_lifecycle_prescreen": "pending_complete_rgbd_archive",
                "complete_sequence_two_model_admitted": False,
                "source_count_credit": 0,
            }
        )
    candidates.sort(key=lambda value: (value["priority"], -value["path_length_m"], value["trajectory_id"]))
    return {
        "schema": SCHEMA,
        "authority": "discovery_candidate_only",
        "config_sha256": sha256(config_path),
        "frozen_r3_prereg_sha256": actual_prereg_sha256,
        "groundtruth_archive_sha256": actual_groundtruth_archive_sha256,
        "candidate_trajectory_count": len(candidates),
        "admitted_trajectory_count": 0,
        "three_source_count_credit": False,
        "unchanged_review_anchor_tolerance_frames": 15,
        "candidates": candidates,
        "next_gate": "download only ranked RGB-D candidates, inspect complete consecutive clips, then require both isolated model admissions before counting any trajectory",
        "production_authority": False,
        "hardware_selection_authorized": False,
        "u0_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--groundtruth-root", type=Path, required=True)
    parser.add_argument("--groundtruth-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = build_report(
            args.repo.resolve(),
            args.config.resolve(),
            args.groundtruth_root.resolve(),
            args.groundtruth_archive.resolve(),
        )
        write_json(args.output.resolve(), report)
        print(json.dumps({
            "candidate_trajectory_count": report["candidate_trajectory_count"],
            "admitted_trajectory_count": report["admitted_trajectory_count"],
            "three_source_count_credit": report["three_source_count_credit"],
        }))
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
