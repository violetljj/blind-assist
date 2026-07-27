"""Freeze pose-ranked ETH3D 10-second window queues without assigning roles."""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any

import numpy as np

from scripts.research.egomotion_compensated_looming.real_positive_approach_role_admission_r2_cid_sims import (
    producer as geometry,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON_OBJECT_REQUIRED")
    return value


def depth_rows(path: Path, sequence_id: str) -> list[tuple[Decimal, str]]:
    rows: list[tuple[Decimal, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        timestamp, member = line.split()
        normalized = member.replace("\\", "/").lstrip("/")
        if not normalized.startswith(f"{sequence_id}/"):
            normalized = f"{sequence_id}/{normalized}"
        rows.append((Decimal(timestamp), normalized))
    if not rows or any(left[0] >= right[0] for left, right in zip(rows, rows[1:])):
        raise ValueError(f"DEPTH_INDEX_NOT_MONOTONIC:{path}")
    return rows


def longest(durations: list[float], predicate: list[bool]) -> float:
    best = current = 0.0
    for duration, matches in zip(durations, predicate):
        current = current + duration if matches else 0.0
        best = max(best, current)
    return best


def analyze_window(
    sequence_id: str,
    start: Decimal,
    rows: list[tuple[Decimal, str]],
    poses: Any,
) -> dict[str, Any] | None:
    end = start + Decimal("10")
    selected = [(timestamp, member) for timestamp, member in rows if start <= timestamp < end]
    if len(selected) < 250:
        return None
    deltas = [float(right[0] - left[0]) for left, right in zip(selected, selected[1:])]
    if not deltas or max(deltas) > 0.1:
        return None
    pose_samples = []
    for timestamp, _ in selected:
        try:
            pose_samples.append((timestamp, geometry._interpolate_pose(poses, timestamp)))
        except ValueError:
            pose_samples.append((timestamp, None))
    speeds: list[float] = []
    speed_durations: list[float] = []
    for (left_time, left_pose), (right_time, right_pose) in zip(pose_samples, pose_samples[1:]):
        if left_pose is None or right_pose is None:
            continue
        dt = float(right_time - left_time)
        displacement = geometry._rotation(left_pose[1]).T @ (right_pose[0] - left_pose[0])
        speeds.append(float(displacement[2]) / dt)
        speed_durations.append(dt)
    coverage = len(speeds) / (len(selected) - 1)
    if coverage < 0.8:
        return None
    positive_flags = [value >= 0.05 for value in speeds]
    below_flags = [value < 0.01 for value in speeds]
    start_pose = next((pose for _, pose in pose_samples if pose is not None), None)
    end_pose = next((pose for _, pose in reversed(pose_samples) if pose is not None), None)
    if start_pose is None or end_pose is None:
        return None
    net = geometry._rotation(start_pose[1]).T @ (end_pose[0] - start_pose[0])
    return {
        "window_id": f"{sequence_id}@{start}",
        "sequence_id": sequence_id,
        "start_timestamp_s": str(start),
        "end_timestamp_s": str(end),
        "frame_count": len(selected),
        "pair_count": len(selected) - 1,
        "depth_members": [member for _, member in selected],
        "pose_pair_coverage": coverage,
        "proxy_forward_speed_positive_fraction": sum(positive_flags) / len(speeds),
        "proxy_forward_speed_below_fraction": sum(below_flags) / len(speeds),
        "proxy_longest_positive_duration_s": longest(speed_durations, positive_flags),
        "proxy_longest_below_duration_s": longest(speed_durations, below_flags),
        "proxy_median_forward_speed_m_s": statistics.median(speeds),
        "proxy_net_translation_start_camera_xyz_m": [float(value) for value in net],
        "motion_role": "UNKNOWN_UNTIL_DEPTH_GEOMETRY",
    }


def overlaps(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left["sequence_id"] != right["sequence_id"]:
        return False
    return not (
        Decimal(left["end_timestamp_s"]) <= Decimal(right["start_timestamp_s"])
        or Decimal(right["end_timestamp_s"]) <= Decimal(left["start_timestamp_s"])
    )


def queue(
    candidates: list[dict[str, Any]],
    key: Any,
    limit: int,
) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    for row in sorted(candidates, key=key):
        if not any(overlaps(row, prior) for prior in chosen):
            chosen.append(row)
        if len(chosen) == limit:
            break
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queue-limit", type=int, default=16)
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    contract = load(contract_path)
    if contract["search_policy"]["trajectory_role_authority"] is not False:
        raise ValueError("TRAJECTORY_ROLE_AUTHORITY")
    metadata_root = args.metadata_root.resolve()
    candidates: list[dict[str, Any]] = []
    source_receipts = []
    for source in contract["source_family"]["predeclared_sequence_order"]:
        sequence_id = source["sequence_id"]
        root = metadata_root / sequence_id / sequence_id
        depth_index = root / "depth.txt"
        groundtruth = root / "groundtruth.txt"
        calibration = root / "calibration.txt"
        rows = depth_rows(depth_index, sequence_id)
        poses = geometry._parse_poses(groundtruth.read_bytes())
        geometry._parse_intrinsic(calibration.read_bytes())
        anchor = rows[0][0]
        final_start = rows[-1][0] - Decimal("10")
        start = anchor
        count = 0
        while start <= final_start:
            candidate = analyze_window(sequence_id, start, rows, poses)
            if candidate is not None:
                candidate["sequence_order"] = count
                candidates.append(candidate)
            start += Decimal("1")
            count += 1
        source_receipts.append(
            {
                "sequence_id": sequence_id,
                "depth_index_sha256": sha(depth_index),
                "groundtruth_sha256": sha(groundtruth),
                "calibration_sha256": sha(calibration),
                "depth_index_count": len(rows),
            }
        )
    positive = queue(
        candidates,
        lambda row: (
            -row["proxy_longest_positive_duration_s"],
            -row["proxy_forward_speed_positive_fraction"],
            -row["proxy_median_forward_speed_m_s"],
            row["sequence_id"],
            Decimal(row["start_timestamp_s"]),
        ),
        args.queue_limit,
    )
    below = queue(
        candidates,
        lambda row: (
            -row["proxy_longest_below_duration_s"],
            -row["proxy_forward_speed_below_fraction"],
            row["proxy_median_forward_speed_m_s"],
            row["sequence_id"],
            Decimal(row["start_timestamp_s"]),
        ),
        args.queue_limit,
    )
    result = {
        "schema": "rcle.motion_diverse_rgbd.source_search.pose_queue.v1",
        "protocol_id": contract["protocol_id"],
        "contract_sha256": sha(contract_path),
        "trajectory_role_authority": False,
        "candidate_window_count": len(candidates),
        "source_receipts": source_receipts,
        "positive_proxy_queue": positive,
        "below_proxy_queue": below,
        "queue_policy": "Frozen before any R1 depth geometry; non-overlapping within each queue.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_window_count": len(candidates), "positive_head": [row["window_id"] for row in positive[:4]], "below_head": [row["window_id"] for row in below[:4]]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
