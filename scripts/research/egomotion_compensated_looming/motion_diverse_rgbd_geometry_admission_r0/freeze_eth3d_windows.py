"""Freeze all complete ETH3D desk_3 10-second window identities pose-only."""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import statistics
from typing import Any

import numpy as np

from scripts.research.egomotion_compensated_looming.motion_diverse_rgbd_geometry_admission_r0.template import (
    validate_execution_contract,
)
from scripts.research.egomotion_compensated_looming.real_positive_approach_role_admission_r2_cid_sims import (
    producer as geometry,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON_OBJECT_REQUIRED")
    return value


def depth_index(path: Path) -> list[tuple[Decimal, str]]:
    rows: list[tuple[Decimal, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        values = stripped.split()
        if len(values) != 2:
            raise ValueError("DEPTH_INDEX_COLUMNS")
        member = PurePosixPath(values[1])
        if (
            member.is_absolute()
            or ".." in member.parts
            or len(member.parts) != 2
            or member.parts[0] != "depth"
            or member.suffix != ".png"
        ):
            raise ValueError("DEPTH_INDEX_MEMBER")
        rows.append((Decimal(values[0]), f"desk_3/{member.as_posix()}"))
    if not rows or any(left[0] >= right[0] for left, right in zip(rows, rows[1:])):
        raise ValueError("DEPTH_INDEX_NOT_MONOTONIC")
    return rows


def write_exclusive(path: Path, value: object) -> None:
    payload = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "wb", closefd=True) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--claim", type=Path, required=True)
    parser.add_argument("--depth-index", type=Path, required=True)
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    contract = load_object(contract_path)
    validate_execution_contract(contract)
    claim = load_object(args.claim.resolve())
    if claim.get("candidate_id") != "ETH3D_SLAM_DESK_3":
        raise ValueError("CLAIM_CANDIDATE")
    if claim.get("contract_sha256") != sha256_file(contract_path):
        raise ValueError("CLAIM_CONTRACT_IDENTITY")
    rows = depth_index(args.depth_index.resolve())
    poses = geometry._parse_poses(args.groundtruth.resolve().read_bytes())
    geometry._parse_intrinsic(args.calibration.resolve().read_bytes())
    duration = Decimal(contract["geometry_only_selection"]["window_duration_s"])
    anchor = rows[0][0]
    complete_count = int((rows[-1][0] - anchor) // duration)
    if complete_count < 4:
        raise ValueError("FEWER_THAN_FOUR_COMPLETE_WINDOWS")
    windows = []
    for index in range(complete_count):
        start = anchor + Decimal(index) * duration
        end = start + duration
        selected = [(timestamp, member) for timestamp, member in rows if start <= timestamp < end]
        if len(selected) < 250:
            raise ValueError(f"WINDOW_FRAME_COUNT:{index}:{len(selected)}")
        deltas = [
            float(right[0] - left[0])
            for left, right in zip(selected, selected[1:])
        ]
        identity_eligible = bool(deltas) and max(deltas) <= 0.1
        pose_samples: list[tuple[Decimal, tuple[np.ndarray, np.ndarray]]] = []
        for timestamp, _ in selected:
            try:
                pose_samples.append(
                    (timestamp, geometry._interpolate_pose(poses, timestamp))
                )
            except ValueError as error:
                if str(error) not in {
                    "R2_POSE_NOT_BRACKETED",
                    "R2_POSE_BRACKET_TOO_WIDE",
                }:
                    raise
        if len(pose_samples) < 2:
            raise ValueError(f"WINDOW_POSE_COVERAGE_EMPTY:{index}")
        start_pose = pose_samples[0][1]
        end_pose = pose_samples[-1][1]
        start_rotation = geometry._rotation(start_pose[1])
        displacement_start_camera = start_rotation.T @ (
            end_pose[0] - start_pose[0]
        )
        pose_centers = [pose for _, (pose, _) in pose_samples]
        path_length = sum(
            float(np.linalg.norm(right - left))
            for left, right in zip(pose_centers, pose_centers[1:])
        )
        windows.append(
            {
                "window_index": index,
                "start_timestamp_s": str(start),
                "end_timestamp_s": str(end),
                "frame_count": len(selected),
                "pair_count": len(selected) - 1,
                "identity_eligible": identity_eligible,
                "identity_reason": (
                    None
                    if identity_eligible
                    else "SOURCE_CONSECUTIVE_DEPTH_DT_GT_0P1"
                ),
                "depth_members": [member for _, member in selected],
                "pose_only_diagnostic_not_role": {
                    "net_translation_start_camera_xyz_m": [
                        float(value) for value in displacement_start_camera
                    ],
                    "net_translation_norm_m": float(
                        np.linalg.norm(displacement_start_camera)
                    ),
                    "path_length_m": path_length,
                    "pose_interpolable_frame_count": len(pose_samples),
                    "pose_interpolable_fraction": (
                        len(pose_samples) / len(selected)
                    ),
                    "median_frame_dt_s": statistics.median(deltas),
                    "maximum_frame_dt_s": max(deltas),
                },
                "motion_role": "UNKNOWN_UNTIL_DEPTH_GEOMETRY",
            }
        )
    result = {
        "schema": "rcle.motion_diverse_rgbd.eth3d_window_freeze.v1",
        "candidate_id": "ETH3D_SLAM_DESK_3",
        "contract_sha256": sha256_file(contract_path),
        "claim_sha256": sha256_file(args.claim.resolve()),
        "depth_index_sha256": sha256_file(args.depth_index.resolve()),
        "groundtruth_sha256": sha256_file(args.groundtruth.resolve()),
        "calibration_sha256": sha256_file(args.calibration.resolve()),
        "window_duration_s": str(duration),
        "complete_window_count": complete_count,
        "trajectory_role_authority": False,
        "post_geometry_window_addition_allowed": False,
        "windows": windows,
    }
    write_exclusive(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "candidate_id": result["candidate_id"],
                "complete_window_count": complete_count,
                "frame_counts": [window["frame_count"] for window in windows],
                "trajectory_role_authority": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
