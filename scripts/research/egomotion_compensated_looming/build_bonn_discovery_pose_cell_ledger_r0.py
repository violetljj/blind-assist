#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import json
import math
import zipfile
from pathlib import Path
from typing import Any


WINDOW_SECONDS = 10.0
MAX_POSE_JOIN_DELTA_SECONDS = 0.040
MIN_JOINED_SAMPLES = 250
C1_MAX_TRANSLATION_PATH_METERS = 0.30
C1_MAX_TRANSLATION_ENDPOINT_METERS = 0.15
C1_MIN_ANGULAR_PATH_RADIANS = math.radians(20.0)
C2_MIN_TRANSLATION_PATH_METERS = 0.75
C2_MIN_TRANSLATION_ENDPOINT_METERS = 0.50


def rows(data: bytes, columns: int) -> list[list[str]]:
    result = [
        line.split()
        for line in data.decode("utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    if not result or any(len(row) != columns for row in result):
        raise ValueError("invalid source index")
    times = [float(row[0]) for row in result]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("timestamps must be strictly increasing")
    return result


def nearest_pose(
    poses: list[list[str]], pose_times: list[float], timestamp: float
) -> tuple[list[str], float] | None:
    index = bisect.bisect_left(pose_times, timestamp)
    candidates = poses[max(0, index - 1) : min(len(poses), index + 1)]
    if not candidates:
        return None
    selected = min(candidates, key=lambda row: abs(float(row[0]) - timestamp))
    delta = abs(float(selected[0]) - timestamp)
    if delta > MAX_POSE_JOIN_DELTA_SECONDS:
        return None
    return selected, delta


def norm3(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def quaternion_step(left: list[float], right: list[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("zero quaternion")
    dot = abs(
        sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
    )
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


def audit_sequence(
    archive_path: Path, sequence_id: str, expected_archive_sha256: str
) -> dict[str, Any]:
    import hashlib

    digest = hashlib.sha256()
    with archive_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_archive_sha256:
        raise ValueError("archive SHA-256 mismatch")

    root = f"{sequence_id}/"
    with zipfile.ZipFile(archive_path) as archive:
        rgb = rows(archive.read(f"{root}rgb.txt"), 2)
        poses = rows(archive.read(f"{root}groundtruth.txt"), 8)
    pose_times = [float(row[0]) for row in poses]
    rgb_times = [float(row[0]) for row in rgb]

    windows: list[dict[str, Any]] = []
    start = rgb_times[0]
    index = 0
    while start + WINDOW_SECONDS <= rgb_times[-1]:
        end = start + WINDOW_SECONDS
        frame_times = [value for value in rgb_times if start <= value < end]
        joined: list[tuple[float, list[float], list[float], float]] = []
        for timestamp in frame_times:
            match = nearest_pose(poses, pose_times, timestamp)
            if match is None:
                continue
            pose, delta = match
            joined.append(
                (
                    timestamp,
                    [float(value) for value in pose[1:4]],
                    [float(value) for value in pose[4:8]],
                    delta,
                )
            )
        if len(joined) < MIN_JOINED_SAMPLES:
            windows.append(
                {
                    "window_index": index,
                    "start_timestamp": start,
                    "end_timestamp_exclusive": end,
                    "rgb_frame_count": len(frame_times),
                    "joined_pose_count": len(joined),
                    "eligible": False,
                    "abstained": True,
                    "abstention_reason": "INSUFFICIENT_POSE_JOINED_SAMPLES",
                }
            )
        else:
            translations = [item[1] for item in joined]
            quaternions = [item[2] for item in joined]
            translation_steps = [
                norm3(left, right)
                for left, right in zip(translations, translations[1:])
            ]
            angular_steps = [
                quaternion_step(left, right)
                for left, right in zip(quaternions, quaternions[1:])
            ]
            translation_path = sum(translation_steps)
            translation_endpoint = norm3(translations[0], translations[-1])
            angular_path = sum(angular_steps)
            angular_endpoint = quaternion_step(quaternions[0], quaternions[-1])
            c1_candidate = (
                translation_path <= C1_MAX_TRANSLATION_PATH_METERS
                and translation_endpoint <= C1_MAX_TRANSLATION_ENDPOINT_METERS
                and angular_path >= C1_MIN_ANGULAR_PATH_RADIANS
            )
            c2_candidate = (
                translation_path >= C2_MIN_TRANSLATION_PATH_METERS
                and translation_endpoint >= C2_MIN_TRANSLATION_ENDPOINT_METERS
            )
            windows.append(
                {
                    "window_index": index,
                    "start_timestamp": start,
                    "end_timestamp_exclusive": end,
                    "rgb_frame_count": len(frame_times),
                    "joined_pose_count": len(joined),
                    "eligible": True,
                    "abstained": False,
                    "abstention_reason": None,
                    "maximum_pose_join_delta_seconds": max(
                        item[3] for item in joined
                    ),
                    "translation_path_meters": translation_path,
                    "translation_endpoint_meters": translation_endpoint,
                    "angular_path_radians": angular_path,
                    "angular_endpoint_radians": angular_endpoint,
                    "c1_pose_mechanics_candidate": c1_candidate,
                    "c2_translation_mechanics_candidate": c2_candidate,
                    "cell_truth_proven": False,
                }
            )
        index += 1
        start += WINDOW_SECONDS
    return {
        "sequence_id": sequence_id,
        "window_count": len(windows),
        "windows": windows,
    }


def build_ledger(
    acquisition: dict[str, Any], archive_dir: Path
) -> dict[str, Any]:
    sequences = [
        audit_sequence(
            archive_dir / item["archive_filename"],
            item["sequence_id"],
            item["archive_sha256"],
        )
        for item in acquisition["archives"]
    ]
    windows = [window for sequence in sequences for window in sequence["windows"]]
    return {
        "schema_version": "bonn_discovery_pose_cell_ledger_r0",
        "goal_id": "EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R1",
        "source_family": "BONN_RGBD_DYNAMIC",
        "window_contract": {
            "window_seconds": WINDOW_SECONDS,
            "non_overlapping_from_first_rgb_timestamp": True,
            "maximum_pose_join_delta_seconds": MAX_POSE_JOIN_DELTA_SECONDS,
            "minimum_joined_samples": MIN_JOINED_SAMPLES,
        },
        "proposal_contract": {
            "c1_max_translation_path_meters": C1_MAX_TRANSLATION_PATH_METERS,
            "c1_max_translation_endpoint_meters": C1_MAX_TRANSLATION_ENDPOINT_METERS,
            "c1_min_angular_path_radians": C1_MIN_ANGULAR_PATH_RADIANS,
            "c2_min_translation_path_meters": C2_MIN_TRANSLATION_PATH_METERS,
            "c2_min_translation_endpoint_meters": C2_MIN_TRANSLATION_ENDPOINT_METERS,
            "pose_mechanics_only": True,
            "cell_truth_proven": False,
        },
        "sequences": sequences,
        "counts": {
            "sequence_count": len(sequences),
            "window_count": len(windows),
            "eligible_window_count": sum(
                bool(window.get("eligible")) for window in windows
            ),
            "c1_pose_mechanics_candidate_count": sum(
                bool(window.get("c1_pose_mechanics_candidate")) for window in windows
            ),
            "c2_translation_mechanics_candidate_count": sum(
                bool(window.get("c2_translation_mechanics_candidate"))
                for window in windows
            ),
        },
        "read_firewall": {
            "validation_or_holdout_read_count": 0,
            "image_member_read_or_decode_count": 0,
            "old_window_selection_tuning_acceptance_reads": 0,
            "candidate_signal_computed": False,
        },
        "terminal": "BONN_DISCOVERY_POSE_MECHANICS_LEDGER_AVAILABLE_CELL_TRUTH_PENDING",
        "status": "VALID",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acquisition", required=True, type=Path)
    parser.add_argument("--archive-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    acquisition = json.loads(args.acquisition.read_text(encoding="utf-8"))
    ledger = build_ledger(acquisition, args.archive_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": ledger["status"],
                "terminal": ledger["terminal"],
                **ledger["counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
