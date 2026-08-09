#!/usr/bin/env python3
"""Audit whether every frozen ARKitScenes frame is inside its pose domain."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "blindassist_assistive_geometry_b0_arkitscenes_pose_coverage_audit_v1"
EXPECTED_MANIFEST_SCHEMA = "blindassist_assistive_geometry_b0_arkitscenes_media_manifest_v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def frame_timestamp(stem: str) -> float:
    value = float(stem.rsplit("_", 1)[1])
    require(math.isfinite(value), f"non-finite frame timestamp: {stem}")
    return value


def read_trajectory_timestamps(path: Path) -> list[float]:
    timestamps: list[float] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        require(len(fields) == 7, f"trajectory row {line_number} does not have 7 fields: {path}")
        values = [float(field) for field in fields]
        require(all(math.isfinite(value) for value in values), f"non-finite trajectory row {line_number}: {path}")
        timestamps.append(values[0])
    require(len(timestamps) >= 2, f"trajectory has fewer than two rows: {path}")
    require(all(left < right for left, right in zip(timestamps, timestamps[1:])), f"trajectory is not strictly increasing: {path}")
    return timestamps


def audit_video(video: dict[str, Any]) -> dict[str, Any]:
    selected = [frame_timestamp(str(stem)) for stem in video["selected_frame_stems"]]
    require(len(selected) == int(video["selected_frame_count"]), "selected frame count drift")
    require(all(left < right for left, right in zip(selected, selected[1:])), "frame timestamps are not strictly increasing")
    rgb_entries = video["extracted"]["lowres_wide"]
    require(rgb_entries, "RGB entries missing")
    video_root = Path(rgb_entries[0]["path"]).parent.parent
    trajectory_path = video_root / "lowres_wide.traj"
    require(trajectory_path.is_file(), f"trajectory missing: {trajectory_path}")
    trajectory = read_trajectory_timestamps(trajectory_path)
    covered = [timestamp for timestamp in selected if trajectory[0] <= timestamp <= trajectory[-1]]
    return {
        "role": str(video["role"]),
        "visit_id": str(video["visit_id"]),
        "video_id": str(video["video_id"]),
        "selected_frame_count": len(selected),
        "pose_covered_frame_count": len(covered),
        "all_selected_frames_pose_covered": len(covered) == len(selected),
        "selected_start_timestamp": selected[0],
        "selected_end_timestamp": selected[-1],
        "trajectory_start_timestamp": trajectory[0],
        "trajectory_end_timestamp": trajectory[-1],
        "selected_start_minus_trajectory_start_seconds": selected[0] - trajectory[0],
        "selected_end_minus_trajectory_end_seconds": selected[-1] - trajectory[-1],
        "trajectory_row_count": len(trajectory),
        "trajectory_sha256": sha256_file(trajectory_path),
    }


def build_receipt(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema") == EXPECTED_MANIFEST_SCHEMA, "manifest schema drift")
    require(manifest.get("task_outcome_opened") is False, "task outcome boundary drift")
    require(manifest.get("model_outputs_read") is False, "model output boundary drift")
    videos = [audit_video(video) for video in manifest["videos"]]
    failed = [video for video in videos if not video["all_selected_frames_pose_covered"]]
    return {
        "schema": SCHEMA,
        "producer": {
            "path": "scripts/research/assistive_geometry/audit_b0_arkitscenes_pose_coverage.py",
            "sha256": sha256_file(Path(__file__)),
        },
        "manifest": {"path": str(manifest_path.resolve()), "sha256": sha256_file(manifest_path)},
        "video_count": len(videos),
        "all_selected_frames_pose_covered_video_count": len(videos) - len(failed),
        "failed_video_count": len(failed),
        "minimum_pose_covered_frame_count": min(video["pose_covered_frame_count"] for video in videos),
        "videos": videos,
        "task_outcome_opened": False,
        "model_outputs_read": False,
        "terminal": (
            "B0_ARKIT_MEDIA_POSE_COVERAGE_PASS"
            if not failed
            else "B0_ARKIT_MEDIA_POSE_COVERAGE_FAIL_REMATERIALIZATION_REQUIRED"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_receipt(args.manifest.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in receipt.items() if key != "videos"}, indent=2))
    return 0 if receipt["failed_video_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
