"""Freeze a pose-ranked positive queue from a cached TartanAir archive."""

from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
from pathlib import Path
import statistics
import tarfile
from typing import Any

import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON_OBJECT_REQUIRED")
    return value


def longest(flags: list[bool], dt: float) -> float:
    best = current = 0.0
    for flag in flags:
        current = current + dt if flag else 0.0
        best = max(best, current)
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queue-limit", type=int, default=16)
    args = parser.parse_args()
    amendment = load(args.amendment.resolve())
    archive_path = args.archive.resolve()
    archive_sha256 = sha(archive_path)
    if archive_sha256 != amendment["source"]["archive_sha256"]:
        raise ValueError("TARTANAIR_ARCHIVE_IDENTITY")
    frame_rate = float(amendment["source"]["frame_rate_hz"])
    frames_per_window = int(float(amendment["selection"]["window_duration_s"]) * frame_rate)
    by_trajectory: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith("_cam.npz"):
                continue
            trajectory, filename = member.name.rsplit("/", 1)
            frame_id = int(filename.removesuffix("_cam.npz"))
            payload = archive.extractfile(member).read()
            with np.load(BytesIO(payload)) as camera:
                by_trajectory.setdefault(trajectory, {})[frame_id] = (
                    np.asarray(camera["camera_pose"], dtype=np.float64),
                    np.asarray(camera["camera_intrinsics"], dtype=np.float64),
                )
    candidates = []
    for trajectory, frames in sorted(by_trajectory.items()):
        frame_ids = sorted(frames)
        poses = {frame_id: frames[frame_id][0] for frame_id in frame_ids}
        intrinsics = {frame_id: frames[frame_id][1] for frame_id in frame_ids}
        for offset in range(0, max(0, len(frame_ids) - frames_per_window + 1), 10):
            selected = frame_ids[offset : offset + frames_per_window]
            if len(selected) != frames_per_window or selected != list(range(selected[0], selected[0] + frames_per_window)):
                continue
            speeds = []
            for left_id, right_id in zip(selected, selected[1:]):
                left, right = poses[left_id], poses[right_id]
                displacement = left[:3, :3].T @ (right[:3, 3] - left[:3, 3])
                speeds.append(float(displacement[2]) * frame_rate)
            flags = [value >= 0.05 for value in speeds]
            candidates.append(
                {
                    "window_id": f"{trajectory}@{selected[0]:06d}",
                    "trajectory": trajectory,
                    "start_frame_id": f"{selected[0]:06d}",
                    "end_frame_id_exclusive": f"{selected[-1] + 1:06d}",
                    "frame_ids": [f"{frame_id:06d}" for frame_id in selected],
                    "pair_count": len(selected) - 1,
                    "proxy_forward_speed_positive_fraction": sum(flags) / len(flags),
                    "proxy_longest_positive_duration_s": longest(flags, 1.0 / frame_rate),
                    "proxy_median_forward_speed_m_s": statistics.median(speeds),
                    "intrinsics": intrinsics[selected[0]].tolist(),
                    "motion_role": "UNKNOWN_UNTIL_DEPTH_GEOMETRY",
                }
            )
    ordered = sorted(
        candidates,
        key=lambda row: (
            -row["proxy_longest_positive_duration_s"],
            -row["proxy_forward_speed_positive_fraction"],
            -row["proxy_median_forward_speed_m_s"],
            row["window_id"],
        ),
    )
    queue = []
    for row in ordered:
        start = int(row["start_frame_id"])
        end = int(row["end_frame_id_exclusive"])
        if any(
            prior["trajectory"] == row["trajectory"]
            and not (
                end <= int(prior["start_frame_id"])
                or int(prior["end_frame_id_exclusive"]) <= start
            )
            for prior in queue
        ):
            continue
        queue.append(row)
        if len(queue) == args.queue_limit:
            break
    result = {
        "schema": "rcle.motion_diverse_rgbd.source_search.tartanair_pose_queue.v1",
        "protocol_id": amendment["protocol_id"],
        "amendment_sha256": sha(args.amendment.resolve()),
        "archive_sha256": archive_sha256,
        "trajectory_role_authority": False,
        "candidate_window_count": len(candidates),
        "positive_proxy_queue": queue,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_window_count": len(candidates), "positive_head": [row["window_id"] for row in queue[:4]]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
