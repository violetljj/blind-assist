#!/usr/bin/env python3
"""Qualify CrowdBot source capacity without running any frozen candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SOURCE_PREFIXES = {
    "crowdbot_0325_shared_control": "0325_shared_control_defaced_processed",
    "crowdbot_0424_rds": "0424_rds_defaced_processed",
}
ROUTE_HORIZON_SECONDS = 24.0 / 15.0
CORRIDOR_HALF_WIDTH_METERS = 0.45
MIN_ALERT_FRAMES = 2
MIN_CLEAR_FRAMES = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_name(path: Path) -> str:
    name = path.name
    marker = "__defaced_"
    start = name.index(marker) + 2
    for suffix in ("_tfqolo_sampled.npy", "_stamped.npy", ".npy"):
        if name.endswith(suffix):
            return name[start : -len(suffix)]
    raise RuntimeError(f"unsupported metadata filename: {name}")


def point_to_polyline_distances(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    if len(polyline) == 1:
        return np.linalg.norm(points - polyline[0], axis=1)
    starts = polyline[:-1]
    vectors = polyline[1:] - starts
    denom = np.sum(vectors * vectors, axis=1)
    delta = points[:, None, :] - starts[None, :, :]
    numer = np.sum(delta * vectors[None, :, :], axis=2)
    fraction = np.divide(numer, denom[None, :], out=np.zeros_like(numer), where=denom[None, :] > 1e-12)
    fraction = np.clip(fraction, 0.0, 1.0)
    closest = starts[None, :, :] + fraction[:, :, None] * vectors[None, :, :]
    return np.min(np.linalg.norm(points[:, None, :] - closest, axis=2), axis=1)


def longest_consecutive(frames: list[int]) -> int:
    best = current = 0
    previous: int | None = None
    for frame in frames:
        current = current + 1 if previous is not None and frame == previous + 1 else 1
        best = max(best, current)
        previous = frame
    return best


def split_episodes(frames: list[int]) -> list[list[int]]:
    episodes: list[list[int]] = []
    for frame in sorted(frames):
        if not episodes or frame - episodes[-1][-1] >= MIN_CLEAR_FRAMES:
            episodes.append([frame])
        else:
            episodes[-1].append(frame)
    return episodes


def qualify_source(metadata_root: Path, source_id: str, prefix: str) -> dict[str, Any]:
    track_paths = sorted(metadata_root.glob(f"{prefix}__alg_res__tracks_2D__*.npy"))
    pose_paths = sorted(metadata_root.glob(f"{prefix}__source_data__tf_qolo__*_tfqolo_sampled.npy"))
    timestamp_paths = sorted(metadata_root.glob(f"{prefix}__source_data__timestamp__*_stamped.npy"))
    tracks_by_sequence = {sequence_name(path): path for path in track_paths}
    poses_by_sequence = {sequence_name(path): path for path in pose_paths}
    timestamps_by_sequence = {sequence_name(path): path for path in timestamp_paths}
    sequence_ids = sorted(set(tracks_by_sequence) & set(poses_by_sequence) & set(timestamps_by_sequence))
    if not sequence_ids or sequence_ids != sorted(timestamps_by_sequence):
        raise RuntimeError(f"metadata coverage incomplete for {source_id}")
    source_events: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    total_exposure = 0.0
    total_scorable = 0.0
    total_negative = 0.0
    total_frames = 0
    total_scorable_frames = 0
    total_cooccurrence_frames = 0
    total_active_frames = 0
    input_bindings: list[dict[str, str]] = []
    for sequence_id in sequence_ids:
        track_path = tracks_by_sequence[sequence_id]
        pose_path = poses_by_sequence[sequence_id]
        timestamp_path = timestamps_by_sequence[sequence_id]
        tracks = np.load(track_path, allow_pickle=True).item()
        pose = np.load(pose_path, allow_pickle=True).item()
        stamped = np.load(timestamp_path, allow_pickle=True).item()
        timestamps = np.asarray(stamped["timestamp"], dtype=float)
        positions = np.asarray(pose["position"], dtype=float)[:, :2]
        if len(timestamps) != len(positions) or set(tracks) != set(range(len(timestamps))):
            raise RuntimeError(f"frame alignment mismatch: {sequence_id}")
        dt = np.diff(timestamps, append=timestamps[-1])
        if len(dt) > 1:
            dt[-1] = float(np.median(dt[:-1]))
        scorable = np.zeros(len(timestamps), dtype=bool)
        active = np.zeros(len(timestamps), dtype=bool)
        cooccurrence = np.zeros(len(timestamps), dtype=bool)
        active_frames_by_track: dict[int, list[int]] = defaultdict(list)
        for frame_id, timestamp in enumerate(timestamps):
            end = int(np.searchsorted(timestamps, timestamp + ROUTE_HORIZON_SECONDS, side="left"))
            if end >= len(timestamps):
                continue
            scorable[frame_id] = True
            rows = np.asarray(tracks[frame_id])
            cooccurrence[frame_id] = len(rows) >= 2
            if len(rows) == 0:
                continue
            distances = point_to_polyline_distances(rows[:, :2], positions[frame_id : end + 1])
            intersecting = distances <= CORRIDOR_HALF_WIDTH_METERS
            if np.any(intersecting):
                active[frame_id] = True
                for track_id in rows[intersecting, -1].astype(int):
                    active_frames_by_track[int(track_id)].append(frame_id)
        events = []
        for track_id, frames in sorted(active_frames_by_track.items()):
            for episode_index, episode in enumerate(split_episodes(frames)):
                consecutive = longest_consecutive(episode)
                if consecutive < MIN_ALERT_FRAMES:
                    continue
                event = {
                    "event_id": f"{source_id}:{sequence_id}:{track_id}:{episode_index}",
                    "sequence_id": sequence_id,
                    "published_track_id": track_id,
                    "first_frame": episode[0],
                    "last_frame": episode[-1],
                    "intersecting_observation_count": len(episode),
                    "longest_consecutive_intersection_frames": consecutive,
                    "critical_capacity_proxy": consecutive >= MIN_CLEAR_FRAMES,
                }
                events.append(event)
                source_events.append(event)
        exposure = float(np.sum(dt))
        scorable_exposure = float(np.sum(dt[scorable]))
        negative_exposure = float(np.sum(dt[scorable & ~active]))
        sequence_rows.append(
            {
                "sequence_id": sequence_id,
                "frame_count": len(timestamps),
                "exposure_seconds": exposure,
                "scorable_route_seconds": scorable_exposure,
                "negative_route_seconds": negative_exposure,
                "cooccurrence_frame_count": int(np.sum(cooccurrence & scorable)),
                "active_route_frame_count": int(np.sum(active & scorable)),
                "positive_event_capacity_proxy": len(events),
                "critical_event_capacity_proxy": sum(row["critical_capacity_proxy"] for row in events),
            }
        )
        total_exposure += exposure
        total_scorable += scorable_exposure
        total_negative += negative_exposure
        total_frames += len(timestamps)
        total_scorable_frames += int(np.sum(scorable))
        total_cooccurrence_frames += int(np.sum(cooccurrence & scorable))
        total_active_frames += int(np.sum(active & scorable))
        for path in (track_path, pose_path, timestamp_path):
            input_bindings.append({"path": path.as_posix(), "sha256": sha256_file(path)})
    return {
        "source_id": source_id,
        "sequence_count": len(sequence_ids),
        "frame_count": total_frames,
        "exposure_minutes": total_exposure / 60.0,
        "scorable_route_minutes": total_scorable / 60.0,
        "negative_route_minutes_proxy": total_negative / 60.0,
        "cooccurrence_frame_rate_proxy": total_cooccurrence_frames / total_scorable_frames,
        "active_route_frame_rate_proxy": total_active_frames / total_scorable_frames,
        "positive_event_capacity_proxy": len(source_events),
        "critical_event_capacity_proxy": sum(row["critical_capacity_proxy"] for row in source_events),
        "sequence_metrics": sequence_rows,
        "input_bindings": input_bindings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", action="append", default=[], metavar="SOURCE_ID=PREFIX")
    args = parser.parse_args()
    source_prefixes = dict(SOURCE_PREFIXES)
    if args.source:
        source_prefixes = {}
        for value in args.source:
            if "=" not in value:
                raise RuntimeError("--source must use SOURCE_ID=PREFIX")
            source_id, prefix = value.split("=", 1)
            if not source_id or not prefix or source_id in source_prefixes:
                raise RuntimeError(f"invalid or duplicate --source: {value}")
            source_prefixes[source_id] = prefix
    sources = [qualify_source(args.metadata_root, source_id, prefix) for source_id, prefix in source_prefixes.items()]
    payload = {
        "schema": "blindassist_crowdbot_holdout_content_capacity_proxy_r1",
        "authority": "source_admission_capacity_only_not_route_role_truth_not_candidate_score",
        "candidate_outputs_executed": False,
        "uses_published_model_tracks": True,
        "forward_camera_visibility_verified": False,
        "route_proxy": {
            "definition": "published_track_center_distance_to_actual_future_robot_polyline",
            "horizon_seconds": ROUTE_HORIZON_SECONDS,
            "corridor_half_width_m": CORRIDOR_HALF_WIDTH_METERS,
            "positive_capacity": "same published track intersects on two consecutive frames",
            "critical_capacity": "same published track intersects on three consecutive frames",
            "new_scalar_tuned": False,
        },
        "sources": sources,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({source["source_id"]: {key: source[key] for key in ("exposure_minutes", "scorable_route_minutes", "negative_route_minutes_proxy", "cooccurrence_frame_rate_proxy", "positive_event_capacity_proxy", "critical_event_capacity_proxy")} for source in sources}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
