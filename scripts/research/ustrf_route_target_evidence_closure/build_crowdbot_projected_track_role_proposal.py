#!/usr/bin/env python3
"""Project published CrowdBot tracks into RGB and propose candidate-blind route roles."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

from build_crowdbot_causal_route_ledger import (
    find_static_transform,
    load_json,
    metadata_sequence_name,
    rigid_matrix,
    sha256_file,
)


ROUTE_HORIZON_SECONDS = 24.0 / 15.0
CORRIDOR_HALF_WIDTH_METERS = 0.45
ROLE_TREND_OBSERVATIONS = 3
ROLE_TREND_DELTA_METERS = 0.10
MIN_ALERT_FRAMES = 2
MIN_CLEAR_FRAMES = 3


def point_to_polyline_distance(point: np.ndarray, polyline: np.ndarray) -> float:
    if len(polyline) == 1:
        return float(np.linalg.norm(point - polyline[0]))
    starts = polyline[:-1]
    vectors = polyline[1:] - starts
    denominators = np.sum(vectors * vectors, axis=1)
    deltas = point[None, :] - starts
    numerators = np.sum(deltas * vectors, axis=1)
    fractions = np.divide(
        numerators,
        denominators,
        out=np.zeros_like(numerators),
        where=denominators > 1e-12,
    )
    fractions = np.clip(fractions, 0.0, 1.0)
    closest = starts + fractions[:, None] * vectors
    return float(np.min(np.linalg.norm(point[None, :] - closest, axis=1)))


def project_point(point_world: np.ndarray, camera_to_world: np.ndarray, camera_info: dict[str, Any]) -> list[float] | None:
    camera = np.linalg.inv(camera_to_world) @ np.r_[point_world, 1.0]
    if camera[2] <= 0.03:
        return None
    k = camera_info["K"]
    u = float(k[0] * camera[0] / camera[2] + k[2])
    v = float(k[4] * camera[1] / camera[2] + k[5])
    if not (0.0 <= u < float(camera_info["width"]) and 0.0 <= v < float(camera_info["height"])):
        return None
    return [u, v]


def derive_event_proposals(sequence_id: str, frame_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observations: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for frame_number, frame in enumerate(frame_rows):
        for track in frame["projected_tracks"]:
            observations[track["published_track_id"]].append((frame_number, track))
    events = []
    for track_id, rows in sorted(observations.items()):
        previous_frame: int | None = None
        active_run: list[tuple[int, dict[str, Any]]] = []
        release_run = 0
        intersecting_run = 0
        current_event: dict[str, Any] | None = None
        event_index = 0
        for frame_number, track in rows:
            if previous_frame is None or frame_number != previous_frame + 1:
                active_run = []
                release_run = 0
                intersecting_run = 0
            role = track["role_proposal"]
            if role in {"route_intersecting", "approaching_route"}:
                active_run.append((frame_number, track))
                active_run = active_run[-MIN_ALERT_FRAMES:]
                release_run = 0
                intersecting_run = intersecting_run + 1 if role == "route_intersecting" else 0
                if current_event is None and len(active_run) >= MIN_ALERT_FRAMES:
                    event_id = f"{sequence_id}:{track_id}:{event_index}"
                    event_index += 1
                    current_event = {
                        "event_id": event_id,
                        "published_track_id": track_id,
                        "onset_frame": active_run[0][0],
                        "alertable_start_frame": frame_number,
                        "clear_frame": None,
                        "critical": False,
                        "censored_without_clear": True,
                    }
                    for _, active_track in active_run:
                        active_track["event_id"] = event_id
                    events.append(current_event)
                if current_event is not None:
                    track["event_id"] = current_event["event_id"]
                    if intersecting_run >= MIN_CLEAR_FRAMES:
                        current_event["critical"] = True
            else:
                active_run = []
                intersecting_run = 0
                if current_event is not None:
                    release_run += 1
                    track["event_id"] = current_event["event_id"]
                    if release_run >= MIN_CLEAR_FRAMES:
                        track["role_proposal"] = "cleared"
                        current_event["clear_frame"] = frame_number
                        current_event["censored_without_clear"] = False
                        current_event = None
                        release_run = 0
                else:
                    release_run = 0
            previous_frame = frame_number
    return events


def build_sequence_proposal(
    sequence_dir: Path,
    pose_path: Path,
    track_path: Path,
    *,
    maximum_age_ms: float = 200.0,
) -> dict[str, Any]:
    bundle_path = sequence_dir / "bundle.json"
    frames_path = sequence_dir / "frames.jsonl"
    tf_path = sequence_dir / "tf-frame-inventory.json"
    bundle = load_json(bundle_path)
    frames = [json.loads(line) for line in frames_path.read_text(encoding="utf-8").splitlines() if line]
    tf_inventory = load_json(tf_path)
    if bundle.get("candidate_outputs_executed") is not False or tf_inventory.get("candidate_outputs_executed") is not False:
        raise RuntimeError(f"candidate output leak in {sequence_dir}")
    if bundle.get("frames_sha256") != sha256_file(frames_path) or bundle.get("tf_frame_inventory_sha256") != sha256_file(tf_path):
        raise RuntimeError(f"sequence binding mismatch: {sequence_dir}")
    camera_frame = bundle.get("camera_info", {}).get("frame_id")
    if not camera_frame:
        raise RuntimeError(f"camera frame missing: {sequence_dir}")
    qolo_from_camera, static_path = find_static_transform(tf_inventory, target="tf_qolo", source=camera_frame)
    pose = np.load(pose_path, allow_pickle=True).item()
    timestamps = np.asarray(pose["timestamp"], dtype=np.float64)
    positions = np.asarray(pose["position"], dtype=np.float64)
    orientations = np.asarray(pose["orientation"], dtype=np.float64)
    tracks = np.load(track_path, allow_pickle=True).item()
    if len(timestamps) != len(positions) or len(positions) != len(orientations) or set(tracks) != set(range(len(timestamps))):
        raise RuntimeError(f"pose/track alignment mismatch: {sequence_dir.name}")
    histories: dict[int, deque[tuple[int, float]]] = defaultdict(lambda: deque(maxlen=ROLE_TREND_OBSERVATIONS))
    track_positions = {
        index: {int(row[-1]): np.asarray(row[:2], dtype=np.float64) for row in np.asarray(tracks[index])}
        for index in range(len(timestamps))
    }
    ever_intersected: set[int] = set()
    frame_rows = []
    for frame_number, frame in enumerate(frames):
        rgb_timestamp_s = float(frame["source_capture_timestamp_ns"]) / 1e9
        pose_index = bisect.bisect_right(timestamps, rgb_timestamp_s) - 1
        base = {
            "frame_id": frame["frame_id"],
            "source_capture_timestamp_ns": frame["source_capture_timestamp_ns"],
            "status": "unknown",
            "pose_index": pose_index if pose_index >= 0 else None,
            "pose_age_ms": None,
            "projected_tracks": [],
        }
        if pose_index < 0:
            frame_rows.append(base)
            continue
        age_ms = (rgb_timestamp_s - float(timestamps[pose_index])) * 1000.0
        base["pose_age_ms"] = age_ms
        future_end = int(np.searchsorted(timestamps, rgb_timestamp_s + ROUTE_HORIZON_SECONDS, side="left"))
        if age_ms < -1e-6 or age_ms > maximum_age_ms or future_end >= len(timestamps):
            frame_rows.append(base)
            continue
        world_from_qolo = rigid_matrix(positions[pose_index], orientations[pose_index])
        camera_to_world = world_from_qolo @ qolo_from_camera
        future_polyline = positions[pose_index : future_end + 1, :2]
        current_ids: set[int] = set()
        for track in np.asarray(tracks[pose_index]):
            track_id = int(track[-1])
            current_ids.add(track_id)
            center_world = np.asarray([float(track[0]), float(track[1]), float(track[2])])
            foot_world = np.asarray([float(track[0]), float(track[1]), float(track[2] - track[5] / 2.0)])
            center_uv = project_point(center_world, camera_to_world, bundle["camera_info"])
            if center_uv is None:
                continue
            ground_uv = project_point(foot_world, camera_to_world, bundle["camera_info"])
            distance = point_to_polyline_distance(foot_world[:2], future_polyline)
            future_track_distances = [
                point_to_polyline_distance(track_positions[index][track_id], future_polyline)
                for index in range(pose_index, future_end + 1)
                if track_id in track_positions[index]
            ]
            future_track_route_distance = min(future_track_distances) if future_track_distances else None
            history = histories[track_id]
            if history and frame_number != history[-1][0] + 1:
                history.clear()
            history.append((frame_number, distance))
            if distance <= CORRIDOR_HALF_WIDTH_METERS:
                role = "route_intersecting"
                ever_intersected.add(track_id)
            elif (
                len(history) == ROLE_TREND_OBSERVATIONS
                and history[0][1] - distance >= ROLE_TREND_DELTA_METERS
                and future_track_route_distance is not None
                and future_track_route_distance <= CORRIDOR_HALF_WIDTH_METERS
            ):
                role = "approaching_route"
            elif (
                track_id in ever_intersected
                and len(history) == ROLE_TREND_OBSERVATIONS
                and distance - history[0][1] >= ROLE_TREND_DELTA_METERS
            ):
                role = "receding"
            else:
                role = "adjacent_safe"
            base["projected_tracks"].append({
                "published_track_id": track_id,
                "projected_track_center_uv": center_uv,
                "projected_ground_uv": ground_uv,
                "world_ground_xy": foot_world[:2].tolist(),
                "route_distance_m": distance,
                "future_track_route_distance_m": future_track_route_distance,
                "role_proposal": role,
            })
        for track_id in set(histories) - current_ids:
            histories[track_id].clear()
        base["status"] = "known"
        base["track_timestamp_ns"] = int(round(float(timestamps[pose_index]) * 1e9))
        frame_rows.append(base)
    event_proposals = derive_event_proposals(sequence_dir.name, frame_rows)
    role_counts: dict[str, int] = defaultdict(int)
    visible_ids: set[int] = set()
    for frame in frame_rows:
        for track in frame["projected_tracks"]:
            visible_ids.add(track["published_track_id"])
            role_counts[track["role_proposal"]] += 1
    return {
        "sequence_id": sequence_dir.name,
        "frames_sha256": sha256_file(frames_path),
        "bundle_sha256": sha256_file(bundle_path),
        "tf_frame_inventory_sha256": sha256_file(tf_path),
        "pose_metadata_path": pose_path.as_posix(),
        "pose_metadata_sha256": sha256_file(pose_path),
        "track_metadata_path": track_path.as_posix(),
        "track_metadata_sha256": sha256_file(track_path),
        "camera_frame_id": camera_frame,
        "static_tf_path": static_path,
        "frame_count": len(frame_rows),
        "known_frame_count": sum(row["status"] == "known" for row in frame_rows),
        "projected_track_identity_count": len(visible_ids),
        "projected_person_frame_count": sum(role_counts.values()),
        "role_proposal_counts": dict(sorted(role_counts.items())),
        "positive_event_proposal_count": len(event_proposals),
        "critical_event_proposal_count": sum(row["critical"] for row in event_proposals),
        "cleared_event_proposal_count": sum(not row["censored_without_clear"] for row in event_proposals),
        "event_proposals": event_proposals,
        "frames": frame_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--metadata-root", required=True, type=Path)
    parser.add_argument("--source", action="append", required=True, metavar="SOURCE_ID=PREFIX")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--replacement", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite projected track proposal")
    config = load_json(args.config)
    replacement = load_json(args.replacement) if args.replacement else None
    planned_output = (
        replacement["planned_outputs"]["projected_track_role"]
        if replacement
        else config["sealed_holdout"]["holdout_truth_freeze_contract"]["all_person_presence"]["planned_projected_track_output_path"]
    )
    if args.output.resolve() != (Path.cwd() / planned_output).resolve():
        raise RuntimeError("projected track proposal output differs from preregistration")
    state = load_json(args.state)
    if state.get("status") != "complete" or state.get("candidate_outputs_executed") is not False:
        raise RuntimeError("holdout materialization must be complete and candidate blind")
    if replacement and state.get("replacement_preregistration_sha256") != sha256_file(args.replacement):
        raise RuntimeError("replacement materialization state binding mismatch")
    sources = []
    for source_arg in args.source:
        if "=" not in source_arg:
            raise RuntimeError("--source must use SOURCE_ID=PREFIX")
        source_id, prefix = source_arg.split("=", 1)
        poses = {
            metadata_sequence_name(path): path
            for path in args.metadata_root.glob(f"{prefix}__source_data__tf_qolo__*_tfqolo_sampled.npy")
        }
        tracks = {
            path.name.split("__")[-1][:-4]: path
            for path in args.metadata_root.glob(f"{prefix}__alg_res__tracks_2D__*.npy")
        }
        sequence_dirs = sorted(path for path in (args.dataset_root / source_id / "sequences").iterdir() if path.is_dir())
        sequence_ids = {path.name for path in sequence_dirs}
        if sequence_ids != set(poses) or sequence_ids != set(tracks):
            raise RuntimeError(f"RGB-D, pose, and track sequence coverage mismatch: {source_id}")
        rows = [build_sequence_proposal(path, poses[path.name], tracks[path.name]) for path in sequence_dirs]
        sources.append({
            "source_id": source_id,
            "sequence_count": len(rows),
            "frame_count": sum(row["frame_count"] for row in rows),
            "known_frame_count": sum(row["known_frame_count"] for row in rows),
            "projected_person_frame_count": sum(row["projected_person_frame_count"] for row in rows),
            "sequences": rows,
        })
    payload = {
        "schema": "blindassist_crowdbot_projected_track_role_proposal_r1",
        "authority": "candidate_blind_model_proxy_proposal_not_final_all_person_truth",
        "candidate_outputs_executed": False,
        "published_tracks_can_declare_visual_person_absent": False,
        "future_robot_polyline_used_only_for_annotation_role_proposal": True,
        "route_horizon_seconds": ROUTE_HORIZON_SECONDS,
        "corridor_half_width_m": CORRIDOR_HALF_WIDTH_METERS,
        "role_trend_observations": ROLE_TREND_OBSERVATIONS,
        "role_trend_delta_m": ROLE_TREND_DELTA_METERS,
        "materialization_state_sha256": sha256_file(args.state),
        "replacement_preregistration_sha256": sha256_file(args.replacement) if args.replacement else None,
        "sources": sources,
        "production_authority": False,
        "candidate_h2_authority": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PROJECTED_TRACK_ROLE_PROPOSAL_MATERIALIZED",
        "sources": [
            {
                "source_id": source["source_id"],
                "frame_count": source["frame_count"],
                "known_frame_count": source["known_frame_count"],
                "projected_person_frame_count": source["projected_person_frame_count"],
            }
            for source in sources
        ],
        "output_sha256": sha256_file(args.output),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
