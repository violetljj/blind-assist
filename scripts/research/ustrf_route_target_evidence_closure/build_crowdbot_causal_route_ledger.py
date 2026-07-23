#!/usr/bin/env python3
"""Build candidate-blind causal route UV ledgers from CrowdBot pose prefixes and bag TF."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def quaternion_matrix_xyzw(values: list[float] | np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(values, dtype=np.float64)
    norm = float(x * x + y * y + z * z + w * w)
    if norm <= 1e-15:
        raise RuntimeError("zero-norm quaternion")
    scale = 2.0 / norm
    xx, yy, zz = x * x * scale, y * y * scale, z * z * scale
    xy, xz, yz = x * y * scale, x * z * scale, y * z * scale
    wx, wy, wz = w * x * scale, w * y * scale, w * z * scale
    return np.asarray([
        [1.0 - yy - zz, xy - wz, xz + wy],
        [xy + wz, 1.0 - xx - zz, yz - wx],
        [xz - wy, yz + wx, 1.0 - xx - yy],
    ])


def rigid_matrix(translation: list[float] | np.ndarray, quaternion: list[float] | np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_matrix_xyzw(quaternion)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


def static_pair(row: dict[str, Any], tolerance: float = 1e-6) -> bool:
    first = row["first"]
    last = row["last"]
    translation_delta = np.max(np.abs(
        np.asarray(first["translation_xyz"], dtype=np.float64)
        - np.asarray(last["translation_xyz"], dtype=np.float64)
    ))
    first_q = np.asarray(first["quaternion_xyzw"], dtype=np.float64)
    last_q = np.asarray(last["quaternion_xyzw"], dtype=np.float64)
    quaternion_delta = min(float(np.linalg.norm(first_q - last_q)), float(np.linalg.norm(first_q + last_q)))
    return translation_delta <= tolerance and quaternion_delta <= tolerance


def find_static_transform(inventory: dict[str, Any], *, target: str, source: str) -> tuple[np.ndarray, list[str]]:
    target = target.strip("/")
    source = source.strip("/")
    graph: dict[str, list[tuple[str, np.ndarray]]] = {}
    banned_frames = {"odom", "map", "world"}
    for row in inventory["frame_pairs"]:
        parent = row["parent_frame"].strip("/")
        child = row["child_frame"].strip("/")
        if parent in banned_frames or child in banned_frames or not static_pair(row):
            continue
        parent_from_child = rigid_matrix(
            row["first"]["translation_xyz"],
            row["first"]["quaternion_xyzw"],
        )
        graph.setdefault(child, []).append((parent, parent_from_child))
        graph.setdefault(parent, []).append((child, np.linalg.inv(parent_from_child)))
    queue = deque([(source, np.eye(4, dtype=np.float64), [source])])
    visited = {source}
    while queue:
        frame, frame_from_source, path = queue.popleft()
        if frame == target:
            return frame_from_source, path
        for neighbor, neighbor_from_frame in graph.get(frame, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, neighbor_from_frame @ frame_from_source, [*path, neighbor]))
    raise RuntimeError(f"no static TF path from {source!r} to {target!r}")


def metadata_sequence_name(path: Path) -> str:
    marker = "__"
    name = path.name
    start = name.rfind(marker) + len(marker)
    suffix = "_tfqolo_sampled.npy"
    if start < len(marker) or not name.endswith(suffix):
        raise RuntimeError(f"unexpected pose metadata name: {name}")
    return name[start:-len(suffix)]


def project(
    world_point: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsics: dict[str, Any],
    *,
    require_in_frame: bool = True,
) -> list[float] | None:
    camera_point = np.linalg.inv(camera_to_world) @ np.r_[world_point, 1.0]
    if camera_point[2] <= 0.03:
        return None
    k = intrinsics["K"]
    u = float(k[0] * camera_point[0] / camera_point[2] + k[2])
    v = float(k[4] * camera_point[1] / camera_point[2] + k[5])
    if require_in_frame and not (
        0.0 <= u < float(intrinsics["width"])
        and 0.0 <= v < float(intrinsics["height"])
    ):
        return None
    return [u, v]


def build_sequence(
    sequence_dir: Path,
    pose_path: Path,
    *,
    history_frames: int,
    horizon_frames: int,
    minimum_displacement_m: float,
    maximum_pose_age_ms: float,
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
        raise RuntimeError(f"sequence input hash mismatch: {sequence_dir}")
    camera_frame = bundle.get("camera_info", {}).get("frame_id")
    if not camera_frame:
        raise RuntimeError(f"camera frame ID missing: {bundle_path}")
    qolo_from_camera, tf_path_frames = find_static_transform(
        tf_inventory,
        target="tf_qolo",
        source=camera_frame,
    )
    pose = np.load(pose_path, allow_pickle=True).item()
    pose_timestamps = np.asarray(pose["timestamp"], dtype=np.float64)
    positions = np.asarray(pose["position"], dtype=np.float64)
    orientations = np.asarray(pose["orientation"], dtype=np.float64)
    if len(pose_timestamps) != len(positions) or len(positions) != len(orientations):
        raise RuntimeError(f"pose metadata length mismatch: {pose_path}")
    camera_to_world: list[np.ndarray | None] = []
    joined_pose_indices: list[int | None] = []
    pose_ages_ms: list[float | None] = []
    for frame in frames:
        timestamp_s = float(frame["source_capture_timestamp_ns"]) / 1e9
        pose_index = bisect.bisect_right(pose_timestamps, timestamp_s) - 1
        if pose_index < 0:
            camera_to_world.append(None)
            joined_pose_indices.append(None)
            pose_ages_ms.append(None)
            continue
        age_ms = (timestamp_s - float(pose_timestamps[pose_index])) * 1000.0
        if age_ms < -1e-6 or age_ms > maximum_pose_age_ms:
            camera_to_world.append(None)
            joined_pose_indices.append(None)
            pose_ages_ms.append(age_ms)
            continue
        world_from_qolo = rigid_matrix(positions[pose_index], orientations[pose_index])
        camera_to_world.append(world_from_qolo @ qolo_from_camera)
        joined_pose_indices.append(pose_index)
        pose_ages_ms.append(age_ms)
    predictions = []
    truth = []
    for index, frame in enumerate(frames):
        base = {
            "frame_id": frame["frame_id"],
            "source_capture_timestamp_ns": frame["source_capture_timestamp_ns"],
        }
        prediction = {
            **base,
            "predicted_at_ns": frame["source_capture_timestamp_ns"],
            "status": "unknown",
            "pose_index": joined_pose_indices[index],
            "pose_age_ms": pose_ages_ms[index],
        }
        current = camera_to_world[index]
        history_index = index - history_frames
        if current is not None and history_index >= 0 and camera_to_world[history_index] is not None:
            past = camera_to_world[history_index]
            assert past is not None
            delta = current[:3, 3] - past[:3, 3]
            target = current[:3, 3] + delta * (horizon_frames / history_frames)
            uv = project(target, current, bundle["camera_info"])
            if float(np.linalg.norm(delta)) >= minimum_displacement_m and uv is not None:
                prediction.update({
                    "status": "known",
                    "uv": uv,
                    "history_start_frame_id": frames[history_index]["frame_id"],
                })
        predictions.append(prediction)

        truth_row = {**base, "status": "unknown"}
        future_index = index + horizon_frames
        if current is not None and future_index < len(frames) and camera_to_world[future_index] is not None:
            future = camera_to_world[future_index]
            assert future is not None
            displacement = float(np.linalg.norm(future[:3, 3] - current[:3, 3]))
            future_pose_points = [
                camera_to_world[future_pose_index][:3, 3]
                for future_pose_index in range(index + 1, future_index + 1)
                if camera_to_world[future_pose_index] is not None
            ]
            uv_polyline = [
                projected
                for point in future_pose_points
                if (projected := project(
                    point,
                    current,
                    bundle["camera_info"],
                    require_in_frame=False,
                )) is not None
            ]
            uv = uv_polyline[-1] if uv_polyline else None
            if displacement >= minimum_displacement_m and len(uv_polyline) >= 2:
                truth_row.update({
                    "status": "known",
                    "uv": uv,
                    "uv_polyline": uv_polyline,
                    "future_frame_id": frames[future_index]["frame_id"],
                })
        truth.append(truth_row)
    known_ages = [row["pose_age_ms"] for row in predictions if row["status"] == "known"]
    return {
        "sequence_id": sequence_dir.name,
        "frames_sha256": sha256_file(frames_path),
        "bundle_sha256": sha256_file(bundle_path),
        "tf_frame_inventory_sha256": sha256_file(tf_path),
        "pose_metadata_path": pose_path.as_posix(),
        "pose_metadata_sha256": sha256_file(pose_path),
        "camera_frame_id": camera_frame,
        "qolo_from_camera_static_tf_path": tf_path_frames,
        "qolo_from_camera_matrix": qolo_from_camera.tolist(),
        "frame_count": len(frames),
        "known_route_prediction_count": sum(row["status"] == "known" for row in predictions),
        "known_route_truth_count": sum(row["status"] == "known" for row in truth),
        "known_prediction_pose_age_p95_ms": float(np.percentile(known_ages, 95)) if known_ages else None,
        "route_predictions": predictions,
        "route_truth_annotation_only": truth,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--metadata-root", required=True, type=Path)
    parser.add_argument("--source", action="append", required=True, metavar="SOURCE_ID=PREFIX")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--replacement", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite causal route ledger")
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    replacement = load_json(args.replacement) if args.replacement else None
    planned_output = (
        replacement["planned_outputs"]["causal_route_ledger"]
        if replacement
        else config["sealed_holdout"]["causal_route_input_contract"]["planned_output_path"]
    )
    if args.output.resolve() != (Path.cwd() / planned_output).resolve():
        raise RuntimeError("causal route ledger output differs from preregistration")
    state = load_json(args.state)
    if state.get("status") != "complete" or state.get("candidate_outputs_executed") is not False:
        raise RuntimeError("holdout materialization must be complete and candidate blind")
    if replacement and state.get("replacement_preregistration_sha256") != sha256_file(args.replacement):
        raise RuntimeError("replacement materialization state binding mismatch")
    route = config["sealed_holdout"]["causal_route_input_contract"]
    sources = []
    for source_arg in args.source:
        source_id, prefix = source_arg.split("=", 1)
        pose_paths = sorted(args.metadata_root.glob(f"{prefix}__source_data__tf_qolo__*_tfqolo_sampled.npy"))
        poses = {metadata_sequence_name(path): path for path in pose_paths}
        sequence_root = args.dataset_root / source_id / "sequences"
        sequence_dirs = sorted(path for path in sequence_root.iterdir() if path.is_dir())
        if {path.name for path in sequence_dirs} != set(poses):
            raise RuntimeError(f"pose and RGB-D sequence coverage mismatch: {source_id}")
        rows = [
            build_sequence(
                sequence_dir,
                poses[sequence_dir.name],
                history_frames=int(route["causal_history_frames"]),
                horizon_frames=int(route["projection_horizon_frames"]),
                minimum_displacement_m=float(route["minimum_forward_displacement_m"]),
                maximum_pose_age_ms=float(route["maximum_pose_age_ms"]),
            )
            for sequence_dir in sequence_dirs
        ]
        frame_count = sum(row["frame_count"] for row in rows)
        known_count = sum(row["known_route_prediction_count"] for row in rows)
        ages = [
            prediction["pose_age_ms"]
            for row in rows
            for prediction in row["route_predictions"]
            if prediction["status"] == "known"
        ]
        sources.append({
            "source_id": source_id,
            "sequence_count": len(rows),
            "frame_count": frame_count,
            "known_route_prediction_count": known_count,
            "route_unknown_rate": 1.0 - known_count / frame_count,
            "evidence_age_p95_ms": float(np.percentile(ages, 95)) if ages else None,
            "sequences": rows,
        })
    payload = {
        "schema": "blindassist_crowdbot_holdout_causal_route_ledger_r1",
        "authority": "sealed_holdout_input_not_route_role_truth_not_candidate_score",
        "candidate_outputs_executed": False,
        "future_pose_access_for_candidate": False,
        "actual_future_pose_used_only_in": "route_truth_annotation_only",
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "replacement_preregistration_sha256": sha256_file(args.replacement) if args.replacement else None,
        "materialization_state_sha256": sha256_file(args.state),
        "route_contract": route,
        "sources": sources,
        "production_authority": False,
        "candidate_h2_authority": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "CAUSAL_ROUTE_LEDGER_MATERIALIZED",
        "sources": [
            {
                "source_id": source["source_id"],
                "frame_count": source["frame_count"],
                "known_route_prediction_count": source["known_route_prediction_count"],
                "route_unknown_rate": source["route_unknown_rate"],
                "evidence_age_p95_ms": source["evidence_age_p95_ms"],
            }
            for source in sources
        ],
        "output_sha256": sha256_file(args.output),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
