#!/usr/bin/env python3
"""Materialize source-native local route supervision from THOR-MAGNI."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

FOLD_COUNT = 5
HISTORY_FRAME_OFFSETS = (-24, -18, -12, -6, 0)
ANCHOR_STRIDE_SCENE_FRAMES = 30
MIN_WEARER_SPEED_MPS = 0.25
FUTURE_HORIZON_SECONDS = 2.0
FUTURE_SAMPLE_SECONDS = 0.10
CORRIDOR_HALF_WIDTH_M = 0.90
CORRIDOR_FORWARD_LIMIT_M = 4.0
PROXIMITY_THRESHOLD_M = 1.25
DISTANCE_EDGES_M = (0.0, 1.0, 2.0, 3.0, 4.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optional_float(value: str) -> float:
    value = value.strip()
    return float(value) if value not in {"", "N/A"} else math.nan


def stable_fold(source_session_id: str) -> int:
    digest = hashlib.sha256(
        source_session_id.encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % FOLD_COUNT


def stable_sample_id(session: str, frame: int) -> str:
    digest = hashlib.sha256(
        f"{session}\x1f{frame}".encode("utf-8")
    ).hexdigest()
    return f"thor-route-{digest[:20]}"


def video_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Unreadable THOR-MAGNI video: {path}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if count <= 0:
        raise ValueError(f"Invalid video frame count: {path}")
    return count


def read_scenario(
    path: Path,
    camera_body: str,
    scene_column: str,
) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        metadata = [next(reader) for _ in range(16)]
        header = next(reader)
        index = {name: position for position, name in enumerate(header)}
        body_names = [value for value in metadata[12][1:] if value]
        body_roles = metadata[13][1:]
        role_by_body = {
            body: body_roles[position]
            for position, body in enumerate(body_names)
            if position < len(body_roles)
        }
        required = [
            "Frame",
            "Time",
            scene_column,
            *[
                f"{camera_body} Centroid_{axis}"
                for axis in "XYZ"
            ],
        ]
        missing = [name for name in required if name not in index]
        if missing:
            raise ValueError(f"Missing scenario columns: {missing}")
        other_bodies = [
            body
            for body in body_names
            if body != camera_body
            and all(
                f"{body} Centroid_{axis}" in index
                for axis in "XYZ"
            )
        ]
        columns = {
            "frame": index["Frame"],
            "time": index["Time"],
            "scene": index[scene_column],
            "camera": [
                index[f"{camera_body} Centroid_{axis}"]
                for axis in "XYZ"
            ],
            "other": {
                body: [
                    index[f"{body} Centroid_{axis}"]
                    for axis in "XYZ"
                ]
                for body in other_bodies
            },
        }
        frames = []
        times = []
        scenes = []
        camera = []
        others = {body: [] for body in other_bodies}
        for row in reader:
            frames.append(int(row[columns["frame"]]))
            times.append(float(row[columns["time"]]))
            scenes.append(optional_float(row[columns["scene"]]))
            camera.append(
                [
                    optional_float(row[column])
                    for column in columns["camera"]
                ]
            )
            for body, body_columns in columns["other"].items():
                others[body].append(
                    [
                        optional_float(row[column])
                        for column in body_columns
                    ]
                )
    return {
        "frames": np.asarray(frames, dtype=np.int64),
        "times": np.asarray(times, dtype=np.float64),
        "scenes": np.asarray(scenes, dtype=np.float64),
        "camera": np.asarray(camera, dtype=np.float64) / 1000.0,
        "others": {
            body: np.asarray(values, dtype=np.float64) / 1000.0
            for body, values in others.items()
        },
        "roles": role_by_body,
    }


def route_target(
    times: np.ndarray,
    camera: np.ndarray,
    others: dict[str, np.ndarray],
    roles: dict[str, str],
    index: int,
) -> dict[str, Any] | None:
    before = index - 25
    after = index + 25
    if before < 0 or after >= len(times):
        return None
    if not (
        np.isfinite(camera[[before, index, after], :2]).all()
        and 0.35 <= times[after] - times[before] <= 0.65
    ):
        return None
    velocity = (
        camera[after, :2] - camera[before, :2]
    ) / (times[after] - times[before])
    speed = float(np.linalg.norm(velocity))
    if speed < MIN_WEARER_SPEED_MPS:
        return None
    forward = velocity / speed
    lateral_axis = np.asarray((-forward[1], forward[0]))
    end_time = times[index] + FUTURE_HORIZON_SECONDS
    future_end = int(np.searchsorted(times, end_time, side="right"))
    step = max(
        1,
        int(
            round(
                FUTURE_SAMPLE_SECONDS
                / np.median(np.diff(times[max(0, index - 20): index + 21]))
            )
        ),
    )
    occupancy = np.zeros((2, 6, 4), dtype=np.int8)
    minimum_distance = math.inf
    minimum_row = None
    corridor_intrusion = False
    observed_pairs = 0
    for future_index in range(index, future_end, step):
        if not np.isfinite(camera[future_index, :2]).all():
            continue
        delta_time = times[future_index] - times[index]
        horizon = 0 if delta_time <= 1.0 else 1
        for body, positions in others.items():
            if not np.isfinite(positions[future_index, :2]).all():
                continue
            relative = (
                positions[future_index, :2]
                - camera[future_index, :2]
            )
            distance = float(np.linalg.norm(relative))
            longitudinal = float(np.dot(relative, forward))
            lateral = float(np.dot(relative, lateral_axis))
            observed_pairs += 1
            if distance < minimum_distance:
                minimum_distance = distance
                minimum_row = {
                    "body": body,
                    "role": roles.get(body),
                    "time_offset_seconds": float(delta_time),
                    "distance_m": distance,
                    "longitudinal_m": longitudinal,
                    "lateral_m": lateral,
                }
            if (
                0.0 <= longitudinal <= CORRIDOR_FORWARD_LIMIT_M
                and abs(lateral) <= CORRIDOR_HALF_WIDTH_M
            ):
                corridor_intrusion = True
            if distance >= DISTANCE_EDGES_M[-1]:
                continue
            distance_bin = int(
                np.searchsorted(
                    DISTANCE_EDGES_M,
                    distance,
                    side="right",
                )
                - 1
            )
            angle = math.atan2(lateral, longitudinal)
            direction = min(
                5,
                int((angle + math.pi) / (2.0 * math.pi) * 6),
            )
            occupancy[horizon, direction, distance_bin] = 1
    if minimum_row is None or observed_pairs == 0:
        return None
    return {
        "wearer_speed_mps": speed,
        "future_minimum_synchronized_distance_m": minimum_distance,
        "future_proximity_le_1_25m": (
            minimum_distance <= PROXIMITY_THRESHOLD_M
        ),
        "future_corridor_intrusion": corridor_intrusion,
        "closest": minimum_row,
        "observed_future_body_time_pairs": observed_pairs,
        "occupancy_target": occupancy.tolist(),
        "occupancy_positive_cells": int(occupancy.sum()),
    }


def materialize_session(
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenario_path = Path(manifest["scenario_csv"])
    video_path = Path(manifest["rgb_path"])
    if sha256(scenario_path) != manifest["scenario_csv_sha256"]:
        raise ValueError(f"Scenario hash mismatch: {manifest_path}")
    if sha256(video_path) != manifest["rgb_sha256"]:
        raise ValueError(f"Video hash mismatch: {manifest_path}")
    data = read_scenario(
        scenario_path,
        str(manifest["camera_body"]),
        str(manifest["scene_frame_column"]),
    )
    frame_count = video_frame_count(video_path)
    by_scene: dict[int, int] = {}
    for index, scene in enumerate(data["scenes"]):
        if not math.isfinite(scene):
            continue
        scene_frame = int(scene)
        if scene_frame < 1 or scene_frame > frame_count:
            continue
        if not np.isfinite(data["camera"][index, :2]).all():
            continue
        current = by_scene.get(scene_frame)
        if current is None:
            by_scene[scene_frame] = index
            continue
        current_other_count = sum(
            np.isfinite(values[current, :2]).all()
            for values in data["others"].values()
        )
        candidate_other_count = sum(
            np.isfinite(values[index, :2]).all()
            for values in data["others"].values()
        )
        if candidate_other_count > current_other_count:
            by_scene[scene_frame] = index

    samples = []
    rejected = Counter()
    for scene_frame, index in sorted(by_scene.items()):
        if scene_frame % ANCHOR_STRIDE_SCENE_FRAMES != 0:
            continue
        history = [
            scene_frame + offset
            for offset in HISTORY_FRAME_OFFSETS
        ]
        if history[0] < 1 or history[-1] > frame_count:
            rejected["history_out_of_range"] += 1
            continue
        target = route_target(
            data["times"],
            data["camera"],
            data["others"],
            data["roles"],
            index,
        )
        if target is None:
            rejected["route_target_unavailable"] += 1
            continue
        samples.append(
            {
                "sample_id": stable_sample_id(
                    str(manifest["source_session_id"]),
                    scene_frame,
                ),
                "dataset_id": "THOR-MAGNI",
                "source_session_id": manifest["source_session_id"],
                "ancestry_group": manifest["ancestry_group"],
                "source_file_id": manifest["file_id"],
                "camera_body": manifest["camera_body"],
                "camera_body_role": manifest["camera_body_role"],
                "video_path": str(video_path.resolve()),
                "video_sha256": manifest["rgb_sha256"],
                "scenario_csv_path": str(scenario_path.resolve()),
                "scenario_csv_sha256": manifest[
                    "scenario_csv_sha256"
                ],
                "scene_frame_index_base": 1,
                "anchor_scene_frame": scene_frame,
                "history_scene_frames": history,
                "qtm_frame": int(data["frames"][index]),
                "qtm_time_seconds": float(data["times"][index]),
                "fold": stable_fold(
                    str(manifest["source_session_id"])
                ),
                "target": target,
                "authority": {
                    "source_native_geometric_proxy": True,
                    "human_event_truth": False,
                    "app_or_safety": False,
                    "promotion": False,
                },
            }
        )
    return samples, {
        "source_session_id": manifest["source_session_id"],
        "file_id": manifest["file_id"],
        "camera_body": manifest["camera_body"],
        "camera_body_role": manifest["camera_body_role"],
        "video_frame_count": frame_count,
        "scene_frames_with_camera_geometry": len(by_scene),
        "sample_count": len(samples),
        "rejected": dict(sorted(rejected.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path(
            r"F:\ba-data\hftf-d7-public-real\manifests"
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise ValueError("Refusing to overwrite THOR route supervision")
    manifests = sorted(
        args.manifest_dir.glob(
            "thor_magni_window_manifest_"
            "d7-r1-thor-magni-window-pupil-*.json"
        )
    )
    if len(manifests) != 19:
        raise ValueError(f"Expected 19 THOR Pupil manifests, got {len(manifests)}")
    all_samples = []
    sessions = []
    for manifest_path in manifests:
        samples, session = materialize_session(manifest_path)
        all_samples.extend(samples)
        sessions.append(session)
        print(json.dumps(session), flush=True)
    if len({row["sample_id"] for row in all_samples}) != len(all_samples):
        raise RuntimeError("Duplicate THOR local-route sample IDs")
    if len({row["source_session_id"] for row in all_samples}) != 19:
        raise RuntimeError("Not every THOR source session produced samples")
    fold_counts = Counter(row["fold"] for row in all_samples)
    proximity = sum(
        row["target"]["future_proximity_le_1_25m"]
        for row in all_samples
    )
    intrusion = sum(
        row["target"]["future_corridor_intrusion"]
        for row in all_samples
    )
    distances = np.asarray(
        [
            row["target"]["future_minimum_synchronized_distance_m"]
            for row in all_samples
        ],
        dtype=np.float64,
    )
    args.output_root.mkdir(parents=True)
    samples_path = args.output_root / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8", newline="\n") as output:
        for row in all_samples:
            output.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    report = {
        "schema": (
            "blindassist_hftf_stage_c_d8_thor_magni_"
            "local_route_supervision_v0"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "THOR_MAGNI_LOCAL_ROUTE_SUPERVISION_MATERIALIZED",
        "design": {
            "source_sessions": 19,
            "anchor_stride_scene_frames": ANCHOR_STRIDE_SCENE_FRAMES,
            "history_frame_offsets": list(HISTORY_FRAME_OFFSETS),
            "minimum_wearer_speed_mps": MIN_WEARER_SPEED_MPS,
            "future_horizon_seconds": FUTURE_HORIZON_SECONDS,
            "future_sample_seconds": FUTURE_SAMPLE_SECONDS,
            "corridor_half_width_m": CORRIDOR_HALF_WIDTH_M,
            "corridor_forward_limit_m": CORRIDOR_FORWARD_LIMIT_M,
            "proximity_threshold_m": PROXIMITY_THRESHOLD_M,
            "field_shape": [2, 6, 4],
            "field_semantics": (
                "future horizon x wearer-motion-relative direction "
                "x radial distance occupancy"
            ),
            "split": "SHA-256(source_session_id) modulo 5",
        },
        "counts": {
            "sample_count": len(all_samples),
            "source_session_count": len(
                {row["source_session_id"] for row in all_samples}
            ),
            "fold_counts": {
                str(key): value
                for key, value in sorted(fold_counts.items())
            },
            "future_proximity_le_1_25m_count": int(proximity),
            "future_corridor_intrusion_count": int(intrusion),
            "proximity_fraction": proximity / len(all_samples),
            "corridor_intrusion_fraction": intrusion / len(all_samples),
        },
        "minimum_synchronized_distance_m": {
            "minimum": float(np.min(distances)),
            "p10": float(np.quantile(distances, 0.10)),
            "median": float(np.median(distances)),
            "p90": float(np.quantile(distances, 0.90)),
            "maximum": float(np.max(distances)),
        },
        "sessions": sessions,
        "samples": {
            "path": str(samples_path.resolve()),
            "sha256": sha256(samples_path),
        },
        "authority": {
            "training_proxy": (
                "source-native geometric Development supervision"
            ),
            "human_event_truth": False,
            "app_or_safety": False,
            "promotion": False,
        },
        "next_scientific_gate": (
            "source-session-held-out RGB-history student must exceed "
            "current-frame and prior-only baselines on local occupancy "
            "AUROC/AP and minimum-distance ranking"
        ),
    }
    report_path = args.output_root / "report.json"
    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    digest = sha256(report_path)
    Path(str(report_path) + ".sha256").write_text(
        f"{digest}  {report_path.name}\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "counts": report["counts"],
                "distance": report[
                    "minimum_synchronized_distance_m"
                ],
                "report_sha256": digest,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
