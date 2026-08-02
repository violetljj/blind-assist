#!/usr/bin/env python3
"""Run a repairable TartanGround HFTF geometry-opportunity pilot.

The pilot deliberately uses outcome-open Development windows. It asks whether
future depth observations create non-redundant, aligned future traversability
labels relative to current-depth geometry. It does not evaluate a student or
support a system/safety claim.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

import cv2
import fsspec
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_swept_envelope_label_mechanics import (  # noqa: E402
    _swept_prism_counts,
    _swept_prism_probes_world,
)


REPO_ID = "theairlabcmu/TartanGround"
REVISION = "388faf9c800568cfc6828fa47e063f8369397eb3"
DEFAULT_ROOT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d5-s0b-p0c-provider-resolution-20260802/sentinel"
)
WINDOWS = (
    ("AbandonedCable/Data_diff/P1000", 292),
    ("MiddleEast/Data_diff/P1002", 585),
    ("WaterMillNight/Data_diff/P1002", 328),
)
CAMERA = "lcam_front"
RAW_FRAME_COUNT = 25
ANCHOR_OFFSETS = tuple(range(0, 17, 2))
HORIZON_OFFSETS = {"current": 0, "near": 4, "far": 8}
HORIZON_SECONDS = {"current": 0.0, "near": 0.4, "far": 0.8}
THETA_EDGES = np.linspace(math.radians(-45.0), math.radians(45.0), 7)
DISTANCE_EDGES_M = np.asarray(
    [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0],
    dtype=np.float64,
)
HEIGHT_BANDS_M = [(0.05, 0.35), (0.35, 1.35), (1.35, 2.05)]
HEIGHT_NAMES = ("foot", "body", "head")
HALF_WIDTHS_M = np.asarray([0.30, 0.40, 0.28], dtype=np.float64)
HUMAN_SPEED_MPS = 1.0
DEPTH_TOLERANCE_M = 0.20
MINIMUM_PASSING_PROBES = 5
POINT_STRIDE = 8
POINT_OFFSET = 4
RISK_SATURATION_POINTS = 8


def decode_depth(payload: bytes) -> np.ndarray:
    encoded = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None or image.shape != (640, 640, 4):
        raise ValueError(f"Unexpected depth image shape: {None if image is None else image.shape}")
    return np.squeeze(image.view("<f4"), axis=-1).astype(np.float64)


def pose_matrix(row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if row.shape != (7,) or not np.all(np.isfinite(row)):
        raise ValueError("Pose must contain seven finite values")
    return row[:3], Rotation.from_quat(row[3:]).as_matrix()


def anchor_basis(
    pose: np.ndarray,
    robot_height_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    translation, rotation = pose_matrix(pose)
    up = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    origin = translation - up * robot_height_m
    forward = rotation @ np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    forward -= float(forward @ up) * up
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    return origin, forward, right, up


def depth_points_world(
    depth: np.ndarray,
    pose: np.ndarray,
) -> np.ndarray:
    translation, rotation = pose_matrix(pose)
    v, u = np.mgrid[
        POINT_OFFSET : depth.shape[0] : POINT_STRIDE,
        POINT_OFFSET : depth.shape[1] : POINT_STRIDE,
    ]
    distance = depth[v, u]
    valid = (
        np.isfinite(distance)
        & (distance > 0.05)
        & (distance < 100.0)
    )
    distance = distance[valid]
    u = u[valid].astype(np.float64)
    v = v[valid].astype(np.float64)
    local_ned = np.vstack(
        [
            distance,
            (u - 320.0) * distance / 320.0,
            (v - 320.0) * distance / 320.0,
        ]
    )
    return rotation @ local_ned + translation[:, None]


def reprojection_consistency(
    source_depth: np.ndarray,
    source_pose: np.ndarray,
    target_depth: np.ndarray,
    target_pose: np.ndarray,
) -> dict[str, float | int | None]:
    points_world = depth_points_world(source_depth, source_pose)
    target_translation, target_rotation = pose_matrix(target_pose)
    target_ned = target_rotation.T @ (
        points_world - target_translation[:, None]
    )
    predicted = target_ned[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.floor(320.0 * target_ned[1] / predicted + 320.5).astype(
            np.int64
        )
        v = np.floor(320.0 * target_ned[2] / predicted + 320.5).astype(
            np.int64
        )
    inside = (
        np.isfinite(predicted)
        & (predicted > 0.05)
        & (predicted < 100.0)
        & (u >= 0)
        & (u < target_depth.shape[1])
        & (v >= 0)
        & (v < target_depth.shape[0])
    )
    observed = np.zeros(predicted.shape, dtype=np.float64)
    observed[inside] = target_depth[v[inside], u[inside]]
    comparable = (
        inside
        & np.isfinite(observed)
        & (observed > 0.05)
        & (observed < 100.0)
    )
    if not np.any(comparable):
        return {
            "comparable_points": 0,
            "median_relative_depth_error": None,
            "fraction_within_5_percent": None,
        }
    relative_error = np.abs(observed[comparable] - predicted[comparable]) / np.maximum(
        observed[comparable],
        predicted[comparable],
    )
    return {
        "comparable_points": int(comparable.sum()),
        "median_relative_depth_error": float(np.median(relative_error)),
        "fraction_within_5_percent": float(np.mean(relative_error <= 0.05)),
    }


def known_field(
    probes_world: np.ndarray,
    depth: np.ndarray,
    pose: np.ndarray,
) -> np.ndarray:
    translation, rotation = pose_matrix(pose)
    flat = probes_world.transpose(1, 0, 2).reshape(3, -1)
    camera_ned = rotation.T @ (flat - translation[:, None])
    forward = camera_ned[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.floor(320.0 * camera_ned[1] / forward + 320.5).astype(
            np.int64
        )
        v = np.floor(320.0 * camera_ned[2] / forward + 320.5).astype(
            np.int64
        )
    inside = (
        np.isfinite(forward)
        & (forward > 0.0)
        & (u >= 0)
        & (u < depth.shape[1])
        & (v >= 0)
        & (v < depth.shape[0])
    )
    observed = np.zeros(forward.shape, dtype=np.float64)
    observed[inside] = depth[v[inside], u[inside]]
    passing = (
        inside
        & np.isfinite(observed)
        & (observed > 0.0)
        & (observed + DEPTH_TOLERANCE_M >= forward)
    )
    shape = (
        len(THETA_EDGES) - 1,
        len(DISTANCE_EDGES_M) - 1,
        len(HEIGHT_BANDS_M),
    )
    return (
        passing.reshape(-1, 9).sum(axis=1) >= MINIMUM_PASSING_PROBES
    ).reshape(shape)


def field_from_observation(
    depth: np.ndarray,
    observation_pose: np.ndarray,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    horizon_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    shifted_edges = DISTANCE_EDGES_M + HUMAN_SPEED_MPS * horizon_seconds
    probes = _swept_prism_probes_world(
        basis,
        THETA_EDGES,
        shifted_edges,
        HEIGHT_BANDS_M,
        HALF_WIDTHS_M,
    )
    known = known_field(probes, depth, observation_pose)
    points = depth_points_world(depth, observation_pose)
    origin, _, _, up = basis
    height = up @ (points - origin[:, None])
    obstacle = points[
        :,
        (height >= HEIGHT_BANDS_M[0][0])
        & (height <= HEIGHT_BANDS_M[-1][1]),
    ]
    counts, _ = _swept_prism_counts(
        obstacle,
        np.zeros(obstacle.shape[1], dtype=bool),
        basis,
        THETA_EDGES,
        shifted_edges,
        HEIGHT_BANDS_M,
        HALF_WIDTHS_M,
    )
    risk = np.minimum(1.0, counts / float(RISK_SATURATION_POINTS))
    return known, risk


def remote_archive(parent_id: str, modality: str) -> str:
    return (
        f"hf://datasets/{REPO_ID}@{REVISION}/{parent_id}/"
        f"{modality}_{CAMERA}.zip"
    )


def member_frame_id(name: str) -> int:
    match = re.search(r"/(\d{6})_", name)
    if not match:
        raise ValueError(f"Cannot parse frame id from {name!r}")
    return int(match.group(1))


def fetch_window(root: Path, parent_id: str, start: int) -> None:
    wanted = {
        start + anchor_offset + future_offset
        for anchor_offset in ANCHOR_OFFSETS
        for future_offset in HORIZON_OFFSETS.values()
    }
    output = root / parent_id / "pilot_window"
    output.mkdir(parents=True, exist_ok=True)
    for modality in ("depth",):
        archive = remote_archive(parent_id, modality)
        with fsspec.open(archive, "rb", block_size=1024 * 1024) as source:
            with zipfile.ZipFile(source) as zipped:
                members = {
                    member_frame_id(name): name
                    for name in zipped.namelist()
                    if member_frame_id(name) in wanted
                }
                missing = sorted(wanted - set(members))
                if missing:
                    raise ValueError(
                        f"{parent_id} {modality} missing frames {missing}"
                    )
                for frame_id in sorted(wanted):
                    destination = (
                        output / modality / f"{frame_id:06d}.png"
                    )
                    if destination.exists():
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(zipped.read(members[frame_id]))


def load_metadata(
    root: Path,
    parent_id: str,
) -> tuple[dict[str, Any], np.ndarray]:
    archive = root / parent_id / "metadata.zip"
    with zipfile.ZipFile(archive) as zipped:
        metadata_name = next(
            name for name in zipped.namelist() if name.endswith("_metadata.json")
        )
        metadata = json.loads(zipped.read(metadata_name))
        pose = np.loadtxt(
            io.BytesIO(zipped.read(f"pose_{CAMERA}.txt")),
            dtype=np.float64,
        )
    return metadata, pose


def load_window_depth(
    root: Path,
    parent_id: str,
    frame_id: int,
) -> np.ndarray:
    path = (
        root
        / parent_id
        / "pilot_window"
        / "depth"
        / f"{frame_id:06d}.png"
    )
    return decode_depth(path.read_bytes())


def compare_fields(
    baseline_known: np.ndarray,
    baseline_risk: np.ndarray,
    oracle_known: np.ndarray,
    oracle_risk: np.ndarray,
) -> dict[str, Any]:
    common = baseline_known & oracle_known
    baseline_binary = baseline_risk > 0.0
    oracle_binary = oracle_risk > 0.0
    changed = common & (baseline_binary != oracle_binary)
    onsets = common & ~baseline_binary & oracle_binary
    clearances = common & baseline_binary & ~oracle_binary
    return {
        "baseline_known_cells": int(baseline_known.sum()),
        "oracle_known_cells": int(oracle_known.sum()),
        "oracle_newly_known_cells": int((oracle_known & ~baseline_known).sum()),
        "common_known_cells": int(common.sum()),
        "risk_state_changed_cells": int(changed.sum()),
        "risk_onset_cells": int(onsets.sum()),
        "risk_clearance_cells": int(clearances.sum()),
        "mean_absolute_risk_delta_common": (
            float(np.mean(np.abs(oracle_risk[common] - baseline_risk[common])))
            if np.any(common)
            else None
        ),
        "oracle_risk_cells_by_height": {
            name: int((oracle_known[:, :, index] & oracle_binary[:, :, index]).sum())
            for index, name in enumerate(HEIGHT_NAMES)
        },
    }


def analyze_parent(
    root: Path,
    parent_id: str,
    start: int,
) -> dict[str, Any]:
    metadata, poses = load_metadata(root, parent_id)
    robot_height = float(metadata["robot_height"])
    anchor_results = []
    for anchor_offset in ANCHOR_OFFSETS:
        anchor_frame = start + anchor_offset
        anchor_depth = load_window_depth(root, parent_id, anchor_frame)
        basis = anchor_basis(poses[anchor_frame], robot_height)
        horizons = {}
        for horizon_name, future_offset in HORIZON_OFFSETS.items():
            future_frame = anchor_frame + future_offset
            future_depth = load_window_depth(root, parent_id, future_frame)
            horizon_seconds = HORIZON_SECONDS[horizon_name]
            baseline_known, baseline_risk = field_from_observation(
                anchor_depth,
                poses[anchor_frame],
                basis,
                horizon_seconds,
            )
            oracle_known, oracle_risk = field_from_observation(
                future_depth,
                poses[future_frame],
                basis,
                horizon_seconds,
            )
            comparison = compare_fields(
                baseline_known,
                baseline_risk,
                oracle_known,
                oracle_risk,
            )
            comparison["pose_depth_reprojection"] = reprojection_consistency(
                anchor_depth,
                poses[anchor_frame],
                future_depth,
                poses[future_frame],
            )
            horizons[horizon_name] = comparison
        anchor_results.append(
            {"anchor_frame_id": anchor_frame, "horizons": horizons}
        )

    summary = {}
    for horizon_name in HORIZON_OFFSETS:
        rows = [
            anchor["horizons"][horizon_name] for anchor in anchor_results
        ]
        summary[horizon_name] = {
            key: sum(int(row[key]) for row in rows)
            for key in (
                "baseline_known_cells",
                "oracle_known_cells",
                "oracle_newly_known_cells",
                "common_known_cells",
                "risk_state_changed_cells",
                "risk_onset_cells",
                "risk_clearance_cells",
            )
        }
        deltas = [
            row["mean_absolute_risk_delta_common"]
            for row in rows
            if row["mean_absolute_risk_delta_common"] is not None
        ]
        summary[horizon_name]["mean_absolute_risk_delta_common"] = (
            float(np.mean(deltas)) if deltas else None
        )
        summary[horizon_name]["oracle_risk_cells_by_height"] = {
            name: sum(
                int(row["oracle_risk_cells_by_height"][name]) for row in rows
            )
            for name in HEIGHT_NAMES
        }
        reprojections = [
            row["pose_depth_reprojection"]
            for row in rows
            if row["pose_depth_reprojection"]["median_relative_depth_error"]
            is not None
        ]
        summary[horizon_name]["pose_depth_reprojection"] = {
            "pair_count": len(reprojections),
            "median_of_pair_median_relative_depth_error": (
                float(
                    np.median(
                        [
                            row["median_relative_depth_error"]
                            for row in reprojections
                        ]
                    )
                )
                if reprojections
                else None
            ),
            "median_fraction_within_5_percent": (
                float(
                    np.median(
                        [
                            row["fraction_within_5_percent"]
                            for row in reprojections
                        ]
                    )
                )
                if reprojections
                else None
            ),
        }

    return {
        "parent_id": parent_id,
        "window": {
            "start_frame_id": start,
            "end_frame_id": start + RAW_FRAME_COUNT - 1,
            "anchor_frame_ids": [
                start + offset for offset in ANCHOR_OFFSETS
            ],
        },
        "metadata": {
            "robot_height_m": robot_height,
            "time_step_s": float(metadata["time_step"]),
            "trajectory_type": metadata["type"],
        },
        "summary": summary,
        "anchors": anchor_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Analyze an already materialized pilot window.",
    )
    args = parser.parse_args()

    if not args.skip_fetch:
        for parent_id, start in WINDOWS:
            fetch_window(args.root, parent_id, start)

    parents = [
        analyze_parent(args.root, parent_id, start)
        for parent_id, start in WINDOWS
    ]
    near_far = [
        parent["summary"][horizon]
        for parent in parents
        for horizon in ("near", "far")
    ]
    result = {
        "schema": "blindassist_hftf_stage_c_d5_tartanground_development_pilot",
        "status": "DEVELOPMENT_PILOT_COMPLETE",
        "provider": {"repo_id": REPO_ID, "revision": REVISION},
        "design": {
            "outcome_open_exploratory_windows": True,
            "repairable_after_engineering_failure": True,
            "one_shot": False,
            "window_count": len(WINDOWS),
            "raw_frames_per_window": RAW_FRAME_COUNT,
            "depth_frames_read_per_window": len(
                {
                    anchor_offset + future_offset
                    for anchor_offset in ANCHOR_OFFSETS
                    for future_offset in HORIZON_OFFSETS.values()
                }
            ),
            "anchor_count_per_window": len(ANCHOR_OFFSETS),
            "horizons_s": HORIZON_SECONDS,
            "human_speed_mps": HUMAN_SPEED_MPS,
            "field_shape": [6, 6, 3],
            "height_bands_m": dict(zip(HEIGHT_NAMES, HEIGHT_BANDS_M)),
        },
        "parents": parents,
        "aggregate": {
            "future_common_known_cells": sum(
                row["common_known_cells"] for row in near_far
            ),
            "future_risk_state_changed_cells": sum(
                row["risk_state_changed_cells"] for row in near_far
            ),
            "future_risk_onset_cells": sum(
                row["risk_onset_cells"] for row in near_far
            ),
            "future_risk_clearance_cells": sum(
                row["risk_clearance_cells"] for row in near_far
            ),
            "future_oracle_newly_known_cells": sum(
                row["oracle_newly_known_cells"] for row in near_far
            ),
        },
        "interpretation": {
            "aligned_geometry_teacher_feasible": True,
            "future_label_nonredundancy_observed": any(
                row["risk_state_changed_cells"] > 0 for row in near_far
            ),
            "student_effect_established": False,
            "system_or_safety_claim_authorized": False,
        },
    }
    output = args.root / "development_pilot.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(result["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
