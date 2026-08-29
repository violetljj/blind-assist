"""Run the X1 causal rear-route Floxel source canary.

X0 localized the Huang-lane miss to absent M1-PD support.  The responsible
pedestrian is visible in raw LiDAR but remains behind the frozen -1 m source
crop throughout the early warning window.  X1 therefore changes two source
properties only: it admits the rear body-route field and replaces pairwise
nearest matching with a past-only, multi-scan voxel flow optimizer inspired by
Floxels.  The R7 route-entry geometry, motion bounds, and scorer are unchanged.

This is an independent causal adaptation, not the authors' Floxels code: the
published objective uses adjacent scans on both sides of a reference frame,
whereas an online avoidance source may consume current and past scans only.
Evaluator OBB identity and velocity are opened only by ``score`` after both
source ledgers have been sealed.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import dtr_m1_raw_point_direct_velocity as direct  # noqa: E402
import dtr_r7_occupancy_flow_canary as r7  # noqa: E402
from dtr_c1_global_obb_cohort_admission import require, sha256_file, write_json  # noqa: E402
from dtr_m1_point_velocity_oracle import (  # noqa: E402
    _box_history,
    load_native_boxes,
    load_world_clouds,
)
from dtr_x0_motion_source_attribution import (  # noqa: E402
    ASSOCIATION_MARGIN_M,
    FLOW_ERROR_LIMIT_MPS,
    _cell_clearance,
    _cells,
    _target_velocity,
)
from dtr_r7_occupancy_flow_canary import (  # noqa: E402
    FROZEN_FLOW_CONFIG,
    _causal_pose,
    _entry_s,
    _world_to_ego_xy,
    atomic_npz,
)
from jrdb_rgb_bridge import read_bag_pose_and_rgb  # noqa: E402


SCHEMA = "blindassist-dtr-x1-causal-floxel-source-canary-v1"
LEDGER_SCHEMA = "blindassist-dtr-x1-causal-floxel-ledger-v1"
SEQUENCE = "huang-lane-2019-02-12_0"
RESPONSIBLE_ID = "pedestrian:56"
SUPPORT_FIRST_FRAME = 169
TARGET_FIRST_FRAME = 173
TARGET_LAST_FRAME = 191
SUPPORT_SCANS = 4
REAR_ROI_FORWARD_M = (-10.5, FROZEN_FLOW_CONFIG.roi_forward_m[1])

# Fixed from the Floxels paper unless explicitly marked as the causal adapter.
FLOW_GRID_M = 0.5
DT_GRID_M = 1.0 / 6.0
LEARNING_RATE = 0.05
MAX_EPOCHS = 500
EARLY_PATIENCE = 250
EARLY_MIN_DELTA = 0.01
DBSCAN_EPS_M = 0.5
DBSCAN_MIN_POINTS = 4
CLUSTER_WEIGHT = 0.6  # public OpenSceneFlow integration, fixed before this run
FLOW_NORM_START = 0.1
FLOW_NORM_END = 0.01
FLOW_NORM_END_EPOCH = 100
TRUNCATE_DISTANCE_M = 5.0
MATCH_VOXEL_M = direct.MATCH_VOXEL_M


def _source_paths(root: Path) -> dict[str, Path]:
    return {
        "direct_npz": root / "rear-direct.npz",
        "direct_manifest": root / "rear-direct.json",
        "direct_backend": root / "rear-direct-backend.json",
        "floxel_npz": root / "causal-floxel.npz",
        "floxel_manifest": root / "causal-floxel.json",
        "result": root / "result.json",
    }


def _voxel_centroids(local_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not len(local_xyz):
        return np.empty((0, 3), np.float32), np.empty(0, np.int32)
    cells = np.floor(local_xyz / MATCH_VOXEL_M).astype(np.int32)
    _unique, inverse, counts = np.unique(
        cells, axis=0, return_inverse=True, return_counts=True
    )
    sums = np.zeros((len(counts), 3), dtype=np.float64)
    np.add.at(sums, inverse, local_xyz)
    return (sums / counts[:, None]).astype(np.float32), counts.astype(np.int32)


def _local_cloud(
    world_xyz: np.ndarray,
    pose: Mapping[str, Any],
    *,
    margin_m: float = 0.0,
) -> np.ndarray:
    local_xy = _world_to_ego_xy(world_xyz[:, :2], dict(pose))
    local = np.column_stack((local_xy, world_xyz[:, 2]))
    keep = (
        (local[:, 0] >= REAR_ROI_FORWARD_M[0] - margin_m)
        & (local[:, 0] <= REAR_ROI_FORWARD_M[1] + margin_m)
        & (local[:, 1] >= FROZEN_FLOW_CONFIG.roi_left_m[0] - margin_m)
        & (local[:, 1] <= FROZEN_FLOW_CONFIG.roi_left_m[1] + margin_m)
        & (local[:, 2] >= FROZEN_FLOW_CONFIG.roi_height_m[0] - margin_m)
        & (local[:, 2] <= FROZEN_FLOW_CONFIG.roi_height_m[1] + margin_m)
    )
    return local[keep]


def _normalized(points: Any, lower: Sequence[float], upper: Sequence[float]) -> Any:
    import torch

    low = torch.as_tensor(lower, dtype=points.dtype, device=points.device)
    high = torch.as_tensor(upper, dtype=points.dtype, device=points.device)
    return 2.0 * (points - low) / (high - low) - 1.0


def _sample_volume(volume: Any, points: Any, lower: Sequence[float], upper: Sequence[float]) -> Any:
    import torch.nn.functional as functional

    coordinates = _normalized(points, lower, upper)
    return functional.grid_sample(
        volume,
        coordinates.reshape(1, -1, 1, 1, 3),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).reshape(volume.shape[1], -1).transpose(0, 1)


def _distance_volume(
    points: np.ndarray,
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    device: Any,
) -> Any:
    import torch
    from scipy.ndimage import distance_transform_edt

    lower_array = np.asarray(lower, dtype=np.float64)
    upper_array = np.asarray(upper, dtype=np.float64)
    shape_xyz = np.ceil((upper_array - lower_array) / DT_GRID_M).astype(int) + 1
    occupancy = np.ones(
        (int(shape_xyz[2]), int(shape_xyz[1]), int(shape_xyz[0])), dtype=bool
    )
    indices = np.rint((points - lower_array) / DT_GRID_M).astype(int)
    indices = np.clip(indices, 0, shape_xyz - 1)
    occupancy[indices[:, 2], indices[:, 1], indices[:, 0]] = False
    distance = distance_transform_edt(
        occupancy, sampling=(DT_GRID_M, DT_GRID_M, DT_GRID_M)
    ).astype(np.float32)
    np.minimum(distance, TRUNCATE_DISTANCE_M, out=distance)
    return torch.as_tensor(distance, device=device).reshape(
        1, 1, distance.shape[0], distance.shape[1], distance.shape[2]
    )


def _cluster_labels(points: np.ndarray) -> np.ndarray:
    from sklearn.cluster import DBSCAN

    return DBSCAN(eps=DBSCAN_EPS_M, min_samples=DBSCAN_MIN_POINTS).fit_predict(points)


def _cluster_loss(flow: Any, labels: Any) -> Any:
    import torch

    keep = labels >= 0
    if not bool(keep.any()):
        return flow.new_zeros(())
    selected = flow[keep]
    unique, inverse = torch.unique(labels[keep], return_inverse=True)
    sums = torch.zeros((len(unique), 3), dtype=flow.dtype, device=flow.device)
    counts = torch.zeros((len(unique), 1), dtype=flow.dtype, device=flow.device)
    sums.scatter_add_(0, inverse[:, None].expand(-1, 3), selected)
    counts.scatter_add_(0, inverse[:, None], torch.ones_like(selected[:, :1]))
    means = sums / counts.clamp_min(1.0)
    return (selected - means[inverse]).norm(dim=1).mean()


def _optimize_frame(
    current: np.ndarray,
    supports: Sequence[np.ndarray],
    *,
    scan_offsets: Sequence[int] | None = None,
    device: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    lower = (
        REAR_ROI_FORWARD_M[0],
        FROZEN_FLOW_CONFIG.roi_left_m[0],
        FROZEN_FLOW_CONFIG.roi_height_m[0],
    )
    upper = (
        REAR_ROI_FORWARD_M[1],
        FROZEN_FLOW_CONFIG.roi_left_m[1],
        FROZEN_FLOW_CONFIG.roi_height_m[1],
    )
    dt_margin = 1.5
    dt_lower = tuple(value - dt_margin for value in lower)
    dt_upper = tuple(value + dt_margin for value in upper)
    shape_xyz = np.ceil(
        (np.asarray(upper) - np.asarray(lower)) / FLOW_GRID_M
    ).astype(int) + 1
    grid = torch.nn.Parameter(
        torch.zeros(
            (1, 3, int(shape_xyz[2]), int(shape_xyz[1]), int(shape_xyz[0])),
            dtype=torch.float32,
            device=device,
        )
    )
    current_tensor = torch.as_tensor(current, dtype=torch.float32, device=device)
    labels_numpy = _cluster_labels(current)
    labels = torch.as_tensor(labels_numpy, dtype=torch.int64, device=device)
    distance_volumes = [
        _distance_volume(points, dt_lower, dt_upper, device=device)
        for points in supports
    ]
    optimizer = torch.optim.Adam((grid,), lr=LEARNING_RATE, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=(10, 20), gamma=0.5
    )
    best_loss = math.inf
    best_flow: Any = None
    early_best = math.inf
    stale = 0
    started = time.perf_counter()
    epochs = 0
    for epoch in range(MAX_EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        flow = _sample_volume(grid, current_tensor, lower, upper)
        distance_loss = flow.new_zeros(())
        offsets = (
            tuple(scan_offsets)
            if scan_offsets is not None
            else tuple(-index for index in range(1, len(distance_volumes) + 1))
        )
        require(len(offsets) == len(distance_volumes), "x1_scan_offset_count")
        for scan_offset, distance in zip(offsets, distance_volumes):
            projected = current_tensor + float(scan_offset) * flow
            sampled = _sample_volume(distance, projected, dt_lower, dt_upper)[:, 0]
            distance_loss = distance_loss + sampled.mean() / float(abs(scan_offset) ** 2)
        multiplier = float(len(distance_volumes))
        cluster = _cluster_loss(flow, labels)
        flow_norm = flow.norm(dim=1).mean()
        gamma = float(
            np.interp(
                epoch,
                (0, FLOW_NORM_END_EPOCH),
                (FLOW_NORM_START, FLOW_NORM_END),
            )
        )
        loss = distance_loss + multiplier * (
            CLUSTER_WEIGHT * cluster + gamma * flow_norm
        )
        value = float(loss.detach().cpu())
        if value < best_loss:
            best_loss = value
            best_flow = flow.detach().clone()
        if value < early_best - EARLY_MIN_DELTA:
            early_best = value
            stale = 0
        else:
            stale += 1
        epochs = epoch + 1
        if stale >= EARLY_PATIENCE:
            break
        loss.backward()
        optimizer.step()
        scheduler.step()
    require(best_flow is not None, "x1_best_flow_missing")
    require(best_flow.device.type == "cuda", "x1_gpu_execution_not_verified")
    return best_flow.cpu().numpy(), {
        "epochs": epochs,
        "best_loss": best_loss,
        "seconds": time.perf_counter() - started,
        "points": int(len(current)),
        "clusters": int(len(set(labels_numpy)) - (1 if -1 in labels_numpy else 0)),
    }


def _aggregate(
    points: np.ndarray, velocity: np.ndarray, counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    speed = np.linalg.norm(velocity[:, :2], axis=1)
    keep = (
        (speed >= FROZEN_FLOW_CONFIG.minimum_dynamic_speed_mps)
        & (speed <= FROZEN_FLOW_CONFIG.maximum_dynamic_speed_mps)
    )
    points = points[keep]
    velocity = velocity[keep]
    counts = counts[keep]
    if not len(points):
        return (
            np.empty((0, 2), np.float32),
            np.empty((0, 2), np.float32),
            np.empty(0, np.int32),
        )
    cells = np.floor(points[:, :2] / FROZEN_FLOW_CONFIG.voxel_size_m).astype(int)
    unique, inverse = np.unique(cells, axis=0, return_inverse=True)
    centers = (unique.astype(np.float64) + 0.5) * FROZEN_FLOW_CONFIG.voxel_size_m
    velocities = np.asarray(
        [np.median(velocity[inverse == index, :2], axis=0) for index in range(len(unique))],
        dtype=np.float32,
    )
    source_counts = np.asarray(
        [int(counts[inverse == index].sum()) for index in range(len(unique))],
        dtype=np.int32,
    )
    return centers.astype(np.float32), velocities, source_counts


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    require(torch.cuda.is_available(), "x1_cuda_unavailable")
    device = torch.device("cuda:0")
    paths = _source_paths(args.root.resolve())
    args.root.resolve().mkdir(parents=True, exist_ok=True)
    baseline_npz = args.baseline_ledger.resolve(strict=True)
    with np.load(baseline_npz, allow_pickle=False) as values:
        timestamps = {
            int(frame): float(stamp)
            for frame, stamp in zip(values["frames"], values["frame_time_s"])
            if SUPPORT_FIRST_FRAME <= int(frame) <= TARGET_LAST_FRAME
        }
    frames, frame_times, poses, world_clouds, lidar = load_world_clouds(
        bag_path=args.bag.resolve(strict=True),
        timestamps_path=args.timestamps.resolve(strict=True),
        calibration_dir=args.calibration_dir.resolve(strict=True),
        timestamps_override=timestamps,
    )
    require(frames.tolist() == list(range(SUPPORT_FIRST_FRAME, TARGET_LAST_FRAME + 1)), "x1_frame_window")

    # Same pairwise source with only the source crop changed: a necessary
    # control that separates missing coverage from better flow estimation.
    rear_config = replace(FROZEN_FLOW_CONFIG, roi_forward_m=REAR_ROI_FORWARD_M)
    prior_r7_config, prior_direct_config = r7.FROZEN_FLOW_CONFIG, direct.FROZEN_FLOW_CONFIG
    try:
        r7.FROZEN_FLOW_CONFIG = rear_config
        direct.FROZEN_FLOW_CONFIG = rear_config
        direct_manifest = direct.materialize(
            bag_path=args.bag.resolve(strict=True),
            timestamps_path=args.timestamps.resolve(strict=True),
            calibration_dir=args.calibration_dir.resolve(strict=True),
            output_path=paths["direct_npz"],
            manifest_path=paths["direct_manifest"],
            backend_receipt_path=paths["direct_backend"],
            sequence=SEQUENCE,
            timestamps_override=timestamps,
        )
    finally:
        r7.FROZEN_FLOW_CONFIG = prior_r7_config
        direct.FROZEN_FLOW_CONFIG = prior_direct_config

    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    diagnostics = []
    for index, frame_value in enumerate(frames):
        frame = int(frame_value)
        if index < SUPPORT_SCANS or frame < TARGET_FIRST_FRAME:
            rows.append(
                (
                    np.empty((0, 2), np.float32),
                    np.empty((0, 2), np.float32),
                    np.empty(0, np.int32),
                )
            )
            continue
        pose = poses[frame]
        current_local = _local_cloud(world_clouds[index], pose)
        current, counts = _voxel_centroids(current_local)
        supports = []
        for support_index in range(index - 1, index - SUPPORT_SCANS - 1, -1):
            support_local = _local_cloud(world_clouds[support_index], pose, margin_m=1.5)
            support, _support_counts = _voxel_centroids(support_local)
            supports.append(support)
        require(len(current) > 0 and all(len(row) > 0 for row in supports), f"x1_empty_cloud:{frame}")
        displacement, detail = _optimize_frame(current, supports, device=device)
        one_step_s = float(
            np.median(
                [
                    frame_times[int(frames[offset])] - frame_times[int(frames[offset - 1])]
                    for offset in range(index - SUPPORT_SCANS + 1, index + 1)
                ]
            )
        )
        require(one_step_s > 0.0, f"x1_nonpositive_step:{frame}")
        positions, velocities, source_counts = _aggregate(
            current, displacement / one_step_s, counts
        )
        rows.append((positions, velocities, source_counts))
        diagnostics.append(
            {
                "frame": frame,
                "one_step_s": one_step_s,
                "output_cells": int(len(positions)),
                **detail,
            }
        )

    offsets = np.cumsum([0] + [len(row[0]) for row in rows], dtype=np.int64)
    arrays = {
        "frames": np.asarray(frames, dtype=np.int32),
        "frame_time_s": np.asarray([frame_times[int(frame)] for frame in frames], dtype=np.float64),
        "offsets": offsets,
        "forward_m": np.concatenate([row[0][:, 0] for row in rows]).astype(np.float32),
        "left_m": np.concatenate([row[0][:, 1] for row in rows]).astype(np.float32),
        "velocity_forward_mps": np.concatenate([row[1][:, 0] for row in rows]).astype(np.float32),
        "velocity_left_mps": np.concatenate([row[1][:, 1] for row in rows]).astype(np.float32),
        "component_id": np.concatenate([np.arange(len(row[0]), dtype=np.int32) for row in rows]),
        "source_point_count": np.concatenate([row[2] for row in rows]),
        "flow_support": np.concatenate([np.ones(len(row[0]), dtype=np.float32) for row in rows]),
    }
    atomic_npz(paths["floxel_npz"], **arrays)
    manifest = {
        "schema": LEDGER_SCHEMA,
        "truth_blind": True,
        "oracle": False,
        "sequence": SEQUENCE,
        "motion_source": "causal past-only multi-scan explicit voxel scene flow",
        "frame_window": [SUPPORT_FIRST_FRAME, TARGET_LAST_FRAME],
        "target_window": [TARGET_FIRST_FRAME, TARGET_LAST_FRAME],
        "config": {
            "support_scans": SUPPORT_SCANS,
            "future_scans": 0,
            "rear_roi_forward_m": list(REAR_ROI_FORWARD_M),
            "flow_grid_m": FLOW_GRID_M,
            "distance_grid_m": DT_GRID_M,
            "learning_rate": LEARNING_RATE,
            "maximum_epochs": MAX_EPOCHS,
            "early_stopping": {"patience": EARLY_PATIENCE, "minimum_delta": EARLY_MIN_DELTA},
            "dbscan": {"epsilon_m": DBSCAN_EPS_M, "minimum_points": DBSCAN_MIN_POINTS},
            "cluster_weight": CLUSTER_WEIGHT,
            "flow_norm_weight": {
                "epoch_0": FLOW_NORM_START,
                f"epoch_{FLOW_NORM_END_EPOCH}": FLOW_NORM_END,
            },
            "distance_truncation_m": TRUNCATE_DISTANCE_M,
            "source_motion_bounds_mps": [
                FROZEN_FLOW_CONFIG.minimum_dynamic_speed_mps,
                FROZEN_FLOW_CONFIG.maximum_dynamic_speed_mps,
            ],
            "bev_aggregation_m": FROZEN_FLOW_CONFIG.voxel_size_m,
        },
        "backend": {
            "required": "cuda",
            "verified": True,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "python": platform.python_version(),
        },
        "frozen_downstream": {
            "route_entry_geometry": "UNCHANGED_R7",
            "motion_bounds": "UNCHANGED_R7",
            "event_scorer": "UNCHANGED_R7",
        },
        "independent_adaptation_boundary": (
            "Floxels-inspired explicit voxel field, multi-scan distance loss, DBSCAN consistency, "
            "and flow norm; causal past-only adapter; not official Floxels code or benchmark reproduction"
        ),
        "source": {
            "bag": str(args.bag.resolve(strict=True)),
            "bag_sha256": sha256_file(args.bag.resolve(strict=True)),
            "timestamps_sha256": sha256_file(args.timestamps.resolve(strict=True)),
            "calibration_sha256": sha256_file(args.calibration_dir.resolve(strict=True) / "lidars.yaml"),
            "baseline_ledger": str(baseline_npz),
            "baseline_ledger_sha256": sha256_file(baseline_npz),
            "rear_direct_ledger": str(paths["direct_npz"]),
            "rear_direct_ledger_sha256": sha256_file(paths["direct_npz"]),
        },
        "diagnostics": {
            "frames": diagnostics,
            "total_seconds": sum(float(row["seconds"]) for row in diagnostics),
            "output_cells": int(len(arrays["forward_m"])),
            "lidar": lidar,
        },
        "ledger": str(paths["floxel_npz"]),
        "ledger_sha256": sha256_file(paths["floxel_npz"]),
    }
    write_json(paths["floxel_manifest"], manifest)
    return {"floxel": manifest, "rear_direct": direct_manifest}


def _diagnose(
    ledger: Mapping[str, np.ndarray],
    *,
    frames: Sequence[int],
    timestamps: Mapping[int, float],
    boxes_by_frame: Mapping[int, Sequence[Any]],
    history: Mapping[tuple[int, str], Any],
    poses: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    rows = []
    for frame in range(TARGET_FIRST_FRAME, TARGET_LAST_FRAME + 1):
        boxes = [box for box in boxes_by_frame[frame] if box.label_id == RESPONSIBLE_ID]
        require(len(boxes) == 1, f"x1_target_box:{frame}")
        box = boxes[0]
        target_velocity = _target_velocity(box, history, poses[frame])
        require(target_velocity is not None, f"x1_target_velocity:{frame}")
        target = np.asarray(target_velocity, dtype=np.float64)
        associated = []
        for cell in _cells(ledger, frame):
            if _cell_clearance(cell, box) > ASSOCIATION_MARGIN_M + 1e-9:
                continue
            flow = np.asarray(
                (cell["velocity_forward_mps"], cell["velocity_left_mps"]),
                dtype=np.float64,
            )
            error = float(np.linalg.norm(flow - target))
            associated.append(
                {
                    "error_mps": error,
                    "route_entry_s": _entry_s(
                        cell["forward_m"],
                        cell["left_m"],
                        cell["velocity_forward_mps"],
                        cell["velocity_left_mps"],
                    ),
                }
            )
        rows.append(
            {
                "frame": frame,
                "associated_cells": len(associated),
                "correct_cells": sum(
                    row["error_mps"] <= FLOW_ERROR_LIMIT_MPS + 1e-12
                    for row in associated
                ),
                "correct_route_cells": sum(
                    row["error_mps"] <= FLOW_ERROR_LIMIT_MPS + 1e-12
                    and row["route_entry_s"] is not None
                    for row in associated
                ),
                "minimum_error_mps": min(
                    (row["error_mps"] for row in associated), default=None
                ),
            }
        )
    return {
        "associated_frames": sum(row["associated_cells"] > 0 for row in rows),
        "correct_frames": sum(row["correct_cells"] > 0 for row in rows),
        "correct_route_entry_frames": sum(
            row["correct_route_cells"] > 0 for row in rows
        ),
        "minimum_associated_error_mps": min(
            (
                float(row["minimum_error_mps"])
                for row in rows
                if row["minimum_error_mps"] is not None
            ),
            default=None,
        ),
        "frames": rows,
    }


def _load_sealed(path: Path, manifest_path: Path, schema: str | None) -> dict[str, np.ndarray]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if schema is not None:
        require(manifest.get("schema") == schema, f"x1_manifest_schema:{path}")
    expected_hash = manifest.get("ledger_sha256")
    require(expected_hash == sha256_file(path), f"x1_ledger_hash:{path}")
    with np.load(path, allow_pickle=False) as values:
        return {name: values[name].copy() for name in values.files}


def score(args: argparse.Namespace) -> dict[str, Any]:
    paths = _source_paths(args.root.resolve(strict=True))
    floxel = _load_sealed(paths["floxel_npz"], paths["floxel_manifest"], LEDGER_SCHEMA)
    rear_direct = _load_sealed(paths["direct_npz"], paths["direct_manifest"], None)
    frames = [int(value) for value in floxel["frames"]]
    timestamps = {
        int(frame): float(stamp)
        for frame, stamp in zip(floxel["frames"], floxel["frame_time_s"])
    }
    pose_samples, _rgb, _authority = read_bag_pose_and_rgb(args.bag.resolve(strict=True))
    poses = {
        frame: _causal_pose(pose_samples, round(timestamps[frame] * 1e9))
        for frame in frames
    }
    boxes = load_native_boxes(
        args.labels.resolve(strict=True), timestamps, poses, sequence=SEQUENCE
    )
    history = _box_history(boxes)
    direct_result = _diagnose(
        rear_direct,
        frames=frames,
        timestamps=timestamps,
        boxes_by_frame=boxes,
        history=history,
        poses=poses,
    )
    floxel_result = _diagnose(
        floxel,
        frames=frames,
        timestamps=timestamps,
        boxes_by_frame=boxes,
        history=history,
        poses=poses,
    )
    gate = {
        "at_least_two_correct_frames": floxel_result["correct_frames"] >= 2,
        "at_least_two_correct_route_entry_frames": (
            floxel_result["correct_route_entry_frames"] >= 2
        ),
        "correct_frames_above_rear_direct": (
            floxel_result["correct_frames"] > direct_result["correct_frames"]
        ),
    }
    met = all(gate.values())
    result = {
        "schema": SCHEMA,
        "status": (
            "DTR_X1_CAUSAL_FLOXEL_SOURCE_CANARY_HEADROOM_MET"
            if met
            else "DTR_X1_CAUSAL_FLOXEL_SOURCE_CANARY_HEADROOM_NOT_MET"
        ),
        "question": (
            "After admitting the already visible rear body-route field, can a causal multi-scan "
            "voxel source recover correct route-entering motion that pairwise direct flow cannot?"
        ),
        "opened_truth_scope": {
            "sequence": SEQUENCE,
            "responsible_id": RESPONSIBLE_ID,
            "target_frames": [TARGET_FIRST_FRAME, TARGET_LAST_FRAME],
            "post_outcome_development_canary": True,
        },
        "source_only_comparison": {
            "rear_direct_pairwise": direct_result,
            "causal_floxel": floxel_result,
        },
        "gate": gate,
        "decision": {
            "headroom_met": met,
            "next": (
                "FREEZE_AND_RUN_X0_SIX_SEQUENCE_SOURCE_ONLY_COMPARISON"
                if met
                else "CLOSE_CAUSAL_FLOXEL_ADAPTER_AND_CHANGE_SOURCE_FAMILY"
            ),
        },
        "frozen_contract": {
            "risk_geometry_changed": False,
            "motion_bounds_changed": False,
            "event_scorer_changed": False,
            "pairwise_matcher_tuned": False,
            "source_crop_changed": "forward minimum -1.0 m to -10.5 m",
            "source_estimator_changed": "pairwise reciprocal nearest to causal past-only multi-scan voxel optimization",
            "flow_correct_error_limit_mps": FLOW_ERROR_LIMIT_MPS,
        },
        "sources": {
            "floxel_manifest": str(paths["floxel_manifest"]),
            "floxel_manifest_sha256": sha256_file(paths["floxel_manifest"]),
            "rear_direct_manifest": str(paths["direct_manifest"]),
            "rear_direct_manifest_sha256": sha256_file(paths["direct_manifest"]),
            "labels_sha256": sha256_file(args.labels.resolve(strict=True)),
        },
        "claim_limits": [
            "Opened-truth single-event Development canary; not source-disjoint confirmation or a six-sequence performance result.",
            "The implementation is Floxels-inspired and causal past-only, not official Floxels code or a reproduction of reported Floxels metrics.",
            "This canary tests missed-event source recall only; false-segment suppression remains unevaluated until the frozen X0 comparison.",
            "Native OBB identity and velocity are scorer-only privileged labels and are absent from both materializers.",
        ],
    }
    write_json(paths["result"], result)
    return result


def parse_args() -> argparse.Namespace:
    dataset = REPO / "artifacts.local" / "datasets" / "dtr-r0-jrdb-rgb-bridge-v1"
    evidence = REPO / "artifacts.local" / "evidence" / "dtr-c31" / "fresh-confirmation"
    root = REPO / "artifacts.local" / "evidence" / "dtr-x1" / "causal-floxel-source-canary"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("materialize", "score", "run"))
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--bag",
        type=Path,
        default=REPO
        / "artifacts.local"
        / "datasets"
        / "dtr-c31-jrdb-fresh-confirmation"
        / f"{SEQUENCE}.bag",
    )
    parser.add_argument("--timestamps", type=Path, default=dataset / "train_timestamps.zip")
    parser.add_argument("--labels", type=Path, default=dataset / "train_labels.zip")
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=REPO
        / "artifacts.local"
        / "datasets"
        / "ustrf-canonical-observation-source-authority-data-pack-r0"
        / "jrdb_toolkit"
        / "calibration",
    )
    parser.add_argument(
        "--baseline-ledger",
        type=Path,
        default=evidence
        / "baseline-ledgers"
        / SEQUENCE
        / "m1-pd.raw-point-direct-velocity.npz",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode in {"materialize", "run"}:
        materialized = materialize(args)
        print(
            json.dumps(
                {
                    "materialized": True,
                    "output_cells": materialized["floxel"]["diagnostics"]["output_cells"],
                    "seconds": materialized["floxel"]["diagnostics"]["total_seconds"],
                    "backend": materialized["floxel"]["backend"],
                },
                sort_keys=True,
            )
        )
    if args.mode in {"score", "run"}:
        result = score(args)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "gate": result["gate"],
                    "source_only_comparison": {
                        name: {
                            "associated_frames": value["associated_frames"],
                            "correct_frames": value["correct_frames"],
                            "correct_route_entry_frames": value["correct_route_entry_frames"],
                            "minimum_associated_error_mps": value["minimum_associated_error_mps"],
                        }
                        for name, value in result["source_only_comparison"].items()
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
