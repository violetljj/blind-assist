"""Build a truth-blind raw-point reciprocal direct-velocity ledger.

M1-PD ego-compensates current and historical JRDB LiDAR sweeps, reduces raw
points to 3-D voxel centroids, and estimates a direct velocity only for mutual
nearest correspondences.  Static residuals and implausible speeds are removed
before aggregation to the unchanged R7 BEV route-risk interface.  Evaluator
identity and future data are never used by the matcher.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.research_backend import (  # noqa: E402
    BackendCandidate,
    BackendSelectionError,
    DeviceObservation,
    Workload,
    select_backend,
    torch_observation,
)

from dtr_c1_global_obb_cohort_admission import require, sha256_file, write_json  # noqa: E402
from dtr_m1_point_velocity_oracle import load_world_clouds  # noqa: E402
from dtr_r7_occupancy_flow_canary import (  # noqa: E402
    FROZEN_FLOW_CONFIG,
    FlowLedger,
    _history_index,
    _rotate_world_velocity_to_ego,
    _world_to_ego_xy,
    atomic_npz,
)


SCHEMA = "blindassist-dtr-m1-raw-point-direct-velocity-ledger-v1"
MATCH_VOXEL_M = 2.0 * FROZEN_FLOW_CONFIG.voxel_size_m
GPU_PAIR_BATCH = 4
PROBE_POINTS = 2048


def ledger_paths(output: Path) -> tuple[Path, Path]:
    return (
        output.with_name(output.stem + ".raw-point-direct-velocity.npz"),
        output.with_name(output.stem + ".raw-point-direct-velocity.json"),
    )


def _voxelize(world_xyz: Any, pose: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(world_xyz, dtype=np.float64)
    if not len(points):
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.int32)
    local = _world_to_ego_xy(points[:, :2], pose)
    config = FROZEN_FLOW_CONFIG
    keep = (
        (local[:, 0] >= config.roi_forward_m[0])
        & (local[:, 0] <= config.roi_forward_m[1])
        & (local[:, 1] >= config.roi_left_m[0])
        & (local[:, 1] <= config.roi_left_m[1])
        & (points[:, 2] >= config.roi_height_m[0])
        & (points[:, 2] <= config.roi_height_m[1])
    )
    selected = points[keep]
    if not len(selected):
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.int32)
    cells = np.floor(selected / MATCH_VOXEL_M).astype(np.int32)
    _unique, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((len(counts), 3), dtype=np.float64)
    np.add.at(sums, inverse, selected)
    return (sums / counts[:, None]).astype(np.float32), counts.astype(np.int32)


def _cpu_match_many(
    pairs: Sequence[tuple[np.ndarray, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    from scipy.spatial import cKDTree

    output = []
    for previous, current in pairs:
        if not len(previous) or not len(current):
            output.append(
                (
                    np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.float32),
                    np.empty(0, dtype=bool),
                )
            )
            continue
        distances, nearest_previous = cKDTree(previous).query(current, k=1, workers=1)
        _reverse_distance, nearest_current = cKDTree(current).query(previous, k=1, workers=1)
        mutual = nearest_current[nearest_previous] == np.arange(len(current))
        output.append(
            (
                np.asarray(nearest_previous, dtype=np.int64),
                np.asarray(distances, dtype=np.float32),
                np.asarray(mutual, dtype=bool),
            )
        )
    return output


def _gpu_match_many(
    pairs: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    numpy_output: bool,
) -> Any:
    import torch

    if not pairs:
        return []
    device = torch.device("cuda:0")
    maximum_previous = max(len(previous) for previous, _current in pairs)
    maximum_current = max(len(current) for _previous, current in pairs)
    if maximum_previous == 0 or maximum_current == 0:
        return _cpu_match_many(pairs) if numpy_output else torch.empty(0, device=device)
    previous_batch = torch.zeros(
        (len(pairs), maximum_previous, 3), dtype=torch.float64, device=device
    )
    current_batch = torch.zeros(
        (len(pairs), maximum_current, 3), dtype=torch.float64, device=device
    )
    previous_valid = torch.zeros(
        (len(pairs), maximum_previous), dtype=torch.bool, device=device
    )
    current_valid = torch.zeros(
        (len(pairs), maximum_current), dtype=torch.bool, device=device
    )
    for index, (previous, current) in enumerate(pairs):
        if len(previous):
            previous_batch[index, : len(previous)] = torch.as_tensor(
                previous, dtype=torch.float64, device=device
            )
            previous_valid[index, : len(previous)] = True
        if len(current):
            current_batch[index, : len(current)] = torch.as_tensor(
                current, dtype=torch.float64, device=device
            )
            current_valid[index, : len(current)] = True
    distances = torch.cdist(current_batch, previous_batch)
    distances = distances.masked_fill(~previous_valid[:, None, :], torch.inf)
    nearest_distance, nearest_previous = distances.min(dim=2)
    reverse = distances.masked_fill(~current_valid[:, :, None], torch.inf).min(dim=1).indices
    recovered = reverse.gather(1, nearest_previous.clamp_max(maximum_previous - 1))
    indices = torch.arange(maximum_current, device=device)[None, :]
    mutual = current_valid & (recovered == indices)
    if not numpy_output:
        return nearest_previous, nearest_distance, mutual
    output = []
    for index, (previous, current) in enumerate(pairs):
        count = len(current)
        if not len(previous) or not count:
            output.append(
                (
                    np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.float32),
                    np.empty(0, dtype=bool),
                )
            )
            continue
        output.append(
            (
                nearest_previous[index, :count].cpu().numpy().astype(np.int64),
                nearest_distance[index, :count].cpu().numpy().astype(np.float32),
                mutual[index, :count].cpu().numpy().astype(bool),
            )
        )
    return output


def _select_matching_backend(
    probe_pairs: Sequence[tuple[np.ndarray, np.ndarray]], receipt_path: Path
) -> dict[str, Any]:
    require(bool(probe_pairs), "m1_pd_probe_pairs_missing")
    bounded = [
        (previous[:PROBE_POINTS], current[:PROBE_POINTS])
        for previous, current in probe_pairs[:GPU_PAIR_BATCH]
    ]
    cache: dict[str, Any] = {}

    def cpu_probe() -> Any:
        cache["cpu"] = _cpu_match_many(bounded)
        return cache["cpu"]

    def observe_cpu(_output: Any) -> DeviceObservation:
        return DeviceObservation("cpu", platform.processor() or "CPU", "scipy-cKDTree")

    def gpu_probe() -> Any:
        cache["gpu"] = _gpu_match_many(bounded, numpy_output=False)
        return cache["gpu"]

    def observe_gpu(output: Any) -> DeviceObservation:
        observation = torch_observation(output=output)
        if "cpu" in cache:
            gpu_numpy = _gpu_match_many(bounded, numpy_output=True)
            for cpu, gpu in zip(cache["cpu"], gpu_numpy):
                if not np.allclose(cpu[1], gpu[1], atol=1e-4, rtol=1e-4):
                    raise BackendSelectionError("M1_PD_CPU_GPU_DISTANCE_MISMATCH")
                if not np.array_equal(cpu[2], gpu[2]):
                    raise BackendSelectionError("M1_PD_CPU_GPU_MUTUAL_MISMATCH")
        return observation

    return select_backend(
        Workload.POINT_CLOUD_MATCHING,
        cpu=BackendCandidate("scipy-cKDTree", "cpu", cpu_probe, observe_cpu),
        gpu=BackendCandidate(
            "torch-cuda-cdist",
            "cuda",
            gpu_probe,
            observe_gpu,
            synchronize=lambda: __import__("torch").cuda.synchronize(),
        ),
        record_path=receipt_path,
        warmups=1,
        repeats=3,
    )


def _match_all(
    pairs: Sequence[tuple[np.ndarray, np.ndarray]], backend: str
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    output = []
    for start in range(0, len(pairs), GPU_PAIR_BATCH):
        batch = pairs[start : start + GPU_PAIR_BATCH]
        if backend == "torch-cuda-cdist":
            output.extend(_gpu_match_many(batch, numpy_output=True))
        else:
            output.extend(_cpu_match_many(batch))
    return output


def materialize(
    *,
    bag_path: Path,
    timestamps_path: Path,
    calibration_dir: Path,
    output_path: Path,
    manifest_path: Path,
    backend_receipt_path: Path,
    sequence: str,
    timestamps_override: dict[int, float],
) -> dict[str, Any]:
    frames, timestamps, frame_poses, world_clouds, lidar_diagnostics = load_world_clouds(
        bag_path=bag_path,
        timestamps_path=timestamps_path,
        calibration_dir=calibration_dir,
        timestamps_override=timestamps_override,
    )
    voxels = [
        _voxelize(world, frame_poses[int(frame)])
        for frame, world in zip(frames, world_clouds)
    ]
    times = [float(timestamps[int(frame)]) for frame in frames]
    histories = [_history_index(times, index, FROZEN_FLOW_CONFIG) for index in range(len(frames))]
    eligible = [index for index, history in enumerate(histories) if history is not None]
    probe_pairs = [
        (voxels[int(histories[index])][0], voxels[index][0])
        for index in eligible
        if len(voxels[int(histories[index])][0]) and len(voxels[index][0])
    ]
    selection = _select_matching_backend(probe_pairs, backend_receipt_path)
    pairs = [
        (voxels[int(histories[index])][0], voxels[index][0])
        for index in eligible
    ]
    matches = _match_all(pairs, str(selection["selected_backend"]))
    match_by_index = {index: value for index, value in zip(eligible, matches)}

    rows = []
    offsets = [0]
    admitted_mutual = 0
    for index, frame_value in enumerate(frames):
        history = histories[index]
        current_points, current_counts = voxels[index]
        cell_rows: dict[tuple[int, int], list[tuple[np.ndarray, int, float]]] = {}
        if (
            history is not None
            and len(current_points)
            and len(voxels[int(history)][0])
        ):
            previous_points, previous_counts = voxels[int(history)]
            nearest, _distance, mutual = match_by_index[index]
            span_s = times[index] - times[int(history)]
            displacement = current_points - previous_points[nearest]
            speed = np.linalg.norm(displacement[:, :2], axis=1) / span_s
            keep = (
                mutual
                & (speed >= FROZEN_FLOW_CONFIG.minimum_dynamic_speed_mps)
                & (speed <= FROZEN_FLOW_CONFIG.maximum_dynamic_speed_mps)
            )
            admitted_mutual += int(keep.sum())
            velocity_world = displacement[:, :2] / span_s
            bev = np.floor(current_points[:, :2] / FROZEN_FLOW_CONFIG.voxel_size_m).astype(int)
            for point_index in np.nonzero(keep)[0]:
                support = min(
                    1.0,
                    float(current_counts[point_index]) / 3.0,
                    float(previous_counts[nearest[point_index]]) / 3.0,
                )
                cell_rows.setdefault(tuple(bev[point_index]), []).append(
                    (velocity_world[point_index], int(current_counts[point_index]), support)
                )
        centers_world = []
        velocities_world = []
        source_counts = []
        supports = []
        for cell, values in sorted(cell_rows.items()):
            centers_world.append((np.asarray(cell, dtype=np.float64) + 0.5) * FROZEN_FLOW_CONFIG.voxel_size_m)
            velocities_world.append(np.median([value[0] for value in values], axis=0))
            source_counts.append(sum(value[1] for value in values))
            supports.append(float(np.median([value[2] for value in values])))
        if centers_world:
            positions = _world_to_ego_xy(np.asarray(centers_world), frame_poses[int(frame_value)])
            velocities = _rotate_world_velocity_to_ego(
                np.asarray(velocities_world), frame_poses[int(frame_value)]
            )
        else:
            positions = np.empty((0, 2), dtype=np.float64)
            velocities = np.empty((0, 2), dtype=np.float64)
        rows.append((positions, velocities, source_counts, supports))
        offsets.append(offsets[-1] + len(positions))

    arrays = {
        "frames": np.asarray(frames, dtype=np.int32),
        "frame_time_s": np.asarray(times, dtype=np.float64),
        "frame_ego_x_m": np.asarray([frame_poses[int(frame)]["x_m"] for frame in frames]),
        "frame_ego_y_m": np.asarray([frame_poses[int(frame)]["y_m"] for frame in frames]),
        "frame_ego_yaw_rad": np.asarray([frame_poses[int(frame)]["yaw_rad"] for frame in frames]),
        "offsets": np.asarray(offsets, dtype=np.int64),
        "forward_m": np.concatenate([row[0][:, 0] for row in rows]).astype(np.float32),
        "left_m": np.concatenate([row[0][:, 1] for row in rows]).astype(np.float32),
        "velocity_forward_mps": np.concatenate([row[1][:, 0] for row in rows]).astype(np.float32),
        "velocity_left_mps": np.concatenate([row[1][:, 1] for row in rows]).astype(np.float32),
        "component_id": np.concatenate(
            [np.arange(len(row[0]), dtype=np.int32) for row in rows]
        ),
        "source_point_count": np.concatenate(
            [np.asarray(row[2], dtype=np.int32) for row in rows]
        ),
        "flow_support": np.concatenate(
            [np.asarray(row[3], dtype=np.float32) for row in rows]
        ),
    }
    atomic_npz(output_path, **arrays)
    manifest = {
        "schema_version": SCHEMA,
        "truth_blind": True,
        "oracle": False,
        "sequence": sequence,
        "frames": [int(frame) for frame in frames],
        "motion_source": "ego-compensated raw-LiDAR 3-D voxel reciprocal-nearest direct velocity",
        "config": {
            "match_voxel_m": MATCH_VOXEL_M,
            "bev_voxel_m": FROZEN_FLOW_CONFIG.voxel_size_m,
            "history_target_s": FROZEN_FLOW_CONFIG.history_target_s,
            "history_range_s": [
                FROZEN_FLOW_CONFIG.history_min_s,
                FROZEN_FLOW_CONFIG.history_max_s,
            ],
            "speed_range_mps": [
                FROZEN_FLOW_CONFIG.minimum_dynamic_speed_mps,
                FROZEN_FLOW_CONFIG.maximum_dynamic_speed_mps,
            ],
            "temporal_identity": "mutual nearest only; evaluator identity unused",
        },
        "backend": selection,
        "backend_receipt": str(backend_receipt_path),
        "backend_receipt_sha256": sha256_file(backend_receipt_path),
        "frozen_downstream": {
            "route_geometry_and_lifecycle": "UNCHANGED_R7",
            "confidence_layer": "M1_CT_UNCHANGED",
        },
        "diagnostics": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "raw_frames": len(frames),
            "raw_points": sum(len(world) for world in world_clouds),
            "match_voxels": sum(len(row[0]) for row in voxels),
            "mutual_dynamic_voxels": admitted_mutual,
            "bev_velocity_cells": int(len(arrays["forward_m"])),
            "lidar": lidar_diagnostics,
        },
        "ledger": str(output_path),
        "ledger_sha256": sha256_file(output_path),
    }
    write_json(manifest_path, manifest)
    return manifest


def load_ledger(
    path: Path,
    manifest_path: Path,
    *,
    expected_sequence: str,
    expected_frames: list[int],
) -> FlowLedger:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema_version") == SCHEMA, "m1_pd_schema")
    require(manifest.get("sequence") == expected_sequence, "m1_pd_sequence")
    require(manifest.get("truth_blind") is True, "m1_pd_truth_blind")
    require(sha256_file(path) == manifest["ledger_sha256"], "m1_pd_hash_drift")
    values = np.load(path, allow_pickle=False)
    require(values["frames"].tolist() == expected_frames, "m1_pd_frames")
    return FlowLedger(
        frames=values["frames"],
        offsets=values["offsets"],
        forward_m=values["forward_m"],
        left_m=values["left_m"],
        velocity_forward_mps=values["velocity_forward_mps"],
        velocity_left_mps=values["velocity_left_mps"],
        component_id=values["component_id"],
        manifest=manifest,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--timestamps", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--timestamps-json", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--backend-receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamps = {
        int(frame): float(value)
        for frame, value in json.loads(args.timestamps_json.read_text(encoding="utf-8")).items()
    }
    result = materialize(
        bag_path=args.bag.resolve(strict=True),
        timestamps_path=args.timestamps.resolve(strict=True),
        calibration_dir=args.calibration_dir.resolve(strict=True),
        output_path=args.output.resolve(),
        manifest_path=args.manifest.resolve(),
        backend_receipt_path=args.backend_receipt.resolve(),
        sequence=args.sequence,
        timestamps_override=timestamps,
    )
    print(json.dumps({"status": "M1_PD_MATERIALIZED", "diagnostics": result["diagnostics"]}))


if __name__ == "__main__":
    main()
