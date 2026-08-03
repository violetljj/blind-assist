#!/usr/bin/env python3
"""Compare a Metric3D class-free clearance field with registered RGB-D."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from prepare_bonn_rgbd_metric_depth_manifest import (
    associate_nearest,
    normalize_depth_image,
    read_tum_index,
)
from produce_external_rgb_metric_depth_observations import Metric3DPytorchSource, intrinsics_matrix


SCHEMA = "blindassist_hftf_metric3d_clearance_field_a0"
BANDS = {
    "left": (-1.20, -0.40),
    "center": (-0.40, 0.40),
    "right": (0.40, 1.20),
}
HORIZONS_M = (1.0, 1.5, 2.0)


def depth_to_points(
    depth: np.ndarray, intrinsics: np.ndarray, stride: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth.shape
    rows, columns = np.mgrid[0:height:stride, 0:width:stride]
    z = np.asarray(depth[::stride, ::stride], dtype=np.float64)
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    x = (columns - cx) * z / fx
    y = (rows - cy) * z / fy
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    pixels = np.stack((columns, rows), axis=-1).reshape(-1, 2)
    valid = np.all(np.isfinite(points), axis=1) & (points[:, 2] >= 0.25) & (points[:, 2] <= 6.0)
    return points[valid], pixels[valid]


def fit_ground_plane(
    points: np.ndarray,
    pixels: np.ndarray,
    image_height: int,
    seed: int = 1729,
) -> tuple[np.ndarray, float, float] | None:
    candidates = points[pixels[:, 1] >= 0.55 * image_height]
    if len(candidates) < 100:
        return None
    if len(candidates) > 5000:
        indices = np.linspace(0, len(candidates) - 1, 5000, dtype=int)
        candidates = candidates[indices]
    rng = np.random.default_rng(seed)
    best_inliers = None
    for _ in range(240):
        sample = candidates[rng.choice(len(candidates), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-9:
            continue
        normal /= norm
        if abs(float(normal[1])) < 0.55:
            continue
        offset = -float(np.dot(normal, sample[0]))
        if offset < 0:
            normal = -normal
            offset = -offset
        if not 0.45 <= offset <= 2.20:
            continue
        inliers = np.abs(candidates @ normal + offset) <= 0.045
        if best_inliers is None or int(np.sum(inliers)) > int(np.sum(best_inliers)):
            best_inliers = inliers
    if best_inliers is None or int(np.sum(best_inliers)) < max(80, int(0.08 * len(candidates))):
        return None
    ground = candidates[best_inliers]
    center = np.mean(ground, axis=0)
    _, _, right_vectors = np.linalg.svd(ground - center, full_matrices=False)
    normal = right_vectors[-1]
    offset = -float(np.dot(normal, center))
    if offset < 0:
        normal = -normal
        offset = -offset
    residual = float(np.median(np.abs(ground @ normal + offset)))
    if abs(float(normal[1])) < 0.55 or not 0.45 <= offset <= 2.20:
        return None
    return normal, offset, residual


def fit_normal_guided_ground_plane(
    points: np.ndarray,
    pixels: np.ndarray,
    normal_kappa_map: np.ndarray,
    image_height: int,
) -> tuple[np.ndarray, float, float] | None:
    sampled = normal_kappa_map[pixels[:, 1].astype(int), pixels[:, 0].astype(int)]
    normals = np.asarray(sampled[:, :3], dtype=np.float64)
    kappa = np.asarray(sampled[:, 3], dtype=np.float64)
    candidate = (
        (pixels[:, 1] >= 0.55 * image_height)
        & np.all(np.isfinite(normals), axis=1)
        & np.isfinite(kappa)
        & (kappa > 0)
        & (np.abs(normals[:, 1]) >= 0.55)
    )
    if int(np.sum(candidate)) < 100:
        return None
    confidence_cut = float(np.median(kappa[candidate]))
    candidate &= kappa >= confidence_cut
    vectors = normals[candidate].copy()
    vectors[vectors[:, 1] > 0] *= -1.0
    weights = np.minimum(kappa[candidate], float(np.quantile(kappa[candidate], 0.90)))
    consensus = np.sum(vectors * weights[:, None], axis=0)
    norm = float(np.linalg.norm(consensus))
    if norm <= 1e-9:
        return None
    consensus /= norm
    candidate_points = points[candidate]
    candidate_vectors = vectors
    offsets = -(candidate_points @ consensus)
    valid_offset = (offsets >= 0.45) & (offsets <= 2.20)
    if int(np.sum(valid_offset)) < 80:
        return None
    bins = np.arange(0.45, 2.24 + 1e-9, 0.04)
    counts, edges = np.histogram(offsets[valid_offset], bins=bins)
    if not len(counts) or int(np.max(counts)) < 40:
        return None
    mode_index = int(np.argmax(counts))
    mode_center = float((edges[mode_index] + edges[mode_index + 1]) / 2.0)
    angular = candidate_vectors @ consensus >= math.cos(math.radians(15.0))
    support = valid_offset & angular & (np.abs(offsets - mode_center) <= 0.08)
    if int(np.sum(support)) < 80:
        return None
    ground = candidate_points[support]
    center = np.mean(ground, axis=0)
    _, _, right_vectors = np.linalg.svd(ground - center, full_matrices=False)
    normal = right_vectors[-1]
    offset = -float(np.dot(normal, center))
    if offset < 0:
        normal = -normal
        offset = -offset
    residual = float(np.median(np.abs(ground @ normal + offset)))
    if abs(float(normal[1])) < 0.55 or not 0.45 <= offset <= 2.20:
        return None
    return normal, offset, residual


def fit_gravity_guided_ground_plane(
    points: np.ndarray,
    pixels: np.ndarray,
    up_camera: np.ndarray,
    image_height: int,
) -> tuple[np.ndarray, float, float] | None:
    up = np.asarray(up_camera, dtype=np.float64)
    norm = float(np.linalg.norm(up))
    if norm <= 1e-9:
        return None
    up /= norm
    candidates = points[pixels[:, 1] >= 0.55 * image_height]
    if len(candidates) < 100:
        return None
    offsets = -(candidates @ up)
    if float(np.median(offsets)) < 0:
        up = -up
        offsets = -offsets
    valid = (offsets >= 0.45) & (offsets <= 2.20)
    if int(np.sum(valid)) < 80:
        return None
    bins = np.arange(0.45, 2.24 + 1e-9, 0.04)
    counts, edges = np.histogram(offsets[valid], bins=bins)
    if not len(counts) or int(np.max(counts)) < 40:
        return None
    mode = int(np.argmax(counts))
    center = float((edges[mode] + edges[mode + 1]) / 2.0)
    support = valid & (np.abs(offsets - center) <= 0.08)
    if int(np.sum(support)) < 80:
        return None
    offset = float(np.median(offsets[support]))
    residual = float(np.median(np.abs(offsets[support] - offset)))
    return up, offset, residual


def clearance_field(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    normal_kappa_map: np.ndarray | None = None,
    up_camera: np.ndarray | None = None,
    plane_override: tuple[np.ndarray, float, float] | None = None,
) -> dict[str, Any]:
    points, pixels = depth_to_points(depth, intrinsics)
    plane = plane_override or (
        fit_gravity_guided_ground_plane(
            points, pixels, up_camera, depth.shape[0]
        )
        if up_camera is not None
        else fit_normal_guided_ground_plane(
            points, pixels, normal_kappa_map, depth.shape[0]
        )
        if normal_kappa_map is not None
        else fit_ground_plane(points, pixels, depth.shape[0])
    )
    if plane is None:
        return {"status": "UNKNOWN_GROUND"}
    up, camera_height, plane_residual = plane
    optical_forward = np.asarray([0.0, 0.0, 1.0])
    forward_axis = optical_forward - float(np.dot(optical_forward, up)) * up
    forward_norm = float(np.linalg.norm(forward_axis))
    if forward_norm <= 1e-6:
        return {"status": "UNKNOWN_GROUND_FORWARD"}
    forward_axis /= forward_norm
    lateral_axis = np.cross(forward_axis, up)
    lateral_axis /= np.linalg.norm(lateral_axis)
    heights = points @ up + camera_height
    forward = points @ forward_axis
    lateral = points @ lateral_axis
    obstacle = (heights >= 0.08) & (heights <= 2.00) & (forward >= 0.20) & (forward <= 4.00)
    band_output = {}
    for name, (minimum, maximum) in BANDS.items():
        distances = forward[obstacle & (lateral >= minimum) & (lateral < maximum)]
        if len(distances) < 20:
            band_output[name] = {
                "clearance_m": None,
                "obstacle_points": int(len(distances)),
                "occupied_by_horizon": {str(value): None for value in HORIZONS_M},
            }
            continue
        clearance = float(np.quantile(distances, 0.02))
        band_output[name] = {
            "clearance_m": clearance,
            "obstacle_points": int(len(distances)),
            "occupied_by_horizon": {
                str(value): clearance <= value for value in HORIZONS_M
            },
        }
    return {
        "status": "VALID",
        "ground_source": (
            "tum_fixed_world_floor_oracle"
            if plane_override is not None
            else "tum_mocap_gravity_oracle"
            if up_camera is not None
            else "metric3d_prediction_normal"
            if normal_kappa_map is not None
            else "depth_ransac"
        ),
        "camera_height_m": camera_height,
        "ground_plane_median_residual_m": plane_residual,
        "bands": band_output,
    }


def load_clearance_manifests(manifest_paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in manifest_paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = sorted({"sequence_id", "frame_path", "intrinsics_fx_fy_cx_cy"} - row.keys())
            if missing:
                raise ValueError(f"{path}:{line_number}: missing {missing}")
            frame_path = Path(str(row["frame_path"]))
            if not frame_path.is_absolute():
                frame_path = (path.parent / frame_path).resolve()
            row["frame_path"] = str(frame_path)
            rows.append(row)
    if not rows:
        raise ValueError("clearance manifests contain no frames")
    return rows


def _unique_frames(manifest_paths: list[Path]) -> list[dict[str, Any]]:
    rows = load_clearance_manifests(manifest_paths)
    by_path = {}
    for row in rows:
        key = str(Path(str(row["frame_path"])).resolve())
        by_path.setdefault(key, row)
    return [by_path[key] for key in sorted(by_path)]


def _depth_lookup(sequence_root: Path) -> dict[str, Path]:
    pairs = associate_nearest(
        read_tum_index(sequence_root / "rgb.txt"),
        read_tum_index(sequence_root / "depth.txt"),
        0.02,
    )
    return {
        str((sequence_root / rgb_relative).resolve()): (sequence_root / depth_relative).resolve()
        for _, rgb_relative, _, depth_relative in pairs
    }


def tum_depth_metres(depth_raw: np.ndarray) -> np.ndarray:
    """Apply the published TUM factor: a stored value of 5000 is one metre."""
    return np.asarray(depth_raw, dtype=np.float32) / 5000.0


def infer_metric3d_depth_normal(
    source: Metric3DPytorchSource, rgb: np.ndarray, row: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    height, width = rgb.shape[:2]
    scale = min(source.input_height / height, source.input_width / width)
    resized_height, resized_width = int(height * scale), int(width * scale)
    resized = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_height = source.input_height - resized_height
    pad_width = source.input_width - resized_width
    pad_top, pad_left = pad_height // 2, pad_width // 2
    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_height - pad_top,
        pad_left,
        pad_width - pad_left,
        cv2.BORDER_CONSTANT,
        value=[123.675, 116.28, 103.53],
    )


def load_tum_camera_poses(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps = []
    translations = []
    quaternions = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 8:
            raise ValueError(f"{path}:{line_number}: expected 8 fields")
        values = [float(value) for value in fields]
        quaternion = np.asarray(values[4:8], dtype=np.float64)
        quaternion /= np.linalg.norm(quaternion)
        timestamps.append(values[0])
        translations.append(values[1:4])
        quaternions.append(quaternion)
    if not timestamps:
        raise ValueError(f"{path}: no camera orientations")
    return np.asarray(timestamps), np.asarray(translations), np.stack(quaternions)


def quaternion_rotation_matrix_xyzw(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(value) for value in quaternion)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def tum_world_up_in_camera(
    timestamp: float, poses: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray | None:
    timestamps, _, quaternions = poses
    index = int(np.argmin(np.abs(timestamps - timestamp)))
    if abs(float(timestamps[index]) - timestamp) > 0.03:
        return None
    camera_to_world = quaternion_rotation_matrix_xyzw(quaternions[index])
    return camera_to_world.T @ np.asarray([0.0, 0.0, 1.0])


def tum_fixed_world_floor_in_camera(
    timestamp: float,
    poses: tuple[np.ndarray, np.ndarray, np.ndarray],
    floor_world_z_m: float,
) -> tuple[np.ndarray, float, float] | None:
    timestamps, translations, quaternions = poses
    index = int(np.argmin(np.abs(timestamps - timestamp)))
    if abs(float(timestamps[index]) - timestamp) > 0.03:
        return None
    camera_to_world = quaternion_rotation_matrix_xyzw(quaternions[index])
    up_camera = camera_to_world.T @ np.asarray([0.0, 0.0, 1.0])
    camera_height = float(translations[index, 2] - floor_world_z_m)
    if not 0.45 <= camera_height <= 2.20:
        return None
    return up_camera, camera_height, 0.0
    tensor = source.torch.from_numpy(
        np.ascontiguousarray(padded.transpose(2, 0, 1))
    ).float().to(source.device)
    tensor = ((tensor - source.mean) / source.std)[None]
    with source.torch.inference_mode():
        prediction, _, outputs = source.model.inference({"input": tensor})
    depth = prediction.squeeze()[
        pad_top : source.input_height - (pad_height - pad_top),
        pad_left : source.input_width - (pad_width - pad_left),
    ]
    depth = source.torch.nn.functional.interpolate(
        depth[None, None], (height, width), mode="bilinear", align_corners=False
    ).squeeze()
    canonical_to_real = float(intrinsics_matrix(row)[0, 0]) * scale / 1000.0
    depth = source.torch.clamp(depth * canonical_to_real, 0, 300)
    normal = outputs["prediction_normal"].squeeze(0)[
        :,
        pad_top : source.input_height - (pad_height - pad_top),
        pad_left : source.input_width - (pad_width - pad_left),
    ]
    normal = source.torch.nn.functional.interpolate(
        normal[None], (height, width), mode="bilinear", align_corners=False
    ).squeeze(0)
    directions = normal[:3]
    directions = directions / source.torch.clamp(
        source.torch.linalg.vector_norm(directions, dim=0, keepdim=True), min=1e-8
    )
    normal = source.torch.cat((directions, normal[3:4]), dim=0)
    return (
        depth.detach().cpu().numpy(),
        normal.permute(1, 2, 0).detach().cpu().numpy(),
    )


def evaluate_rows(
    rows: list[dict[str, Any]],
    source: Metric3DPytorchSource,
    ground_mode: str = "depth_ransac",
    reference_ground_mode: str = "depth_ransac",
    reference_floor_world_z_m: float | None = None,
) -> dict[str, Any]:
    lookup_cache: dict[Path, dict[str, Path]] = {}
    pose_cache: dict[
        Path, tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    frame_results = []
    for row in rows:
        frame_path = Path(str(row["frame_path"])).resolve()
        sequence_root = frame_path.parent.parent
        if sequence_root not in lookup_cache:
            lookup_cache[sequence_root] = _depth_lookup(sequence_root)
            pose_cache[sequence_root] = load_tum_camera_poses(
                sequence_root / "groundtruth.txt"
            )
        depth_path = lookup_cache[sequence_root].get(str(frame_path))
        if depth_path is None:
            raise ValueError(f"no registered depth for {frame_path}")
        bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError(f"cannot read {frame_path}")
        sensor = tum_depth_metres(
            normalize_depth_image(
                cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED), depth_path
            )
        )
        intrinsics = intrinsics_matrix(row)
        reference_up = (
            tum_world_up_in_camera(
                float(frame_path.stem), pose_cache[sequence_root]
            )
            if reference_ground_mode == "tum_gravity_oracle"
            else None
        )
        reference_plane = (
            tum_fixed_world_floor_in_camera(
                float(frame_path.stem),
                pose_cache[sequence_root],
                float(reference_floor_world_z_m),
            )
            if reference_ground_mode == "tum_fixed_world_floor"
            and reference_floor_world_z_m is not None
            else None
        )
        started = time.perf_counter()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if ground_mode == "metric3d_normal_guided":
            model, normal_kappa = infer_metric3d_depth_normal(source, rgb, row)
        else:
            model, _ = source.infer(rgb, row)
            normal_kappa = None
        up_camera = (
            tum_world_up_in_camera(
                float(frame_path.stem), pose_cache[sequence_root]
            )
            if ground_mode == "tum_gravity_oracle"
            else None
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        model_field = (
            {"status": "UNKNOWN_GRAVITY"}
            if ground_mode == "tum_gravity_oracle" and up_camera is None
            else clearance_field(model, intrinsics, normal_kappa, up_camera)
        )
        frame_results.append(
            {
                "sequence_root": sequence_root.name,
                "sequence_id": str(row["sequence_id"]),
                "timestamp": float(frame_path.stem),
                "frame_path": str(frame_path),
                "latency_ms": latency_ms,
                "sensor": (
                    {"status": "UNKNOWN_GRAVITY"}
                    if (
                        reference_ground_mode == "tum_gravity_oracle"
                        and reference_up is None
                    )
                    or (
                        reference_ground_mode == "tum_fixed_world_floor"
                        and reference_plane is None
                    )
                    else clearance_field(
                        sensor,
                        intrinsics,
                        up_camera=reference_up,
                        plane_override=reference_plane,
                    )
                ),
                "metric3d": model_field,
            }
        )
    return summarize(frame_results)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    paired_valid = [row for row in rows if row["sensor"]["status"] == "VALID" and row["metric3d"]["status"] == "VALID"]
    clear_errors = []
    decisions = []
    false_clear = []
    height_errors = []
    delta_errors = []
    by_band = {
        band: {"clearance_errors": [], "decisions": [], "false_clear": []}
        for band in BANDS
    }
    previous: dict[tuple[str, str], tuple[float, float]] = {}
    for row in paired_valid:
        height_errors.append(abs(row["metric3d"]["camera_height_m"] - row["sensor"]["camera_height_m"]))
        for band in BANDS:
            truth_band = row["sensor"]["bands"][band]
            model_band = row["metric3d"]["bands"][band]
            truth_clearance = truth_band["clearance_m"]
            model_clearance = model_band["clearance_m"]
            if truth_clearance is not None and model_clearance is not None:
                clearance_error = abs(model_clearance - truth_clearance)
                clear_errors.append(clearance_error)
                by_band[band]["clearance_errors"].append(clearance_error)
                key = (row["sequence_id"], band)
                if key in previous:
                    previous_truth, previous_model = previous[key]
                    delta_errors.append(abs((model_clearance - previous_model) - (truth_clearance - previous_truth)))
                previous[key] = (truth_clearance, model_clearance)
            for horizon in HORIZONS_M:
                truth = truth_band["occupied_by_horizon"][str(horizon)]
                predicted = model_band["occupied_by_horizon"][str(horizon)]
                if truth is not None and predicted is not None:
                    decisions.append(predicted == truth)
                    false_clear.append(bool(truth and not predicted))
                    by_band[band]["decisions"].append(predicted == truth)
                    by_band[band]["false_clear"].append(bool(truth and not predicted))
    valid_fraction = len(paired_valid) / len(rows) if rows else 0.0
    clearance_mae = statistics.fmean(clear_errors) if clear_errors else None
    decision_agreement = statistics.fmean(decisions) if decisions else None
    false_clear_rate = statistics.fmean(false_clear) if false_clear else None
    temporal_delta_mae = statistics.fmean(delta_errors) if delta_errors else None
    gates = {
        "paired_valid_fraction_at_least_0_90": valid_fraction >= 0.90,
        "clearance_mae_at_most_0_25m": clearance_mae is not None and clearance_mae <= 0.25,
        "collision_agreement_at_least_0_90": decision_agreement is not None and decision_agreement >= 0.90,
        "false_clear_rate_at_most_0_05": false_clear_rate is not None and false_clear_rate <= 0.05,
        "temporal_delta_mae_at_most_0_15m": temporal_delta_mae is not None and temporal_delta_mae <= 0.15,
    }
    per_band = {}
    for band, values in by_band.items():
        per_band[band] = {
            "known_clearance_pairs": len(values["clearance_errors"]),
            "clearance_mae_m": statistics.fmean(values["clearance_errors"]) if values["clearance_errors"] else None,
            "known_collision_pairs": len(values["decisions"]),
            "collision_agreement": statistics.fmean(values["decisions"]) if values["decisions"] else None,
            "false_clear_rate": statistics.fmean(values["false_clear"]) if values["false_clear"] else None,
        }
    return {
        "schema": SCHEMA,
        "unique_frames": len(rows),
        "paired_valid_frames": len(paired_valid),
        "paired_valid_fraction": valid_fraction,
        "known_clearance_pairs": len(clear_errors),
        "known_collision_pairs": len(decisions),
        "clearance_mae_m": clearance_mae,
        "collision_agreement": decision_agreement,
        "false_clear_rate": false_clear_rate,
        "temporal_clearance_delta_mae_m": temporal_delta_mae,
        "camera_height_mae_m": statistics.fmean(height_errors) if height_errors else None,
        "per_band": per_band,
        "latency_mean_ms": statistics.fmean(row["latency_ms"] for row in rows),
        "gates": gates,
        "status": "METRIC3D_CLEARANCE_FIELD_A0_DEVELOPMENT_PASS" if all(gates.values()) else "METRIC3D_CLEARANCE_FIELD_A0_DEVELOPMENT_FAIL",
        "frames": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--metric3d-repo", type=Path, required=True)
    parser.add_argument("--metric3d-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--ground-mode",
        choices=(
            "depth_ransac",
            "metric3d_normal_guided",
            "tum_gravity_oracle",
        ),
        default="depth_ransac",
    )
    parser.add_argument(
        "--reference-ground-mode",
        choices=(
            "depth_ransac",
            "tum_gravity_oracle",
            "tum_fixed_world_floor",
        ),
        default="depth_ransac",
    )
    parser.add_argument("--reference-floor-world-z-m", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Metric3DPytorchSource(args.metric3d_repo, args.metric3d_checkpoint, args.device)
    report = evaluate_rows(
        _unique_frames(args.manifest),
        source,
        args.ground_mode,
        args.reference_ground_mode,
        args.reference_floor_world_z_m,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "frames"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
