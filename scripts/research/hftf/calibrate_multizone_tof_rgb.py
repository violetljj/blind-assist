#!/usr/bin/env python3
"""Calibrate a multi-zone ToF sensor into a rectified RGB camera frame."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from external_camera_calibration import CameraCalibration, load_calibration
from multizone_tof_anchor import REGISTRATION_SCHEMA

GEOMETRY_SCHEMA = "hftf_multizone_tof_geometry_r0"
CORRESPONDENCE_SCHEMA = "hftf_tof_rgb_correspondence_r0"


def load_geometry(path: Path) -> tuple[str, dict[str, np.ndarray], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != GEOMETRY_SCHEMA:
        raise ValueError("unsupported ToF geometry schema")
    sensor_id = str(payload["tof_sensor_id"])
    rays = {}
    for item in payload["zones"]:
        zone_id = str(item["zone_id"])
        ray = np.asarray(item["ray_tof_unit"], dtype=np.float64)
        if not zone_id or zone_id in rays or ray.shape != (3,):
            raise ValueError("geometry zone IDs must be unique with 3D rays")
        norm = float(np.linalg.norm(ray))
        if not np.all(np.isfinite(ray)) or not math.isclose(norm, 1.0, abs_tol=1e-5):
            raise ValueError("geometry rays must be finite unit vectors")
        rays[zone_id] = ray
    if not sensor_id or len(rays) < 3:
        raise ValueError("ToF geometry requires an identity and at least three zones")
    return sensor_id, rays, payload


def load_correspondences(
    path: Path,
    sensor_id: str,
    rgb_calibration_id: str,
    rays: dict[str, np.ndarray],
    calibration: CameraCalibration,
) -> tuple[np.ndarray, np.ndarray, list[str], list[float]]:
    object_points = []
    image_points = []
    zone_ids = []
    ranges = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("schema") != CORRESPONDENCE_SCHEMA:
            raise ValueError(f"line {line_number}: unsupported correspondence schema")
        if str(row["tof_sensor_id"]) != sensor_id:
            raise ValueError(f"line {line_number}: ToF sensor identity mismatch")
        if str(row["rgb_calibration_id"]) != rgb_calibration_id:
            raise ValueError(f"line {line_number}: RGB calibration identity mismatch")
        zone_id = str(row["zone_id"])
        if zone_id not in rays:
            raise ValueError(f"line {line_number}: unknown ToF zone")
        distance = float(row["range_m"])
        u, v = (float(value) for value in row["rgb_uv_px"])
        if not math.isfinite(distance) or distance <= 0:
            raise ValueError(f"line {line_number}: range must be finite and positive")
        if not (math.isfinite(u) and math.isfinite(v)):
            raise ValueError(f"line {line_number}: RGB point must be finite")
        if not (0 <= u < calibration.width and 0 <= v < calibration.height):
            raise ValueError(f"line {line_number}: RGB point lies outside calibration")
        object_points.append(rays[zone_id] * distance)
        image_points.append([u, v])
        zone_ids.append(zone_id)
        ranges.append(distance)
    if not object_points:
        raise ValueError("correspondence file is empty")
    return (
        np.asarray(object_points, dtype=np.float64),
        np.asarray(image_points, dtype=np.float64),
        zone_ids,
        ranges,
    )


def calibrate(
    object_points: np.ndarray,
    image_points: np.ndarray,
    zone_ids: list[str],
    ranges: list[float],
    camera_matrix: np.ndarray,
    *,
    minimum_observations: int,
    minimum_zones: int,
    minimum_range_span_m: float,
    minimum_inlier_fraction: float,
    maximum_rmse_px: float,
    ransac_threshold_px: float,
) -> tuple[np.ndarray | None, dict[str, Any], dict[str, bool]]:
    if minimum_observations < 6 or minimum_zones < 3:
        raise ValueError("calibration requires at least six observations and three zones")
    if not 0 < minimum_inlier_fraction <= 1:
        raise ValueError("minimum inlier fraction must be in (0, 1]")
    if any(
        not math.isfinite(value) or value <= 0
        for value in (minimum_range_span_m, maximum_rmse_px, ransac_threshold_px)
    ):
        raise ValueError("calibration thresholds must be finite and positive")
    if object_points.shape != (len(image_points), 3) or image_points.shape[1:] != (2,):
        raise ValueError("invalid correspondence array shapes")
    enough_for_solver = len(object_points) >= 6
    transform = None
    inlier_count = 0
    rmse = None
    maximum_error = None
    if enough_for_solver:
        ok, rotation_vector, translation, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            camera_matrix,
            np.zeros(5, dtype=np.float64),
            iterationsCount=200,
            reprojectionError=ransac_threshold_px,
            confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if ok and inliers is not None and len(inliers) >= 6:
            indices = inliers.reshape(-1)
            rotation_vector, translation = cv2.solvePnPRefineLM(
                object_points[indices],
                image_points[indices],
                camera_matrix,
                np.zeros(5, dtype=np.float64),
                rotation_vector,
                translation,
            )
            rotation, _ = cv2.Rodrigues(rotation_vector)
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = rotation
            transform[:3, 3] = translation.reshape(3)
            projected, _ = cv2.projectPoints(
                object_points,
                rotation_vector,
                translation,
                camera_matrix,
                np.zeros(5, dtype=np.float64),
            )
            errors = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
            inlier_count = len(indices)
            rmse = float(np.sqrt(np.mean(np.square(errors))))
            maximum_error = float(np.max(errors))
    observation_count = len(object_points)
    distinct_zones = len(set(zone_ids))
    range_span = float(max(ranges) - min(ranges)) if ranges else 0.0
    inlier_fraction = inlier_count / observation_count if observation_count else 0.0
    metrics = {
        "observation_count": observation_count,
        "distinct_zone_count": distinct_zones,
        "range_span_m": range_span,
        "inlier_count": inlier_count,
        "inlier_fraction": inlier_fraction,
        "reprojection_rmse_px": rmse,
        "maximum_reprojection_error_px": maximum_error,
    }
    gates = {
        "observation_count": observation_count >= minimum_observations,
        "zone_coverage": distinct_zones >= minimum_zones,
        "range_span": range_span >= minimum_range_span_m,
        "solver": transform is not None,
        "inlier_fraction": inlier_fraction >= minimum_inlier_fraction,
        "reprojection_rmse": rmse is not None and rmse <= maximum_rmse_px,
    }
    return transform, metrics, gates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tof-geometry", type=Path, required=True)
    parser.add_argument("--rgb-calibration", type=Path, required=True)
    parser.add_argument("--correspondences", type=Path, required=True)
    parser.add_argument("--registration-id", required=True)
    parser.add_argument("--minimum-observations", type=int, required=True)
    parser.add_argument("--minimum-zones", type=int, required=True)
    parser.add_argument("--minimum-range-span-m", type=float, required=True)
    parser.add_argument("--minimum-inlier-fraction", type=float, required=True)
    parser.add_argument("--maximum-rmse-px", type=float, required=True)
    parser.add_argument("--ransac-threshold-px", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sensor_id, rays, geometry = load_geometry(args.tof_geometry)
    calibration = load_calibration(args.rgb_calibration)
    object_points, image_points, zone_ids, ranges = load_correspondences(
        args.correspondences,
        sensor_id,
        calibration.source_id,
        rays,
        calibration,
    )
    transform, metrics, gates = calibrate(
        object_points,
        image_points,
        zone_ids,
        ranges,
        calibration.camera_matrix,
        minimum_observations=args.minimum_observations,
        minimum_zones=args.minimum_zones,
        minimum_range_span_m=args.minimum_range_span_m,
        minimum_inlier_fraction=args.minimum_inlier_fraction,
        maximum_rmse_px=args.maximum_rmse_px,
        ransac_threshold_px=args.ransac_threshold_px,
    )
    admitted = all(gates.values())
    result = {
        "schema": REGISTRATION_SCHEMA,
        "admitted": admitted,
        "registration_id": args.registration_id,
        "tof_sensor_id": sensor_id,
        "rgb_calibration_id": calibration.source_id,
        "transform_rgb_from_tof": transform.tolist() if transform is not None else None,
        "zones": geometry["zones"],
        "metrics": metrics,
        "gates": gates,
        "policy": {
            "minimum_observations": args.minimum_observations,
            "minimum_zones": args.minimum_zones,
            "minimum_range_span_m": args.minimum_range_span_m,
            "minimum_inlier_fraction": args.minimum_inlier_fraction,
            "maximum_rmse_px": args.maximum_rmse_px,
            "ransac_threshold_px": args.ransac_threshold_px,
        },
        "claim_ceiling": "rig-specific spatial registration only; no task quality, synchronization, safety, or production authority",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not admitted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
