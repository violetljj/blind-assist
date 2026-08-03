#!/usr/bin/env python3
"""Calibrate an external RGB camera from a fixed chessboard image manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from external_camera_calibration import CALIBRATION_SCHEMA

MIN_USABLE_VIEWS = 8
MAX_RMS_REPROJECTION_ERROR_PX = 1.0
MAX_VIEW_RMSE_PX = 1.5
MIN_FOCAL_TO_MAX_DIMENSION = 0.2
MAX_FOCAL_TO_MAX_DIMENSION = 5.0


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = (path.parent / image_path).resolve()
        row["image_path"] = str(image_path)
    return rows


def board_object_points(cols: int, rows: int, square_size_m: float) -> np.ndarray:
    points = np.zeros((rows * cols, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points[:, :2] *= square_size_m
    return points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--pattern-cols", type=int, required=True)
    parser.add_argument("--pattern-rows", type=int, required=True)
    parser.add_argument("--square-size-mm", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pattern_cols < 3 or args.pattern_rows < 3:
        raise ValueError("chessboard must have at least 3x3 inner corners")
    if not np.isfinite(args.square_size_mm) or args.square_size_mm <= 0:
        raise ValueError("square size must be finite and positive")

    manifest = load_manifest(args.image_manifest)
    pattern_size = (args.pattern_cols, args.pattern_rows)
    square_size_m = args.square_size_mm / 1000.0
    object_template = board_object_points(
        args.pattern_cols, args.pattern_rows, square_size_m
    )
    object_points = []
    image_points = []
    accepted_rows = []
    audit_rows = []
    image_size = None
    seen_hashes: dict[str, str] = {}
    flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    for row in manifest:
        path = Path(row["image_path"])
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest().upper()
        audit = {
            "image_path": str(path),
            "sha256": digest,
            "corners_found": False,
            "duplicate_of": seen_hashes.get(digest),
        }
        if digest in seen_hashes:
            audit_rows.append(audit)
            continue
        seen_hashes[digest] = str(path)
        bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"cannot decode calibration image {path}")
        height, width = bgr.shape[:2]
        if image_size is None:
            image_size = (width, height)
        elif image_size != (width, height):
            raise ValueError("calibration images must share one resolution")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=flags)
        audit["corners_found"] = bool(found)
        audit_rows.append(audit)
        if not found:
            continue
        object_points.append(object_template.copy())
        image_points.append(np.asarray(corners, dtype=np.float32))
        accepted_rows.append(audit)

    if image_size is None:
        raise ValueError("calibration manifest is empty")
    report: dict[str, Any] = {
        "schema": CALIBRATION_SCHEMA,
        "image_size_px": list(image_size),
        "pattern_inner_corners": [args.pattern_cols, args.pattern_rows],
        "square_size_m": square_size_m,
        "input_images": len(manifest),
        "unique_images": len(seen_hashes),
        "usable_views": len(object_points),
        "gates": {
            "min_usable_views": MIN_USABLE_VIEWS,
            "max_rms_reprojection_error_px": MAX_RMS_REPROJECTION_ERROR_PX,
            "max_view_rmse_px": MAX_VIEW_RMSE_PX,
            "min_focal_to_max_dimension": MIN_FOCAL_TO_MAX_DIMENSION,
            "max_focal_to_max_dimension": MAX_FOCAL_TO_MAX_DIMENSION,
        },
        "images": audit_rows,
        "camera_matrix": None,
        "distortion_model": "opencv_plumb_bob",
        "distortion_coefficients": None,
        "rms_reprojection_error_px": None,
        "mean_view_rmse_px": None,
        "max_view_rmse_px": None,
        "gate_results": {"minimum_usable_views": len(object_points) >= MIN_USABLE_VIEWS},
        "admitted": False,
        "claim_ceiling": "intrinsic calibration for the exact camera, focus, zoom, and resolution only",
    }
    if len(object_points) >= MIN_USABLE_VIEWS:
        rms, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None,
        )
        view_errors = []
        for objects, observed, rotation, translation in zip(
            object_points, image_points, rvecs, tvecs, strict=True
        ):
            projected, _ = cv2.projectPoints(
                objects, rotation, translation, matrix, distortion
            )
            residual = observed.reshape(-1, 2) - projected.reshape(-1, 2)
            view_errors.append(float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))))
        max_dimension = max(image_size)
        fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
        cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
        report.update(
            {
                "camera_matrix": matrix.tolist(),
                "distortion_coefficients": distortion.reshape(-1).tolist(),
                "rms_reprojection_error_px": float(rms),
                "mean_view_rmse_px": statistics.fmean(view_errors),
                "max_view_rmse_px": max(view_errors),
            }
        )
        report["gate_results"].update(
            {
                "rms_reprojection_error": float(rms)
                <= MAX_RMS_REPROJECTION_ERROR_PX,
                "max_view_rmse": max(view_errors) <= MAX_VIEW_RMSE_PX,
                "focal_range": all(
                    MIN_FOCAL_TO_MAX_DIMENSION
                    <= focal / max_dimension
                    <= MAX_FOCAL_TO_MAX_DIMENSION
                    for focal in (fx, fy)
                ),
                "principal_point": 0 <= cx <= image_size[0]
                and 0 <= cy <= image_size[1],
            }
        )
        report["admitted"] = all(report["gate_results"].values())
    report["terminal"] = (
        "EXTERNAL_CAMERA_INTRINSIC_CALIBRATION_ADMITTED"
        if report["admitted"]
        else "EXTERNAL_CAMERA_INTRINSIC_CALIBRATION_NOT_ADMITTED"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: report[key] for key in (
                "terminal",
                "admitted",
                "usable_views",
                "rms_reprojection_error_px",
                "max_view_rmse_px",
                "gate_results",
            )},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
