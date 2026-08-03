"""Validated external-camera calibration profile and frame rectification."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

CALIBRATION_SCHEMA = "hftf_external_camera_calibration_r0"


@dataclass(frozen=True)
class CameraCalibration:
    width: int
    height: int
    camera_matrix: np.ndarray
    distortion: np.ndarray
    source_id: str
    rectification_required: bool

    @property
    def intrinsics(self) -> list[float]:
        return [
            float(self.camera_matrix[0, 0]),
            float(self.camera_matrix[1, 1]),
            float(self.camera_matrix[0, 2]),
            float(self.camera_matrix[1, 2]),
        ]

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height


def validate_calibration(calibration: CameraCalibration) -> None:
    if calibration.width <= 0 or calibration.height <= 0:
        raise ValueError("calibration image dimensions must be positive")
    if calibration.camera_matrix.shape != (3, 3):
        raise ValueError("camera matrix must be 3x3")
    if calibration.distortion.ndim != 1 or calibration.distortion.size not in {
        4,
        5,
        8,
        12,
        14,
    }:
        raise ValueError("unsupported distortion coefficient count")
    if not np.all(np.isfinite(calibration.camera_matrix)) or not np.all(
        np.isfinite(calibration.distortion)
    ):
        raise ValueError("calibration values must be finite")
    fx, fy, cx, cy = calibration.intrinsics
    if fx <= 0 or fy <= 0:
        raise ValueError("calibration focal lengths must be positive")
    if not (0 <= cx <= calibration.width and 0 <= cy <= calibration.height):
        raise ValueError("calibration principal point must lie inside the frame")


def load_calibration(path: Path) -> CameraCalibration:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if payload.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError("unsupported calibration schema")
    if payload.get("admitted") is not True:
        raise ValueError("calibration profile is not admitted")
    width, height = (int(value) for value in payload["image_size_px"])
    matrix = np.asarray(payload["camera_matrix"], dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(
        payload["distortion_coefficients"], dtype=np.float64
    ).reshape(-1)
    calibration = CameraCalibration(
        width=width,
        height=height,
        camera_matrix=matrix,
        distortion=distortion,
        source_id=f"json:{hashlib.sha256(raw).hexdigest().upper()}",
        rectification_required=bool(np.any(np.abs(distortion) > 1e-12)),
    )
    validate_calibration(calibration)
    return calibration


def pinhole_calibration(
    intrinsics: list[float], calibration_size: list[int]
) -> CameraCalibration:
    fx, fy, cx, cy = (float(value) for value in intrinsics)
    width, height = (int(value) for value in calibration_size)
    matrix = np.asarray(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    calibration = CameraCalibration(
        width=width,
        height=height,
        camera_matrix=matrix,
        distortion=np.zeros(5, dtype=np.float64),
        source_id="cli:pinhole-zero-distortion",
        rectification_required=False,
    )
    validate_calibration(calibration)
    return calibration


class FrameRectifier:
    def __init__(self, calibration: CameraCalibration) -> None:
        validate_calibration(calibration)
        self.calibration = calibration
        self._map_x: np.ndarray | None = None
        self._map_y: np.ndarray | None = None
        self._valid_mask: np.ndarray | None = None
        if calibration.rectification_required:
            self._map_x, self._map_y = cv2.initUndistortRectifyMap(
                calibration.camera_matrix,
                calibration.distortion,
                None,
                calibration.camera_matrix,
                calibration.size,
                cv2.CV_32FC1,
            )
            source_valid = np.full(
                (calibration.height, calibration.width), 255, dtype=np.uint8
            )
            self._valid_mask = (
                cv2.remap(
                    source_valid,
                    self._map_x,
                    self._map_y,
                    cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                > 0
            )

    def rectify(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        height, width = bgr.shape[:2]
        if (width, height) != self.calibration.size:
            raise ValueError(
                f"frame size {(width, height)} differs from calibration size "
                f"{self.calibration.size}"
            )
        if self._map_x is None or self._map_y is None or self._valid_mask is None:
            return bgr, np.ones((height, width), dtype=bool)
        rectified = cv2.remap(
            bgr,
            self._map_x,
            self._map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(104, 116, 124),
        )
        return rectified, self._valid_mask


def finite_ratio(mask: np.ndarray) -> float:
    value = float(np.mean(mask))
    if not math.isfinite(value):
        raise ValueError("invalid rectification valid-mask fraction")
    return value
