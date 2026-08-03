#!/usr/bin/env python3
"""Shared frozen primitives for dense propagation and calibration distillation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
BANDS = ("left", "center", "right")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPO_ROOT / path


def validate_file(receipt: dict[str, Any], path_key: str, hash_key: str) -> Path:
    path = resolve(str(receipt[path_key]))
    expected = str(receipt[hash_key]).upper()
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual} != {expected}")
    return path


def frame_key(frame: dict[str, Any]) -> tuple[str, float, str]:
    return (
        str(frame["sequence_id"]),
        float(frame["timestamp"]),
        str(frame["frame_path"]),
    )


def report_frames(report: dict[str, Any]) -> list[dict[str, Any]]:
    frames = report.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("bound report has no frames")
    return sorted(frames, key=frame_key)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    return ordered[round(fraction * (len(ordered) - 1))]


def fit_dense_affine(
    fast_depth: np.ndarray,
    metric_depth: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Fit the single frozen OLS/MAD/OLS affine mapping."""
    if fast_depth.shape != metric_depth.shape or fast_depth.ndim != 2:
        raise ValueError("dense affine inputs must be equal-shape 2D arrays")
    stride = int(config["sample_stride_px"])
    low, high = (float(value) for value in config["valid_depth_range_m"])
    x = np.asarray(fast_depth[::stride, ::stride], dtype=np.float64).reshape(-1)
    y = np.asarray(metric_depth[::stride, ::stride], dtype=np.float64).reshape(-1)
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= low)
        & (x <= high)
        & (y >= low)
        & (y <= high)
    )
    x, y = x[valid], y[valid]
    minimum_pairs = int(config["minimum_pairs"])
    if len(x) < minimum_pairs:
        return {"status": "UNKNOWN_AFFINE_PAIRS", "pairs": len(x)}
    design = np.column_stack((x, np.ones_like(x)))
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - (slope * x + intercept)
    center = float(np.median(residual))
    mad = float(np.median(np.abs(residual - center)))
    robust_sigma = max(1e-6, 1.4826 * mad)
    inliers = np.abs(residual - center) <= 3.0 * robust_sigma
    inlier_fraction = float(np.mean(inliers))
    if inlier_fraction < float(config["minimum_inlier_fraction"]):
        return {
            "status": "UNKNOWN_AFFINE_INLIERS",
            "pairs": len(x),
            "inlier_fraction": inlier_fraction,
        }
    slope, intercept = np.linalg.lstsq(design[inliers], y[inliers], rcond=None)[0]
    inlier_residual = y[inliers] - (slope * x[inliers] + intercept)
    median_absolute_residual = float(np.median(np.abs(inlier_residual)))
    lower_slope, upper_slope = (float(value) for value in config["slope_bounds"])
    status = "VALID"
    if not all(math.isfinite(float(value)) for value in (slope, intercept)):
        status = "UNKNOWN_AFFINE_NONFINITE"
    elif not lower_slope <= float(slope) <= upper_slope:
        status = "UNKNOWN_AFFINE_SLOPE"
    elif median_absolute_residual > float(
        config["maximum_inlier_median_absolute_residual_m"]
    ):
        status = "UNKNOWN_AFFINE_RESIDUAL"
    return {
        "status": status,
        "pairs": len(x),
        "inliers": int(np.sum(inliers)),
        "inlier_fraction": inlier_fraction,
        "slope": float(slope),
        "intercept_m": float(intercept),
        "median_absolute_residual_m": median_absolute_residual,
    }


def affine_depth(depth: np.ndarray, fit: dict[str, Any]) -> np.ndarray:
    if fit.get("status") != "VALID":
        raise ValueError("cannot apply invalid affine fit")
    return np.asarray(
        float(fit["slope"]) * np.asarray(depth, dtype=np.float32)
        + float(fit["intercept_m"]),
        dtype=np.float32,
    )


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
