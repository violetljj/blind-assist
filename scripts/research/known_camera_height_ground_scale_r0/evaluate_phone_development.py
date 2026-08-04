"""Evaluate fixed-height phone captures with a consumed Samsung AR reference.

This is a Development-only diagnostic.  Samsung Quick Measure uses the same phone
camera/AR stack, so its distance is not an independent P0/R2 ground truth.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import core as scale_core


REPO_ROOT = Path(__file__).resolve().parents[3]
HFTF_DIR = REPO_ROOT / "scripts" / "research" / "hftf"
sys.path.insert(0, str(HFTF_DIR))

from produce_external_rgb_metric_depth_observations import (  # noqa: E402
    DepthAnythingV2MetricSource,
)


EXPECTED_CHECKPOINT_SHA256 = (
    "B782898D8A3E8BE1F639DE33837ED85E9B4B73E40F8F5E5CD99067588D722545"
)
DEVELOPMENT_PROTOCOL_ID = "KNOWN_HEIGHT_PHONE_DEVELOPMENT_CAPTURE_R0"
DEVELOPMENT_STATUS = "DEVELOPMENT_CAPTURED_CONSUMED_REFERENCE"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(
            descriptor,
            (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            ),
        )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def scaled_raw_intrinsics(receipt: dict[str, Any], width: int, height: int) -> np.ndarray:
    values = receipt.get("intrinsic_calibration")
    active = str(receipt.get("active_array", "")).split()
    if not isinstance(values, list) or len(values) < 4 or len(active) != 4:
        raise ValueError("missing intrinsic calibration or active array")
    left, top, right, bottom = map(float, active)
    active_width = right - left
    active_height = bottom - top
    if active_width <= 0 or active_height <= 0:
        raise ValueError("invalid active array")
    sx = width / active_width
    sy = height / active_height
    fx, fy, cx, cy = map(float, values[:4])
    return np.asarray(
        [[fx * sx, 0.0, (cx - left) * sx], [0.0, fy * sy, (cy - top) * sy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def rotate_clockwise_with_intrinsics(
    image: np.ndarray, intrinsics: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if image.ndim != 3:
        raise ValueError("expected HWC image")
    height, _ = image.shape[:2]
    matrix = np.asarray(intrinsics, dtype=np.float64)
    rotated = np.ascontiguousarray(np.rot90(image, k=3))
    transformed = np.asarray(
        [
            [matrix[1, 1], 0.0, (height - 1.0) - matrix[1, 2]],
            [0.0, matrix[0, 0], matrix[0, 2]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return rotated, transformed


def center_median(depth: np.ndarray, fraction: float = 0.10) -> float | None:
    values = np.asarray(depth, dtype=np.float64)
    if values.ndim != 2 or not 0.0 < fraction <= 1.0:
        raise ValueError("invalid depth or center fraction")
    height, width = values.shape
    half_h = max(2, int(round(height * fraction / 2.0)))
    half_w = max(2, int(round(width * fraction / 2.0)))
    center_y, center_x = height // 2, width // 2
    roi = values[
        max(0, center_y - half_h) : min(height, center_y + half_h),
        max(0, center_x - half_w) : min(width, center_x + half_w),
    ]
    valid = roi[np.isfinite(roi) & (roi > 0.0)]
    return float(np.median(valid)) if len(valid) >= 16 else None


def session_metric(values: list[float], reference_m: float) -> dict[str, Any]:
    if not values:
        return {"coverage": 0.0, "median_m": None, "absolute_relative_error": None, "temporal_instability": None}
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    return {
        "coverage": None,
        "median_m": median,
        "absolute_relative_error": abs(median - reference_m) / reference_m,
        "temporal_instability": float(np.median(np.abs(array - median)) / max(median, 1e-9)),
    }


def load_session(session_root: Path) -> dict[str, Any]:
    receipt = json.loads((session_root / "receipt.json").read_text(encoding="utf-8"))
    reference = json.loads(
        (session_root / "reference" / "reference.json").read_text(encoding="utf-8")
    )
    frames = json.loads((session_root / "frames.json").read_text(encoding="utf-8"))
    intrinsics = json.loads((session_root / "intrinsics.json").read_text(encoding="utf-8"))
    if receipt.get("protocol_id") != DEVELOPMENT_PROTOCOL_ID or receipt.get("status") != DEVELOPMENT_STATUS:
        raise ValueError(f"{session_root.name}: not an admitted Development capture")
    if reference.get("truth_firewall") != "DEVELOPMENT_LABEL_ONLY_NOT_FORMAL_GROUND_TRUTH":
        raise ValueError(f"{session_root.name}: missing Development truth firewall")
    if reference.get("reference_method") != "samsung_quick_measure_ar":
        raise ValueError(f"{session_root.name}: unexpected reference method")
    points = reference.get("reference_points")
    if not isinstance(points, list) or len(points) != 1:
        raise ValueError(f"{session_root.name}: expected one current-target reference")
    reference_m = float(points[0]["measured_distance_m"])
    if not 0.1 <= reference_m <= 10.0:
        raise ValueError(f"{session_root.name}: reference out of Development range")
    if len(frames) != 25:
        raise ValueError(f"{session_root.name}: expected 25 frames")
    timestamps = [int(row["capture_timestamp_ns"]) for row in frames]
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError(f"{session_root.name}: non-monotonic timestamps")
    for row in frames:
        image_path = session_root / row["rgb_file"]
        if not image_path.is_file() or sha256(image_path) != str(row["rgb_sha256"]).upper():
            raise ValueError(f"{session_root.name}: RGB identity failure")
    return {
        "receipt": receipt,
        "reference": reference,
        "reference_m": reference_m,
        "frames": frames,
        "intrinsics": intrinsics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--captures-root", required=True, type=Path)
    parser.add_argument("--dav2-repo", required=True, type=Path)
    parser.add_argument("--dav2-checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--camera-height-override-m", type=float)
    parser.add_argument("--camera-height-uncertainty-override-m", type=float)
    parser.add_argument("--height-correction-note")
    args = parser.parse_args()
    height_override_requested = args.camera_height_override_m is not None
    if height_override_requested != (args.camera_height_uncertainty_override_m is not None):
        raise ValueError("height and uncertainty overrides must be provided together")
    if height_override_requested and not args.height_correction_note:
        raise ValueError("a height correction note is required with an override")
    if height_override_requested and (
        args.camera_height_override_m <= 0.0
        or args.camera_height_uncertainty_override_m < 0.0
    ):
        raise ValueError("height override must be positive and uncertainty non-negative")
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    if sha256(args.dav2_checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("unexpected DA V2 checkpoint")
    session_roots = sorted(path for path in args.captures_root.iterdir() if path.is_dir())
    if len(session_roots) != 3:
        raise ValueError("expected exactly three Development phone sessions")
    sessions = [(root, load_session(root)) for root in session_roots]

    source = DepthAnythingV2MetricSource(
        args.dav2_repo,
        args.dav2_checkpoint,
        args.device,
        input_size=518,
        precision="fp16" if args.device.startswith("cuda") else "fp32",
    )
    args.output_root.mkdir(parents=True)
    rows_path = args.output_root / "frames.jsonl"
    session_results = []
    all_rows: list[dict[str, Any]] = []
    for session_root, session in sessions:
        receipt = session["receipt"]
        reference_m = session["reference_m"]
        raw_values: list[float] = []
        known_values: list[float] = []
        valid_scales: list[float] = []
        unique_hashes = {str(row["rgb_sha256"]) for row in session["frames"]}
        for row in session["frames"]:
            image_path = session_root / row["rgb_file"]
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"unable to decode {image_path}")
            raw_intrinsics = scaled_raw_intrinsics(session["intrinsics"], bgr.shape[1], bgr.shape[0])
            oriented_bgr, oriented_intrinsics = rotate_clockwise_with_intrinsics(bgr, raw_intrinsics)
            started = time.perf_counter()
            da_depth, _ = source.infer(cv2.cvtColor(oriented_bgr, cv2.COLOR_BGR2RGB), {})
            latency_ms = (time.perf_counter() - started) * 1000.0
            raw_center = center_median(da_depth)
            if raw_center is not None:
                raw_values.append(raw_center)
            height_receipt = scale_core.CameraHeightReceipt(
                camera_profile_id=f"{receipt['device_model']}/camera-{receipt['camera_id']}",
                mount_profile_id=str(receipt["mount_profile_id"]),
                height_m=(
                    float(args.camera_height_override_m)
                    if height_override_requested
                    else float(receipt["camera_height_m"])
                ),
                uncertainty_m=(
                    float(args.camera_height_uncertainty_override_m)
                    if height_override_requested
                    else float(receipt["camera_height_uncertainty_m"])
                ),
            )
            recovered = scale_core.recover_metric_scale(
                da_depth,
                oriented_intrinsics,
                height_receipt,
                height_receipt.camera_profile_id,
                height_receipt.mount_profile_id,
            )
            known_center = None
            scale = None
            reason = None
            ground = None
            if recovered["status"] == "VALID":
                scale = float(recovered["scale"])
                valid_scales.append(scale)
                known_center = center_median(np.asarray(recovered["metric_depth"]))
                if known_center is not None:
                    known_values.append(known_center)
                plane = recovered["ground"]
                ground = {
                    "relative_height": plane.relative_height,
                    "normalized_median_residual": plane.normalized_median_residual,
                    "candidate_count": plane.candidate_count,
                    "inlier_count": plane.inlier_count,
                    "inlier_fraction": plane.inlier_fraction,
                }
            else:
                reason = str(recovered.get("reason"))
            all_rows.append(
                {
                    "session_id": session_root.name,
                    "frame_id": int(row["frame_id"]),
                    "rgb_sha256": row["rgb_sha256"],
                    "reference_m": reference_m,
                    "raw_center_m": raw_center,
                    "known_height_center_m": known_center,
                    "known_height_scale": scale,
                    "known_height_status": recovered["status"],
                    "known_height_reason": reason,
                    "ground": ground,
                    "latency_ms": latency_ms,
                }
            )
        raw_metric = session_metric(raw_values, reference_m)
        known_metric = session_metric(known_values, reference_m)
        raw_metric["coverage"] = len(raw_values) / len(session["frames"])
        known_metric["coverage"] = len(known_values) / len(session["frames"])
        session_results.append(
            {
                "session_id": session_root.name,
                "reference_m": reference_m,
                "frame_count": len(session["frames"]),
                "unique_rgb_count": len(unique_hashes),
                "raw_da": raw_metric,
                "known_height": known_metric,
                "known_height_scale_median": float(np.median(valid_scales)) if valid_scales else None,
            }
        )

    rows_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in all_rows),
        encoding="utf-8",
    )
    raw_errors = [row["raw_da"]["absolute_relative_error"] for row in session_results]
    known_errors = [
        row["known_height"]["absolute_relative_error"]
        for row in session_results
        if row["known_height"]["absolute_relative_error"] is not None
    ]
    raw_macro = float(np.mean(raw_errors))
    known_macro = float(np.mean(known_errors)) if known_errors else None
    improved_sessions = sum(
        1
        for row in session_results
        if row["known_height"]["absolute_relative_error"] is not None
        and row["known_height"]["absolute_relative_error"]
        < row["raw_da"]["absolute_relative_error"]
    )
    known_coverage = float(
        np.mean([row["known_height"]["coverage"] for row in session_results])
    )
    unknown_reason_counts = dict(
        sorted(
            Counter(
                str(row["known_height_reason"])
                for row in all_rows
                if row["known_height_status"] != "VALID"
            ).items()
        )
    )
    if known_coverage == 0.0 and unknown_reason_counts == {"INVALID_HEIGHT_RECEIPT": len(all_rows)}:
        recommendation = "DEVELOPMENT_NOT_EVALUABLE_CAMERA_HEIGHT_OUT_OF_PROTOCOL"
    elif known_macro is not None and known_macro < raw_macro and improved_sessions >= 2 and known_coverage >= 0.8:
        recommendation = "DEVELOPMENT_PROMISING_NEEDS_INDEPENDENT_FRESH_CAPTURE"
    else:
        recommendation = "DEVELOPMENT_NOT_PROMISING_DO_NOT_PROMOTE"
    result = {
        "schema": "blindassist_known_height_phone_development_evaluation_v1",
        "status": "DEVELOPMENT_DIAGNOSTIC_COMPLETE",
        "authority": {
            "development_only": True,
            "consumed_same_phone_ar_reference": True,
            "formal_p0_r2": False,
            "production": False,
        },
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "camera_height": {
            "recorded_receipt_values_m": sorted(
                {float(session["receipt"]["camera_height_m"]) for _, session in sessions}
            ),
            "effective_height_m": (
                float(args.camera_height_override_m)
                if height_override_requested
                else None
            ),
            "effective_uncertainty_m": (
                float(args.camera_height_uncertainty_override_m)
                if height_override_requested
                else None
            ),
            "correction_note": args.height_correction_note,
            "original_receipts_unchanged": True,
            "protocol_range_m": list(scale_core.CAMERA_HEIGHT_RANGE_M),
        },
        "session_count": len(session_results),
        "frame_count": len(all_rows),
        "session_results": session_results,
        "macro": {
            "raw_da_absolute_relative_error": raw_macro,
            "known_height_absolute_relative_error": known_macro,
            "known_height_coverage": known_coverage,
            "improved_sessions": improved_sessions,
        },
        "recommendation": recommendation,
        "unknown_reason_counts": unknown_reason_counts,
        "limitations": [
            "Samsung Quick Measure is a same-phone AR reference, not independent ground truth.",
            "The center ROI is a proxy for the Quick Measure center target.",
            "Three fixed phone sessions are Development diagnostics, not generalization evidence.",
        ],
        "frames_jsonl": rows_path.name,
        "frames_jsonl_sha256": sha256(rows_path),
    }
    write_json_new(args.output_root / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
