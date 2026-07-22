#!/usr/bin/env python3
"""Fail-closed host validation for the benchmark-only ARCore frame-bound canary."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


RAW_SCHEMA = "blindassist_ustrf_arcore_single_frame_observation_v1"
SUMMARY_SCHEMA = "blindassist_ustrf_arcore_frame_bound_canary_summary_v1"
DEVICE_SCHEMA = "blindassist_ustrf_arcore_frame_bound_device_receipt_v1"
AUDIT_SCHEMA = "blindassist_ustrf_arcore_frame_bound_canary_audit_v1"
EXPECTED_MODEL = "SM-S9280"
EXPECTED_PACKAGE = "com.linnan.blindassist.ustrfbenchmark"
EXPECTED_ARCORE_SDK = "1.33.0"
EXPECTED_REFERENCE_MODE = "INTER_FRAME_STABLE"
MIN_QUALIFYING_FRAMES = 100
MIN_FRACTION = 0.95
MAX_ANCHOR_TRANSLATION_DRIFT_M = 0.25
MAX_ANCHOR_ROTATION_DRIFT_DEG = 20.0
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ContractError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot parse JSON {path}: {error}") from error
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"missing JSONL file: {path}")
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
            require(bool(line.strip()), f"blank JSONL row {line_number}")
            value = json.loads(line)
            require(isinstance(value, dict), f"JSONL row {line_number} must be an object")
            rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot parse JSONL {path}: {error}") from error
    require(rows, "raw frame JSONL is empty")
    return rows


def boundary_is_safe(value: Any) -> bool:
    return isinstance(value, dict) and value == {
        "benchmark_only": True,
        "app_runtime_involved": False,
        "navigation_output_issued": False,
        "training_authority": False,
        "production_authorized": False,
        "human_truth": False,
    }


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def finite_array(value: Any, length: int) -> bool:
    return isinstance(value, list) and len(value) == length and all(finite_number(item) for item in value)


def valid_pose(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and finite_array(value.get("translation_m"), 3)
        and finite_array(value.get("rotation_quaternion_xyzw"), 4)
        and finite_array(value.get("matrix_4x4_column_major"), 16)
    )


def valid_intrinsics(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        positive_int(value.get("width_px"))
        and positive_int(value.get("height_px"))
        and all(finite_number(value.get(key)) and float(value[key]) > 0.0 for key in ("focal_x_px", "focal_y_px"))
        and all(finite_number(value.get(key)) for key in ("principal_x_px", "principal_y_px"))
    )


def valid_image(value: Any) -> bool:
    if not isinstance(value, dict) or value.get("available") is not True:
        return False
    digest = value.get("content_sha256")
    return (
        positive_int(value.get("timestamp_ns"))
        and positive_int(value.get("width_px"))
        and positive_int(value.get("height_px"))
        and positive_int(value.get("plane_count"))
        and isinstance(digest, str)
        and SHA256_RE.fullmatch(digest) is not None
    )


def pose_translation(value: dict[str, Any]) -> tuple[float, float, float]:
    translation = value["translation_m"]
    return float(translation[0]), float(translation[1]), float(translation[2])


def pose_quaternion(value: dict[str, Any]) -> tuple[float, float, float, float]:
    quaternion = [float(item) for item in value["rotation_quaternion_xyzw"]]
    norm = math.sqrt(sum(item * item for item in quaternion))
    require(norm > 1e-9, "anchor quaternion has zero norm")
    return tuple(item / norm for item in quaternion)  # type: ignore[return-value]


def translation_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def quaternion_distance_degrees(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    dot = min(1.0, max(-1.0, abs(sum(a * b for a, b in zip(left, right)))))
    return math.degrees(2.0 * math.acos(dot))


def validate(
    raw_path: Path,
    summary_path: Path,
    device_receipt_path: Path,
) -> dict[str, Any]:
    raw_path = raw_path.resolve()
    summary_path = summary_path.resolve()
    device_receipt_path = device_receipt_path.resolve()
    require(len({raw_path, summary_path, device_receipt_path}) == 3, "input paths must be distinct")

    summary = load_json(summary_path)
    device = load_json(device_receipt_path)
    rows = load_jsonl(raw_path)

    require(summary.get("schema") == SUMMARY_SCHEMA, "unexpected summary schema")
    require(device.get("schema") == DEVICE_SCHEMA, "unexpected device receipt schema")
    run_id = summary.get("run_id")
    require(isinstance(run_id, str) and bool(run_id), "summary run_id missing")
    require(device.get("run_id") == run_id, "device receipt run_id mismatch")
    require(all(row.get("run_id") == run_id for row in rows), "raw row run_id mismatch")
    require(summary.get("raw_frames_file") == "raw_frames.jsonl", "summary raw filename is not frozen")
    require(summary.get("device_receipt_file") == "device_receipt.json", "summary device filename is not frozen")
    require(summary.get("raw_frames_sha256") == sha256_file(raw_path), "raw frame SHA-256 mismatch")
    require(summary.get("device_receipt_sha256") == sha256_file(device_receipt_path), "device receipt SHA-256 mismatch")
    require(summary.get("capture_completed") is True, "device capture did not complete")
    require(summary.get("capture_error") is None, "device capture recorded an error")
    require(summary.get("raw_frame_row_count") == len(rows), "summary row count mismatch")
    require(summary.get("session_update_attempt_count") == len(rows), "session update count mismatch")
    require(summary.get("frame_attempts_requested") == len(rows), "requested frame count mismatch")
    require(summary.get("depth_mode_automatic_supported") is True, "raw depth mode unavailable")
    require(boundary_is_safe(summary.get("evidence_boundary")), "summary evidence boundary is unsafe")

    device_identity = device.get("device")
    require(isinstance(device_identity, dict), "device identity missing")
    require(device_identity.get("model") == EXPECTED_MODEL, f"device model must be {EXPECTED_MODEL}")
    arcore = device.get("arcore")
    require(isinstance(arcore, dict), "ARCore device receipt missing")
    require(arcore.get("availability") == "SUPPORTED_INSTALLED", "ARCore is not supported and installed")
    require(arcore.get("sdk_dependency_version") == EXPECTED_ARCORE_SDK, "ARCore SDK version drifted")
    require(device.get("capture_package") == EXPECTED_PACKAGE, "capture package is not benchmark-isolated")
    require(device.get("session_ownership") == "EXCLUSIVE_SINGLE_SESSION", "ARCore Session ownership is not exclusive")
    require(device.get("autonomous_capture") is True, "capture was not autonomous")
    require(device.get("user_motion_instruction") is False, "capture requested user motion")
    require(boundary_is_safe(device.get("evidence_boundary")), "device evidence boundary is unsafe")

    camera_timestamps: list[int] = []
    timestamped_row_count = 0
    camera_pair_count = 0
    depth_confidence_pair_count = 0
    source_aligned_count = 0
    tracking_count = 0
    anchor_tracking_count = 0
    valid_pair_count = 0
    missing_camera_image_count = 0
    missing_raw_depth_count = 0
    missing_raw_confidence_count = 0
    anchor_ids: set[str] = set()
    anchor_created_indices: set[int] = set()
    anchor_created_timestamps: set[int] = set()
    anchor_poses: list[dict[str, Any]] = []
    anchor_reference_modes: set[str] = set()

    for expected_index, row in enumerate(rows):
        require(row.get("schema") == RAW_SCHEMA, f"row {expected_index} schema mismatch")
        require(row.get("frame_index") == expected_index, f"row {expected_index} frame_index is not contiguous")
        require(row.get("session_update_index") == expected_index, f"row {expected_index} Session.update index mismatch")
        require(boundary_is_safe(row.get("evidence_boundary")), f"row {expected_index} evidence boundary is unsafe")
        frame_timestamp = row.get("frame_timestamp_ns")
        camera_timestamp = row.get("android_camera_timestamp_ns")
        timestamped = positive_int(frame_timestamp) and positive_int(camera_timestamp)
        if timestamped:
            timestamped_row_count += 1
            camera_timestamps.append(int(camera_timestamp))

        camera_image = row.get("camera_image")
        depth_image = row.get("raw_depth_image")
        confidence_image = row.get("raw_confidence_image")
        camera_available = valid_image(camera_image)
        depth_available = valid_image(depth_image)
        confidence_available = valid_image(confidence_image)
        missing_camera_image_count += int(not camera_available)
        missing_raw_depth_count += int(not depth_available)
        missing_raw_confidence_count += int(not confidence_available)
        camera_pair = bool(
            timestamped
            and camera_available
            # ARCore's acquired camera Image is bound to Frame.getTimestamp().
            # getAndroidCameraTimestamp() is a separate Camera2 metadata-correlation clock.
            and camera_image.get("timestamp_ns") == frame_timestamp
        )
        depth_confidence_pair = bool(depth_available and confidence_available)
        source_aligned = bool(
            timestamped
            and depth_confidence_pair
            and depth_image.get("timestamp_ns") == frame_timestamp
            and confidence_image.get("timestamp_ns") == frame_timestamp
        )
        camera_pair_count += int(camera_pair)
        depth_confidence_pair_count += int(depth_confidence_pair)
        source_aligned_count += int(source_aligned)

        tracking = row.get("tracking_state") == "TRACKING"
        tracking_count += int(tracking)
        intrinsics = row.get("intrinsics")
        transforms = row.get("transforms")
        transform_valid = bool(
            isinstance(intrinsics, dict)
            and valid_intrinsics(intrinsics.get("image"))
            and valid_intrinsics(intrinsics.get("texture"))
            and isinstance(transforms, dict)
            and valid_pose(transforms.get("world_from_camera"))
            and finite_array(transforms.get("camera_view_matrix"), 16)
            and finite_array(transforms.get("camera_projection_matrix"), 16)
            and valid_pose(transforms.get("world_from_android_sensor"))
        )

        anchor = row.get("anchor")
        anchor_tracking = False
        if isinstance(anchor, dict) and anchor.get("available") is True:
            anchor_id = anchor.get("anchor_id")
            created_index = anchor.get("created_frame_index")
            created_timestamp = anchor.get("created_frame_timestamp_ns")
            require(isinstance(anchor_id, str) and bool(anchor_id), f"row {expected_index} anchor id missing")
            require(isinstance(created_index, int) and created_index >= 0, f"row {expected_index} anchor creation index invalid")
            require(positive_int(created_timestamp), f"row {expected_index} anchor creation timestamp invalid")
            require(valid_pose(anchor.get("world_from_anchor")), f"row {expected_index} anchor pose invalid")
            require(valid_pose(anchor.get("anchor_from_camera")), f"row {expected_index} anchor-camera transform invalid")
            anchor_ids.add(anchor_id)
            anchor_created_indices.add(created_index)
            anchor_created_timestamps.add(created_timestamp)
            anchor_reference_modes.add(str(anchor.get("reference_mode")))
            anchor_poses.append(anchor["world_from_anchor"])
            anchor_tracking = anchor.get("tracking_state") == "TRACKING"
        anchor_tracking_count += int(anchor_tracking)
        valid_pair_count += int(camera_pair and source_aligned and tracking and anchor_tracking and transform_valid)

    unique_camera_timestamp_count = len(set(camera_timestamps))
    duplicate_camera_timestamp_count = len(camera_timestamps) - unique_camera_timestamp_count
    tracking_fraction = tracking_count / timestamped_row_count if timestamped_row_count else 0.0
    source_aligned_fraction = source_aligned_count / timestamped_row_count if timestamped_row_count else 0.0
    anchor_tracking_fraction = anchor_tracking_count / tracking_count if tracking_count else 0.0

    max_anchor_translation_drift_m: float | None = None
    max_anchor_rotation_drift_deg: float | None = None
    if anchor_poses:
        origin_translation = pose_translation(anchor_poses[0])
        origin_rotation = pose_quaternion(anchor_poses[0])
        max_anchor_translation_drift_m = max(
            translation_distance(origin_translation, pose_translation(pose)) for pose in anchor_poses
        )
        max_anchor_rotation_drift_deg = max(
            quaternion_distance_degrees(origin_rotation, pose_quaternion(pose)) for pose in anchor_poses
        )

    checks = {
        "unique_android_camera_timestamp_count_gte_100": unique_camera_timestamp_count >= MIN_QUALIFYING_FRAMES,
        "duplicate_android_camera_timestamp_count_eq_0": duplicate_camera_timestamp_count == 0,
        "camera_image_pair_count_gte_100": camera_pair_count >= MIN_QUALIFYING_FRAMES,
        "raw_depth_confidence_pair_count_gte_100": depth_confidence_pair_count >= MIN_QUALIFYING_FRAMES,
        "source_aligned_fraction_gte_0_95": source_aligned_fraction >= MIN_FRACTION,
        "tracking_denominator_gte_100": timestamped_row_count >= MIN_QUALIFYING_FRAMES,
        "tracking_fraction_gte_0_95": tracking_fraction >= MIN_FRACTION,
        "valid_pair_count_gte_100": valid_pair_count >= MIN_QUALIFYING_FRAMES,
        "one_persistent_anchor": len(anchor_ids) == 1 and len(anchor_created_indices) == 1 and len(anchor_created_timestamps) == 1,
        "anchor_reference_mode_inter_frame_stable": anchor_reference_modes == {EXPECTED_REFERENCE_MODE},
        "anchor_tracking_fraction_gte_0_95": anchor_tracking_fraction >= MIN_FRACTION,
        "anchor_tracking_count_gte_100": anchor_tracking_count >= MIN_QUALIFYING_FRAMES,
        "anchor_translation_drift_lte_0_25m": max_anchor_translation_drift_m is not None
        and max_anchor_translation_drift_m <= MAX_ANCHOR_TRANSLATION_DRIFT_M,
        "anchor_rotation_drift_lte_20deg": max_anchor_rotation_drift_deg is not None
        and max_anchor_rotation_drift_deg <= MAX_ANCHOR_ROTATION_DRIFT_DEG,
        "all_evidence_is_benchmark_only": True,
    }
    gate_open = all(checks.values())
    return {
        "schema": AUDIT_SCHEMA,
        "run_id": run_id,
        "verdict": "PASS_BENCHMARK_ONLY" if gate_open else "FREEZE_FRAME_BOUND_METRIC_GEOMETRY",
        "gate_open": gate_open,
        "input_bindings": {
            "raw_frames_sha256": sha256_file(raw_path),
            "summary_sha256": sha256_file(summary_path),
            "device_receipt_sha256": sha256_file(device_receipt_path),
        },
        "device": {
            "model": device_identity.get("model"),
            "android_sdk_int": device_identity.get("android_sdk_int"),
            "build_fingerprint": device_identity.get("build_fingerprint"),
            "arcore_sdk_dependency_version": arcore.get("sdk_dependency_version"),
        },
        "recomputed_metrics": {
            "raw_frame_row_count": len(rows),
            "unique_android_camera_timestamp_count": unique_camera_timestamp_count,
            "duplicate_android_camera_timestamp_count": duplicate_camera_timestamp_count,
            "camera_image_pair_count": camera_pair_count,
            "raw_depth_confidence_pair_count": depth_confidence_pair_count,
            "source_aligned_count": source_aligned_count,
            "source_aligned_fraction": source_aligned_fraction,
            "tracking_denominator": timestamped_row_count,
            "tracking_count": tracking_count,
            "tracking_fraction": tracking_fraction,
            "valid_pair_count": valid_pair_count,
            "anchor_tracking_count": anchor_tracking_count,
            "anchor_tracking_fraction": anchor_tracking_fraction,
            "anchor_id_count": len(anchor_ids),
            "anchor_reference_modes": sorted(anchor_reference_modes),
            "max_anchor_translation_drift_m": max_anchor_translation_drift_m,
            "max_anchor_rotation_drift_deg": max_anchor_rotation_drift_deg,
            "missing_camera_image_count": missing_camera_image_count,
            "missing_raw_depth_count": missing_raw_depth_count,
            "missing_raw_confidence_count": missing_raw_confidence_count,
        },
        "checks": checks,
        "thresholds": {
            "minimum_qualifying_frames": MIN_QUALIFYING_FRAMES,
            "minimum_fraction": MIN_FRACTION,
            "required_pose_reference_mode": EXPECTED_REFERENCE_MODE,
            "maximum_anchor_translation_drift_m": MAX_ANCHOR_TRANSLATION_DRIFT_M,
            "maximum_anchor_rotation_drift_deg": MAX_ANCHOR_ROTATION_DRIFT_DEG,
        },
        "evidence_boundary": {
            "benchmark_only": True,
            "app_runtime_involved": False,
            "navigation_output_issued": False,
            "training_authority": False,
            "production_authorized": False,
            "human_truth": False,
        },
    }
