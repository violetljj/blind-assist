"""Extract response-blind signed-pose and flow-direction measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping

import cv2
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from scripts.research.egomotion_compensated_looming.ecological_response_discovery_r0 import (
    runner as r3,
)


PROTOCOL_ID = "RCLE_TEMPORAL_STRUCTURE_DIAGNOSTIC_R1"
CONTRACT_STATUS = "FROZEN_BEFORE_NEW_FLOW_DIRECTION_EXTRACTION"
CONTRACT_RELATIVE_PATH = Path(
    "docs/research/rcle/"
    "RCLE_TEMPORAL_STRUCTURE_DIAGNOSTIC_R1_CONTRACT_2026-07-28.json"
)
PAIR_COUNT = 601
ALLOWED_SESSIONS = frozenset({13, 14, 15, 17})
SEALED_SESSION = 16
RESIZE_SCALE = 0.5
FB_LIMIT_PX = 1.0
DIRECTION_MIN_MAGNITUDE_PX = 0.25
GATE_FB_MEDIAN_LIMIT_PX = 0.75
GATE_MIN_FEATURES = 60
GATE_MIN_FB_TRACKS = 60
GATE_MIN_FB_FRACTION = 0.5
GATE_MIN_OCCUPIED_CELLS = 5
OPENCV_THREADS = 1
OPENCV_RNG_SEED = 20260728


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path.name}")
    return value


def validate_contract(
    contract_path: Path, session: int
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    """Validate the exact frozen contract and return its session input lock."""

    expected_path = (
        Path(__file__).resolve().parents[4] / CONTRACT_RELATIVE_PATH
    ).resolve()
    if contract_path.resolve() != expected_path:
        raise ValueError("CONTRACT_PATH_MISMATCH")
    contract = _load_json_object(expected_path)
    if contract.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("CONTRACT_PROTOCOL_ID_MISMATCH")
    if contract.get("status") != CONTRACT_STATUS:
        raise ValueError("CONTRACT_NOT_FROZEN_FOR_STAGE_1")

    frozen_inputs = contract.get("frozen_inputs")
    if not isinstance(frozen_inputs, dict):
        raise ValueError("CONTRACT_FROZEN_INPUTS_MISSING")
    if frozen_inputs.get("sessions") != sorted(ALLOWED_SESSIONS):
        raise ValueError("CONTRACT_SESSION_SET_MISMATCH")
    if frozen_inputs.get("sealed_session") != SEALED_SESSION:
        raise ValueError("CONTRACT_SEALED_SESSION_MISMATCH")
    by_session = frozen_inputs.get("by_session")
    if not isinstance(by_session, dict):
        raise ValueError("CONTRACT_SESSION_INPUTS_MISSING")
    session_lock = by_session.get(str(session))
    if not isinstance(session_lock, dict):
        raise ValueError("CONTRACT_SESSION_INPUT_LOCK_MISSING")
    return contract, session_lock


def validate_input_identity(
    source_root: Path, session_lock: Mapping[str, Any]
) -> tuple[dict[str, Path], dict[str, str]]:
    """Require all three Stage-1 inputs and match their frozen SHA-256 values."""

    paths = {
        "frames.mov": source_root / "iphone/frames.mov",
        "frames.csv": source_root / "iphone/frames.csv",
        "pose.csv": source_root / "ground-truth/pose.csv",
    }
    expected_keys = {
        "frames.mov": "frames_mov_sha256",
        "frames.csv": "frames_csv_sha256",
        "pose.csv": "pose_csv_sha256",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    hashes = {name: sha256_file(path) for name, path in paths.items()}
    for name, expected_key in expected_keys.items():
        expected = session_lock.get(expected_key)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"CONTRACT_INPUT_SHA256_MISSING:{name}")
        if hashes[name] != expected:
            raise ValueError(f"INPUT_SHA256_MISMATCH:{name}")
    return paths, hashes


def configure_and_validate_runtime(
    runtime_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Set deterministic OpenCV state and enforce the frozen runtime versions."""

    cv2.setNumThreads(OPENCV_THREADS)
    cv2.setRNGSeed(OPENCV_RNG_SEED)
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "opencv_threads": int(cv2.getNumThreads()),
        "opencv_rng_seed": OPENCV_RNG_SEED,
    }
    for key in ("python", "numpy", "opencv"):
        if runtime_lock.get(key) != versions[key]:
            raise RuntimeError(f"RUNTIME_VERSION_MISMATCH:{key}")
    if runtime_lock.get("opencv_threads") != OPENCV_THREADS:
        raise RuntimeError("CONTRACT_OPENCV_THREADS_MISMATCH")
    if runtime_lock.get("opencv_rng_seed") != OPENCV_RNG_SEED:
        raise RuntimeError("CONTRACT_OPENCV_RNG_SEED_MISMATCH")
    if versions["opencv_threads"] != OPENCV_THREADS:
        raise RuntimeError("OPENCV_THREADS_NOT_LOCKED")
    return versions


def valid_laplacian_variance(gray: np.ndarray, valid: np.ndarray) -> float:
    pixels = cv2.Laplacian(gray, cv2.CV_64F)[valid > 0]
    if pixels.size == 0:
        return float("nan")
    return float(np.var(pixels))


def _empty_flow_metrics(
    detected: int, valid_pixels: int
) -> dict[str, Any]:
    return {
        "detected_feature_count": detected,
        "detected_features_per_valid_megapixel": (
            detected / (valid_pixels / 1_000_000.0)
            if valid_pixels
            else None
        ),
        "forward_track_fraction": 0.0,
        "forward_backward_consistent_count": 0,
        "forward_backward_consistent_fraction": 0.0,
        "median_forward_backward_error_px": None,
        "occupied_grid_cells": 0,
        "direction_track_count": 0,
        "median_flow_dx_px": None,
        "median_flow_dy_px": None,
        "median_flow_magnitude_px": None,
        "spatial_direction_resultant": None,
        "radial_direction_track_count": 0,
        "median_radial_flow_px": None,
        "radial_direction_consistency": None,
    }


def flow_direction_metrics(
    previous: np.ndarray,
    current: np.ndarray,
    previous_valid: np.ndarray,
) -> dict[str, Any]:
    """Compute the unchanged R0 quality scalars plus signed flow direction."""

    points = cv2.goodFeaturesToTrack(
        previous,
        maxCorners=400,
        qualityLevel=0.01,
        minDistance=8.0,
        mask=previous_valid,
        blockSize=7,
    )
    detected = 0 if points is None else int(len(points))
    valid_pixels = int(np.count_nonzero(previous_valid))
    metrics = _empty_flow_metrics(detected, valid_pixels)
    if points is None or detected == 0:
        return metrics

    lk = {
        "winSize": (21, 21),
        "maxLevel": 3,
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    }
    forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        previous, current, points, None, **lk
    )
    if forward is None or forward_status is None:
        return metrics
    forward_ok = forward_status.reshape(-1).astype(bool)
    metrics["forward_track_fraction"] = float(np.mean(forward_ok))

    backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current, previous, forward, None, **lk
    )
    if backward is None or backward_status is None:
        return metrics
    backward_ok = backward_status.reshape(-1).astype(bool)
    initial_xy = points.reshape(-1, 2)
    forward_xy = forward.reshape(-1, 2)
    backward_xy = backward.reshape(-1, 2)
    errors = np.linalg.norm(backward_xy - initial_xy, axis=1)
    jointly_tracked = forward_ok & backward_ok & np.isfinite(errors)
    consistent = jointly_tracked & (errors <= FB_LIMIT_PX)
    consistent_count = int(np.count_nonzero(consistent))
    metrics["forward_backward_consistent_count"] = consistent_count
    metrics["forward_backward_consistent_fraction"] = (
        consistent_count / detected
    )
    if np.any(jointly_tracked):
        metrics["median_forward_backward_error_px"] = float(
            np.median(errors[jointly_tracked])
        )

    if consistent_count:
        xy = initial_xy[consistent]
        height, width = previous.shape
        columns = np.minimum((xy[:, 0] * 3 / width).astype(int), 2)
        rows = np.minimum((xy[:, 1] * 3 / height).astype(int), 2)
        metrics["occupied_grid_cells"] = int(
            len(set((int(y), int(x)) for x, y in zip(columns, rows)))
        )

        vectors = forward_xy[consistent] - initial_xy[consistent]
        magnitudes = np.linalg.norm(vectors, axis=1)
        direction_valid = (
            np.isfinite(vectors).all(axis=1)
            & np.isfinite(magnitudes)
            & (magnitudes >= DIRECTION_MIN_MAGNITUDE_PX)
        )
        vectors = vectors[direction_valid].astype(np.float64)
        magnitudes = magnitudes[direction_valid].astype(np.float64)
        metrics["direction_track_count"] = int(len(vectors))
        if len(vectors):
            unit_vectors = vectors / magnitudes[:, None]
            metrics["median_flow_dx_px"] = float(np.median(vectors[:, 0]))
            metrics["median_flow_dy_px"] = float(np.median(vectors[:, 1]))
            metrics["median_flow_magnitude_px"] = float(
                np.median(magnitudes)
            )
            metrics["spatial_direction_resultant"] = float(
                np.linalg.norm(np.mean(unit_vectors, axis=0))
            )

            direction_xy = xy[direction_valid].astype(np.float64)
            image_center = np.asarray(
                ((width - 1.0) / 2.0, (height - 1.0) / 2.0),
                dtype=np.float64,
            )
            radial_offsets = direction_xy - image_center
            radial_norms = np.linalg.norm(radial_offsets, axis=1)
            noncentral = radial_norms > np.finfo(np.float64).eps
            radial_components = np.full(
                len(vectors), np.nan, dtype=np.float64
            )
            radial_components[noncentral] = np.sum(
                vectors[noncentral]
                * (
                    radial_offsets[noncentral]
                    / radial_norms[noncentral, None]
                ),
                axis=1,
            )
            radial_valid = (
                np.isfinite(radial_components)
                & (
                    np.abs(radial_components)
                    >= DIRECTION_MIN_MAGNITUDE_PX
                )
            )
            radial_components = radial_components[radial_valid]
            metrics["radial_direction_track_count"] = int(
                len(radial_components)
            )
            if len(radial_components):
                metrics["median_radial_flow_px"] = float(
                    np.median(radial_components)
                )
                metrics["radial_direction_consistency"] = float(
                    abs(np.mean(np.sign(radial_components)))
                )
    return metrics


def gate_reasons(metrics: Mapping[str, Any]) -> list[str]:
    """Apply the unchanged R0 flow-quality gate."""

    reasons: list[str] = []
    if metrics["detected_feature_count"] < GATE_MIN_FEATURES:
        reasons.append("FEATURES_LT_60")
    if metrics["forward_backward_consistent_count"] < GATE_MIN_FB_TRACKS:
        reasons.append("FB_TRACKS_LT_60")
    if (
        metrics["forward_backward_consistent_fraction"]
        < GATE_MIN_FB_FRACTION
    ):
        reasons.append("FB_FRACTION_LT_0P50")
    median_error = metrics["median_forward_backward_error_px"]
    if (
        median_error is None
        or median_error > GATE_FB_MEDIAN_LIMIT_PX
    ):
        reasons.append("FB_MEDIAN_GT_0P75")
    if metrics["occupied_grid_cells"] < GATE_MIN_OCCUPIED_CELLS:
        reasons.append("OCCUPIED_CELLS_LT_5")
    return reasons


def direction_evaluable(metrics: Mapping[str, Any]) -> bool:
    resultant = metrics["spatial_direction_resultant"]
    radial_consistency = metrics["radial_direction_consistency"]
    median_error = metrics["median_forward_backward_error_px"]
    direction_structure = bool(
        (resultant is not None and resultant >= 0.5)
        or (
            metrics["radial_direction_track_count"] >= GATE_MIN_FB_TRACKS
            and radial_consistency is not None
            and radial_consistency >= 0.5
        )
    )
    return bool(
        metrics["direction_track_count"] >= GATE_MIN_FB_TRACKS
        and metrics["forward_backward_consistent_fraction"]
        >= GATE_MIN_FB_FRACTION
        and median_error is not None
        and median_error <= GATE_FB_MEDIAN_LIMIT_PX
        and direction_structure
    )


def signed_camera_velocities(
    previous_pose: tuple[np.ndarray, np.ndarray],
    current_pose: tuple[np.ndarray, np.ndarray],
    dt_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return contract-defined camera-basis angular and translation velocities."""

    if not math.isfinite(dt_seconds) or dt_seconds <= 0.0:
        raise ValueError("PAIR_DT_NONPOSITIVE")
    previous_position, previous_quaternion = previous_pose
    current_position, current_quaternion = current_pose
    previous_rotation = r3.quaternion_rotation_wxyz(previous_quaternion)
    current_rotation = r3.quaternion_rotation_wxyz(current_quaternion)
    current_from_previous_imu = current_rotation.T @ previous_rotation
    basis = r3.T_CAM_IMU_ROTATION
    current_from_previous_camera = (
        basis @ current_from_previous_imu @ basis.T
    )
    rotation_vector, _ = cv2.Rodrigues(current_from_previous_camera)
    angular_velocity = (
        np.degrees(rotation_vector.reshape(3).astype(np.float64))
        / dt_seconds
    )

    world_position_delta = (
        np.asarray(current_position, dtype=np.float64)
        - np.asarray(previous_position, dtype=np.float64)
    )
    translation_velocity = (
        basis @ current_rotation.T @ world_position_delta
    ) / dt_seconds
    return angular_velocity, translation_velocity


def _firewall_flags() -> dict[str, bool]:
    return {
        "stage_1_response_blind": True,
        "response_accessed_during_extraction": False,
        "r3_pair_ledger_accessed": False,
        "r0_proxy_ledger_accessed": False,
        "risk_or_obstacle_label_accessed": False,
        "manual_gait_phase_accessed": False,
        "sealed_session_accessed": False,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def extract(
    session: int,
    source_root: Path,
    output_dir: Path,
    contract_path: Path,
) -> dict[str, Any]:
    if session == SEALED_SESSION:
        raise PermissionError("SEALED_UNSEEN_SESSION_ACCESS_FORBIDDEN")
    if session not in ALLOWED_SESSIONS:
        raise ValueError("SESSION_NOT_IN_FROZEN_SET")

    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    contract_path = contract_path.resolve()
    if output_dir.exists():
        raise FileExistsError("OUTPUT_DIRECTORY_EXISTS")

    contract, session_lock = validate_contract(contract_path, session)
    input_paths, input_hashes = validate_input_identity(
        source_root, session_lock
    )
    runtime_lock = contract.get("runtime_lock")
    if not isinstance(runtime_lock, dict):
        raise ValueError("CONTRACT_RUNTIME_LOCK_MISSING")
    runtime_versions = configure_and_validate_runtime(runtime_lock)

    frame_rows = r3.load_csv(input_paths["frames.csv"], 2)
    poses = r3.load_pose_series(input_paths["pose.csv"])
    timestamps = frame_rows[: PAIR_COUNT + 1, 0]
    if len(timestamps) != PAIR_COUNT + 1:
        raise ValueError("FROZEN_FRAME_COUNT_UNAVAILABLE")

    output_dir.mkdir(parents=True, exist_ok=False)
    capture = cv2.VideoCapture(os.fspath(input_paths["frames.mov"]))
    if not capture.isOpened():
        raise ValueError("VIDEO_OPEN_FAILED")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (width, height) != (720, 1280):
            raise ValueError(f"VIDEO_SHAPE:{width}x{height}")
        undistort_maps = r3.build_undistort_maps(width, height)
        ok, previous_bgr = capture.read()
        if not ok:
            raise ValueError("VIDEO_FIRST_FRAME_DECODE_FAILED")
        previous, previous_valid = r3.preprocess_frame_with_mask(
            previous_bgr, RESIZE_SCALE, undistort_maps
        )
        previous_sharpness = valid_laplacian_variance(
            previous, previous_valid
        )

        rows: list[dict[str, Any]] = []
        for pair_index in range(PAIR_COUNT):
            ok, current_bgr = capture.read()
            if not ok:
                raise ValueError(
                    f"VIDEO_FRAME_DECODE_FAILED:{pair_index + 1}"
                )
            current, current_valid = r3.preprocess_frame_with_mask(
                current_bgr, RESIZE_SCALE, undistort_maps
            )
            current_sharpness = valid_laplacian_variance(
                current, current_valid
            )
            previous_timestamp = float(timestamps[pair_index])
            current_timestamp = float(timestamps[pair_index + 1])
            dt = current_timestamp - previous_timestamp
            if not 0.0 < dt <= 0.1:
                raise ValueError(f"PAIR_DT:{pair_index}:{dt}")
            previous_pose = r3.interpolate_pose(poses, previous_timestamp)
            current_pose = r3.interpolate_pose(poses, current_timestamp)
            angular_velocity, translation_velocity = (
                signed_camera_velocities(
                    previous_pose, current_pose, dt
                )
            )
            metrics = flow_direction_metrics(
                previous, current, previous_valid
            )
            reasons = gate_reasons(metrics)
            rows.append(
                {
                    "session": session,
                    "pair_index": pair_index,
                    "previous_timestamp_s": previous_timestamp,
                    "current_timestamp_s": current_timestamp,
                    "dt_s": dt,
                    "sharpness_laplacian_variance": float(
                        np.mean(
                            [previous_sharpness, current_sharpness]
                        )
                    ),
                    "camera_angular_velocity_x_deg_per_s": float(
                        angular_velocity[0]
                    ),
                    "camera_angular_velocity_y_deg_per_s": float(
                        angular_velocity[1]
                    ),
                    "camera_angular_velocity_z_deg_per_s": float(
                        angular_velocity[2]
                    ),
                    "camera_translation_velocity_x_m_per_s": float(
                        translation_velocity[0]
                    ),
                    "camera_translation_velocity_y_m_per_s": float(
                        translation_velocity[1]
                    ),
                    "camera_translation_velocity_z_m_per_s": float(
                        translation_velocity[2]
                    ),
                    **metrics,
                    "flow_quality_gate_accept": not reasons,
                    "flow_quality_gate_reasons": reasons,
                    "direction_evaluable": direction_evaluable(metrics),
                    **_firewall_flags(),
                }
            )
            previous = current
            previous_valid = current_valid
            previous_sharpness = current_sharpness
    finally:
        capture.release()

    ledger_path = output_dir / "direction_ledger.jsonl"
    _write_jsonl(ledger_path, rows)
    flags = _firewall_flags()
    summary = {
        "schema": "rcle.temporal_structure_diagnostic.direction_summary.v1",
        "protocol_id": PROTOCOL_ID,
        "session": session,
        "pair_count": len(rows),
        "contract_path": CONTRACT_RELATIVE_PATH.as_posix(),
        "contract_sha256": sha256_file(contract_path),
        "input_sha256": input_hashes,
        "direction_ledger_sha256": sha256_file(ledger_path),
        "runtime_versions": runtime_versions,
        "direction_evaluable_fraction": float(
            np.mean([row["direction_evaluable"] for row in rows])
        ),
        "firewall_flags": flags,
        **flags,
    }
    _write_json(output_dir / "direction_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=int, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    result = extract(
        args.session,
        args.source_root,
        args.output_dir,
        args.contract,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
