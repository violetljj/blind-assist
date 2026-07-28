"""Run a bounded RCLE Discovery pass on one continuous ADVIO video."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ..rgb_algorithm_development_canary_cid_sims_r0 import producer as rgb_core
from ..rcle_low_reference_false_trigger_r1.temporal_confirmation import (
    REQUIRED_CONSECUTIVE_PAIRS,
    THRESHOLD,
)


PROTOCOL_ID = "RCLE_ECOLOGICAL_RESPONSE_DISCOVERY_R0"
SOURCE_ID = "ADVIO_OFFICE03_SEQUENCE15_IPHONE"
CALIBRATION_URL = (
    "https://github.com/AaltoVision/ADVIO/blob/master/calibration/iphone-03.yaml"
)
CALIBRATION_PINNED_URL = (
    "https://raw.githubusercontent.com/AaltoVision/ADVIO/"
    "4db6093d4bc632a8c10ae00f99a98cce0699bd0a/calibration/iphone-03.yaml"
)
CALIBRATION_SHA256 = (
    "725aa78baf117ef150c2c43a1161d51994f812e28b2bbfefb5c0224809f55cf2"
)
INTRINSIC = np.asarray(
    ((1082.4, 0.0, 364.6778), (0.0, 1084.4, 643.3080), (0.0, 0.0, 1.0)),
    dtype=np.float64,
)
DISTORTION = np.asarray(
    (0.0366, 0.0803, 0.000783, -0.000215), dtype=np.float64
)
T_CAM_IMU_ROTATION = np.asarray(
    (
        (
            0.9999763379093255,
            -0.004079205042965442,
            -0.005539287650170447,
        ),
        (
            -0.004066386342107199,
            -0.9999890330121858,
            0.0023234365646622014,
        ),
        (
            -0.00554870467502187,
            -0.0023008567036498766,
            -0.9999819588046867,
        ),
    ),
    dtype=np.float64,
)
SEGMENT_SECONDS = 5.0


@dataclass(frozen=True)
class PoseSeries:
    timestamps: np.ndarray
    positions: np.ndarray
    quaternions_xyzw: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_exclusive(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "wb", closefd=True) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> str:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )
    return write_exclusive(path, payload)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    payload = b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        + b"\n"
        for row in rows
    )
    return write_exclusive(path, payload)


def load_csv(path: Path, columns: int) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != columns:
        raise ValueError(f"CSV_SHAPE:{path.name}:{values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"CSV_NONFINITE:{path.name}")
    if np.any(np.diff(values[:, 0]) <= 0):
        raise ValueError(f"CSV_TIMESTAMP_NOT_STRICT:{path.name}")
    return values


def load_pose_series(path: Path) -> PoseSeries:
    values = load_csv(path, 8)
    quaternions = values[:, 4:8]
    norms = np.linalg.norm(quaternions, axis=1)
    if np.max(np.abs(norms - 1.0)) > 1e-5:
        raise ValueError("POSE_QUATERNION_NORM")
    return PoseSeries(
        timestamps=values[:, 0],
        positions=values[:, 1:4],
        quaternions_xyzw=quaternions / norms[:, None],
    )


def quaternion_rotation_xyzw(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    return np.asarray(
        (
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ),
            (
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ),
            (
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ),
        ),
        dtype=np.float64,
    )


def quaternion_rotation_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return np.asarray(
        (
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ),
            (
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ),
            (
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ),
        ),
        dtype=np.float64,
    )


def quaternion_rotation(
    quaternion: np.ndarray, component_order: str
) -> np.ndarray:
    if component_order == "xyzw":
        return quaternion_rotation_xyzw(quaternion)
    if component_order == "wxyz":
        return quaternion_rotation_wxyz(quaternion)
    raise ValueError(f"UNSUPPORTED_QUATERNION_COMPONENT_ORDER:{component_order}")


def slerp_xyzw(left: np.ndarray, right: np.ndarray, fraction: float) -> np.ndarray:
    q0 = np.asarray(left, dtype=np.float64)
    q1 = np.asarray(right, dtype=np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        value = q0 + fraction * (q1 - q0)
        return value / np.linalg.norm(value)
    angle = math.acos(dot)
    sine = math.sin(angle)
    return (
        math.sin((1.0 - fraction) * angle) / sine * q0
        + math.sin(fraction * angle) / sine * q1
    )


def interpolate_pose(
    series: PoseSeries, timestamp: float
) -> tuple[np.ndarray, np.ndarray]:
    right = int(np.searchsorted(series.timestamps, timestamp, side="left"))
    if right == 0:
        if abs(float(series.timestamps[0]) - timestamp) > 1e-6:
            raise ValueError("POSE_NOT_BRACKETED")
        return series.positions[0], series.quaternions_xyzw[0]
    if right >= len(series.timestamps):
        raise ValueError("POSE_NOT_BRACKETED")
    left = right - 1
    t0 = float(series.timestamps[left])
    t1 = float(series.timestamps[right])
    if t1 - t0 > 0.03:
        raise ValueError("POSE_BRACKET_TOO_WIDE")
    fraction = (timestamp - t0) / (t1 - t0)
    position = series.positions[left] + fraction * (
        series.positions[right] - series.positions[left]
    )
    quaternion = slerp_xyzw(
        series.quaternions_xyzw[left],
        series.quaternions_xyzw[right],
        fraction,
    )
    return position, quaternion


def pair_geometry(
    previous_pose: tuple[np.ndarray, np.ndarray],
    current_pose: tuple[np.ndarray, np.ndarray],
    dt_seconds: float,
    *,
    quaternion_component_order: str = "xyzw",
    pose_to_camera_rotation: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    previous_position, previous_quaternion = previous_pose
    current_position, current_quaternion = current_pose
    previous_rotation = quaternion_rotation(
        previous_quaternion, quaternion_component_order
    )
    current_rotation = quaternion_rotation(
        current_quaternion, quaternion_component_order
    )
    current_from_previous = current_rotation.T @ previous_rotation
    if pose_to_camera_rotation is not None:
        basis = np.asarray(pose_to_camera_rotation, dtype=np.float64)
        if basis.shape != (3, 3):
            raise ValueError("POSE_TO_CAMERA_ROTATION_SHAPE")
        current_from_previous = (
            basis @ current_from_previous @ basis.T
        )
    cosine = min(
        1.0,
        max(-1.0, (float(np.trace(current_from_previous)) - 1.0) / 2.0),
    )
    angular_speed = math.degrees(math.acos(cosine)) / dt_seconds
    translation_speed = (
        float(np.linalg.norm(current_position - previous_position)) / dt_seconds
    )
    homography = INTRINSIC @ current_from_previous @ np.linalg.inv(INTRINSIC)
    return homography, angular_speed, translation_speed


def build_undistort_maps(
    width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    return cv2.initUndistortRectifyMap(
        INTRINSIC,
        DISTORTION,
        None,
        INTRINSIC,
        (width, height),
        cv2.CV_32FC1,
    )


def preprocess_frame(
    bgr: np.ndarray,
    resize_scale: float,
    undistort_maps: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    return preprocess_frame_with_mask(
        bgr, resize_scale, undistort_maps
    )[0]


def preprocess_frame_with_mask(
    bgr: np.ndarray,
    resize_scale: float,
    undistort_maps: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray]:
    prepared = bgr
    valid = np.full(bgr.shape[:2], 255, dtype=np.uint8)
    if undistort_maps is not None:
        prepared = cv2.remap(
            prepared,
            undistort_maps[0],
            undistort_maps[1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        map_x, map_y = undistort_maps
        height, width = bgr.shape[:2]
        valid = np.where(
            (map_x >= 0.0)
            & (map_x < width - 1.0)
            & (map_y >= 0.0)
            & (map_y < height - 1.0),
            255,
            0,
        ).astype(np.uint8)
    if resize_scale != 1.0:
        prepared = cv2.resize(
            prepared,
            None,
            fx=resize_scale,
            fy=resize_scale,
            interpolation=cv2.INTER_AREA,
        )
        valid = cv2.resize(
            valid,
            None,
            fx=resize_scale,
            fy=resize_scale,
            interpolation=cv2.INTER_NEAREST,
        )
    return (
        cv2.cvtColor(prepared, cv2.COLOR_BGR2GRAY),
        np.ascontiguousarray(valid),
    )


def global_image_scale_proxy(
    previous: np.ndarray, current: np.ndarray, dt_seconds: float
) -> tuple[float | None, int, str | None]:
    points = cv2.goodFeaturesToTrack(
        previous,
        maxCorners=400,
        qualityLevel=0.01,
        minDistance=8.0,
        blockSize=7,
    )
    if points is None or len(points) < 12:
        return None, 0, "IMAGE_SCALE_FEATURES_BELOW_12"
    forward, status, _ = cv2.calcOpticalFlowPyrLK(
        previous,
        current,
        points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    )
    if forward is None or status is None:
        return None, 0, "IMAGE_SCALE_FORWARD_LK_FAILED"
    backward, back_status, _ = cv2.calcOpticalFlowPyrLK(
        current,
        previous,
        forward,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    )
    if backward is None or back_status is None:
        return None, 0, "IMAGE_SCALE_BACKWARD_LK_FAILED"
    mask = (
        status.reshape(-1).astype(bool)
        & back_status.reshape(-1).astype(bool)
        & (np.linalg.norm(backward - points, axis=2).reshape(-1) <= 1.0)
    )
    source = points.reshape(-1, 2)[mask]
    target = forward.reshape(-1, 2)[mask]
    if len(source) < 12:
        return None, int(len(source)), "IMAGE_SCALE_TRACKS_BELOW_12"
    affine, inliers = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=1.5,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    if affine is None or inliers is None or int(inliers.sum()) < 12:
        return (
            None,
            int(inliers.sum()) if inliers is not None else 0,
            "IMAGE_SCALE_FIT_FAILED",
        )
    determinant = float(np.linalg.det(affine[:, :2]))
    if not math.isfinite(determinant) or determinant <= 0:
        return (
            None,
            int(inliers.sum()),
            "IMAGE_SCALE_NON_POSITIVE_DETERMINANT",
        )
    scale = math.sqrt(determinant)
    return math.log(scale) / dt_seconds, int(inliers.sum()), None


def update_confirmation(row: dict[str, Any], streak: int, prefix: str) -> int:
    value = row.get(f"{prefix}_expansion_median_per_s")
    above = bool(
        row.get("evaluable") is True
        and value is not None
        and float(value) > THRESHOLD
    )
    streak = streak + 1 if above else 0
    row[f"{prefix}_above_threshold"] = above
    row[f"{prefix}_consecutive_above_threshold_pair_count"] = streak
    row[f"{prefix}_three_pair_trigger"] = (
        streak >= REQUIRED_CONSECUTIVE_PAIRS
    )
    return streak


def finite_values(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    values = [
        float(row[field])
        for row in rows
        if row.get(field) is not None and math.isfinite(float(row[field]))
    ]
    return np.asarray(values, dtype=np.float64)


def rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(values):
        end = index + 1
        while (
            end < len(values)
            and values[order[end]] == values[order[index]]
        ):
            end += 1
        ranks[order[index:end]] = (index + end - 1) / 2.0
        index = end
    return ranks


def correlation(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    selected = [
        (float(row["angular_speed_deg_per_s"]), float(row[field]))
        for row in rows
        if row.get(field) is not None
        and math.isfinite(float(row[field]))
        and math.isfinite(float(row["angular_speed_deg_per_s"]))
    ]
    if len(selected) < 3:
        return {"pair_count": len(selected), "pearson": None, "spearman": None}
    values = np.asarray(selected, dtype=np.float64)
    if np.std(values[:, 0]) == 0 or np.std(values[:, 1]) == 0:
        return {"pair_count": len(selected), "pearson": None, "spearman": None}
    return {
        "pair_count": len(selected),
        "pearson": float(np.corrcoef(values[:, 0], values[:, 1])[0, 1]),
        "spearman": float(
            np.corrcoef(
                rank_average(values[:, 0]),
                rank_average(values[:, 1]),
            )[0, 1]
        ),
    }


def longest_true_run(
    rows: list[dict[str, Any]], field: str
) -> dict[str, Any]:
    best_count = current_count = 0
    best_duration = 0.0
    start: float | None = None
    for row in rows:
        if row.get(field) is True:
            if current_count == 0:
                start = float(row["previous_timestamp_s"])
            current_count += 1
            duration = float(row["current_timestamp_s"]) - float(start)
            if current_count > best_count or (
                current_count == best_count and duration > best_duration
            ):
                best_count = current_count
                best_duration = duration
        else:
            current_count = 0
            start = None
    return {"pair_count": best_count, "duration_s": best_duration}


def method_summary(
    rows: list[dict[str, Any]], value_field: str, trigger_field: str
) -> dict[str, Any]:
    values = finite_values(rows, value_field)
    denominator = len(rows)
    return {
        "evaluable_value_count": int(len(values)),
        "median_per_s": float(np.median(values)) if len(values) else None,
        "median_abs_per_s": (
            float(np.median(np.abs(values))) if len(values) else None
        ),
        "p10_per_s": float(np.quantile(values, 0.1)) if len(values) else None,
        "p90_per_s": float(np.quantile(values, 0.9)) if len(values) else None,
        "positive_pair_fraction_fixed_denominator": (
            sum(
                row.get(value_field) is not None
                and float(row[value_field]) > THRESHOLD
                for row in rows
            )
            / denominator
            if denominator
            else None
        ),
        "three_pair_trigger_fraction_fixed_denominator": (
            sum(row.get(trigger_field) is True for row in rows) / denominator
            if denominator
            else None
        ),
        "three_pair_trigger_onset_count": sum(
            row.get(trigger_field) is True
            and (
                index == 0
                or rows[index - 1].get(trigger_field) is not True
            )
            for index, row in enumerate(rows)
        ),
        "longest_three_pair_trigger_run": longest_true_run(
            rows, trigger_field
        ),
    }


def segment_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    origin = float(rows[0]["previous_timestamp_s"])
    groups: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        index = int(
            (float(row["previous_timestamp_s"]) - origin) // SEGMENT_SECONDS
        )
        groups.setdefault(index, []).append(row)
    summaries = []
    for index, selected in sorted(groups.items()):
        summaries.append(
            {
                "segment_index": index,
                "start_timestamp_s": float(
                    selected[0]["previous_timestamp_s"]
                ),
                "end_timestamp_s": float(
                    selected[-1]["current_timestamp_s"]
                ),
                "candidate_pair_count": len(selected),
                "evaluable_pair_count": sum(
                    row.get("evaluable") is True for row in selected
                ),
                "raw": method_summary(
                    selected,
                    "raw_expansion_median_per_s",
                    "raw_three_pair_trigger",
                ),
                "compensated": method_summary(
                    selected,
                    "compensated_expansion_median_per_s",
                    "compensated_three_pair_trigger",
                ),
                "image_scale_proxy": method_summary(
                    selected,
                    "image_scale_expansion_per_s",
                    "image_scale_three_pair_trigger",
                ),
                "median_angular_speed_deg_per_s": float(
                    np.median(
                        [
                            row["angular_speed_deg_per_s"]
                            for row in selected
                        ]
                    )
                ),
                "median_translation_speed_m_per_s": float(
                    np.median(
                        [
                            row["translation_speed_m_per_s"]
                            for row in selected
                        ]
                    )
                ),
            }
        )
    return summaries


def render_curves(path: Path, rows: list[dict[str, Any]]) -> None:
    width, height = 1400, 760
    x0, x1 = 80, width - 30
    upper_top, upper_bottom = 40, 500
    lower_top, lower_bottom = 570, 710
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    if not rows:
        image.save(path)
        return
    times = np.asarray(
        [float(row["current_timestamp_s"]) for row in rows],
        dtype=np.float64,
    )
    times = times - times[0]
    fields = (
        ("raw_expansion_median_per_s", (70, 90, 180), "raw"),
        (
            "compensated_expansion_median_per_s",
            (210, 70, 60),
            "compensated",
        ),
        (
            "image_scale_expansion_per_s",
            (45, 150, 80),
            "image-scale proxy",
        ),
    )
    arrays = [
        finite_values(rows, field)
        for field, _, _ in fields
        if len(finite_values(rows, field))
    ]
    all_values = np.concatenate(arrays) if arrays else np.asarray([0.0])
    bound = max(0.05, float(np.quantile(np.abs(all_values), 0.98)))
    draw.rectangle(
        (x0, upper_top, x1, upper_bottom), outline=(80, 80, 80)
    )

    def x_of(value: float) -> int:
        return int(
            x0 + (x1 - x0) * value / max(float(times[-1]), 1e-9)
        )

    def y_of(value: float) -> int:
        clipped = min(bound, max(-bound, value))
        return int(
            upper_top
            + (upper_bottom - upper_top)
            * (bound - clipped)
            / (2.0 * bound)
        )

    draw.line(
        (x0, y_of(0.0), x1, y_of(0.0)),
        fill=(100, 100, 100),
        width=1,
    )
    draw.line(
        (x0, y_of(THRESHOLD), x1, y_of(THRESHOLD)),
        fill=(160, 120, 30),
        width=1,
    )
    draw.text((8, upper_top), f"+{bound:.3f}/s", fill=(40, 40, 40))
    draw.text(
        (8, upper_bottom - 15), f"-{bound:.3f}/s", fill=(40, 40, 40)
    )
    draw.text(
        (x0, 12),
        "RCLE ecological Discovery response curves",
        fill=(20, 20, 20),
    )
    for field_index, (field, color, label) in enumerate(fields):
        points: list[tuple[int, int]] = []
        for timestamp, row in zip(times, rows):
            value = row.get(field)
            if value is None or not math.isfinite(float(value)):
                if len(points) >= 2:
                    draw.line(points, fill=color, width=2)
                points = []
                continue
            points.append((x_of(float(timestamp)), y_of(float(value))))
        if len(points) >= 2:
            draw.line(points, fill=color, width=2)
        draw.text(
            (x0 + field_index * 230, upper_bottom + 10),
            label,
            fill=color,
        )

    angular = np.asarray(
        [float(row["angular_speed_deg_per_s"]) for row in rows]
    )
    angular_bound = max(1.0, float(np.quantile(angular, 0.98)))
    draw.rectangle(
        (x0, lower_top, x1, lower_bottom), outline=(80, 80, 80)
    )
    angular_points = [
        (
            x_of(float(timestamp)),
            int(
                lower_bottom
                - (lower_bottom - lower_top)
                * min(angular_bound, value)
                / angular_bound
            ),
        )
        for timestamp, value in zip(times, angular)
    ]
    if len(angular_points) >= 2:
        draw.line(angular_points, fill=(120, 70, 155), width=2)
    draw.text(
        (8, lower_top), f"{angular_bound:.1f} deg/s", fill=(60, 40, 80)
    )
    draw.text(
        (x0, lower_bottom + 10),
        "source-pose angular speed",
        fill=(120, 70, 155),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def load_protocol(repo_root: Path) -> dict[str, Any]:
    path = (
        repo_root
        / "scripts/research/egomotion_compensated_looming/configs/"
        "phase_a_synthetic_signal_audit_r0.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def run(
    source_root: Path,
    output_dir: Path,
    *,
    max_pairs: int | None = None,
    start_time_s: float | None = None,
    start_frame: int | None = None,
    duration_s: float | None = None,
    progress_every: int = 100,
    resize_scale: float = 1.0,
    quaternion_component_order: str = "xyzw",
    distortion_correction: bool = False,
    pose_to_camera_rotation: np.ndarray | None = None,
    evidence_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError("OUTPUT_DIRECTORY_EXISTS")
    if not (0.0 < resize_scale <= 1.0):
        raise ValueError("RESIZE_SCALE_MUST_BE_IN_OPEN_CLOSED_UNIT_INTERVAL")
    if quaternion_component_order not in {"xyzw", "wxyz"}:
        raise ValueError("UNSUPPORTED_QUATERNION_COMPONENT_ORDER")
    if start_time_s is not None and start_frame is not None:
        raise ValueError("START_TIME_AND_START_FRAME_ARE_MUTUALLY_EXCLUSIVE")
    if start_frame is not None and start_frame < 0:
        raise ValueError("START_FRAME_MUST_BE_NONNEGATIVE")
    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    video_path = source_root / "iphone/frames.mov"
    frame_csv_path = source_root / "iphone/frames.csv"
    pose_path = source_root / "ground-truth/pose.csv"
    for path in (video_path, frame_csv_path, pose_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    frame_rows = load_csv(frame_csv_path, 2)
    if not np.array_equal(
        frame_rows[:, 1], np.arange(1, len(frame_rows) + 1)
    ):
        raise ValueError("FRAME_ORDINAL_NOT_CONTIGUOUS_ONE_BASED")
    poses = load_pose_series(pose_path)
    capture = cv2.VideoCapture(os.fspath(video_path))
    if not capture.isOpened():
        raise ValueError("VIDEO_OPEN_FAILED")
    video_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_frame_count != len(frame_rows):
        raise ValueError(
            f"VIDEO_FRAME_COUNT:{video_frame_count}:{len(frame_rows)}"
        )
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (width, height) != (720, 1280):
        raise ValueError(f"VIDEO_SHAPE:{width}x{height}")
    timestamps = frame_rows[:, 0]
    undistort_maps = (
        build_undistort_maps(width, height)
        if distortion_correction
        else None
    )
    first_index = (
        start_frame
        if start_frame is not None
        else (
            int(np.searchsorted(timestamps, start_time_s, side="left"))
            if start_time_s is not None
            else 0
        )
    )
    if first_index >= len(timestamps) - 1:
        raise ValueError("START_TIME_OUTSIDE_VIDEO")
    last_index_exclusive = len(timestamps)
    if duration_s is not None:
        start_value = float(timestamps[first_index])
        last_index_exclusive = int(
            np.searchsorted(
                timestamps, start_value + duration_s, side="right"
            )
        )
    if max_pairs is not None:
        last_index_exclusive = min(
            last_index_exclusive, first_index + max_pairs + 1
        )
    if last_index_exclusive - first_index < 2:
        raise ValueError("PAIR_DENOMINATOR_ZERO")
    candidate_pairs = last_index_exclusive - first_index - 1
    capture.set(cv2.CAP_PROP_POS_FRAMES, first_index)
    ok, previous_bgr = capture.read()
    if not ok:
        raise ValueError("VIDEO_FIRST_FRAME_DECODE_FAILED")
    previous, previous_valid = preprocess_frame_with_mask(
        previous_bgr, resize_scale, undistort_maps
    )
    repo_root = Path(__file__).resolve().parents[4]
    protocol = load_protocol(repo_root)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(20260728)
    state = rgb_core.PairState()
    raw_streak = compensated_streak = image_scale_streak = 0
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=False)
    progress_path = output_dir / "progress.json"
    for pair_offset in range(candidate_pairs):
        current_index = first_index + pair_offset + 1
        ok, current_bgr = capture.read()
        if not ok:
            raise ValueError(f"VIDEO_FRAME_DECODE_FAILED:{current_index}")
        current, current_valid = preprocess_frame_with_mask(
            current_bgr, resize_scale, undistort_maps
        )
        previous_timestamp = float(timestamps[current_index - 1])
        current_timestamp = float(timestamps[current_index])
        dt_seconds = current_timestamp - previous_timestamp
        if not (0.0 < dt_seconds <= 0.1):
            raise ValueError(f"PAIR_DT:{current_index}:{dt_seconds}")
        previous_pose = interpolate_pose(poses, previous_timestamp)
        current_pose = interpolate_pose(poses, current_timestamp)
        homography, angular_speed, translation_speed = pair_geometry(
            previous_pose,
            current_pose,
            dt_seconds,
            quaternion_component_order=quaternion_component_order,
            pose_to_camera_rotation=pose_to_camera_rotation,
        )
        if resize_scale != 1.0:
            scale_matrix = np.array(
                [
                    [resize_scale, 0.0, 0.0],
                    [0.0, resize_scale, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            homography = (
                scale_matrix
                @ homography
                @ np.linalg.inv(scale_matrix)
            )
        row = rgb_core._evaluate_pair(
            pair_offset,
            previous,
            current,
            Decimal(str(previous_timestamp)),
            Decimal(str(current_timestamp)),
            homography,
            protocol,
            state,
            previous_valid,
            current_valid,
        )
        image_scale, image_scale_support, image_scale_reason = (
            global_image_scale_proxy(previous, current, dt_seconds)
        )
        row.update(
            source_id=SOURCE_ID,
            frame_index_previous_zero_based=current_index - 1,
            frame_index_current_zero_based=current_index,
            angular_speed_deg_per_s=angular_speed,
            translation_speed_m_per_s=translation_speed,
            image_scale_expansion_per_s=image_scale,
            image_scale_support_count=image_scale_support,
            image_scale_reason=image_scale_reason,
            bbox_growth=None,
            bbox_growth_reason="NOT_EVALUABLE_NO_FROZEN_OBJECT_BOXES",
        )
        raw_streak = update_confirmation(row, raw_streak, "raw")
        compensated_streak = update_confirmation(
            row, compensated_streak, "compensated"
        )
        image_scale_above = (
            image_scale is not None and image_scale > THRESHOLD
        )
        image_scale_streak = (
            image_scale_streak + 1 if image_scale_above else 0
        )
        row["image_scale_above_threshold"] = image_scale_above
        row[
            "image_scale_consecutive_above_threshold_pair_count"
        ] = image_scale_streak
        row["image_scale_three_pair_trigger"] = (
            image_scale_streak >= REQUIRED_CONSECUTIVE_PAIRS
        )
        rows.append(row)
        previous = current
        previous_valid = current_valid
        completed = pair_offset + 1
        if completed % progress_every == 0 or completed == candidate_pairs:
            elapsed = time.perf_counter() - started
            progress = {
                "schema": "rcle.ecological_response.discovery.progress.v1",
                "protocol_id": (
                    evidence_context or {}
                ).get("protocol_id", PROTOCOL_ID),
                "phase": "PAIR_EVALUATION",
                "completed_units": completed,
                "total_units": candidate_pairs,
                "throughput": completed / elapsed if elapsed else None,
                "eta_seconds": (
                    (candidate_pairs - completed) * elapsed / completed
                    if completed
                    else None
                ),
                "last_progress_at": datetime.now(timezone.utc).isoformat(),
                "status": (
                    "COMPLETE"
                    if completed == candidate_pairs
                    else "RUNNING"
                ),
                "completed_pairs": completed,
                "total_pairs": candidate_pairs,
                "elapsed_s": elapsed,
                "pairs_per_s": completed / elapsed if elapsed else None,
                "eta_s": (
                    (candidate_pairs - completed) * elapsed / completed
                    if completed
                    else None
                ),
                "terminal": completed == candidate_pairs,
            }
            progress_path.write_text(
                json.dumps(progress, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                f"completed={completed}/{candidate_pairs} "
                f"pairs_per_s={progress['pairs_per_s']:.3f} "
                f"eta_s={progress['eta_s']:.1f}",
                flush=True,
            )
    capture.release()
    elapsed = time.perf_counter() - started
    segments = segment_summaries(rows)
    abstentions = Counter(
        str(row["reason"])
        for row in rows
        if row.get("evaluable") is not True
    )
    common_cells = [
        int(row["common_cell_count"])
        for row in rows
        if row.get("common_cell_count") is not None
    ]
    raw_abs_rows = [
        {
            "angular_speed_deg_per_s": row["angular_speed_deg_per_s"],
            "_value": abs(float(row["raw_expansion_median_per_s"])),
        }
        for row in rows
        if row.get("raw_expansion_median_per_s") is not None
    ]
    compensated_abs_rows = [
        {
            "angular_speed_deg_per_s": row["angular_speed_deg_per_s"],
            "_value": abs(
                float(row["compensated_expansion_median_per_s"])
            ),
        }
        for row in rows
        if row.get("compensated_expansion_median_per_s") is not None
    ]
    result = {
        "schema": "rcle.ecological_response.discovery.summary.v1",
        "protocol_id": (evidence_context or {}).get(
            "protocol_id", PROTOCOL_ID
        ),
        "governance_policy_id": (
            "DATA_CAPABILITY_DRIVEN_RESEARCH_GOVERNANCE_R2"
        ),
        "research_track": (evidence_context or {}).get(
            "research_track", "CAPABILITY_DISCOVERY"
        ),
        "outcome_access_state": (evidence_context or {}).get(
            "outcome_access_state", "OUTPUT_INSPECTED"
        ),
        "stage": (evidence_context or {}).get(
            "stage", "CAPABILITY_DISCOVERY"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "source_id": SOURCE_ID,
            "observation_unit": "ONE_CAPTURE_SESSION",
            "access_state": [
                "CONTENT_INSPECTED",
                "OUTPUT_INSPECTED_FOR_PRIOR_NON_RCLE_PROBES",
                "RCLE_OUTPUT_INSPECTED_BY_THIS_DISCOVERY",
            ],
            "license": "CC-BY-NC-4.0",
            "video_path": video_path.as_posix(),
            "video_sha256": sha256_file(video_path),
            "frame_timestamps_path": frame_csv_path.as_posix(),
            "frame_timestamps_sha256": sha256_file(frame_csv_path),
            "pose_path": pose_path.as_posix(),
            "pose_sha256": sha256_file(pose_path),
            "calibration_url": CALIBRATION_URL,
            "calibration_pinned_url": CALIBRATION_PINNED_URL,
            "calibration_sha256": CALIBRATION_SHA256,
            "intrinsic": INTRINSIC.tolist(),
            "distortion": DISTORTION.tolist(),
            "distortion_model": "radial_tangential_k1_k2_p1_p2",
            "distortion_correction_applied": distortion_correction,
            "distortion_correction_recipe": (
                "initUndistortRectifyMap_same_K_then_remap_with_valid_mask_before_resize"
                if distortion_correction
                else "none"
            ),
            "pose_quaternion_component_order": (
                quaternion_component_order
            ),
            "pose_to_camera_rotation": (
                np.asarray(pose_to_camera_rotation).tolist()
                if pose_to_camera_rotation is not None
                else None
            ),
        },
        "execution": {
            "frame_index_start_zero_based": first_index,
            "frame_index_end_exclusive_zero_based": last_index_exclusive,
            "candidate_pair_count": candidate_pairs,
            "evaluable_pair_count": sum(
                row.get("evaluable") is True for row in rows
            ),
            "evaluable_pair_fraction": sum(
                row.get("evaluable") is True for row in rows
            )
            / len(rows),
            "abstention_reasons": dict(sorted(abstentions.items())),
            "duration_s": float(
                timestamps[last_index_exclusive - 1]
                - timestamps[first_index]
            ),
            "runtime_s": elapsed,
            "pairs_per_s": candidate_pairs / elapsed,
            "native_frame_rate_preserved": True,
            "spatial_resize_scale": resize_scale,
            "processed_frame_shape": [
                int(round(height * resize_scale)),
                int(round(width * resize_scale)),
            ],
            "threshold_per_s": THRESHOLD,
            "required_consecutive_pairs": REQUIRED_CONSECUTIVE_PAIRS,
            "single_process_pair_state_continuous": True,
            "support_manager_baseline_pair_count": sum(
                bool(row.get("support_manager", {}).get("baseline_only"))
                for row in rows
            ),
        },
        "methods": {
            "raw_local_expansion": method_summary(
                rows,
                "raw_expansion_median_per_s",
                "raw_three_pair_trigger",
            ),
            "source_pose_rotation_compensated_local_expansion": method_summary(
                rows,
                "compensated_expansion_median_per_s",
                "compensated_three_pair_trigger",
            ),
            "global_image_scale_proxy": method_summary(
                rows,
                "image_scale_expansion_per_s",
                "image_scale_three_pair_trigger",
            ),
            "bbox_growth": {
                "status": "NOT_EVALUABLE",
                "reason": "NO_FROZEN_OBJECT_BOXES",
            },
        },
        "diagnostics": {
            "median_angular_speed_deg_per_s": float(
                np.median(
                    [row["angular_speed_deg_per_s"] for row in rows]
                )
            ),
            "p90_angular_speed_deg_per_s": float(
                np.quantile(
                    [row["angular_speed_deg_per_s"] for row in rows], 0.9
                )
            ),
            "median_translation_speed_m_per_s": float(
                np.median(
                    [row["translation_speed_m_per_s"] for row in rows]
                )
            ),
            "p90_translation_speed_m_per_s": float(
                np.quantile(
                    [row["translation_speed_m_per_s"] for row in rows], 0.9
                )
            ),
            "angular_speed_correlation": {
                "raw_expansion": correlation(
                    rows, "raw_expansion_median_per_s"
                ),
                "compensated_expansion": correlation(
                    rows, "compensated_expansion_median_per_s"
                ),
                "raw_abs_expansion": correlation(raw_abs_rows, "_value"),
                "compensated_abs_expansion": correlation(
                    compensated_abs_rows, "_value"
                ),
            },
            "median_common_cell_count": (
                float(np.median(common_cells)) if common_cells else None
            ),
            "segment_count": len(segments),
        },
        "claim_ceiling": {
            "allowed": [
                "single-session response characterization",
                "support and abstention characterization",
                "failure-mode hypothesis generation",
                "runtime characterization",
            ],
            "forbidden": [
                "performance",
                "generalization",
                "causal mechanism confirmation",
                "risk or alert efficacy",
                "Android",
                "product",
                "safety",
            ],
            "old_rgb_segment_confirmation_r1_terminal_changed": False,
        },
    }
    if evidence_context:
        result["evidence_context"] = evidence_context
    ledger_sha = write_jsonl(output_dir / "pair_ledger.jsonl", rows)
    segment_sha = write_jsonl(
        output_dir / "segment_summary.jsonl", segments
    )
    result["artifacts"] = {
        "pair_ledger_sha256": ledger_sha,
        "segment_summary_sha256": segment_sha,
    }
    render_curves(output_dir / "response_curves.png", rows)
    result["artifacts"]["response_curves_sha256"] = sha256_file(
        output_dir / "response_curves.png"
    )
    write_json(output_dir / "summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--start-time-s", type=float)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--resize-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(
        args.source_root,
        args.output_dir,
        max_pairs=args.max_pairs,
        start_time_s=args.start_time_s,
        start_frame=args.start_frame,
        duration_s=args.duration_s,
        progress_every=args.progress_every,
        resize_scale=args.resize_scale,
    )
    print(
        json.dumps(
            {
                "stage": result["stage"],
                "candidate_pair_count": result["execution"][
                    "candidate_pair_count"
                ],
                "evaluable_pair_fraction": result["execution"][
                    "evaluable_pair_fraction"
                ],
                "runtime_s": result["execution"]["runtime_s"],
                "methods": result["methods"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
