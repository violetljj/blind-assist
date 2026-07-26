from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

from ..pb_h1_role_proxy.geometry import (
    summarize_translation_induced_geometry,
    translation_induced_geometry,
)


SEQUENCE_ID = "rgbd_dataset_freiburg2_rpy"
IMAGE_SIZE_WH = (640, 480)
K_TUM_DEFAULT = np.asarray(
    ((525.0, 0.0, 319.5), (0.0, 525.0, 239.5), (0.0, 0.0, 1.0)),
    dtype=np.float64,
)
DEPTH_UNITS_PER_METER = 5000.0
MAX_RGB_DEPTH_DELTA = Decimal("0.020")
MIN_PAIR_DT = Decimal("0.020")
MAX_PAIR_DT = Decimal("0.050")
MAX_POSE_BRACKET = Decimal("0.050")
WINDOW_SECONDS = Decimal("10")
QUANTILES = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0)
QUANTILE_NAMES = ("min", "q10", "q25", "median", "q75", "q90", "q95", "max")


@dataclass(frozen=True)
class IndexRow:
    timestamp: Decimal
    relative_path: str


@dataclass(frozen=True)
class PoseRow:
    timestamp: Decimal
    center_world_m: np.ndarray
    quaternion_xyzw: np.ndarray


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_lines(path: Path) -> Iterable[list[str]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        yield stripped.split()


def parse_index(path: Path) -> list[IndexRow]:
    rows: list[IndexRow] = []
    for tokens in _data_lines(path):
        if len(tokens) != 2:
            raise ValueError("TUM_INDEX_COLUMNS")
        rows.append(IndexRow(Decimal(tokens[0]), tokens[1]))
    if not rows or any(a.timestamp >= b.timestamp for a, b in zip(rows, rows[1:])):
        raise ValueError("TUM_INDEX_NOT_STRICTLY_MONOTONIC")
    if len({row.relative_path for row in rows}) != len(rows):
        raise ValueError("TUM_INDEX_DUPLICATE_PATH")
    return rows


def parse_poses_with_diagnostics(path: Path) -> tuple[list[PoseRow], dict[str, Any]]:
    raw_rows: list[PoseRow] = []
    for tokens in _data_lines(path):
        if len(tokens) != 8:
            raise ValueError("TUM_POSE_COLUMNS")
        values = np.asarray([float(value) for value in tokens[1:]], dtype=np.float64)
        quaternion = values[3:7]
        norm = float(np.linalg.norm(quaternion))
        if not np.all(np.isfinite(values)) or abs(norm - 1.0) > 0.001:
            raise ValueError("TUM_POSE_INVALID")
        raw_rows.append(
            PoseRow(
                timestamp=Decimal(tokens[0]),
                center_world_m=values[:3],
                quaternion_xyzw=quaternion / norm,
            )
        )
    if not raw_rows or any(
        a.timestamp > b.timestamp for a, b in zip(raw_rows, raw_rows[1:])
    ):
        raise ValueError("TUM_POSE_NOT_STRICTLY_MONOTONIC")
    groups: dict[Decimal, list[PoseRow]] = {}
    for row in raw_rows:
        groups.setdefault(row.timestamp, []).append(row)
    duplicate_groups = [group for group in groups.values() if len(group) > 1]
    maximum_translation_spread = 0.0
    maximum_orientation_spread = 0.0
    for group in duplicate_groups:
        for left_index, left in enumerate(group):
            for right in group[left_index + 1 :]:
                maximum_translation_spread = max(
                    maximum_translation_spread,
                    float(np.linalg.norm(left.center_world_m - right.center_world_m)),
                )
                quaternion_dot = abs(
                    float(
                        np.dot(
                            _normalize_quaternion(left.quaternion_xyzw),
                            _normalize_quaternion(right.quaternion_xyzw),
                        )
                    )
                )
                maximum_orientation_spread = max(
                    maximum_orientation_spread,
                    math.degrees(
                        2.0 * math.acos(float(np.clip(quaternion_dot, -1.0, 1.0)))
                    ),
                )
    rows = [group[-1] for group in groups.values()]
    diagnostics = {
        "raw_pose_row_count": len(raw_rows),
        "unique_pose_timestamp_count": len(rows),
        "duplicate_timestamp_group_count": len(duplicate_groups),
        "duplicate_extra_row_count": len(raw_rows) - len(rows),
        "maximum_duplicate_translation_spread_m": maximum_translation_spread,
        "maximum_duplicate_orientation_spread_deg": maximum_orientation_spread,
        "duplicate_resolution": "last_text_row_wins",
        "duplicate_spread_pass": (
            maximum_translation_spread <= 0.001
            and maximum_orientation_spread <= 0.5
        ),
    }
    return rows, diagnostics


def parse_poses(path: Path) -> list[PoseRow]:
    return parse_poses_with_diagnostics(path)[0]


def associate_unique_nearest(
    first: Sequence[IndexRow],
    second: Sequence[IndexRow],
    maximum_delta: Decimal = MAX_RGB_DEPTH_DELTA,
) -> dict[int, int]:
    """Match TUM associate.py semantics: greedy unique minimum time difference."""

    potential: list[tuple[Decimal, int, int]] = []
    second_timestamps = [row.timestamp for row in second]
    for first_index, first_row in enumerate(first):
        insertion = bisect_left(second_timestamps, first_row.timestamp)
        left = insertion - 1
        while left >= 0 and first_row.timestamp - second[left].timestamp <= maximum_delta:
            potential.append(
                (
                    abs(first_row.timestamp - second[left].timestamp),
                    first_index,
                    left,
                )
            )
            left -= 1
        right = insertion
        while (
            right < len(second)
            and second[right].timestamp - first_row.timestamp <= maximum_delta
        ):
            potential.append(
                (
                    abs(first_row.timestamp - second[right].timestamp),
                    first_index,
                    right,
                )
            )
            right += 1
    used_first: set[int] = set()
    used_second: set[int] = set()
    matches: dict[int, int] = {}
    for _, first_index, second_index in sorted(potential):
        if first_index in used_first or second_index in used_second:
            continue
        matches[first_index] = second_index
        used_first.add(first_index)
        used_second.add(second_index)
    return matches


def _normalize_quaternion(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)) or norm <= 0:
        raise ValueError("TUM_POSE_INVALID_QUATERNION")
    return quaternion / norm


def slerp(left_raw: np.ndarray, right_raw: np.ndarray, fraction: float) -> np.ndarray:
    left = _normalize_quaternion(left_raw)
    right = _normalize_quaternion(right_raw)
    dot = float(np.dot(left, right))
    if dot < 0.0:
        right = -right
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return _normalize_quaternion(left + fraction * (right - left))
    theta = math.acos(dot)
    result = (
        math.sin((1.0 - fraction) * theta) / math.sin(theta) * left
        + math.sin(fraction * theta) / math.sin(theta) * right
    )
    return _normalize_quaternion(result)


def interpolate_pose(
    poses: Sequence[PoseRow],
    pose_timestamps: Sequence[Decimal],
    timestamp: Decimal,
) -> tuple[np.ndarray, np.ndarray]:
    insertion = bisect_left(pose_timestamps, timestamp)
    if insertion < len(poses) and poses[insertion].timestamp == timestamp:
        row = poses[insertion]
        return row.center_world_m.copy(), row.quaternion_xyzw.copy()
    if insertion == 0 or insertion == len(poses):
        raise ValueError("POSE_NOT_BRACKETED")
    left = poses[insertion - 1]
    right = poses[insertion]
    span = right.timestamp - left.timestamp
    if span > MAX_POSE_BRACKET:
        raise ValueError("POSE_BRACKET_GT_0P050_S")
    fraction = float((timestamp - left.timestamp) / span)
    center = left.center_world_m + fraction * (
        right.center_world_m - left.center_world_m
    )
    return center, slerp(left.quaternion_xyzw, right.quaternion_xyzw, fraction)


def quaternion_rotation(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = _normalize_quaternion(quaternion_xyzw)
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def relative_geometry(
    previous_pose: tuple[np.ndarray, np.ndarray],
    current_pose: tuple[np.ndarray, np.ndarray],
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    previous_center, previous_quaternion = previous_pose
    current_center, current_quaternion = current_pose
    previous_rotation = quaternion_rotation(previous_quaternion)
    current_rotation = quaternion_rotation(current_quaternion)
    rotation_current_from_previous = current_rotation.T @ previous_rotation
    translation_current_from_previous = current_rotation.T @ (
        previous_center - current_center
    )
    angle = math.acos(
        float(
            np.clip(
                (np.trace(rotation_current_from_previous) - 1.0) / 2.0,
                -1.0,
                1.0,
            )
        )
    )
    return (
        rotation_current_from_previous,
        translation_current_from_previous,
        angle / dt_s,
    )


def fixed_windows(
    rgb: Sequence[IndexRow], depth: Sequence[IndexRow], poses: Sequence[PoseRow]
) -> list[tuple[Decimal, Decimal]]:
    shared_start = max(rgb[0].timestamp, depth[0].timestamp, poses[0].timestamp)
    shared_end = min(rgb[-1].timestamp, depth[-1].timestamp, poses[-1].timestamp)
    anchor = Decimal(math.ceil(float(shared_start)))
    windows: list[tuple[Decimal, Decimal]] = []
    start = anchor
    while start + WINDOW_SECONDS <= shared_end:
        windows.append((start, start + WINDOW_SECONDS))
        start += WINDOW_SECONDS
    return windows


def decode_and_sample_depth(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, float]:
    with Image.open(path) as image:
        image.load()
        raw = np.asarray(image)
    if raw.shape != (480, 640) or raw.dtype != np.uint16:
        raise ValueError("DEPTH_MEMBER_MISSING_OR_INVALID")
    ys = np.arange(0, 480, 8, dtype=np.int64)
    xs = np.arange(0, 640, 8, dtype=np.int64)
    xx, yy = np.meshgrid(xs, ys)
    sampled = raw[yy, xx].ravel()
    valid = sampled > 0
    pixels = np.column_stack((xx.ravel()[valid], yy.ravel()[valid])).astype(
        np.float64
    )
    return (
        pixels,
        sampled[valid].astype(np.float64) / DEPTH_UNITS_PER_METER,
        float(np.mean(valid)),
    )


def distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, **{name: None for name in QUANTILE_NAMES}}
    array = np.asarray(values, dtype=np.float64)
    result: dict[str, float | int | None] = {"count": int(array.size)}
    result.update(
        {
            name: float(value)
            for name, value in zip(
                QUANTILE_NAMES, np.quantile(array, QUANTILES), strict=True
            )
        }
    )
    return result


def _summarize_pairs(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "raw_translation_speed_m_s",
        "angular_rate_rad_s",
        "median_signed_radial_expansion_per_s",
        "median_absolute_radial_expansion_per_s",
        "radial_expansion_positive_fraction",
        "q90_time_normalized_parallax_rad_per_s",
        "valid_fraction",
        "source_depth_nonzero_fraction",
        "rgb_depth_delta_previous_s",
        "rgb_depth_delta_current_s",
    )
    return {
        metric: distribution([float(row[metric]) for row in rows])
        for metric in metrics
    }


def _classify_window(summary: dict[str, Any]) -> dict[str, Any]:
    distributions = summary["distributions"]
    angular = float(distributions["angular_rate_rad_s"]["median"])
    raw_speed = float(distributions["raw_translation_speed_m_s"]["median"])
    signed = float(
        distributions["median_signed_radial_expansion_per_s"]["median"]
    )
    absolute = float(
        distributions["median_absolute_radial_expansion_per_s"]["median"]
    )
    positive = float(
        distributions["radial_expansion_positive_fraction"]["median"]
    )
    parallax = float(
        distributions["q90_time_normalized_parallax_rad_per_s"]["median"]
    )
    ratio = parallax / angular if angular > 0.0 else math.inf
    rotation_active = math.degrees(angular) >= 5.0
    rotation_like = (
        absolute <= 0.02
        and abs(signed) < 0.01
        and 0.40 <= positive <= 0.60
        and ratio <= 0.10
    )
    counterexample = rotation_active and rotation_like and raw_speed > 0.02
    return {
        "median_angular_speed_deg_s": math.degrees(angular),
        "q90_parallax_to_median_angular_rate_ratio": ratio,
        "rotation_active": rotation_active,
        "rotation_like_direct_geometry": rotation_like,
        "passes_old_raw_speed_gate": raw_speed <= 0.02,
        "old_raw_speed_gate_counterexample": counterexample,
    }


def _verify_image_inventory(
    root: Path, rgb: Sequence[IndexRow], depth: Sequence[IndexRow]
) -> dict[str, Any]:
    failures: list[str] = []
    for row in rgb:
        path = root / row.relative_path
        try:
            with Image.open(path) as image:
                if image.size != IMAGE_SIZE_WH or image.mode != "RGB" or image.format != "PNG":
                    failures.append(row.relative_path)
        except (OSError, ValueError):
            failures.append(row.relative_path)
    for row in depth:
        path = root / row.relative_path
        try:
            with Image.open(path) as image:
                if (
                    image.size != IMAGE_SIZE_WH
                    or image.mode not in ("I;16", "I;16B")
                    or image.format != "PNG"
                ):
                    failures.append(row.relative_path)
        except (OSError, ValueError):
            failures.append(row.relative_path)
    return {
        "rgb_index_rows": len(rgb),
        "depth_index_rows": len(depth),
        "all_indexed_members_valid_png": not failures,
        "invalid_member_count": len(failures),
        "invalid_members": failures,
    }


def run_audit(root: Path, archive_path: Path) -> dict[str, Any]:
    rgb = parse_index(root / "rgb.txt")
    depth = parse_index(root / "depth.txt")
    poses, pose_diagnostics = parse_poses_with_diagnostics(
        root / "groundtruth.txt"
    )
    matches = associate_unique_nearest(rgb, depth)
    windows = fixed_windows(rgb, depth, poses)
    pose_timestamps = [row.timestamp for row in poses]
    image_inventory = _verify_image_inventory(root, rgb, depth)
    window_results: list[dict[str, Any]] = []
    all_evaluable_pairs: list[dict[str, Any]] = []
    total_candidate_pairs = 0
    abstentions: Counter[str] = Counter()

    for window_index, (start, end) in enumerate(windows):
        rgb_indices = [
            index for index, row in enumerate(rgb) if start <= row.timestamp < end
        ]
        candidate_pairs = [
            (previous, current)
            for previous, current in zip(rgb_indices, rgb_indices[1:])
            if MIN_PAIR_DT
            <= rgb[current].timestamp - rgb[previous].timestamp
            <= MAX_PAIR_DT
        ]
        total_candidate_pairs += len(candidate_pairs)
        pair_rows: list[dict[str, Any]] = []
        window_abstentions: Counter[str] = Counter()
        for previous_index, current_index in candidate_pairs:
            previous_rgb = rgb[previous_index]
            current_rgb = rgb[current_index]
            previous_depth_index = matches.get(previous_index)
            current_depth_index = matches.get(current_index)
            if previous_depth_index is None or current_depth_index is None:
                reason = "RGB_DEPTH_UNMATCHED_OR_REUSED"
                abstentions[reason] += 1
                window_abstentions[reason] += 1
                continue
            dt_s = float(current_rgb.timestamp - previous_rgb.timestamp)
            try:
                previous_pose = interpolate_pose(
                    poses, pose_timestamps, previous_rgb.timestamp
                )
                current_pose = interpolate_pose(
                    poses, pose_timestamps, current_rgb.timestamp
                )
                rotation, translation, angular_rate = relative_geometry(
                    previous_pose, current_pose, dt_s
                )
                pixels, depth_m, source_depth_nonzero_fraction = (
                    decode_and_sample_depth(
                    root / depth[previous_depth_index].relative_path
                    )
                )
                geometry = summarize_translation_induced_geometry(
                    translation_induced_geometry(
                        pixels,
                        depth_m,
                        K_TUM_DEFAULT,
                        rotation,
                        translation,
                        dt_s,
                        image_size_wh=IMAGE_SIZE_WH,
                        minimum_radius_px=8.0,
                        zbuffer=True,
                    )
                )
                if not geometry["evaluable"]:
                    raise ValueError("PB_H1_NO_VISIBLE_DEPTH_SUPPORT")
            except ValueError as error:
                reason = str(error)
                abstentions[reason] += 1
                window_abstentions[reason] += 1
                continue
            row = {
                "window_index": window_index,
                "previous_rgb_timestamp": str(previous_rgb.timestamp),
                "current_rgb_timestamp": str(current_rgb.timestamp),
                "previous_depth_timestamp": str(
                    depth[previous_depth_index].timestamp
                ),
                "current_depth_timestamp": str(depth[current_depth_index].timestamp),
                "rgb_depth_delta_previous_s": float(
                    abs(
                        previous_rgb.timestamp
                        - depth[previous_depth_index].timestamp
                    )
                ),
                "rgb_depth_delta_current_s": float(
                    abs(
                        current_rgb.timestamp - depth[current_depth_index].timestamp
                    )
                ),
                "dt_s": dt_s,
                "angular_rate_rad_s": angular_rate,
                "source_depth_nonzero_fraction": (
                    source_depth_nonzero_fraction
                ),
                **geometry,
            }
            pair_rows.append(row)
            all_evaluable_pairs.append(row)
        pair_coverage = (
            len(pair_rows) / len(candidate_pairs) if candidate_pairs else 0.0
        )
        median_pb_h1_visibility = (
            float(np.median([row["valid_fraction"] for row in pair_rows]))
            if pair_rows
            else 0.0
        )
        median_source_depth = (
            float(
                np.median(
                    [
                        row["source_depth_nonzero_fraction"]
                        for row in pair_rows
                    ]
                )
            )
            if pair_rows
            else 0.0
        )
        evaluable = (
            bool(pair_rows)
            and pair_coverage >= 0.80
            and median_source_depth >= 0.50
            and median_pb_h1_visibility >= 0.50
        )
        window_coverage_abstentions: list[str] = []
        if not pair_rows:
            window_coverage_abstentions.append("NO_FORMULA_EVALUABLE_PAIR")
        if pair_coverage < 0.80:
            window_coverage_abstentions.append("PAIR_COVERAGE_LT_0P80")
        if median_source_depth < 0.50:
            window_coverage_abstentions.append(
                "SOURCE_DEPTH_COVERAGE_LT_0P50"
            )
        if median_pb_h1_visibility < 0.50:
            window_coverage_abstentions.append(
                "PB_H1_VISIBILITY_COVERAGE_LT_0P50"
            )
        window_summary: dict[str, Any] = {
            "window_index": window_index,
            "start": str(start),
            "end": str(end),
            "candidate_pair_count": len(candidate_pairs),
            "evaluable_pair_count": len(pair_rows),
            "pair_coverage": pair_coverage,
            "median_valid_depth_fraction": median_source_depth,
            "median_source_depth_nonzero_fraction": median_source_depth,
            "median_pb_h1_visibility_fraction": median_pb_h1_visibility,
            "evaluable": evaluable,
            "window_coverage_abstentions": window_coverage_abstentions,
            "abstentions": dict(sorted(window_abstentions.items())),
            "distributions": _summarize_pairs(pair_rows),
            "pairs": pair_rows,
        }
        window_summary["diagnostics"] = (
            _classify_window(window_summary) if evaluable else None
        )
        window_results.append(window_summary)

    evaluable_windows = [window for window in window_results if window["evaluable"]]
    window_abstention_counts = Counter(
        reason
        for window in window_results
        for reason in window["window_coverage_abstentions"]
    )
    evaluable_window_fraction = (
        len(evaluable_windows) / len(window_results) if window_results else 0.0
    )
    association_fraction = len(matches) / max(len(rgb), len(depth))
    coverage_pass = (
        association_fraction >= 0.95
        and len(evaluable_windows) >= 5
        and evaluable_window_fraction >= 0.80
        and image_inventory["all_indexed_members_valid_png"]
        and pose_diagnostics["duplicate_spread_pass"]
    )
    rotation_windows = [
        window
        for window in evaluable_windows
        if window["diagnostics"]["rotation_active"]
        and window["diagnostics"]["rotation_like_direct_geometry"]
    ]
    counterexample_windows = [
        window
        for window in rotation_windows
        if window["diagnostics"]["old_raw_speed_gate_counterexample"]
    ]
    source_pass = coverage_pass and bool(rotation_windows)
    return {
        "schema_version": "rcle.tum_fr2_rpy.source_native_geometry_audit.result.v1",
        "protocol_id": "RCLE-TUM-FR2-RPY-SOURCE-NATIVE-GEOMETRY-AUDIT-R0",
        "sequence_id": SEQUENCE_ID,
        "archive": {
            "path": str(archive_path),
            "bytes": archive_path.stat().st_size,
            "sha256": sha256_file(archive_path),
        },
        "groundtruth_sha256": sha256_file(root / "groundtruth.txt"),
        "pose_inventory": pose_diagnostics,
        "inventory": image_inventory,
        "time_coverage": {
            "rgb_first": str(rgb[0].timestamp),
            "rgb_last": str(rgb[-1].timestamp),
            "depth_first": str(depth[0].timestamp),
            "depth_last": str(depth[-1].timestamp),
            "pose_first": str(poses[0].timestamp),
            "pose_last": str(poses[-1].timestamp),
            "fixed_window_count": len(windows),
        },
        "association": {
            "matched_rgb_depth_count": len(matches),
            "association_fraction_over_max_stream_count": association_fraction,
            "maximum_delta_s": float(MAX_RGB_DEPTH_DELTA),
        },
        "coverage": {
            "candidate_pair_count": total_candidate_pairs,
            "evaluable_pair_count": len(all_evaluable_pairs),
            "pair_coverage": (
                len(all_evaluable_pairs) / total_candidate_pairs
                if total_candidate_pairs
                else 0.0
            ),
            "evaluable_window_count": len(evaluable_windows),
            "evaluable_window_fraction": evaluable_window_fraction,
            "abstentions": dict(sorted(abstentions.items())),
            "window_abstentions": dict(sorted(window_abstention_counts.items())),
            "coverage_pass": coverage_pass,
        },
        "continuous_pair_distributions": _summarize_pairs(all_evaluable_pairs),
        "rotation_dominant_window_indices": [
            window["window_index"] for window in rotation_windows
        ],
        "old_raw_speed_gate_counterexample_window_indices": [
            window["window_index"] for window in counterexample_windows
        ],
        "conclusions": {
            "fr2_rpy_provides_rotation_dominant_real_windows": source_pass,
            "old_0p02_m_s_gate_wrong_kill_observed": bool(counterexample_windows),
            "source_evaluable": coverage_pass,
        },
        "terminal": (
            "PASS_SOURCE_NATIVE_GEOMETRY_CANARY_DESIGN_MAY_FOLLOW"
            if source_pass
            else "HOLD_NOT_EVALUABLE_FR2_RPY_LOCAL"
        ),
        "windows": window_results,
        "authority_boundary": (
            "Single-source discovery characterization only; no RCLE RGB "
            "algorithm, confirmation, safety, or product authority."
        ),
    }
