from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path, PurePosixPath
import tarfile
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

from ..pb_h1_role_proxy.geometry import (
    summarize_translation_induced_geometry,
    translation_induced_geometry,
)


QUANTILES = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0)
QUANTILE_NAMES = ("min", "q10", "q25", "median", "q75", "q90", "q95", "max")
METRIC_KEYS = (
    "raw_translation_speed_m_s",
    "median_angular_speed_deg_s",
    "median_signed_radial_expansion_per_s",
    "median_absolute_radial_expansion_per_s",
    "radial_expansion_positive_fraction",
    "q90_time_normalized_parallax_rad_per_s",
    "valid_depth_fraction",
)


@dataclass(frozen=True)
class IndexRow:
    timestamp: Decimal
    path: str


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


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("CANARY_JSON_OBJECT_REQUIRED")
    return value


def _tokens(text: bytes) -> Iterable[list[str]]:
    for line in text.decode("utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            yield stripped.split()


def _parse_index(text: bytes) -> list[IndexRow]:
    rows: list[IndexRow] = []
    for values in _tokens(text):
        if len(values) != 2:
            raise ValueError("TUM_INDEX_COLUMNS")
        relative = PurePosixPath(values[1])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("TUM_INDEX_UNSAFE_PATH")
        rows.append(IndexRow(Decimal(values[0]), relative.as_posix()))
    if not rows or any(a.timestamp >= b.timestamp for a, b in zip(rows, rows[1:])):
        raise ValueError("TUM_INDEX_NOT_STRICTLY_MONOTONIC")
    if len({row.path for row in rows}) != len(rows):
        raise ValueError("TUM_INDEX_DUPLICATE_PATH")
    return rows


def _normalize_quaternion(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)) or norm <= 0:
        raise ValueError("POSE_INVALID_QUATERNION")
    return quaternion / norm


def _parse_poses(text: bytes) -> list[PoseRow]:
    grouped: dict[Decimal, PoseRow] = {}
    previous: Decimal | None = None
    for values in _tokens(text):
        if len(values) != 8:
            raise ValueError("TUM_POSE_COLUMNS")
        timestamp = Decimal(values[0])
        if previous is not None and timestamp < previous:
            raise ValueError("TUM_POSE_NOT_MONOTONIC")
        numeric = np.asarray([float(item) for item in values[1:]], dtype=np.float64)
        if not np.all(np.isfinite(numeric)):
            raise ValueError("POSE_INVALID_QUATERNION")
        grouped[timestamp] = PoseRow(
            timestamp,
            numeric[:3],
            numeric[3:7],
        )
        previous = timestamp
    if not grouped:
        raise ValueError("TUM_POSE_EMPTY")
    return list(grouped.values())


class ArchiveReader:
    def __init__(self, path: Path) -> None:
        self._bundle = tarfile.open(path, mode="r:*")
        members = [member for member in self._bundle.getmembers() if member.isfile()]
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("TUM_ARCHIVE_DUPLICATE_MEMBER")
        for name in names:
            parsed = PurePosixPath(name)
            if parsed.is_absolute() or ".." in parsed.parts:
                raise ValueError("TUM_ARCHIVE_UNSAFE_MEMBER")
        self._members = {member.name: member for member in members}
        self.rgb_member = self._unique_suffix("rgb.txt")
        self.depth_member = self._unique_suffix("depth.txt")
        self.pose_member = self._unique_suffix("groundtruth.txt")
        roots = {
            member.rsplit("/", 1)[0] if "/" in member else ""
            for member in (self.rgb_member, self.depth_member, self.pose_member)
        }
        if len(roots) != 1:
            raise ValueError("TUM_ARCHIVE_CONTROL_ROOT_MISMATCH")
        self.root = roots.pop()

    def _unique_suffix(self, suffix: str) -> str:
        matches = [name for name in self._members if name.endswith("/" + suffix) or name == suffix]
        if len(matches) != 1:
            raise ValueError(f"TUM_ARCHIVE_CONTROL_MEMBER:{suffix}")
        return matches[0]

    def read(self, member: str) -> bytes:
        handle = self._bundle.extractfile(self._members[member])
        if handle is None:
            raise ValueError("TUM_ARCHIVE_MEMBER_UNREADABLE")
        return handle.read()

    def resolve(self, relative: str) -> str:
        name = f"{self.root}/{relative}" if self.root else relative
        if name not in self._members:
            raise ValueError("DEPTH_MEMBER_MISSING_OR_INVALID")
        return name

    def close(self) -> None:
        self._bundle.close()

    def __enter__(self) -> ArchiveReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _associate(rgb: Sequence[IndexRow], depth: Sequence[IndexRow], limit: Decimal) -> dict[int, int]:
    candidates: list[tuple[Decimal, int, int]] = []
    depth_times = [row.timestamp for row in depth]
    for rgb_index, rgb_row in enumerate(rgb):
        insertion = bisect_left(depth_times, rgb_row.timestamp)
        left = insertion - 1
        while left >= 0 and rgb_row.timestamp - depth[left].timestamp <= limit:
            candidates.append((abs(rgb_row.timestamp - depth[left].timestamp), rgb_index, left))
            left -= 1
        right = insertion
        while right < len(depth) and depth[right].timestamp - rgb_row.timestamp <= limit:
            candidates.append((abs(rgb_row.timestamp - depth[right].timestamp), rgb_index, right))
            right += 1
    result: dict[int, int] = {}
    used_rgb: set[int] = set()
    used_depth: set[int] = set()
    for _, rgb_index, depth_index in sorted(candidates):
        if rgb_index not in used_rgb and depth_index not in used_depth:
            result[rgb_index] = depth_index
            used_rgb.add(rgb_index)
            used_depth.add(depth_index)
    return result


def _slerp(left_raw: np.ndarray, right_raw: np.ndarray, fraction: float) -> np.ndarray:
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
    return _normalize_quaternion(
        math.sin((1.0 - fraction) * theta) / math.sin(theta) * left
        + math.sin(fraction * theta) / math.sin(theta) * right
    )


def _interpolate(poses: Sequence[PoseRow], timestamp: Decimal, maximum: Decimal) -> tuple[np.ndarray, np.ndarray]:
    times = [row.timestamp for row in poses]
    insertion = bisect_left(times, timestamp)
    if insertion < len(poses) and poses[insertion].timestamp == timestamp:
        row = poses[insertion]
        return row.center_world_m.copy(), row.quaternion_xyzw.copy()
    if insertion == 0 or insertion == len(poses):
        raise ValueError("POSE_NOT_BRACKETED")
    left = poses[insertion - 1]
    right = poses[insertion]
    span = right.timestamp - left.timestamp
    if span > maximum:
        raise ValueError("POSE_BRACKET_GT_0P050_S")
    fraction = float((timestamp - left.timestamp) / span)
    return (
        left.center_world_m + fraction * (right.center_world_m - left.center_world_m),
        _slerp(left.quaternion_xyzw, right.quaternion_xyzw, fraction),
    )


def _rotation(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = _normalize_quaternion(quaternion_xyzw)
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _relative(previous: tuple[np.ndarray, np.ndarray], current: tuple[np.ndarray, np.ndarray], dt_s: float) -> tuple[np.ndarray, np.ndarray, float]:
    previous_center, previous_quaternion = previous
    current_center, current_quaternion = current
    previous_rotation = _rotation(previous_quaternion)
    current_rotation = _rotation(current_quaternion)
    rotation = current_rotation.T @ previous_rotation
    translation = current_rotation.T @ (previous_center - current_center)
    angle = math.acos(float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))
    return rotation, translation, math.degrees(angle / dt_s)


def _decode_depth(raw: bytes, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    try:
        with Image.open(BytesIO(raw)) as image:
            image.load()
            array = np.asarray(image)
    except (OSError, ValueError) as error:
        raise ValueError("DEPTH_MEMBER_MISSING_OR_INVALID") from error
    width, height = config["image_size_wh"]
    if array.shape != (height, width) or array.dtype != np.uint16:
        raise ValueError("DEPTH_MEMBER_MISSING_OR_INVALID")
    stride = int(config["depth_sample_stride_px"])
    ys = np.arange(0, height, stride, dtype=np.int64)
    xs = np.arange(0, width, stride, dtype=np.int64)
    xx, yy = np.meshgrid(xs, ys)
    sampled = array[yy, xx].ravel()
    valid = sampled > 0
    pixels = np.column_stack((xx.ravel()[valid], yy.ravel()[valid])).astype(np.float64)
    depth_m = sampled[valid].astype(np.float64) / float(config["depth_units_per_meter"])
    return pixels, depth_m, float(np.mean(valid))


def _blank(pair_keys: Sequence[str], window_index: int, pair_index: int, previous: IndexRow, current: IndexRow) -> dict[str, Any]:
    row = {key: None for key in pair_keys}
    row.update(
        {
            "window_index": window_index,
            "pair_index": pair_index,
            "previous_rgb_timestamp": str(previous.timestamp),
            "current_rgb_timestamp": str(current.timestamp),
            "dt_s": float(current.timestamp - previous.timestamp),
            "evaluable": False,
        }
    )
    return row


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, **{name: None for name in QUANTILE_NAMES}}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        **{
            name: float(value)
            for name, value in zip(QUANTILE_NAMES, np.quantile(array, QUANTILES), strict=True)
        },
    }


def _window_summary(
    window: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    pb_h1_visibility: Sequence[float],
) -> dict[str, Any]:
    evaluable = [row for row in rows if row["evaluable"]]
    count = len(rows)
    coverage = len(evaluable) / count if count else 0.0
    median_depth = float(np.median([row["valid_depth_fraction"] for row in evaluable])) if evaluable else 0.0
    median_visibility = (
        float(np.median(pb_h1_visibility)) if pb_h1_visibility else 0.0
    )
    if not evaluable:
        disposition = "NO_FORMULA_EVALUABLE_PAIR"
    elif coverage < 0.80:
        disposition = "PAIR_COVERAGE_LT_0P80"
    elif median_depth < 0.50:
        disposition = "SOURCE_DEPTH_COVERAGE_LT_0P50"
    elif median_visibility < 0.50:
        disposition = "PB_H1_VISIBILITY_COVERAGE_LT_0P50"
    else:
        disposition = "EVALUABLE"
    return {
        "window_index": int(window["window_index"]),
        "start_unix_s": str(window["start_unix_s"]),
        "end_unix_s": str(window["end_unix_s"]),
        "candidate_pair_count": count,
        "evaluable_pair_count": len(evaluable),
        "pair_coverage": coverage,
        "median_valid_depth_fraction": median_depth,
        "evaluable": disposition == "EVALUABLE",
        "disposition": disposition,
        "abstention_counts": dict(sorted(Counter(row["abstention_reason"] for row in rows if not row["evaluable"]).items())),
        "distributions": {
            metric: _distribution([float(row[metric]) for row in evaluable])
            for metric in METRIC_KEYS
        },
    }


def produce_archive(archive_path: Path, contract: dict[str, Any], config: dict[str, Any], output_schema: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pair_keys = output_schema["pair_record_keys"]
    windows = contract["canary_cohort"]["window_identity"]
    intrinsic = np.asarray(config["intrinsic"], dtype=np.float64)
    minimum_dt = Decimal(config["minimum_pair_dt_s"])
    maximum_dt = Decimal(config["maximum_pair_dt_s"])
    maximum_pose_bracket = Decimal(config["maximum_pose_bracket_s"])
    with ArchiveReader(archive_path) as archive:
        rgb = _parse_index(archive.read(archive.rgb_member))
        depth = _parse_index(archive.read(archive.depth_member))
        poses = _parse_poses(archive.read(archive.pose_member))
        matches = _associate(rgb, depth, Decimal(config["maximum_rgb_depth_delta_s"]))
        all_rows: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        for window in windows:
            start = Decimal(str(window["start_unix_s"]))
            end = Decimal(str(window["end_unix_s"]))
            indices = [index for index, row in enumerate(rgb) if start <= row.timestamp < end]
            rows: list[dict[str, Any]] = []
            visibility_values: list[float] = []
            candidate_pairs = [
                (previous_index, current_index)
                for previous_index, current_index in zip(indices, indices[1:])
                if minimum_dt
                <= rgb[current_index].timestamp - rgb[previous_index].timestamp
                <= maximum_dt
            ]
            for pair_index, (previous_index, current_index) in enumerate(candidate_pairs):
                previous = rgb[previous_index]
                current = rgb[current_index]
                row = _blank(pair_keys, int(window["window_index"]), pair_index, previous, current)
                dt = current.timestamp - previous.timestamp
                previous_depth_index = matches.get(previous_index)
                current_depth_index = matches.get(current_index)
                if previous_depth_index is None or current_depth_index is None:
                    row["abstention_reason"] = "RGB_DEPTH_UNMATCHED_OR_REUSED"
                    rows.append(row)
                    continue
                previous_depth = depth[previous_depth_index]
                current_depth = depth[current_depth_index]
                row["previous_depth_timestamp"] = str(previous_depth.timestamp)
                row["current_depth_timestamp"] = str(current_depth.timestamp)
                try:
                    previous_pose = _interpolate(poses, previous.timestamp, maximum_pose_bracket)
                    current_pose = _interpolate(poses, current.timestamp, maximum_pose_bracket)
                    rotation, translation, angular_speed = _relative(previous_pose, current_pose, float(dt))
                    pixels, depth_m, valid_depth_fraction = _decode_depth(
                        archive.read(archive.resolve(previous_depth.path)), config
                    )
                    geometry = summarize_translation_induced_geometry(
                        translation_induced_geometry(
                            pixels,
                            depth_m,
                            intrinsic,
                            rotation,
                            translation,
                            float(dt),
                            image_size_wh=tuple(config["image_size_wh"]),
                            minimum_radius_px=float(config["minimum_radius_px"]),
                            zbuffer=True,
                        )
                    )
                    if not geometry["evaluable"]:
                        raise ValueError("PB_H1_NO_VISIBLE_DEPTH_SUPPORT")
                except ValueError as error:
                    reason = str(error)
                    allowed = set(output_schema["pair_abstention_reasons"])
                    if reason not in allowed:
                        raise
                    row["abstention_reason"] = reason
                    rows.append(row)
                    continue
                row.update(
                    {
                        "evaluable": True,
                        "abstention_reason": None,
                        "raw_translation_speed_m_s": float(geometry["raw_translation_speed_m_s"]),
                        "median_angular_speed_deg_s": angular_speed,
                        "median_signed_radial_expansion_per_s": float(geometry["median_signed_radial_expansion_per_s"]),
                        "median_absolute_radial_expansion_per_s": float(geometry["median_absolute_radial_expansion_per_s"]),
                        "radial_expansion_positive_fraction": float(geometry["radial_expansion_positive_fraction"]),
                        "q90_time_normalized_parallax_rad_per_s": float(geometry["q90_time_normalized_parallax_rad_per_s"]),
                        "valid_depth_fraction": valid_depth_fraction,
                    }
                )
                visibility_values.append(float(geometry["valid_fraction"]))
                rows.append(row)
            all_rows.extend(rows)
            summaries.append(_window_summary(window, rows, visibility_values))
    return all_rows, summaries
