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


PAIR_METRICS = (
    "raw_translation_speed_m_s",
    "median_angular_speed_deg_s",
    "median_signed_radial_expansion_per_s",
    "median_absolute_radial_expansion_per_s",
    "radial_expansion_positive_fraction",
    "q90_time_normalized_parallax_rad_per_s",
    "valid_depth_fraction",
)
QUANTILE_POINTS = np.asarray(
    (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0),
    dtype=np.float64,
)
QUANTILE_LABELS = (
    "min",
    "q10",
    "q25",
    "median",
    "q75",
    "q90",
    "q95",
    "max",
)


@dataclass(frozen=True)
class FrameReference:
    time: Decimal
    member: str


@dataclass(frozen=True)
class CameraPose:
    time: Decimal
    center: np.ndarray
    quaternion: np.ndarray


def _content_lines(payload: bytes) -> Iterable[list[str]]:
    for line in payload.decode("utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            yield value.split()


def _index(payload: bytes) -> list[FrameReference]:
    result: list[FrameReference] = []
    for fields in _content_lines(payload):
        if len(fields) != 2:
            raise ValueError("TUM_INDEX_COLUMNS")
        member = PurePosixPath(fields[1])
        if member.is_absolute() or ".." in member.parts:
            raise ValueError("TUM_INDEX_UNSAFE_PATH")
        result.append(FrameReference(Decimal(fields[0]), member.as_posix()))
    if not result or any(
        left.time >= right.time for left, right in zip(result, result[1:])
    ):
        raise ValueError("TUM_INDEX_NOT_STRICTLY_MONOTONIC")
    if len({row.member for row in result}) != len(result):
        raise ValueError("TUM_INDEX_DUPLICATE_PATH")
    return result


def _unit_quaternion(raw: np.ndarray) -> np.ndarray:
    value = np.asarray(raw, dtype=np.float64)
    norm = float(np.sqrt(np.dot(value, value)))
    if (
        value.shape != (4,)
        or not np.all(np.isfinite(value))
        or norm <= 0.0
    ):
        raise ValueError("POSE_INVALID_QUATERNION")
    return value / norm


def _poses(payload: bytes) -> list[CameraPose]:
    last_by_time: dict[Decimal, CameraPose] = {}
    prior: Decimal | None = None
    for fields in _content_lines(payload):
        if len(fields) != 8:
            raise ValueError("TUM_POSE_COLUMNS")
        time = Decimal(fields[0])
        if prior is not None and time < prior:
            raise ValueError("TUM_POSE_NOT_MONOTONIC")
        values = np.array(tuple(map(float, fields[1:])), dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("POSE_INVALID_QUATERNION")
        last_by_time[time] = CameraPose(time, values[:3], values[3:])
        prior = time
    if not last_by_time:
        raise ValueError("TUM_POSE_EMPTY")
    return list(last_by_time.values())


class RawTumBundle:
    def __init__(self, path: Path) -> None:
        self.bundle = tarfile.open(path, "r:*")
        files = [item for item in self.bundle.getmembers() if item.isfile()]
        names = [item.name for item in files]
        if len(names) != len(set(names)):
            raise ValueError("TUM_ARCHIVE_DUPLICATE_MEMBER")
        if any(
            PurePosixPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts
            for name in names
        ):
            raise ValueError("TUM_ARCHIVE_UNSAFE_MEMBER")
        self.files = {item.name: item for item in files}
        self.rgb = self._control("rgb.txt")
        self.depth = self._control("depth.txt")
        self.pose = self._control("groundtruth.txt")
        control_roots = {
            name.rsplit("/", 1)[0] if "/" in name else ""
            for name in (self.rgb, self.depth, self.pose)
        }
        if len(control_roots) != 1:
            raise ValueError("TUM_ARCHIVE_CONTROL_ROOT_MISMATCH")
        self.root = control_roots.pop()

    def _control(self, basename: str) -> str:
        matches = [
            name
            for name in self.files
            if name == basename or name.endswith("/" + basename)
        ]
        if len(matches) != 1:
            raise ValueError(f"TUM_ARCHIVE_CONTROL_MEMBER:{basename}")
        return matches[0]

    def bytes(self, name: str) -> bytes:
        stream = self.bundle.extractfile(self.files[name])
        if stream is None:
            raise ValueError("TUM_ARCHIVE_MEMBER_UNREADABLE")
        return stream.read()

    def depth_bytes(self, relative: str) -> bytes:
        name = f"{self.root}/{relative}" if self.root else relative
        if name not in self.files:
            raise ValueError("DEPTH_MEMBER_MISSING_OR_INVALID")
        return self.bytes(name)

    def __enter__(self) -> RawTumBundle:
        return self

    def __exit__(self, *_: object) -> None:
        self.bundle.close()


def _nearest_unique(
    left: Sequence[FrameReference],
    right: Sequence[FrameReference],
    tolerance: Decimal,
) -> dict[int, int]:
    right_times = [item.time for item in right]
    options: list[tuple[Decimal, int, int]] = []
    for left_index, item in enumerate(left):
        pivot = bisect_left(right_times, item.time)
        candidate = pivot - 1
        while (
            candidate >= 0
            and item.time - right[candidate].time <= tolerance
        ):
            options.append(
                (
                    abs(item.time - right[candidate].time),
                    left_index,
                    candidate,
                )
            )
            candidate -= 1
        candidate = pivot
        while (
            candidate < len(right)
            and right[candidate].time - item.time <= tolerance
        ):
            options.append(
                (
                    abs(item.time - right[candidate].time),
                    left_index,
                    candidate,
                )
            )
            candidate += 1
    assigned_left: set[int] = set()
    assigned_right: set[int] = set()
    assignments: dict[int, int] = {}
    for _, left_index, right_index in sorted(options):
        if left_index in assigned_left or right_index in assigned_right:
            continue
        assignments[left_index] = right_index
        assigned_left.add(left_index)
        assigned_right.add(right_index)
    return assignments


def _quaternion_at(
    rows: Sequence[CameraPose],
    time: Decimal,
    max_span: Decimal,
) -> tuple[np.ndarray, np.ndarray]:
    times = [row.time for row in rows]
    place = bisect_left(times, time)
    if place < len(rows) and rows[place].time == time:
        return rows[place].center.copy(), rows[place].quaternion.copy()
    if place == 0 or place == len(rows):
        raise ValueError("POSE_NOT_BRACKETED")
    before, after = rows[place - 1], rows[place]
    span = after.time - before.time
    if span > max_span:
        raise ValueError("POSE_BRACKET_GT_0P050_S")
    weight = float((time - before.time) / span)
    first = _unit_quaternion(before.quaternion)
    second = _unit_quaternion(after.quaternion)
    cosine = float(np.dot(first, second))
    if cosine < 0.0:
        second = -second
        cosine = -cosine
    cosine = float(np.clip(cosine, -1.0, 1.0))
    if cosine > 0.9995:
        orientation = _unit_quaternion(
            first + weight * (second - first)
        )
    else:
        angle = math.acos(cosine)
        orientation = _unit_quaternion(
            math.sin((1.0 - weight) * angle) / math.sin(angle) * first
            + math.sin(weight * angle) / math.sin(angle) * second
        )
    center = before.center + weight * (after.center - before.center)
    return center, orientation


def _matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = _unit_quaternion(quaternion)
    return np.array(
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


def _motion(
    previous: tuple[np.ndarray, np.ndarray],
    current: tuple[np.ndarray, np.ndarray],
    dt: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    previous_rotation = _matrix(previous[1])
    current_rotation = _matrix(current[1])
    rotation = current_rotation.T @ previous_rotation
    translation = current_rotation.T @ (previous[0] - current[0])
    radians = math.acos(
        float(
            np.clip(
                (np.trace(rotation) - 1.0) / 2.0,
                -1.0,
                1.0,
            )
        )
    )
    return rotation, translation, math.degrees(radians / dt)


def _depth(
    payload: bytes,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, float]:
    try:
        with Image.open(BytesIO(payload)) as image:
            image.load()
            array = np.asarray(image)
    except (OSError, ValueError) as error:
        raise ValueError("DEPTH_MEMBER_MISSING_OR_INVALID") from error
    width, height = map(int, config["image_size_wh"])
    if array.shape != (height, width) or array.dtype != np.uint16:
        raise ValueError("DEPTH_MEMBER_MISSING_OR_INVALID")
    step = int(config["depth_sample_stride_px"])
    grid_y, grid_x = np.mgrid[0:height:step, 0:width:step]
    samples = array[grid_y, grid_x].reshape(-1)
    available = samples > 0
    pixels = np.stack(
        (
            grid_x.reshape(-1)[available],
            grid_y.reshape(-1)[available],
        ),
        axis=1,
    ).astype(np.float64)
    depths = (
        samples[available].astype(np.float64)
        / np.float64(config["depth_units_per_meter"])
    )
    return pixels, depths, float(np.mean(available))


def _independent_geometry(
    pixels: np.ndarray,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    dt: float,
    config: dict[str, Any],
) -> tuple[dict[str, float], float]:
    homogeneous = np.column_stack(
        (pixels, np.ones(pixels.shape[0], dtype=np.float64))
    )
    points_before = (
        np.linalg.inv(intrinsic) @ homogeneous.T
    ).T * depth[:, None]
    rotated = (rotation @ points_before.T).T
    moved = rotated + translation[None, :]
    projected_rotated = (intrinsic @ rotated.T).T
    projected_moved = (intrinsic @ moved.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv_rotated = projected_rotated[:, :2] / projected_rotated[:, 2:3]
        uv_moved = projected_moved[:, :2] / projected_moved[:, 2:3]
    width, height = map(int, config["image_size_wh"])
    center = np.array(
        (intrinsic[0, 2], intrinsic[1, 2]),
        dtype=np.float64,
    )
    radius_rotated = np.linalg.norm(
        uv_rotated - center[None, :],
        axis=1,
    )
    radius_moved = np.linalg.norm(
        uv_moved - center[None, :],
        axis=1,
    )
    valid = (
        (depth > 0.0)
        & (rotated[:, 2] > 0.0)
        & (moved[:, 2] > 0.0)
        & (uv_rotated[:, 0] >= 0.0)
        & (uv_rotated[:, 0] < width)
        & (uv_rotated[:, 1] >= 0.0)
        & (uv_rotated[:, 1] < height)
        & (uv_moved[:, 0] >= 0.0)
        & (uv_moved[:, 0] < width)
        & (uv_moved[:, 1] >= 0.0)
        & (uv_moved[:, 1] < height)
        & np.isfinite(radius_rotated)
        & np.isfinite(radius_moved)
        & (radius_rotated >= float(config["minimum_radius_px"]))
        & (radius_moved > 0.0)
    )
    winners: dict[tuple[int, int], int] = {}
    for index in np.flatnonzero(valid):
        target = tuple(
            np.floor(uv_moved[index] + 0.5)
            .astype(np.int64)
            .tolist()
        )
        existing = winners.get(target)
        if existing is None or moved[index, 2] < moved[existing, 2]:
            winners[target] = int(index)
    selected = np.asarray(sorted(winners.values()), dtype=np.int64)
    if selected.size == 0:
        raise ValueError("PB_H1_NO_VISIBLE_DEPTH_SUPPORT")
    rotated_selected = rotated[selected]
    moved_selected = moved[selected]
    rotated_unit = rotated_selected / np.linalg.norm(
        rotated_selected,
        axis=1,
        keepdims=True,
    )
    moved_unit = moved_selected / np.linalg.norm(
        moved_selected,
        axis=1,
        keepdims=True,
    )
    dot = np.sum(rotated_unit * moved_unit, axis=1)
    cross = np.linalg.norm(
        np.cross(rotated_unit, moved_unit),
        axis=1,
    )
    parallax = (
        np.arctan2(cross, np.clip(dot, -1.0, 1.0)) / dt
    )
    radial = (
        np.log(radius_moved[selected] / radius_rotated[selected]) / dt
    )
    return (
        {
            "raw_translation_speed_m_s": float(
                np.linalg.norm(translation) / dt
            ),
            "median_signed_radial_expansion_per_s": float(
                np.median(radial)
            ),
            "median_absolute_radial_expansion_per_s": float(
                np.median(np.abs(radial))
            ),
            "radial_expansion_positive_fraction": float(
                np.mean(radial > 0.0)
            ),
            "q90_time_normalized_parallax_rad_per_s": float(
                np.quantile(parallax, 0.90)
            ),
        },
        float(
            selected.size / pixels.shape[0]
            if pixels.shape[0]
            else 0.0
        ),
    )


def _empty_row(
    keys: Sequence[str],
    window_index: int,
    pair_index: int,
    before: FrameReference,
    after: FrameReference,
) -> dict[str, Any]:
    result = {key: None for key in keys}
    result.update(
        {
            "window_index": window_index,
            "pair_index": pair_index,
            "previous_rgb_timestamp": str(before.time),
            "current_rgb_timestamp": str(after.time),
            "dt_s": float(after.time - before.time),
            "evaluable": False,
        }
    )
    return result


def _summary_distribution(
    values: Sequence[float],
) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            **{label: None for label in QUANTILE_LABELS},
        }
    array = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(array, QUANTILE_POINTS)
    return {
        "count": int(array.size),
        **{
            label: float(value)
            for label, value in zip(
                QUANTILE_LABELS,
                quantiles,
                strict=True,
            )
        },
    }


def _summarize(
    window: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    visibility: Sequence[float],
) -> dict[str, Any]:
    accepted = [row for row in rows if row["evaluable"]]
    coverage = len(accepted) / len(rows) if rows else 0.0
    depth_median = (
        float(
            np.median(
                [row["valid_depth_fraction"] for row in accepted]
            )
        )
        if accepted
        else 0.0
    )
    visibility_median = (
        float(np.median(visibility)) if visibility else 0.0
    )
    if not accepted:
        disposition = "NO_FORMULA_EVALUABLE_PAIR"
    elif coverage < 0.80:
        disposition = "PAIR_COVERAGE_LT_0P80"
    elif depth_median < 0.50:
        disposition = "SOURCE_DEPTH_COVERAGE_LT_0P50"
    elif visibility_median < 0.50:
        disposition = "PB_H1_VISIBILITY_COVERAGE_LT_0P50"
    else:
        disposition = "EVALUABLE"
    return {
        "window_index": int(window["window_index"]),
        "start_unix_s": str(window["start_unix_s"]),
        "end_unix_s": str(window["end_unix_s"]),
        "candidate_pair_count": len(rows),
        "evaluable_pair_count": len(accepted),
        "pair_coverage": coverage,
        "median_valid_depth_fraction": depth_median,
        "evaluable": disposition == "EVALUABLE",
        "disposition": disposition,
        "abstention_counts": dict(
            sorted(
                Counter(
                    row["abstention_reason"]
                    for row in rows
                    if not row["evaluable"]
                ).items()
            )
        ),
        "distributions": {
            metric: _summary_distribution(
                [float(row[metric]) for row in accepted]
            )
            for metric in PAIR_METRICS
        },
    }


def replay_archive(
    archive_path: Path,
    contract: dict[str, Any],
    config: dict[str, Any],
    output_schema: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    keys = output_schema["pair_record_keys"]
    minimum_dt = Decimal(config["minimum_pair_dt_s"])
    maximum_dt = Decimal(config["maximum_pair_dt_s"])
    with RawTumBundle(archive_path) as bundle:
        rgb = _index(bundle.bytes(bundle.rgb))
        depth = _index(bundle.bytes(bundle.depth))
        poses = _poses(bundle.bytes(bundle.pose))
        depth_for_rgb = _nearest_unique(
            rgb,
            depth,
            Decimal(config["maximum_rgb_depth_delta_s"]),
        )
        intrinsic = np.asarray(config["intrinsic"], dtype=np.float64)
        all_rows: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        for window in contract["canary_cohort"]["window_identity"]:
            start = Decimal(str(window["start_unix_s"]))
            end = Decimal(str(window["end_unix_s"]))
            frame_indices = [
                index
                for index, frame in enumerate(rgb)
                if start <= frame.time < end
            ]
            candidates = [
                (before, after)
                for before, after in zip(
                    frame_indices,
                    frame_indices[1:],
                )
                if minimum_dt
                <= rgb[after].time - rgb[before].time
                <= maximum_dt
            ]
            rows: list[dict[str, Any]] = []
            visibility: list[float] = []
            for pair_index, (before_index, after_index) in enumerate(
                candidates
            ):
                before, after = rgb[before_index], rgb[after_index]
                row = _empty_row(
                    keys,
                    int(window["window_index"]),
                    pair_index,
                    before,
                    after,
                )
                before_depth_index = depth_for_rgb.get(before_index)
                after_depth_index = depth_for_rgb.get(after_index)
                if (
                    before_depth_index is None
                    or after_depth_index is None
                ):
                    row["abstention_reason"] = (
                        "RGB_DEPTH_UNMATCHED_OR_REUSED"
                    )
                    rows.append(row)
                    continue
                before_depth = depth[before_depth_index]
                after_depth = depth[after_depth_index]
                row["previous_depth_timestamp"] = str(before_depth.time)
                row["current_depth_timestamp"] = str(after_depth.time)
                try:
                    dt = float(after.time - before.time)
                    previous_pose = _quaternion_at(
                        poses,
                        before.time,
                        Decimal(config["maximum_pose_bracket_s"]),
                    )
                    current_pose = _quaternion_at(
                        poses,
                        after.time,
                        Decimal(config["maximum_pose_bracket_s"]),
                    )
                    rotation, translation, angular = _motion(
                        previous_pose,
                        current_pose,
                        dt,
                    )
                    pixels, depths, source_fraction = _depth(
                        bundle.depth_bytes(before_depth.member),
                        config,
                    )
                    metrics, visible_fraction = _independent_geometry(
                        pixels,
                        depths,
                        intrinsic,
                        rotation,
                        translation,
                        dt,
                        config,
                    )
                except ValueError as error:
                    reason = str(error)
                    if reason not in set(
                        output_schema["pair_abstention_reasons"]
                    ):
                        raise
                    row["abstention_reason"] = reason
                    rows.append(row)
                    continue
                row.update(
                    {
                        "evaluable": True,
                        "abstention_reason": None,
                        **metrics,
                        "median_angular_speed_deg_s": angular,
                        "valid_depth_fraction": source_fraction,
                    }
                )
                rows.append(row)
                visibility.append(visible_fraction)
            all_rows.extend(rows)
            summaries.append(_summarize(window, rows, visibility))
    return all_rows, summaries


def _within(
    left: float,
    right: float,
    absolute: float,
    relative: float,
) -> bool:
    difference = abs(left - right)
    denominator = max(abs(left), abs(right))
    return difference <= absolute or (
        denominator > 0.0
        and difference / denominator <= relative
    )


def validate_materialized(
    archive_path: Path,
    contract: dict[str, Any],
    config: dict[str, Any],
    output_schema: dict[str, Any],
    producer_rows: Sequence[dict[str, Any]],
    producer_summaries: Sequence[dict[str, Any]],
    *,
    enforce_frozen_counts: bool,
) -> dict[str, Any]:
    replay_rows, replay_summaries = replay_archive(
        archive_path,
        contract,
        config,
        output_schema,
    )
    identity_fields = config["pair_identity_fields"]
    expected_keys = set(output_schema["pair_record_keys"])
    identity_mismatch = 0
    key_mismatch = 0
    abstention_mismatch = 0
    numeric_mismatch = 0
    relaxed_mismatch = 0
    first: str | None = None

    def type_valid(value: Any, allowed: Sequence[str]) -> bool:
        if value is None:
            return "null" in allowed
        if isinstance(value, bool):
            return "boolean" in allowed
        if isinstance(value, int):
            return "integer" in allowed
        if isinstance(value, float):
            return "number" in allowed and math.isfinite(value)
        if isinstance(value, str):
            return "string" in allowed
        if isinstance(value, dict):
            return "object" in allowed
        return False

    def remember(message: str) -> None:
        nonlocal first
        if first is None:
            first = message

    if len(producer_rows) != len(replay_rows):
        remember("PAIR_RECORD_COUNT")
    for index in range(max(len(producer_rows), len(replay_rows))):
        if index >= len(producer_rows) or index >= len(replay_rows):
            identity_mismatch += 1
            continue
        produced = producer_rows[index]
        replayed = replay_rows[index]
        if set(produced) != expected_keys or set(replayed) != expected_keys:
            key_mismatch += 1
            remember(f"PAIR_KEY_SET:{index}")
        else:
            type_contract = output_schema["pair_record_type_contract"]
            if any(
                not type_valid(produced[field], type_contract[field])
                or not type_valid(replayed[field], type_contract[field])
                for field in expected_keys
            ):
                key_mismatch += 1
                remember(f"PAIR_TYPE_CONTRACT:{index}")
        if tuple(
            produced.get(field) for field in identity_fields
        ) != tuple(replayed.get(field) for field in identity_fields):
            identity_mismatch += 1
            remember(f"PAIR_IDENTITY:{index}")
        if (
            produced.get("evaluable") != replayed.get("evaluable")
            or produced.get("abstention_reason")
            != replayed.get("abstention_reason")
        ):
            abstention_mismatch += 1
            remember(f"PAIR_ABSTENTION:{index}")
        for metric in output_schema["numeric_pair_metrics"]:
            left = produced.get(metric)
            right = replayed.get(metric)
            if left is None or right is None:
                if left is not right:
                    numeric_mismatch += 1
                    relaxed_mismatch += 1
                    remember(f"PAIR_NUMERIC_NULL:{index}:{metric}")
                continue
            if not _within(
                float(left),
                float(right),
                float(config["strict_absolute_tolerance"]),
                float(config["strict_relative_tolerance"]),
            ):
                numeric_mismatch += 1
                remember(f"PAIR_NUMERIC:{index}:{metric}")
            if not _within(
                float(left),
                float(right),
                float(config["relaxed_absolute_tolerance"]),
                float(config["relaxed_relative_tolerance"]),
            ):
                relaxed_mismatch += 1

    expected_window_keys = set(output_schema["window_summary_keys"])
    if len(producer_summaries) != len(replay_summaries):
        abstention_mismatch += abs(
            len(producer_summaries) - len(replay_summaries)
        )
        remember("WINDOW_COUNT")
    for index, (produced, replayed) in enumerate(
        zip(producer_summaries, replay_summaries)
    ):
        if (
            set(produced) != expected_window_keys
            or set(replayed) != expected_window_keys
        ):
            key_mismatch += 1
            remember(f"WINDOW_KEY_SET:{index}")
        else:
            window_type_contract = output_schema[
                "window_summary_type_contract"
            ]
            if any(
                not type_valid(
                    produced[field],
                    window_type_contract[field],
                )
                or not type_valid(
                    replayed[field],
                    window_type_contract[field],
                )
                for field in expected_window_keys
            ):
                key_mismatch += 1
                remember(f"WINDOW_TYPE_CONTRACT:{index}")
        if (
            produced.get("window_index") != replayed.get("window_index")
            or produced.get("start_unix_s")
            != replayed.get("start_unix_s")
            or produced.get("end_unix_s")
            != replayed.get("end_unix_s")
            or produced.get("candidate_pair_count")
            != replayed.get("candidate_pair_count")
            or produced.get("evaluable_pair_count")
            != replayed.get("evaluable_pair_count")
            or produced.get("evaluable") != replayed.get("evaluable")
            or produced.get("disposition")
            != replayed.get("disposition")
            or produced.get("abstention_counts")
            != replayed.get("abstention_counts")
        ):
            abstention_mismatch += 1
            remember(f"WINDOW_DISPOSITION:{index}")
        for scalar in (
            "pair_coverage",
            "median_valid_depth_fraction",
        ):
            left = produced.get(scalar)
            right = replayed.get(scalar)
            if (
                not isinstance(left, (int, float))
                or isinstance(left, bool)
                or not isinstance(right, (int, float))
                or isinstance(right, bool)
                or not _within(
                    float(left),
                    float(right),
                    float(config["strict_absolute_tolerance"]),
                    float(config["strict_relative_tolerance"]),
                )
            ):
                numeric_mismatch += 1
                remember(f"WINDOW_NUMERIC:{index}:{scalar}")
        produced_distributions = produced.get("distributions", {})
        replayed_distributions = replayed.get("distributions", {})
        expected_metric_keys = set(PAIR_METRICS)
        if (
            set(produced_distributions) != expected_metric_keys
            or set(replayed_distributions) != expected_metric_keys
        ):
            key_mismatch += 1
            remember(f"WINDOW_DISTRIBUTION_KEYS:{index}")
        for metric in expected_metric_keys & set(
            produced_distributions
        ) & set(replayed_distributions):
            produced_distribution = produced_distributions[metric]
            replayed_distribution = replayed_distributions[metric]
            expected_distribution_keys = set(
                output_schema["distribution_keys"]
            )
            if (
                set(produced_distribution)
                != expected_distribution_keys
                or set(replayed_distribution)
                != expected_distribution_keys
            ):
                key_mismatch += 1
                remember(
                    f"WINDOW_DISTRIBUTION_SHAPE:{index}:{metric}"
                )
                continue
            for distribution, label in (
                (produced_distribution, "PRODUCER"),
                (replayed_distribution, "VALIDATOR"),
            ):
                count = distribution["count"]
                if (
                    not isinstance(count, int)
                    or isinstance(count, bool)
                    or count < 0
                    or count
                    != (
                        produced.get("evaluable_pair_count")
                        if label == "PRODUCER"
                        else replayed.get("evaluable_pair_count")
                    )
                ):
                    key_mismatch += 1
                    remember(
                        "WINDOW_DISTRIBUTION_COUNT_TYPE:"
                        f"{index}:{metric}:{label}"
                    )
                for quantile in output_schema[
                    "distribution_keys"
                ][1:]:
                    value = distribution[quantile]
                    valid_quantile = (
                        value is None
                        if count == 0
                        else (
                            isinstance(value, float)
                            and math.isfinite(value)
                        )
                    )
                    if not valid_quantile:
                        key_mismatch += 1
                        remember(
                            "WINDOW_DISTRIBUTION_QUANTILE_TYPE:"
                            f"{index}:{metric}:{quantile}:{label}"
                        )
            if (
                produced_distribution["count"]
                != replayed_distribution["count"]
            ):
                abstention_mismatch += 1
                remember(
                    f"WINDOW_DISTRIBUTION_COUNT:{index}:{metric}"
                )
            for quantile in output_schema["distribution_keys"][1:]:
                left = produced_distribution[quantile]
                right = replayed_distribution[quantile]
                if left is None or right is None:
                    if left is not right:
                        numeric_mismatch += 1
                        remember(
                            "WINDOW_DISTRIBUTION_NULL:"
                            f"{index}:{metric}:{quantile}"
                        )
                elif not _within(
                    float(left),
                    float(right),
                    float(config["strict_absolute_tolerance"]),
                    float(config["strict_relative_tolerance"]),
                ):
                    numeric_mismatch += 1
                    remember(
                        "WINDOW_DISTRIBUTION_NUMERIC:"
                        f"{index}:{metric}:{quantile}"
                    )
        allowed_reasons = set(
            output_schema["pair_abstention_reasons"]
        )
        for summary, rows_label in (
            (produced, "PRODUCER"),
            (replayed, "VALIDATOR"),
        ):
            counts = summary.get("abstention_counts", {})
            valid_counts = (
                isinstance(counts, dict)
                and set(counts).issubset(allowed_reasons)
                and all(
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value >= 1
                    for value in counts.values()
                )
                and sum(counts.values())
                == summary.get("candidate_pair_count", 0)
                - summary.get("evaluable_pair_count", 0)
            )
            if not valid_counts:
                key_mismatch += 1
                remember(
                    f"WINDOW_ABSTENTION_COUNTS:{index}:{rows_label}"
                )

    branch_mismatch = 0
    if enforce_frozen_counts:
        expected_windows = {
            int(item["window_index"]): (
                int(item["candidate_pair_count"]),
                int(item["prior_evaluable_pair_count"]),
                str(item["prior_window_disposition"]),
            )
            for item in contract["canary_cohort"]["window_identity"]
        }
        actual_windows = {
            int(item["window_index"]): (
                int(item["candidate_pair_count"]),
                int(item["evaluable_pair_count"]),
                str(item["disposition"]),
            )
            for item in producer_summaries
        }
        for window_index, expected in expected_windows.items():
            if actual_windows.get(window_index) != expected:
                branch_mismatch += 1
                remember(f"EXPECTED_BRANCH:{window_index}")
        denominator = int(
            contract["canary_cohort"]["candidate_pair_denominator"]
        )
        if (
            len(producer_rows) != denominator
            or len(replay_rows) != denominator
        ):
            branch_mismatch += 1
            remember("EXPECTED_DENOMINATOR")

    counts = {
        "pair_identity_mismatch_count": identity_mismatch,
        "pair_record_key_set_mismatch_count": key_mismatch,
        "abstention_or_window_disposition_mismatch_count": (
            abstention_mismatch
        ),
        "numeric_metric_parity_violation_count": numeric_mismatch,
        "expected_branch_or_count_mismatch_count": branch_mismatch,
    }
    errors = [
        f"{name}:{value}"
        for name, value in counts.items()
        if value
    ]
    gate_pass = not errors
    return {
        "schema_version": (
            "rcle.real_data_geometry_canary.validation.v1"
        ),
        "status": "VALID",
        "terminal": (
            contract["result_model"]["pass_terminal"]
            if gate_pass
            else contract["result_model"]["nonpass_terminal"]
        ),
        "gate_pass": gate_pass,
        "producer_pair_record_count": len(producer_rows),
        "validator_pair_record_count": len(replay_rows),
        "emitted_window_count": len(producer_summaries),
        **counts,
        "relaxed_numeric_metric_parity_violation_count": (
            relaxed_mismatch
        ),
        "first_mismatch": first,
        "errors": errors,
    }


def read_pair_ledger(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("PAIR_LEDGER_OBJECT_REQUIRED")
        rows.append(value)
    return rows


def validate_receipt(
    receipt: dict[str, Any],
    pair_ledger_bytes: bytes,
    window_summary_bytes: bytes,
    output_schema: dict[str, Any],
    expected_bindings: dict[str, str],
    implementation_lock_sha256: str,
    protocol_id: str,
    pair_record_count: int,
    window_count: int,
) -> list[str]:
    errors: list[str] = []
    if set(receipt) != set(output_schema["receipt_keys"]):
        errors.append("RECEIPT_KEY_SET")
    expected = {
        "schema_version": "rcle.real_data_geometry_canary.receipt.v1",
        "protocol_id": protocol_id,
        **expected_bindings,
        "implementation_lock_sha256": implementation_lock_sha256,
        "pair_ledger_sha256": sha256(pair_ledger_bytes).hexdigest(),
        "window_summary_sha256": sha256(window_summary_bytes).hexdigest(),
        "pair_record_count": pair_record_count,
        "window_count": window_count,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            errors.append(f"RECEIPT_BINDING:{key}")
    return errors


def validate_bound_receipt_files(
    repo_root: Path,
    config: dict[str, Any],
    contract: dict[str, Any],
    receipt: dict[str, Any],
    pair_ledger_bytes: bytes,
    window_summary_bytes: bytes,
    output_schema: dict[str, Any],
    implementation_lock_sha256: str,
) -> list[str]:
    def file_hash(path: Path) -> str:
        digest = sha256()
        with path.open("rb") as handle:
            for chunk in iter(
                lambda: handle.read(1024 * 1024),
                b"",
            ):
                digest.update(chunk)
        return digest.hexdigest()

    bindings = contract["prior_evidence_bindings"]
    paths = {
        "contract_sha256": repo_root / config["protocol_contract"],
        "archive_sha256": repo_root / config["source_archive"],
        "source_audit_contract_sha256": (
            repo_root / config["source_audit_contract"]
        ),
        "source_audit_result_sha256": (
            repo_root / config["source_audit_result"]
        ),
        "pb_h1_geometry_sha256": (
            repo_root / config["pb_h1_geometry"]
        ),
    }
    expected_authority = {
        "archive_sha256": bindings["source_archive_sha256"],
        "source_audit_contract_sha256": (
            bindings["source_audit_contract"]["sha256"]
        ),
        "source_audit_result_sha256": (
            bindings["source_audit_result_sha256"]
        ),
        "pb_h1_geometry_sha256": (
            bindings["pb_h1_geometry_implementation_sha256"]
        ),
    }
    errors: list[str] = []
    actual: dict[str, str] = {}
    for key, path in paths.items():
        if not path.is_file():
            errors.append(f"RECEIPT_BOUND_FILE_MISSING:{key}")
            continue
        actual[key] = file_hash(path)
        expected = expected_authority.get(key)
        if expected is not None and actual[key] != expected:
            errors.append(f"RECEIPT_BOUND_FILE_HASH:{key}")
    if len(actual) == len(paths):
        errors.extend(
            validate_receipt(
                receipt,
                pair_ledger_bytes,
                window_summary_bytes,
                output_schema,
                actual,
                implementation_lock_sha256,
                str(contract["protocol_id"]),
                len(pair_ledger_bytes.splitlines()),
                len(json.loads(window_summary_bytes.decode("utf-8"))),
            )
        )
    return errors
