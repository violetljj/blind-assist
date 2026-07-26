from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from io import BytesIO
import math
from pathlib import PurePosixPath
import re
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image


ASCII_WHITESPACE = b" \t\r\n\v\f"
NUMBER_RE = re.compile(
    rb"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
)
K = np.asarray(
    (
        (542.822841, 0.0, 315.593520),
        (0.0, 542.576870, 237.756098),
        (0.0, 0.0, 1.0),
    ),
    dtype=np.float64,
)
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
DEPTH_UNITS_PER_METER = 5000.0


@dataclass(frozen=True)
class IndexRow:
    timestamp: Decimal
    path: str
    source_row_rank: int


@dataclass(frozen=True)
class PoseRow:
    timestamp: Decimal
    translation: np.ndarray
    quaternion_xyzw: np.ndarray
    source_row_rank: int


def parse_decimal(token: bytes) -> Decimal:
    if NUMBER_RE.fullmatch(token) is None:
        raise ValueError("B1A_NUMERIC_GRAMMAR")
    try:
        value = Decimal(token.decode("ascii"))
    except (UnicodeDecodeError, InvalidOperation) as error:
        raise ValueError("B1A_NUMERIC_PARSE") from error
    if not value.is_finite():
        raise ValueError("B1A_NUMERIC_NONFINITE")
    return value


def validate_member_reference(path: str) -> str:
    if (
        not path
        or "\\" in path
        or "\x00" in path
        or any(character.isspace() for character in path)
    ):
        raise ValueError("B1A_MEMBER_REFERENCE_UNSAFE")
    path.encode("utf-8", errors="strict")
    pure = PurePosixPath(path)
    if (
        pure.is_absolute()
        or any(part in ("", ".", "..") for part in pure.parts)
        or any(":" in part for part in pure.parts)
        or str(pure) != path
    ):
        raise ValueError("B1A_MEMBER_REFERENCE_NONCANONICAL")
    return path


def _data_lines(raw: bytes) -> Iterable[list[bytes]]:
    for line in raw.split(b"\n"):
        stripped = line.strip(ASCII_WHITESPACE)
        if not stripped or stripped.startswith(b"#"):
            continue
        yield re.split(rb"[\x09-\x0d\x20]+", stripped)


def parse_index_text(raw: bytes) -> list[IndexRow]:
    rows: list[IndexRow] = []
    for tokens in _data_lines(raw):
        if len(tokens) != 2:
            raise ValueError("B1A_INDEX_TOKEN_COUNT")
        timestamp = parse_decimal(tokens[0])
        try:
            path = tokens[1].decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ValueError("B1A_PATH_UTF8") from error
        validate_member_reference(path)
        if rows and timestamp <= rows[-1].timestamp:
            raise ValueError("B1A_TIMESTAMP_NOT_STRICTLY_INCREASING")
        rows.append(IndexRow(timestamp, path, len(rows)))
    if not rows:
        raise ValueError("B1A_INDEX_EMPTY")
    return rows


def parse_pose_text(raw: bytes) -> list[PoseRow]:
    rows: list[PoseRow] = []
    for tokens in _data_lines(raw):
        if len(tokens) != 8:
            raise ValueError("B1A_POSE_TOKEN_COUNT")
        timestamp = parse_decimal(tokens[0])
        numeric = np.asarray(
            [float(parse_decimal(token)) for token in tokens[1:]],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(numeric)):
            raise ValueError("B1A_POSE_NONFINITE")
        if rows and timestamp <= rows[-1].timestamp:
            raise ValueError("B1A_TIMESTAMP_NOT_STRICTLY_INCREASING")
        rows.append(PoseRow(timestamp, numeric[:3], numeric[3:], len(rows)))
    if not rows:
        raise ValueError("B1A_POSE_EMPTY")
    return rows


def rows_in_window(
    rows: Sequence[IndexRow], start: Decimal, end: Decimal
) -> list[IndexRow]:
    return [row for row in rows if start <= row.timestamp < end]


def adjacent_pairs(
    rows: Sequence[IndexRow], start: Decimal, end: Decimal
) -> list[dict[str, Any]]:
    inside = rows_in_window(rows, start, end)
    result: list[dict[str, Any]] = []
    for previous, current in zip(inside, inside[1:], strict=False):
        dt = current.timestamp - previous.timestamp
        result.append(
            {
                "previous": previous,
                "current": current,
                "dt": dt,
                "candidate": Decimal("0.020") <= dt <= Decimal("0.050"),
            }
        )
    return result


def assign_depth_rows(
    rgb_rows: Sequence[IndexRow],
    depth_rows: Sequence[IndexRow],
    start: Decimal,
    end: Decimal,
) -> dict[int, IndexRow | None]:
    available = {
        row.source_row_rank: row
        for row in depth_rows
        if start <= row.timestamp < end
    }
    assignments: dict[int, IndexRow | None] = {}
    for rgb in rows_in_window(rgb_rows, start, end):
        candidates = [
            row
            for row in available.values()
            if abs(row.timestamp - rgb.timestamp) <= Decimal("0.020")
        ]
        if not candidates:
            assignments[rgb.source_row_rank] = None
            continue
        selected = min(
            candidates,
            key=lambda row: (
                abs(row.timestamp - rgb.timestamp),
                row.timestamp,
                row.source_row_rank,
            ),
        )
        assignments[rgb.source_row_rank] = selected
        del available[selected.source_row_rank]
    return assignments


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,) or not np.all(np.isfinite(value)):
        raise ValueError("B1A_QUATERNION_SHAPE_OR_FINITE")
    norm = float(np.linalg.norm(value))
    if norm <= 0.0 or abs(norm - 1.0) > 0.001:
        raise ValueError("B1A_QUATERNION_NORM")
    return value / norm


def slerp(left_q: np.ndarray, right_q: np.ndarray, fraction: float) -> np.ndarray:
    left = normalize_quaternion(left_q)
    right = normalize_quaternion(right_q)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("B1A_SLERP_FRACTION")
    dot = float(np.dot(left, right))
    if dot < 0.0:
        right = -right
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = left + fraction * (right - left)
        return normalize_quaternion(result)
    theta = math.acos(dot)
    sine = math.sin(theta)
    result = (
        math.sin((1.0 - fraction) * theta) / sine * left
        + math.sin(fraction * theta) / sine * right
    )
    return normalize_quaternion(result)


def interpolate_pose(
    rows: Sequence[PoseRow], timestamp: Decimal
) -> tuple[np.ndarray, np.ndarray]:
    for row in rows:
        if row.timestamp == timestamp:
            return row.translation.copy(), normalize_quaternion(
                row.quaternion_xyzw
            )
    left = next(
        (row for row in reversed(rows) if row.timestamp < timestamp), None
    )
    right = next((row for row in rows if row.timestamp > timestamp), None)
    if left is None or right is None:
        raise ValueError("B1A_POSE_EXTRAPOLATION")
    span = right.timestamp - left.timestamp
    if span <= 0 or span > Decimal("0.050"):
        raise ValueError("B1A_POSE_BRACKET_TOO_WIDE")
    fraction = float((timestamp - left.timestamp) / span)
    translation = left.translation + fraction * (
        right.translation - left.translation
    )
    return translation, slerp(
        left.quaternion_xyzw, right.quaternion_xyzw, fraction
    )


def quaternion_rotation(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quaternion(quaternion_xyzw)
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
    dt_seconds: float,
) -> dict[str, Any]:
    if not math.isfinite(dt_seconds) or dt_seconds <= 0.0:
        raise ValueError("B1A_DT_INVALID")
    previous_center, previous_q = previous_pose
    current_center, current_q = current_pose
    previous_rotation = quaternion_rotation(previous_q)
    current_rotation = quaternion_rotation(current_q)
    rotation = current_rotation.T @ previous_rotation
    translation = current_rotation.T @ (
        np.asarray(previous_center, dtype=np.float64)
        - np.asarray(current_center, dtype=np.float64)
    )
    angle = math.acos(
        float(np.clip((float(np.trace(rotation)) - 1.0) / 2.0, -1.0, 1.0))
    )
    return {
        "R_current_from_previous": rotation,
        "t_current_from_previous": translation,
        "translation_speed_m_s": float(
            np.linalg.norm(
                np.asarray(current_center) - np.asarray(previous_center)
            )
            / dt_seconds
        ),
        "angular_rate_rad_s": angle / dt_seconds,
    }


def rotation_homography(rotation_current_from_previous: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation_current_from_previous, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("B1A_ROTATION_MATRIX")
    return K @ rotation @ np.linalg.inv(K)


def decode_depth_png(payload: bytes) -> np.ndarray:
    with Image.open(BytesIO(payload)) as image:
        image.load()
        if image.format != "PNG" or image.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
            raise ValueError("B1A_DEPTH_PNG_HEADER")
        array = np.asarray(image)
    if array.shape != (IMAGE_HEIGHT, IMAGE_WIDTH) or array.dtype != np.uint16:
        raise ValueError("B1A_DEPTH_PNG_ARRAY")
    return np.ascontiguousarray(array)


def _cell_bounds(index: int, extent: int) -> tuple[int, int]:
    return (
        int(round(index * extent / 3)),
        int(round((index + 1) * extent / 3)),
    )


def grid_index(x: int, y: int) -> int:
    column = next(
        index
        for index in range(3)
        if _cell_bounds(index, IMAGE_WIDTH)[0]
        <= x
        < _cell_bounds(index, IMAGE_WIDTH)[1]
    )
    row = next(
        index
        for index in range(3)
        if _cell_bounds(index, IMAGE_HEIGHT)[0]
        <= y
        < _cell_bounds(index, IMAGE_HEIGHT)[1]
    )
    return row * 3 + column


def zbuffer_winners(
    candidates: Iterable[dict[str, Any]],
) -> dict[tuple[int, int], dict[str, Any]]:
    winners: dict[tuple[int, int], dict[str, Any]] = {}
    for candidate in candidates:
        key = (int(candidate["pixel_x"]), int(candidate["pixel_y"]))
        incumbent = winners.get(key)
        if incumbent is None or (
            float(candidate["z_predicted"]),
            int(candidate["raster_rank"]),
        ) < (
            float(incumbent["z_predicted"]),
            int(incumbent["raster_rank"]),
        ):
            winners[key] = candidate
    return winners


def evaluate_truth(
    previous_depth: np.ndarray,
    current_depth: np.ndarray,
    rotation_current_from_previous: np.ndarray,
    translation_current_from_previous: np.ndarray,
    dt_seconds: float,
) -> list[dict[str, Any]]:
    previous = np.asarray(previous_depth)
    current = np.asarray(current_depth)
    if (
        previous.shape != (IMAGE_HEIGHT, IMAGE_WIDTH)
        or current.shape != (IMAGE_HEIGHT, IMAGE_WIDTH)
        or previous.dtype != np.uint16
        or current.dtype != np.uint16
    ):
        raise ValueError("B1A_DEPTH_ARRAY_CONTRACT")
    if not math.isfinite(dt_seconds) or dt_seconds <= 0.0:
        raise ValueError("B1A_DT_INVALID")
    rotation = np.asarray(rotation_current_from_previous, dtype=np.float64)
    translation = np.asarray(
        translation_current_from_previous, dtype=np.float64
    )
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("B1A_TRANSFORM_SHAPE")

    inverse_k = np.linalg.inv(K)
    candidates: list[dict[str, Any]] = []
    previous_counts = [0] * 9
    for y in range(0, IMAGE_HEIGHT, 4):
        for x in range(0, IMAGE_WIDTH, 4):
            raw_depth = int(previous[y, x])
            if raw_depth == 0:
                continue
            grid = grid_index(x, y)
            previous_counts[grid] += 1
            z_previous = raw_depth / DEPTH_UNITS_PER_METER
            point_previous = inverse_k @ np.asarray(
                (x * z_previous, y * z_previous, z_previous),
                dtype=np.float64,
            )
            point_current = rotation @ point_previous + translation
            z_predicted = float(point_current[2])
            if z_predicted <= 0.0:
                continue
            projected = K @ point_current
            u = float(projected[0] / projected[2])
            v = float(projected[1] / projected[2])
            if u < 0.0 or v < 0.0:
                continue
            pixel_x = math.floor(u + 0.5)
            pixel_y = math.floor(v + 0.5)
            if not (1 <= pixel_x <= 638 and 1 <= pixel_y <= 478):
                continue
            candidates.append(
                {
                    "grid": grid,
                    "raster_rank": y * IMAGE_WIDTH + x,
                    "pixel_x": pixel_x,
                    "pixel_y": pixel_y,
                    "z_predicted": z_predicted,
                    "point_previous": point_previous,
                    "point_current": point_current,
                }
            )

    rows = [
        {
            "grid_index": index,
            "N_previous": previous_counts[index],
            "N_projected": 0,
            "N_observed": 0,
            "N_static": 0,
            "_rates": [],
        }
        for index in range(9)
    ]
    for winner in zbuffer_winners(candidates).values():
        row = rows[int(winner["grid"])]
        row["N_projected"] += 1
        x = int(winner["pixel_x"])
        y = int(winner["pixel_y"])
        patch = current[y - 1 : y + 2, x - 1 : x + 2]
        nonzero = patch[patch != 0]
        if nonzero.size == 0:
            continue
        row["N_observed"] += 1
        observed_z = float(
            np.median(nonzero.astype(np.float64)) / DEPTH_UNITS_PER_METER
        )
        predicted_z = float(winner["z_predicted"])
        if abs(observed_z - predicted_z) > max(
            0.10, 0.05 * predicted_z
        ):
            continue
        row["N_static"] += 1
        previous_range = float(np.linalg.norm(winner["point_previous"]))
        current_range = float(np.linalg.norm(winner["point_current"]))
        row["_rates"].append(
            math.log(previous_range / current_range) / dt_seconds
        )

    result: list[dict[str, Any]] = []
    for row in rows:
        projected_fraction = (
            row["N_projected"] / row["N_previous"]
            if row["N_previous"]
            else 0.0
        )
        static_fraction = (
            row["N_static"] / row["N_projected"]
            if row["N_projected"]
            else 0.0
        )
        eligible = (
            row["N_previous"] >= 30
            and projected_fraction >= 0.50
            and row["N_static"] >= 30
            and static_fraction >= 0.50
        )
        result.append(
            {
                key: value for key, value in row.items() if key != "_rates"
            }
            | {
                "truth_eligible": bool(eligible),
                "c_truth_grid": (
                    float(
                        np.median(
                            np.asarray(row["_rates"], dtype=np.float64)
                        )
                    )
                    if eligible
                    else None
                ),
            }
        )
    return result


def blank_truth_grids(reason: str) -> list[dict[str, Any]]:
    return [
        {
            "grid_index": index,
            "N_previous": 0,
            "N_projected": 0,
            "N_observed": 0,
            "N_static": 0,
            "truth_eligible": False,
            "c_truth_grid": None,
            "abstention_reason": reason,
        }
        for index in range(9)
    ]


def classify_window(
    candidate_pair_count: int,
    covered_pairs: Sequence[dict[str, float]],
) -> dict[str, Any]:
    coverage = (
        len(covered_pairs) / candidate_pair_count
        if candidate_pair_count > 0
        else 0.0
    )
    if candidate_pair_count == 0 or coverage < 0.80:
        return {
            "coverage": coverage,
            "role": "NOT_EVALUABLE_SOURCE_NATIVE_TRUTH_COVERAGE",
        }
    truths = np.asarray(
        [pair["truth"] for pair in covered_pairs], dtype=np.float64
    )
    angular = np.asarray(
        [pair["angular"] for pair in covered_pairs], dtype=np.float64
    )
    translation = np.asarray(
        [pair["translation"] for pair in covered_pairs], dtype=np.float64
    )
    absolute_truth = np.asarray(
        [pair["absolute_truth"] for pair in covered_pairs], dtype=np.float64
    )
    summary = {
        "coverage": coverage,
        "truth": float(np.median(truths)),
        "angular": float(np.median(angular)),
        "translation": float(np.median(translation)),
        "absolute_truth": float(np.median(absolute_truth)),
    }
    if (
        summary["angular"] >= 5.0 * np.pi / 180.0
        and summary["translation"] <= 0.02
        and summary["absolute_truth"] <= 0.02
    ):
        return summary | {"role": "ROTATION_TRUTH_ELIGIBLE"}
    if summary["truth"] >= 0.05:
        return summary | {"role": "STATIC_APPROACH_TRUTH_ELIGIBLE"}
    return summary | {
        "role": "NOT_EVALUABLE_NO_FROZEN_PHASE_B_CONDITION"
    }


def terminal_for_roles(
    windows: Sequence[dict[str, Any]],
) -> tuple[dict[str, int], str]:
    roles = (
        "ROTATION_TRUTH_ELIGIBLE",
        "STATIC_APPROACH_TRUTH_ELIGIBLE",
    )
    counts = {
        role: len(
            {
                str(window["sequence_id"])
                for window in windows
                if window.get("role") == role
            }
        )
        for role in roles
    }
    terminal = (
        "B1A_SOURCE_NATIVE_GEOMETRY_ADMISSION_VALID_"
        "B1B_BRANCH_SCOPE_MAY_BE_REVIEWED"
        if any(count >= 2 for count in counts.values())
        else "HOLD_B1_SOURCE_NATIVE_TRUTH_NOT_EVALUABLE_NO_WINDOW_REPLACEMENT"
    )
    return counts, terminal


__all__ = [
    "IndexRow",
    "K",
    "PoseRow",
    "adjacent_pairs",
    "assign_depth_rows",
    "blank_truth_grids",
    "classify_window",
    "decode_depth_png",
    "evaluate_truth",
    "grid_index",
    "interpolate_pose",
    "normalize_quaternion",
    "parse_decimal",
    "parse_index_text",
    "parse_pose_text",
    "quaternion_rotation",
    "relative_geometry",
    "rotation_homography",
    "rows_in_window",
    "slerp",
    "terminal_for_roles",
    "validate_member_reference",
    "zbuffer_winners",
]
