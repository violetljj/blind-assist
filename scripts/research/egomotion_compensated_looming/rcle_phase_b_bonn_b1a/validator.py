from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from io import BytesIO
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any, Iterable, Sequence
import zipfile

import numpy as np
import PIL
from PIL import Image


PROTOCOL_ID = "RCLE_PHASE_B_BONN_B1A_SOURCE_NATIVE_GEOMETRY_ADMISSION"
DESIGN_LOCK_SHA256 = (
    "c53c9edaf7012df481b2ba286902af87f1716e3a5d4f57f27398303c4f74420e"
)
PREREGISTRATION_SHA256 = (
    "f3974b2c0096dae2334b1d6c8cd563d892b09288df4f2085604b8fee88d4cfd0"
)
B0_RECEIPT_SHA256 = (
    "dc0ffe9a890b539478ff4c035b4dfadea6c21347a11b36f164810a18eb811f86"
)
WINDOW_DENOMINATOR_SHA256 = (
    "f1e6f7f2e54da349d004af744573884e6273089f67bda86d5f0eb812234aa05b"
)
COHORT_IDENTITY_SHA256 = (
    "513b770d18489fd0caf84874e9fb89456eb3a992fc262b037220b66b5caae86e"
)
PASS_TERMINAL = (
    "B1A_SOURCE_NATIVE_GEOMETRY_ADMISSION_VALID_"
    "B1B_BRANCH_SCOPE_MAY_BE_REVIEWED"
)
HOLD_TERMINAL = (
    "HOLD_B1_SOURCE_NATIVE_TRUTH_NOT_EVALUABLE_NO_WINDOW_REPLACEMENT"
)
INVALID_TERMINAL = "INVALID_EXECUTION_CLOSE_B1"
ROTATION_ROLE = "ROTATION_TRUTH_ELIGIBLE"
APPROACH_ROLE = "STATIC_APPROACH_TRUTH_ELIGIBLE"
NO_ROLE = "NOT_EVALUABLE_NO_FROZEN_PHASE_B_CONDITION"
NO_COVERAGE = "NOT_EVALUABLE_SOURCE_NATIVE_TRUTH_COVERAGE"
RECEIPT_KEYS = {
    "schema_version",
    "protocol_id",
    "created_at",
    "environment",
    "design_lock_sha256",
    "preregistration_sha256",
    "implementation_lock_sha256",
    "bootstrap_runner_sha256",
    "b0_receipt_sha256",
    "window_denominator_sha256",
    "cohort_identity_sha256",
    "source_authority_sha256_by_id",
    "archive_sha256_by_sequence",
    "claim",
    "run_claim_sha256",
    "read_firewall",
    "ledger",
    "ledger_identity_sha256",
    "branch_distinct_sequence_counts",
    "gate_pass",
    "terminal_state",
    "b1b_branch_scope_may_be_reviewed",
    "execution_authority_consumed",
    "b1b_implementation_authorized",
}
WINDOWS = (
    (1, "rgbd_bonn_crowd2", 0, "1548339892.26121", "1548339902.26121"),
    (1, "rgbd_bonn_crowd2", 1, "1548339902.26121", "1548339912.26121"),
    (
        2,
        "rgbd_bonn_balloon_tracking",
        0,
        "1548266633.5283",
        "1548266643.5283",
    ),
    (
        3,
        "rgbd_bonn_balloon_tracking2",
        0,
        "1548266677.42642",
        "1548266687.42642",
    ),
    (
        4,
        "rgbd_bonn_moving_obstructing_box2",
        0,
        "1548339380.76065",
        "1548339390.76065",
    ),
    (
        4,
        "rgbd_bonn_moving_obstructing_box2",
        1,
        "1548339390.76065",
        "1548339400.76065",
    ),
    (5, "rgbd_bonn_balloon2", 0, "1548266530.56676", "1548266540.56676"),
    (
        6,
        "rgbd_bonn_moving_nonobstructing_box2",
        0,
        "1548266187.16307",
        "1548266197.16307",
    ),
    (
        6,
        "rgbd_bonn_moving_nonobstructing_box2",
        1,
        "1548266197.16307",
        "1548266207.16307",
    ),
    (
        6,
        "rgbd_bonn_moving_nonobstructing_box2",
        2,
        "1548266207.16307",
        "1548266217.16307",
    ),
)
K = np.asarray(
    (
        (542.822841, 0.0, 315.593520),
        (0.0, 542.576870, 237.756098),
        (0.0, 0.0, 1.0),
    ),
    dtype=np.float64,
)
NUMBER_RE = re.compile(
    rb"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
)
HEX_RE = re.compile(r"^[0-9a-f]{64}$")
ASCII_WS = b" \t\r\n\v\f"
FLOAT_TOLERANCE = 1e-12
SOURCE_AUTHORITY_SHA256 = {
    "bonn_official_page": (
        "2bd8df16acad79c70e1021f1da039c78510034fd9091fd706f8a3f480ea5c186"
    ),
    "tum_file_formats": (
        "721c8df093ade2b0078215c3154f6f1a3641a0c691b5123cd037e87b61b30107"
    ),
}


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


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decimal(token: bytes) -> Decimal:
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
    try:
        path.encode("utf-8", errors="strict")
    except UnicodeError as error:
        raise ValueError("B1A_MEMBER_REFERENCE_UTF8") from error
    pure = PurePosixPath(path)
    if (
        pure.is_absolute()
        or any(part in ("", ".", "..") for part in pure.parts)
        or any(":" in part for part in pure.parts)
        or str(pure) != path
    ):
        raise ValueError("B1A_MEMBER_REFERENCE_NONCANONICAL")
    return path


def parse_index_text(raw: bytes) -> list[IndexRow]:
    rows: list[IndexRow] = []
    for line in raw.split(b"\n"):
        stripped = line.strip(ASCII_WS)
        if not stripped or stripped.startswith(b"#"):
            continue
        tokens = re.split(rb"[\x09-\x0d\x20]+", stripped)
        if len(tokens) != 2:
            raise ValueError("B1A_INDEX_TOKEN_COUNT")
        timestamp = _decimal(tokens[0])
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
    for line in raw.split(b"\n"):
        stripped = line.strip(ASCII_WS)
        if not stripped or stripped.startswith(b"#"):
            continue
        tokens = re.split(rb"[\x09-\x0d\x20]+", stripped)
        if len(tokens) != 8:
            raise ValueError("B1A_POSE_TOKEN_COUNT")
        timestamp = _decimal(tokens[0])
        numeric = np.asarray([float(_decimal(token)) for token in tokens[1:]])
        if numeric.dtype != np.float64 or not np.all(np.isfinite(numeric)):
            raise ValueError("B1A_POSE_NONFINITE")
        if rows and timestamp <= rows[-1].timestamp:
            raise ValueError("B1A_TIMESTAMP_NOT_STRICTLY_INCREASING")
        rows.append(
            PoseRow(timestamp, numeric[:3], numeric[3:], len(rows))
        )
    if not rows:
        raise ValueError("B1A_POSE_EMPTY")
    return rows


def adjacent_pairs(
    rows: Sequence[IndexRow], start: Decimal, end: Decimal
) -> list[dict[str, Any]]:
    inside = [row for row in rows if start <= row.timestamp < end]
    result: list[dict[str, Any]] = []
    for previous, current in zip(inside, inside[1:]):
        dt = current.timestamp - previous.timestamp
        result.append(
            {
                "previous_rgb_source_row_rank": previous.source_row_rank,
                "current_rgb_source_row_rank": current.source_row_rank,
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
) -> dict[int, int | None]:
    available = {
        row.source_row_rank: row
        for row in depth_rows
        if start <= row.timestamp < end
    }
    assignments: dict[int, int | None] = {}
    for rgb in (row for row in rgb_rows if start <= row.timestamp < end):
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
        assignments[rgb.source_row_rank] = selected.source_row_rank
        del available[selected.source_row_rank]
    return assignments


def _normalize_quaternion(q: np.ndarray) -> np.ndarray:
    value = np.asarray(q, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if (
        value.shape != (4,)
        or not np.all(np.isfinite(value))
        or norm <= 0.0
        or abs(norm - 1.0) > 0.001
    ):
        raise ValueError("B1A_QUATERNION_NORM")
    return value / norm


def slerp(q_left: np.ndarray, q_right: np.ndarray, fraction: float) -> np.ndarray:
    left = _normalize_quaternion(q_left)
    right = _normalize_quaternion(q_right)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("B1A_SLERP_FRACTION")
    dot = float(np.dot(left, right))
    if dot < 0.0:
        right = -right
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = left + fraction * (right - left)
        return result / np.linalg.norm(result)
    theta = math.acos(dot)
    sine = math.sin(theta)
    result = (
        math.sin((1.0 - fraction) * theta) / sine * left
        + math.sin(fraction * theta) / sine * right
    )
    return result / np.linalg.norm(result)


def quaternion_rotation(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = _normalize_quaternion(q_xyzw)
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def interpolate_pose(rows: Sequence[PoseRow], timestamp: Decimal) -> tuple[np.ndarray, np.ndarray]:
    exact = [row for row in rows if row.timestamp == timestamp]
    if exact:
        row = exact[0]
        return row.translation.copy(), _normalize_quaternion(row.quaternion_xyzw)
    left = next((row for row in reversed(rows) if row.timestamp < timestamp), None)
    right = next((row for row in rows if row.timestamp > timestamp), None)
    if left is None or right is None:
        raise ValueError("B1A_POSE_EXTRAPOLATION")
    span = right.timestamp - left.timestamp
    if span > Decimal("0.050"):
        raise ValueError("B1A_POSE_BRACKET_TOO_WIDE")
    fraction = float((timestamp - left.timestamp) / span)
    translation = left.translation + fraction * (
        right.translation - left.translation
    )
    return translation, slerp(
        left.quaternion_xyzw, right.quaternion_xyzw, fraction
    )


def relative_geometry(
    previous_pose: tuple[np.ndarray, np.ndarray],
    current_pose: tuple[np.ndarray, np.ndarray],
    dt: float,
) -> dict[str, Any]:
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("B1A_DT_INVALID")
    previous_c, previous_q = previous_pose
    current_c, current_q = current_pose
    previous_r = quaternion_rotation(previous_q)
    current_r = quaternion_rotation(current_q)
    relative_r = current_r.T @ previous_r
    relative_t = current_r.T @ (
        np.asarray(previous_c, dtype=np.float64)
        - np.asarray(current_c, dtype=np.float64)
    )
    trace_value = float(np.trace(relative_r))
    angle = math.acos(float(np.clip((trace_value - 1.0) / 2.0, -1.0, 1.0)))
    return {
        "R_current_from_previous": relative_r,
        "t_current_from_previous": relative_t,
        "translation_speed": float(
            np.linalg.norm(np.asarray(current_c) - np.asarray(previous_c)) / dt
        ),
        "angular_rate_rad_s": angle / dt,
    }


def rotation_homography(rotation_current_from_previous: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation_current_from_previous, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("B1A_ROTATION_MATRIX")
    return K @ rotation @ np.linalg.inv(K)


def decode_depth_png(payload: bytes) -> np.ndarray:
    with Image.open(BytesIO(payload)) as image:
        image.load()
        if image.format != "PNG" or image.size != (640, 480):
            raise ValueError("B1A_DEPTH_PNG_HEADER")
        array = np.asarray(image)
    if array.shape != (480, 640) or array.dtype != np.dtype(np.uint16):
        raise ValueError("B1A_DEPTH_PNG_ARRAY")
    return array


def _cell_bounds(index: int, extent: int) -> tuple[int, int]:
    return int(round(index * extent / 3)), int(round((index + 1) * extent / 3))


def _grid_index(x: int, y: int) -> int:
    column = next(
        index for index in range(3) if _cell_bounds(index, 640)[0] <= x < _cell_bounds(index, 640)[1]
    )
    row = next(
        index for index in range(3) if _cell_bounds(index, 480)[0] <= y < _cell_bounds(index, 480)[1]
    )
    return row * 3 + column


def zbuffer_winners(candidates: Iterable[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
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
    dt: float,
) -> list[dict[str, Any]]:
    previous = np.asarray(previous_depth)
    current = np.asarray(current_depth)
    if (
        previous.shape != (480, 640)
        or current.shape != (480, 640)
        or previous.dtype != np.uint16
        or current.dtype != np.uint16
    ):
        raise ValueError("B1A_DEPTH_ARRAY_CONTRACT")
    rotation = np.asarray(rotation_current_from_previous, dtype=np.float64)
    translation = np.asarray(translation_current_from_previous, dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("B1A_TRANSFORM_SHAPE")
    inv_k = np.linalg.inv(K)
    candidates: list[dict[str, Any]] = []
    n_previous = [0] * 9
    for y in range(0, 480, 4):
        for x in range(0, 640, 4):
            raw_depth = int(previous[y, x])
            if raw_depth == 0:
                continue
            grid = _grid_index(x, y)
            n_previous[grid] += 1
            z_previous = raw_depth / 5000.0
            point_previous = inv_k @ np.asarray(
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
                    "x": x,
                    "y": y,
                    "raster_rank": y * 640 + x,
                    "pixel_x": pixel_x,
                    "pixel_y": pixel_y,
                    "z_predicted": z_predicted,
                    "point_previous": point_previous,
                    "point_current": point_current,
                }
            )
    winners = zbuffer_winners(candidates)
    rows = [
        {
            "grid_index": index,
            "N_previous": n_previous[index],
            "N_projected": 0,
            "N_observed": 0,
            "N_static": 0,
            "_rates": [],
        }
        for index in range(9)
    ]
    for winner in winners.values():
        row = rows[winner["grid"]]
        row["N_projected"] += 1
        x = winner["pixel_x"]
        y = winner["pixel_y"]
        patch = current[y - 1 : y + 2, x - 1 : x + 2]
        nonzero = patch[patch != 0]
        if nonzero.size == 0:
            continue
        row["N_observed"] += 1
        z_observed = float(np.median(nonzero.astype(np.float64)) / 5000.0)
        z_predicted = winner["z_predicted"]
        if abs(z_observed - z_predicted) > max(0.10, 0.05 * z_predicted):
            continue
        row["N_static"] += 1
        previous_range = float(np.linalg.norm(winner["point_previous"]))
        current_range = float(np.linalg.norm(winner["point_current"]))
        row["_rates"].append(math.log(previous_range / current_range) / dt)
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
                "truth_eligible": eligible,
                "c_truth_grid": (
                    float(np.median(np.asarray(row["_rates"], dtype=np.float64)))
                    if eligible
                    else None
                ),
            }
        )
    return result


def classify_window(
    candidate_pair_count: int, covered_pairs: Sequence[dict[str, float]]
) -> dict[str, Any]:
    coverage = (
        len(covered_pairs) / candidate_pair_count
        if candidate_pair_count > 0
        else 0.0
    )
    if candidate_pair_count == 0 or coverage < 0.80:
        return {"coverage": coverage, "role": NO_COVERAGE}
    truths = np.asarray([pair["truth"] for pair in covered_pairs], dtype=np.float64)
    angular = np.asarray([pair["angular"] for pair in covered_pairs], dtype=np.float64)
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
        return summary | {"role": ROTATION_ROLE}
    if summary["truth"] >= 0.05:
        return summary | {"role": APPROACH_ROLE}
    return summary | {"role": NO_ROLE}


def terminal_for_roles(windows: Sequence[dict[str, Any]]) -> tuple[dict[str, int], str]:
    counts = {
        role: len(
            {
                window["sequence_id"]
                for window in windows
                if window.get("role") == role
            }
        )
        for role in (ROTATION_ROLE, APPROACH_ROLE)
    }
    terminal = PASS_TERMINAL if any(value >= 2 for value in counts.values()) else HOLD_TERMINAL
    return counts, terminal


def float_hex(value: float | None) -> str | None:
    if value is None:
        return None
    if not math.isfinite(value):
        raise ValueError("B1A_FLOAT_NONFINITE")
    return float(value).hex()


def parse_float_hex(value: Any) -> float | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("B1A_FLOAT_HEX_TYPE")
    try:
        parsed = float.fromhex(value)
    except ValueError as error:
        raise ValueError("B1A_FLOAT_HEX_PARSE") from error
    if not math.isfinite(parsed) or parsed.hex() != value:
        raise ValueError("B1A_FLOAT_HEX_CANONICAL")
    return parsed


def _canonical_windows() -> list[dict[str, Any]]:
    return [
        {
            "sequence_rank": sequence_rank,
            "sequence_id": sequence_id,
            "window_rank": window_rank,
            "start": start,
            "end": end,
        }
        for sequence_rank, sequence_id, window_rank, start, end in WINDOWS
    ]


def validate_receipt_data(
    receipt: dict[str, Any],
    *,
    claim_raw: bytes | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    if set(receipt) != RECEIPT_KEYS:
        errors.append("RECEIPT_KEYS")
    if (
        receipt.get("schema_version")
        != "rcle.phase_b.bonn_b1a.receipt.v1"
    ):
        errors.append("SCHEMA_VERSION")
    ledger = receipt.get("ledger")
    if not isinstance(ledger, dict):
        raise ValueError("B1A_LEDGER_TYPE")
    if ledger.get("windows") != _canonical_windows():
        errors.append("TEN_WINDOW_DENOMINATOR")
    if receipt.get("ledger_identity_sha256") != canonical_sha256(ledger):
        errors.append("LEDGER_IDENTITY")
    expected_hashes = {
        "design_lock_sha256": DESIGN_LOCK_SHA256,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "b0_receipt_sha256": B0_RECEIPT_SHA256,
        "window_denominator_sha256": WINDOW_DENOMINATOR_SHA256,
        "cohort_identity_sha256": COHORT_IDENTITY_SHA256,
    }
    if any(receipt.get(key) != value for key, value in expected_hashes.items()):
        errors.append("CONTROL_HASH")
    if receipt.get("protocol_id") != PROTOCOL_ID:
        errors.append("PROTOCOL")
    expected_runtime = {
        "python": "3.11.9",
        "numpy": "2.1.3",
        "pillow": "12.2.0",
    }
    actual_runtime = {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "numpy": np.__version__,
        "pillow": PIL.__version__,
    }
    if (
        receipt.get("environment") != expected_runtime
        or actual_runtime != expected_runtime
    ):
        errors.append("RUNTIME")
    if receipt.get("source_authority_sha256_by_id") != SOURCE_AUTHORITY_SHA256:
        errors.append("SOURCE_AUTHORITY")
    if receipt.get("execution_authority_consumed") is not True:
        errors.append("EXECUTION_AUTHORITY")
    if receipt.get("b1b_implementation_authorized") is not False:
        errors.append("B1B_AUTHORITY")
    firewall = receipt.get("read_firewall")
    expected_zero = (
        "rgb_member_bytes_read",
        "rgb_decode_operations",
        "phase_b_metric_operations",
        "static_map_reads",
        "legacy_bonn_outcome_reads",
        "network_operations",
    )
    if not isinstance(firewall, dict) or any(
        firewall.get(key) != 0 for key in expected_zero
    ):
        errors.append("FIREWALL")
    claim = receipt.get("claim")
    if not isinstance(claim, dict) or not (
        claim.get("exclusive_create") is True
        and claim.get("maximum_claims") == 1
        and claim.get("application_data_operations_before_claim") == 0
        and claim.get("claim_permanently_retained") is True
    ):
        errors.append("CLAIM_FIRST")
    elif receipt.get("bootstrap_runner_sha256") != claim.get(
        "bootstrap_runner_sha256"
    ):
        errors.append("BOOTSTRAP_RUNNER_IDENTITY")
    if isinstance(claim, dict) and (
        claim.get("argv") != []
        or claim.get("design_lock_sha256") != DESIGN_LOCK_SHA256
        or claim.get("preregistration_sha256") != PREREGISTRATION_SHA256
        or claim.get("b0_receipt_sha256") != B0_RECEIPT_SHA256
        or claim.get("window_denominator_sha256")
        != WINDOW_DENOMINATOR_SHA256
        or claim.get("implementation_lock_sha256")
        != receipt.get("implementation_lock_sha256")
        or claim.get("delete_replace_or_rewrite_claim") != "FORBIDDEN"
        or claim.get(
            "success_failure_exception_or_interrupt_consumes_claim"
        )
        is not True
    ):
        errors.append("CLAIM_BINDINGS")
    if claim_raw is not None:
        try:
            claim_from_file = json.loads(claim_raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            errors.append("CLAIM_FILE_PARSE")
        else:
            if claim_from_file != claim:
                errors.append("CLAIM_OBJECT_MISMATCH")
        actual_claim_sha256 = hashlib.sha256(claim_raw).hexdigest()
        if receipt.get("run_claim_sha256") != actual_claim_sha256:
            errors.append("CLAIM_FILE_SHA256")
    windows = ledger.get("window_results")
    if not isinstance(windows, list) or len(windows) != 10:
        errors.append("WINDOW_RESULT_CARDINALITY")
        windows = []
    counts, terminal = terminal_for_roles(windows)
    if receipt.get("branch_distinct_sequence_counts") != counts:
        errors.append("BRANCH_COUNTS")
    if receipt.get("terminal_state") != terminal:
        errors.append("TERMINAL")
    gate = terminal == PASS_TERMINAL
    if receipt.get("gate_pass") is not gate:
        errors.append("GATE")
    if receipt.get("b1b_branch_scope_may_be_reviewed") is not gate:
        errors.append("B1B_REVIEW_SCOPE")
    archive_hashes = receipt.get("archive_sha256_by_sequence")
    if not isinstance(archive_hashes, dict) or len(archive_hashes) != 6 or any(
        not isinstance(value, str) or HEX_RE.fullmatch(value) is None
        for value in archive_hashes.values()
    ):
        errors.append("ARCHIVE_HASHES")
    for pair in ledger.get("pairs", []):
        grids = pair.get("grids")
        if not isinstance(grids, list) or len(grids) != 9:
            errors.append("PAIR_GRID_CARDINALITY")
            break
        for index, grid in enumerate(grids):
            if grid.get("grid_index") != index:
                errors.append("GRID_ORDER")
                break
            counts_four = [
                grid.get(name)
                for name in ("N_previous", "N_projected", "N_observed", "N_static")
            ]
            if not all(isinstance(value, int) and value >= 0 for value in counts_four):
                errors.append("TRUTH_COUNTS")
                break
            if not (
                counts_four[3] <= counts_four[2] <= counts_four[1] <= counts_four[0]
            ):
                errors.append("TRUTH_COUNT_ORDER")
                break
            parse_float_hex(grid.get("c_truth_grid_hex"))
    return {
        "status": "VALID" if not errors else "INVALID",
        "errors": sorted(set(errors)),
        "terminal_state": terminal if not errors else INVALID_TERMINAL,
        "gate_pass": not errors and gate,
    }


def _canonical_paths(repo_root: Path) -> dict[str, Path]:
    root = Path(repo_root).resolve()
    package = (
        root
        / "scripts"
        / "research"
        / "egomotion_compensated_looming"
        / "rcle_phase_b_bonn_b1a"
    )
    return {
        "preregistration": root
        / "docs"
        / "research"
        / "rcle"
        / "RCLE_PHASE_B_BONN_B1_PREREGISTRATION_2026-07-26.md",
        "design_lock": root
        / "docs"
        / "research"
        / "rcle"
        / "RCLE_PHASE_B_BONN_B1_DESIGN_LOCK_2026-07-26.json",
        "implementation_lock": package
        / "RCLE_PHASE_B_BONN_B1A_IMPLEMENTATION_LOCK.json",
        "bootstrap_runner": root
        / "scripts"
        / "research"
        / "egomotion_compensated_looming"
        / "run_phase_b_bonn_b1a.py",
        "bonn_official_page": root
        / "artifacts.local"
        / "datasets"
        / "egomotion_compensated_looming_r1"
        / "bonn_metadata_r0"
        / "official_page.html",
        "tum_file_formats": root
        / "artifacts.local"
        / "datasets"
        / "egomotion_compensated_looming_r1"
        / "bonn_b1_authority"
        / "tum_rgbd_file_formats.html",
        "b0_receipt": root
        / "artifacts.local"
        / "evidence"
        / "rcle_phase_b_bonn_b0_r1"
        / "formal_entry_b0_r1"
        / "receipt.json",
        "output": root
        / "artifacts.local"
        / "evidence"
        / "rcle_phase_b_bonn_b1"
        / "b1a_geometry_admission",
    }


def _unique_member(names: Sequence[str], suffix: str) -> str:
    matches = [
        name for name in names if name == suffix or name.endswith("/" + suffix)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"B1A_CONTROL_MEMBER_CARDINALITY:{suffix}:{len(matches)}"
        )
    return matches[0]


def _resolve_reference(root: str, reference: str, members: set[str]) -> str:
    validate_member_reference(reference)
    candidate = f"{root}/{reference}" if root else reference
    if candidate not in members:
        raise ValueError(f"B1A_REFERENCED_MEMBER_ABSENT:{reference}")
    return candidate


def _blank_pair_grids(reason: str) -> list[dict[str, Any]]:
    return [
        {
            "grid_index": index,
            "N_previous": 0,
            "N_projected": 0,
            "N_observed": 0,
            "N_static": 0,
            "truth_eligible": False,
            "abstention_reason": reason,
            "c_truth_grid_hex": None,
        }
        for index in range(9)
    ]


def _serialize_grids(grids: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: value for key, value in grid.items() if key != "c_truth_grid"
        }
        | {"c_truth_grid_hex": float_hex(grid.get("c_truth_grid"))}
        for grid in grids
    ]


def _pair_identity(
    sequence_id: str,
    window_rank: int,
    previous: IndexRow,
    current: IndexRow,
) -> str:
    return (
        f"{sequence_id}:{window_rank}:"
        f"{previous.source_row_rank}:{current.source_row_rank}"
    )


def _window_result(
    base: dict[str, Any],
    adjacent_count: int,
    candidate_count: int,
    covered: Sequence[dict[str, float]],
) -> dict[str, Any]:
    summary = classify_window(candidate_count, covered)
    return {
        **base,
        "all_adjacent_pair_count": adjacent_count,
        "candidate_pair_denominator_count": candidate_count,
        "truth_covered_pair_count": len(covered),
        "truth_coverage_hex": float_hex(float(summary["coverage"])),
        "window_truth_closing_hex": float_hex(summary.get("truth")),
        "window_angular_rate_rad_s_hex": float_hex(summary.get("angular")),
        "window_translation_speed_m_s_hex": float_hex(
            summary.get("translation")
        ),
        "window_absolute_truth_closing_hex": float_hex(
            summary.get("absolute_truth")
        ),
        "role": summary["role"],
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"B1A_JSON_OBJECT_REQUIRED:{path}")
    return value


def _recompute_ledger(
    repo_root: Path,
    receipt: dict[str, Any],
    paths: dict[str, Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file(paths["preregistration"]) != PREREGISTRATION_SHA256:
        raise ValueError("B1A_PREREGISTRATION_HASH_MISMATCH")
    if sha256_file(paths["design_lock"]) != DESIGN_LOCK_SHA256:
        raise ValueError("B1A_DESIGN_LOCK_HASH_MISMATCH")
    implementation_hash = sha256_file(paths["implementation_lock"])
    if receipt.get("implementation_lock_sha256") != implementation_hash:
        raise ValueError("B1A_IMPLEMENTATION_LOCK_HASH_MISMATCH")
    implementation_lock = _read_json_object(paths["implementation_lock"])
    if (
        implementation_lock.get("design_lock_sha256") != DESIGN_LOCK_SHA256
        or implementation_lock.get("preregistration_sha256")
        != PREREGISTRATION_SHA256
        or implementation_lock.get("canonical_execution_authorized") is not True
    ):
        raise ValueError("B1A_IMPLEMENTATION_LOCK_AUTHORITY_INVALID")
    source_files = implementation_lock.get("source_files")
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError("B1A_IMPLEMENTATION_SOURCE_MANIFEST_ABSENT")
    for relative, expected in source_files.items():
        if (
            not isinstance(relative, str)
            or not isinstance(expected, str)
            or sha256_file(Path(repo_root) / relative) != expected
        ):
            raise ValueError(
                f"B1A_IMPLEMENTATION_SOURCE_HASH_MISMATCH:{relative}"
            )
    claim = receipt["claim"]
    output = paths["output"].resolve()
    if (
        Path(claim.get("canonical_output", "")).resolve() != output
        or Path(claim.get("canonical_run_claim", "")).resolve()
        != output / "run_claim.json"
    ):
        raise ValueError("B1A_CLAIM_CANONICAL_PATH_DRIFT")
    if claim.get("implementation_lock_sha256") != implementation_hash:
        raise ValueError("B1A_CLAIM_IMPLEMENTATION_LOCK_HASH_MISMATCH")
    runner_hash = sha256_file(paths["bootstrap_runner"])
    if (
        claim.get("bootstrap_runner_sha256") != runner_hash
        or receipt.get("bootstrap_runner_sha256") != runner_hash
    ):
        raise ValueError("B1A_BOOTSTRAP_RUNNER_HASH_MISMATCH")
    if claim.get("source_authority_sha256") != SOURCE_AUTHORITY_SHA256:
        raise ValueError("B1A_CLAIM_SOURCE_AUTHORITY_DRIFT")
    for key, expected in SOURCE_AUTHORITY_SHA256.items():
        if sha256_file(paths[key]) != expected:
            raise ValueError(f"B1A_SOURCE_AUTHORITY_HASH_MISMATCH:{key}")
    if sha256_file(paths["b0_receipt"]) != B0_RECEIPT_SHA256:
        raise ValueError("B1A_B0_RECEIPT_HASH_MISMATCH")
    b0 = _read_json_object(paths["b0_receipt"])
    if (
        b0.get("window_denominator_sha256") != WINDOW_DENOMINATOR_SHA256
        or b0.get("cohort_identity_sha256") != COHORT_IDENTITY_SHA256
    ):
        raise ValueError("B1A_B0_AUTHORITY_DRIFT")
    sequence_results = {
        row["sequence_id"]: row for row in b0.get("sequence_results", [])
    }
    expected_sequences = {row[1] for row in WINDOWS}
    if set(sequence_results) != expected_sequences:
        raise ValueError("B1A_B0_SEQUENCE_SET_MISMATCH")

    pair_rows: list[dict[str, Any]] = []
    window_results: list[dict[str, Any]] = []
    archive_hashes: dict[str, str] = {}
    depth_decode_operations = 0
    pose_numeric_rows_parsed = 0
    for sequence_rank, sequence_id in sorted(
        {(row[0], row[1]) for row in WINDOWS}
    ):
        source = sequence_results[sequence_id]
        archive = Path(source["archive_path"])
        expected_archive_hash = source["archive_sha256"]
        if (
            claim.get("archive_sha256_by_sequence", {}).get(sequence_id)
            != expected_archive_hash
            or sha256_file(archive) != expected_archive_hash
        ):
            raise ValueError(f"B1A_ARCHIVE_HASH_MISMATCH:{sequence_id}")
        archive_hashes[sequence_id] = expected_archive_hash
        with zipfile.ZipFile(archive, "r") as bundle:
            infos = bundle.infolist()
            names = [info.filename for info in infos if not info.is_dir()]
            if len(names) != len(set(names)):
                raise ValueError(f"B1A_DUPLICATE_ZIP_MEMBER:{sequence_id}")
            members = set(names)
            rgb_member = _unique_member(names, "rgb.txt")
            depth_member = _unique_member(names, "depth.txt")
            pose_member = _unique_member(names, "groundtruth.txt")
            roots = {
                name.rsplit("/", 1)[0] if "/" in name else ""
                for name in (rgb_member, depth_member, pose_member)
            }
            if len(roots) != 1:
                raise ValueError(f"B1A_CONTROL_ROOT_MISMATCH:{sequence_id}")
            root = roots.pop()
            rgb_rows = parse_index_text(bundle.read(rgb_member))
            depth_rows = parse_index_text(bundle.read(depth_member))
            pose_rows = parse_pose_text(bundle.read(pose_member))
            pose_numeric_rows_parsed += len(pose_rows)
            rgb_by_rank = {row.source_row_rank: row for row in rgb_rows}
            depth_by_rank = {row.source_row_rank: row for row in depth_rows}
            for row in WINDOWS:
                if row[1] != sequence_id:
                    continue
                _, _, window_rank, start_text, end_text = row
                start = Decimal(start_text)
                end = Decimal(end_text)
                base = {
                    "sequence_rank": sequence_rank,
                    "sequence_id": sequence_id,
                    "window_rank": window_rank,
                    "start": start_text,
                    "end": end_text,
                }
                pairs = adjacent_pairs(rgb_rows, start, end)
                depth_assignments = assign_depth_rows(
                    rgb_rows, depth_rows, start, end
                )
                covered: list[dict[str, float]] = []
                candidate_count = sum(bool(pair["candidate"]) for pair in pairs)
                for pair in pairs:
                    previous = rgb_by_rank[
                        pair["previous_rgb_source_row_rank"]
                    ]
                    current = rgb_by_rank[
                        pair["current_rgb_source_row_rank"]
                    ]
                    pair_row: dict[str, Any] = {
                        **base,
                        "pair_id": _pair_identity(
                            sequence_id, window_rank, previous, current
                        ),
                        "previous_rgb_source_row_rank": previous.source_row_rank,
                        "current_rgb_source_row_rank": current.source_row_rank,
                        "previous_rgb_timestamp": str(previous.timestamp),
                        "current_rgb_timestamp": str(current.timestamp),
                        "dt": str(pair["dt"]),
                        "candidate": bool(pair["candidate"]),
                        "truth_covered": False,
                        "abstention_reason": None,
                    }
                    if not pair["candidate"]:
                        reason = "DT_OUTSIDE_CANDIDATE_RANGE"
                        pair_row["abstention_reason"] = reason
                        pair_row["grids"] = _blank_pair_grids(reason)
                        pair_rows.append(pair_row)
                        continue
                    try:
                        for rgb in (previous, current):
                            rgb_path = _resolve_reference(
                                root, rgb.path, members
                            )
                            if bundle.getinfo(rgb_path).file_size <= 0:
                                raise ValueError(
                                    "B1A_RGB_DECLARED_SIZE_NONPOSITIVE"
                                )
                        previous_depth_rank = depth_assignments.get(
                            previous.source_row_rank
                        )
                        current_depth_rank = depth_assignments.get(
                            current.source_row_rank
                        )
                        if (
                            previous_depth_rank is None
                            or current_depth_rank is None
                        ):
                            raise ValueError("B1A_DEPTH_JOIN_UNMATCHED")
                        previous_depth = depth_by_rank[previous_depth_rank]
                        current_depth = depth_by_rank[current_depth_rank]
                        previous_depth_member = _resolve_reference(
                            root, previous_depth.path, members
                        )
                        current_depth_member = _resolve_reference(
                            root, current_depth.path, members
                        )
                        previous_pose = interpolate_pose(
                            pose_rows, previous.timestamp
                        )
                        current_pose = interpolate_pose(
                            pose_rows, current.timestamp
                        )
                        geometry = relative_geometry(
                            previous_pose, current_pose, float(pair["dt"])
                        )
                        homography = rotation_homography(
                            geometry["R_current_from_previous"]
                        )
                        previous_array = decode_depth_png(
                            bundle.read(previous_depth_member)
                        )
                        current_array = decode_depth_png(
                            bundle.read(current_depth_member)
                        )
                        depth_decode_operations += 2
                        grids = evaluate_truth(
                            previous_array,
                            current_array,
                            geometry["R_current_from_previous"],
                            geometry["t_current_from_previous"],
                            float(pair["dt"]),
                        )
                        eligible = [
                            grid for grid in grids if grid["truth_eligible"]
                        ]
                        pair_row.update(
                            {
                                "previous_depth_source_row_rank": (
                                    previous_depth.source_row_rank
                                ),
                                "current_depth_source_row_rank": (
                                    current_depth.source_row_rank
                                ),
                                "previous_depth_delta": str(
                                    previous_depth.timestamp
                                    - previous.timestamp
                                ),
                                "current_depth_delta": str(
                                    current_depth.timestamp - current.timestamp
                                ),
                                "R_current_from_previous_hex": [
                                    [float_hex(float(value)) for value in values]
                                    for values in geometry[
                                        "R_current_from_previous"
                                    ]
                                ],
                                "t_current_from_previous_hex": [
                                    float_hex(float(value))
                                    for value in geometry[
                                        "t_current_from_previous"
                                    ]
                                ],
                                "H_previous_to_current_hex": [
                                    [float_hex(float(value)) for value in values]
                                    for values in homography
                                ],
                                "translation_speed_m_s_hex": float_hex(
                                    geometry["translation_speed"]
                                ),
                                "angular_rate_rad_s_hex": float_hex(
                                    geometry["angular_rate_rad_s"]
                                ),
                                "truth_covered": len(eligible) >= 5,
                                "grids": _serialize_grids(grids),
                            }
                        )
                        if len(eligible) >= 5:
                            truth_values = np.asarray(
                                [
                                    grid["c_truth_grid"]
                                    for grid in eligible
                                ],
                                dtype=np.float64,
                            )
                            pair_truth = float(np.median(truth_values))
                            pair_absolute = float(
                                np.median(np.abs(truth_values))
                            )
                            pair_row["pair_truth_closing_hex"] = float_hex(
                                pair_truth
                            )
                            pair_row[
                                "pair_absolute_truth_closing_hex"
                            ] = float_hex(pair_absolute)
                            covered.append(
                                {
                                    "truth": pair_truth,
                                    "absolute_truth": pair_absolute,
                                    "angular": geometry["angular_rate_rad_s"],
                                    "translation": geometry[
                                        "translation_speed"
                                    ],
                                }
                            )
                        else:
                            pair_row["abstention_reason"] = (
                                "TRUTH_ELIGIBLE_GRIDS_BELOW_5"
                            )
                    except (
                        KeyError,
                        OSError,
                        ValueError,
                        zipfile.BadZipFile,
                    ) as error:
                        pair_row["abstention_reason"] = str(error)
                        pair_row["grids"] = _blank_pair_grids(str(error))
                    pair_rows.append(pair_row)
                window_results.append(
                    _window_result(
                        base, len(pairs), candidate_count, covered
                    )
                )
    ledger = {
        "schema_version": "rcle.phase_b.bonn_b1a.ledger.v1",
        "windows": _canonical_windows(),
        "window_results": window_results,
        "pairs": pair_rows,
    }
    metadata = {
        "archive_sha256_by_sequence": archive_hashes,
        "depth_decode_operations": depth_decode_operations,
        "pose_numeric_rows_parsed": pose_numeric_rows_parsed,
    }
    return ledger, metadata


def _float_hex_equal(left: str, right: str) -> bool:
    try:
        left_value = parse_float_hex(left)
        right_value = parse_float_hex(right)
    except ValueError:
        return False
    return (
        left_value is not None
        and right_value is not None
        and abs(left_value - right_value) <= FLOAT_TOLERANCE
    )


def _compare_replay(
    expected: Any, actual: Any, path: str = "$"
) -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            return [f"{path}:KEYS"]
        errors: list[str] = []
        for key in sorted(expected):
            errors.extend(
                _compare_replay(
                    expected[key], actual[key], f"{path}.{key}"
                )
            )
        return errors
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}:LENGTH"]
        errors = []
        for index, (left, right) in enumerate(zip(expected, actual)):
            errors.extend(
                _compare_replay(left, right, f"{path}[{index}]")
            )
        return errors
    if (
        isinstance(expected, str)
        and isinstance(actual, str)
        and ("0x" in expected or "0x" in actual)
    ):
        return [] if _float_hex_equal(expected, actual) else [f"{path}:FLOAT"]
    return [] if expected == actual else [f"{path}:VALUE"]


def validate_existing(repo_root: Path) -> dict[str, Any]:
    paths = _canonical_paths(repo_root)
    output = paths["output"]
    claim_raw = (output / "run_claim.json").read_bytes()
    receipt_path = output / "receipt.json"
    if not receipt_path.is_file():
        failure_path = output / "failure_receipt.json"
        if not failure_path.is_file():
            return {
                "status": "INVALID",
                "errors": ["CLAIM_CONSUMED_WITHOUT_RECEIPT"],
                "terminal_state": INVALID_TERMINAL,
                "gate_pass": False,
            }
        try:
            failure = _read_json_object(failure_path)
            claim = json.loads(claim_raw.decode("utf-8"))
            valid_failure = (
                failure.get("schema_version")
                == "rcle.phase_b.bonn.b1a.failure_receipt.v1"
                and failure.get("terminal_state") == INVALID_TERMINAL
                and failure.get("claim") == claim
                and failure.get("run_claim_sha256")
                == hashlib.sha256(claim_raw).hexdigest()
                and failure.get("bootstrap_runner_sha256")
                == claim.get("bootstrap_runner_sha256")
            )
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
            valid_failure = False
        return {
            "status": "INVALID",
            "errors": [
                "FORMAL_EXECUTION_FAILURE_RECEIPT"
                if valid_failure
                else "FAILURE_RECEIPT_INVALID"
            ],
            "terminal_state": INVALID_TERMINAL,
            "gate_pass": False,
        }
    receipt = _read_json_object(receipt_path)
    structural = validate_receipt_data(receipt, claim_raw=claim_raw)
    errors = list(structural["errors"])
    try:
        persisted_ledger = _read_json_object(output / "ledger.json")
        replayed_ledger, metadata = _recompute_ledger(
            repo_root, receipt, paths
        )
        errors.extend(
            f"REPLAY_MISMATCH:{error}"
            for error in _compare_replay(
                replayed_ledger, receipt.get("ledger")
            )
        )
        errors.extend(
            f"LEDGER_FILE_MISMATCH:{error}"
            for error in _compare_replay(
                receipt.get("ledger"), persisted_ledger
            )
        )
        replay_identity = canonical_sha256(replayed_ledger)
        if receipt.get("ledger_identity_sha256") != replay_identity:
            errors.append("REPLAY_LEDGER_IDENTITY")
        if (
            receipt.get("archive_sha256_by_sequence")
            != metadata["archive_sha256_by_sequence"]
        ):
            errors.append("REPLAY_ARCHIVE_HASHES")
        firewall = receipt.get("read_firewall", {})
        if (
            firewall.get("depth_decode_operations")
            != metadata["depth_decode_operations"]
            or firewall.get("pose_numeric_rows_parsed")
            != metadata["pose_numeric_rows_parsed"]
        ):
            errors.append("REPLAY_FIREWALL_COUNTS")
    except (
        KeyError,
        OSError,
        ValueError,
        zipfile.BadZipFile,
        json.JSONDecodeError,
    ) as error:
        errors.append(f"REPLAY_EXCEPTION:{error}")
    errors = sorted(set(errors))
    if errors:
        return {
            "status": "INVALID",
            "errors": errors,
            "terminal_state": INVALID_TERMINAL,
            "gate_pass": False,
        }
    return {
        "status": "VALID",
        "errors": [],
        "terminal_state": receipt["terminal_state"],
        "gate_pass": bool(receipt["gate_pass"]),
    }


__all__ = [
    "IndexRow",
    "PoseRow",
    "adjacent_pairs",
    "assign_depth_rows",
    "canonical_sha256",
    "classify_window",
    "decode_depth_png",
    "evaluate_truth",
    "float_hex",
    "interpolate_pose",
    "parse_index_text",
    "parse_pose_text",
    "quaternion_rotation",
    "relative_geometry",
    "rotation_homography",
    "slerp",
    "terminal_for_roles",
    "validate_existing",
    "validate_member_reference",
    "validate_receipt_data",
    "zbuffer_winners",
]
