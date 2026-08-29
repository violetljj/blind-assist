"""Score the frozen PB11 P0 metric portal-closure cohort.

This evaluator is deliberately geometry-only.  It verifies every frozen input,
decodes only the explicitly selected SUN RGB-D depth representation, and never
loads a learned model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

np: Any = None


PROTOCOL_SCHEMA = "l10-named-poi-metric-portal-closure-protocol-v1"
PROTOCOL_SHA256 = "041a1be9b744a4f4f83eeaa2b1bae6bc198a0f02f65b1567c3911beee65499b2"
COHORT_SCHEMA = "l10-named-poi-metric-portal-closure-cohort-v1"
RESULT_SCHEMA = "l10-named-poi-metric-portal-closure-p0-result-v1"
COHORT_STATUS = "FROZEN_BEFORE_P0_OUTPUT"
ARM = "P0_PRIVILEGED_SENSOR_DEPTH"

DEPTH_ENCODING = "SUNRGBD_ROTATE_RIGHT_3_MM"
DEPTH_CONTRACT: dict[str, Any] = {
    "container": "PNG_UINT16_GRAYSCALE",
    "encoding": DEPTH_ENCODING,
    "decoded_unit": "millimeter",
    "meters_per_decoded_unit": 0.001,
    "invalid_decoded_values": [0],
    "maximum_depth_m": 8.0,
    "above_maximum_behavior": "CLAMP_TO_MAXIMUM",
}
INTRINSICS_CONTRACT = {"encoding": "SUNRGBD_TXT_MATLAB_3X3"}
PIXEL_COORDINATE_CONTRACT = "ZERO_BASED_INTEGER_PIXEL_CENTERS_XY"
QUAD_CONTRACT = "TOP_LEFT_TOP_RIGHT_BOTTOM_RIGHT_BOTTOM_LEFT"

ROLE_COUNTS = {
    "DOOR_PLANE_POSITIVE": 4,
    "NO_PORTAL_NEGATIVE": 2,
    "LARGE_DOORLESS_OPENING_OOD": 2,
}
POSITIVE_ROLE = "DOOR_PLANE_POSITIVE"
CONTROL_ROLES = {"NO_PORTAL_NEGATIVE", "LARGE_DOORLESS_OPENING_OOD"}
FILE_KINDS = {"rgb", "depth", "intrinsics", "polygon", "scene_metadata"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
NUMBER_RE = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?$"
)


class ContractError(ValueError):
    """A frozen-input or mathematical contract was not satisfied."""


def _fail(code: str, detail: str | None = None) -> None:
    suffix = f":{detail}" if detail else ""
    raise ContractError(f"{code}{suffix}")


def _require_numpy() -> None:
    global np
    if np is not None:
        return
    try:
        import numpy as numpy_module
    except ImportError as exc:
        _fail("NUMPY_UNAVAILABLE", str(exc))
    np = numpy_module


def _json_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("DUPLICATE_JSON_KEY", key)
        result[key] = value
    return result


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_json_no_duplicates,
            parse_constant=lambda token: _fail("NONFINITE_JSON_NUMBER", token),
        )
    except ContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _fail(f"{label}_JSON_READ_FAILED", str(exc))
    if not isinstance(value, dict):
        _fail(f"{label}_JSON_ROOT_NOT_OBJECT")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        _fail("FILE_HASH_READ_FAILED", f"{path}:{exc}")
    return digest.hexdigest()


def _existing_file(value: Path, label: str) -> Path:
    try:
        path = value.resolve(strict=True)
    except OSError as exc:
        _fail(f"{label}_PATH_INVALID", str(exc))
    if not path.is_file():
        _fail(f"{label}_NOT_REGULAR_FILE", str(path))
    return path


def _existing_directory(value: Path, label: str) -> Path:
    try:
        path = value.resolve(strict=True)
    except OSError as exc:
        _fail(f"{label}_PATH_INVALID", str(exc))
    if not path.is_dir():
        _fail(f"{label}_NOT_DIRECTORY", str(path))
    return path


def _output_path(value: Path) -> Path:
    if value.exists() or value.is_symlink():
        _fail("OUTPUT_ALREADY_EXISTS", str(value))
    try:
        parent = value.parent.resolve(strict=True)
    except OSError as exc:
        _fail("OUTPUT_PARENT_INVALID", str(exc))
    if not parent.is_dir():
        _fail("OUTPUT_PARENT_NOT_DIRECTORY", str(parent))
    output = parent / value.name
    if output.exists() or output.is_symlink():
        _fail("OUTPUT_ALREADY_EXISTS", str(output))
    return output


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{label}_NOT_OBJECT")
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        _fail(f"{label}_INVALID")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        _fail(f"{label}_INVALID_SHA256")
    return value


def _require_exact_contract(
    actual: Any, expected: dict[str, Any], label: str
) -> dict[str, Any]:
    obj = _require_object(actual, label)
    if obj != expected:
        _fail(f"{label}_MISMATCH", json.dumps(obj, sort_keys=True))
    return obj


def _normalized_relative_posix(value: Any, label: str) -> str:
    text = _require_nonempty_string(value, label)
    if "\\" in text or "\x00" in text or ":" in text:
        _fail(f"{label}_NOT_CANONICAL_POSIX_RELATIVE", text)
    pure = PurePosixPath(text)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        _fail(f"{label}_NOT_CANONICAL_POSIX_RELATIVE", text)
    if pure.as_posix() != text:
        _fail(f"{label}_NOT_CANONICAL_POSIX_RELATIVE", text)
    return text


def _manifest_file(root: Path, spec: Any, label: str) -> tuple[Path, dict[str, Any]]:
    obj = _require_object(spec, label)
    if set(obj) != {"path", "sha256"}:
        _fail(f"{label}_FIELDS_MISMATCH", ",".join(sorted(obj)))
    relative = _normalized_relative_posix(obj["path"], f"{label}_PATH")
    expected = _require_sha256(obj["sha256"], label)
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        _fail(f"{label}_PATH_OUTSIDE_EXTRACTED_ROOT", str(exc))
    if not resolved.is_file():
        _fail(f"{label}_NOT_REGULAR_FILE", relative)
    size = resolved.stat().st_size
    if size <= 0:
        _fail(f"{label}_EMPTY_FILE", relative)
    actual = _sha256(resolved)
    if actual != expected:
        _fail(f"{label}_HASH_MISMATCH", f"{relative}:{actual}:{expected}")
    return resolved, {"path": relative, "sha256": actual, "bytes": size}


def _validate_protocol(protocol: dict[str, Any], actual_hash: str) -> None:
    if actual_hash != PROTOCOL_SHA256:
        _fail("PROTOCOL_HASH_MISMATCH", f"{actual_hash}:{PROTOCOL_SHA256}")
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        _fail("PROTOCOL_SCHEMA_MISMATCH")
    if protocol.get("status") != "FROZEN_BEFORE_COHORT_IMAGE_EXTRACTION_AND_MODEL_OUTPUT":
        _fail("PROTOCOL_NOT_FROZEN")
    roles = protocol.get("cohort_freeze", {}).get("roles")
    if roles != ROLE_COUNTS:
        _fail("PROTOCOL_ROLE_COUNTS_MISMATCH")
    if ARM not in protocol.get("arms", {}):
        _fail("PROTOCOL_P0_ARM_MISSING")


def _validate_quad(value: Any, width: int | None, height: int | None, label: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != 4:
        _fail(f"{label}_MUST_HAVE_FOUR_VERTICES")
    points: list[list[float]] = []
    for vertex_index, vertex in enumerate(value):
        if not isinstance(vertex, list) or len(vertex) != 2:
            _fail(f"{label}_VERTEX_INVALID", str(vertex_index))
        pair: list[float] = []
        for coordinate in vertex:
            if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
                _fail(f"{label}_COORDINATE_NOT_NUMBER", str(vertex_index))
            number = float(coordinate)
            if not math.isfinite(number):
                _fail(f"{label}_COORDINATE_NONFINITE", str(vertex_index))
            pair.append(number)
        points.append(pair)
    quad = np.asarray(points, dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(quad))))
    normalized_quad = quad / scale
    tolerance = np.finfo(np.float64).eps * 128.0
    edges = np.roll(normalized_quad, -1, axis=0) - normalized_quad
    crosses = np.asarray(
        [
            edges[index, 0] * edges[(index + 1) % 4, 1]
            - edges[index, 1] * edges[(index + 1) % 4, 0]
            for index in range(4)
        ]
    )
    if np.any(crosses <= tolerance):
        _fail(f"{label}_NOT_STRICT_CONVEX_TL_TR_BR_BL")
    top_y = float(np.mean(normalized_quad[[0, 1], 1]))
    bottom_y = float(np.mean(normalized_quad[[2, 3], 1]))
    left_x = float(np.mean(normalized_quad[[0, 3], 0]))
    right_x = float(np.mean(normalized_quad[[1, 2], 0]))
    if not top_y < bottom_y or not left_x < right_x:
        _fail(f"{label}_DECLARED_CORNER_ORDER_INCONSISTENT")
    if width is not None and height is not None:
        if (
            np.any(quad[:, 0] < 0.0)
            or np.any(quad[:, 0] > width - 1)
            or np.any(quad[:, 1] < 0.0)
            or np.any(quad[:, 1] > height - 1)
        ):
            _fail(f"{label}_OUTSIDE_SOURCE_IMAGE")
    return quad


def _validate_cohort(cohort: dict[str, Any], protocol_hash: str) -> list[dict[str, Any]]:
    if cohort.get("schema") != COHORT_SCHEMA:
        _fail("COHORT_SCHEMA_MISMATCH")
    if cohort.get("status") != COHORT_STATUS:
        _fail("COHORT_NOT_FROZEN")
    if cohort.get("arm") != ARM:
        _fail("COHORT_ARM_MISMATCH")
    protocol_ref = _require_object(cohort.get("protocol"), "COHORT_PROTOCOL")
    if set(protocol_ref) != {"schema", "sha256"}:
        _fail("COHORT_PROTOCOL_FIELDS_MISMATCH")
    if protocol_ref.get("schema") != PROTOCOL_SCHEMA:
        _fail("COHORT_PROTOCOL_SCHEMA_MISMATCH")
    if _require_sha256(protocol_ref.get("sha256"), "COHORT_PROTOCOL") != protocol_hash:
        _fail("COHORT_PROTOCOL_HASH_MISMATCH")
    _require_exact_contract(cohort.get("depth_contract"), DEPTH_CONTRACT, "DEPTH_CONTRACT")
    _require_exact_contract(
        cohort.get("intrinsics_contract"), INTRINSICS_CONTRACT, "INTRINSICS_CONTRACT"
    )
    if cohort.get("pixel_coordinate_contract") != PIXEL_COORDINATE_CONTRACT:
        _fail("PIXEL_COORDINATE_CONTRACT_MISMATCH")
    if cohort.get("quad_contract") != QUAD_CONTRACT:
        _fail("QUAD_CONTRACT_MISMATCH")
    frames = cohort.get("frames")
    if not isinstance(frames, list) or len(frames) != 8:
        _fail("COHORT_FRAME_COUNT_MISMATCH")

    required_fields = {
        "index",
        "frame_id",
        "capture_sequence_id",
        "sensor_source_bucket",
        "role",
        "audit_note",
        "canonical_source_path",
        "aperture_quad_xy",
        "files",
    }
    indexes: list[int] = []
    frame_ids: list[str] = []
    captures: list[str] = []
    buckets: list[str] = []
    roles: list[str] = []
    source_paths: list[str] = []
    for row_index, row_value in enumerate(frames):
        row = _require_object(row_value, f"FRAME_{row_index + 1}")
        missing = required_fields - set(row)
        if missing:
            _fail("FRAME_REQUIRED_FIELDS_MISSING", ",".join(sorted(missing)))
        index = row["index"]
        if isinstance(index, bool) or not isinstance(index, int):
            _fail("FRAME_INDEX_INVALID", str(row_index))
        indexes.append(index)
        frame_ids.append(_require_nonempty_string(row["frame_id"], "FRAME_ID"))
        captures.append(
            _require_nonempty_string(row["capture_sequence_id"], "CAPTURE_SEQUENCE_ID")
        )
        buckets.append(
            _require_nonempty_string(row["sensor_source_bucket"], "SENSOR_SOURCE_BUCKET")
        )
        role = _require_nonempty_string(row["role"], "FRAME_ROLE")
        if role not in ROLE_COUNTS:
            _fail("UNKNOWN_FRAME_ROLE", role)
        roles.append(role)
        _require_nonempty_string(row["audit_note"], "FRAME_AUDIT_NOTE")
        source_path = _normalized_relative_posix(
            row["canonical_source_path"], "CANONICAL_SOURCE_PATH"
        )
        source_paths.append(source_path)
        _validate_quad(row["aperture_quad_xy"], None, None, "APERTURE_QUAD")
        files = _require_object(row["files"], "FRAME_FILES")
        if set(files) != FILE_KINDS:
            _fail("FRAME_FILE_KINDS_MISMATCH", ",".join(sorted(files)))
        for kind, spec in files.items():
            file_spec = _require_object(spec, f"FRAME_FILE_{kind.upper()}")
            if set(file_spec) != {"path", "sha256"}:
                _fail("FRAME_FILE_SPEC_FIELDS_MISMATCH", kind)
            relative = _normalized_relative_posix(
                file_spec["path"], f"FRAME_FILE_{kind.upper()}_PATH"
            )
            expected_exact = {
                "intrinsics": f"{source_path}/intrinsics.txt",
                "polygon": f"{source_path}/annotation2Dfinal/index.json",
                "scene_metadata": f"{source_path}/scene.txt",
            }
            if kind in expected_exact and relative != expected_exact[kind]:
                _fail("FRAME_FILE_SOURCE_PATH_MISMATCH", f"{kind}:{relative}")
            if kind == "rgb":
                media = PurePosixPath(relative)
                if media.parent.as_posix() != f"{source_path}/image" or media.suffix.casefold() != ".jpg":
                    _fail("FRAME_FILE_SOURCE_PATH_MISMATCH", f"{kind}:{relative}")
            if kind == "depth":
                media = PurePosixPath(relative)
                if media.parent.as_posix() != f"{source_path}/depth" or media.suffix.casefold() != ".png":
                    _fail("FRAME_FILE_SOURCE_PATH_MISMATCH", f"{kind}:{relative}")
            _require_sha256(file_spec["sha256"], f"FRAME_FILE_{kind.upper()}")

    if sorted(indexes) != list(range(1, 9)):
        _fail("FRAME_INDEX_SET_MISMATCH")
    if len(set(frame_ids)) != 8:
        _fail("FRAME_ID_NOT_UNIQUE")
    if len(set(captures)) != 8:
        _fail("CAPTURE_SEQUENCE_NOT_UNIQUE")
    if len(set(source_paths)) != 8:
        _fail("CANONICAL_SOURCE_PATH_NOT_UNIQUE")
    if dict(Counter(roles)) != ROLE_COUNTS:
        _fail("COHORT_ROLE_COUNTS_MISMATCH", json.dumps(dict(Counter(roles)), sort_keys=True))
    bucket_counts = Counter(buckets)
    if len(bucket_counts) < 3 or max(bucket_counts.values()) > 3:
        _fail("COHORT_SENSOR_SOURCE_DIVERSITY_MISMATCH")
    return sorted(frames, key=lambda row: row["index"])


def _load_rgb_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image, UnidentifiedImageError

        try:
            with Image.open(path) as image:
                image.load()
                width, height = image.size
        except (OSError, UnidentifiedImageError) as exc:
            _fail("RGB_DECODE_FAILED", f"{path}:{exc}")
    except ImportError as exc:
        _fail("PILLOW_UNAVAILABLE", str(exc))
    if width <= 0 or height <= 0:
        _fail("RGB_SIZE_INVALID", str(path))
    return int(width), int(height)


def _png_uint16_header(path: Path) -> tuple[int, int]:
    try:
        header = path.read_bytes()[:33]
    except OSError as exc:
        _fail("DEPTH_PNG_HEADER_READ_FAILED", str(exc))
    if len(header) < 33 or header[:8] != b"\x89PNG\r\n\x1a\n":
        _fail("DEPTH_CONTAINER_NOT_PNG", str(path))
    if int.from_bytes(header[8:12], "big") != 13 or header[12:16] != b"IHDR":
        _fail("DEPTH_PNG_IHDR_INVALID", str(path))
    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    bit_depth, color_type, compression, filtering, interlace = header[24:29]
    if bit_depth != 16 or color_type != 0:
        _fail("DEPTH_PNG_NOT_UINT16_GRAYSCALE", f"bit_depth={bit_depth},color_type={color_type}")
    if compression != 0 or filtering != 0 or interlace not in {0, 1}:
        _fail("DEPTH_PNG_UNSUPPORTED_IHDR")
    if width <= 0 or height <= 0:
        _fail("DEPTH_PNG_SIZE_INVALID")
    return width, height


def _load_depth_raw(path: Path, expected_size: tuple[int, int]) -> np.ndarray:
    header_size = _png_uint16_header(path)
    if header_size != expected_size:
        _fail(
            "RGB_DEPTH_SIZE_MISMATCH",
            f"rgb={expected_size[0]}x{expected_size[1]}:depth={header_size[0]}x{header_size[1]}",
        )
    try:
        from PIL import Image, UnidentifiedImageError

        try:
            with Image.open(path) as image:
                if image.format != "PNG":
                    _fail("DEPTH_CONTAINER_NOT_PNG", str(path))
                image.load()
                array = np.array(image, copy=True)
                decoded_size = image.size
        except (OSError, UnidentifiedImageError) as exc:
            _fail("DEPTH_PNG_DECODE_FAILED", f"{path}:{exc}")
    except ImportError as exc:
        _fail("PILLOW_UNAVAILABLE", str(exc))
    if decoded_size != header_size or array.ndim != 2:
        _fail("DEPTH_PNG_DECODE_CONTRACT_MISMATCH", str(path))
    if array.dtype.kind not in {"u", "i"}:
        _fail("DEPTH_PNG_DECODE_NOT_INTEGER", str(array.dtype))
    if array.size == 0 or int(np.min(array)) < 0 or int(np.max(array)) > 65535:
        _fail("DEPTH_PNG_DECODE_OUTSIDE_UINT16")
    return array.astype(np.uint16, copy=False)


def _decode_sunrgbd_depth(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    if raw.dtype != np.uint16 or raw.ndim != 2:
        _fail("DEPTH_RAW_ARRAY_CONTRACT_MISMATCH")
    widened = raw.astype(np.uint32)
    decoded_mm = ((widened >> 3) | (widened << 13)) & np.uint32(0xFFFF)
    valid = decoded_mm != 0
    above_maximum = decoded_mm > 8000
    depth_m = np.minimum(decoded_mm, np.uint32(8000)).astype(np.float64) * 0.001
    return depth_m, valid, {
        "pixels": int(raw.size),
        "invalid_zero_pixels": int(np.count_nonzero(~valid)),
        "clamped_above_8m_pixels": int(np.count_nonzero(above_maximum)),
    }


def _load_intrinsics(path: Path, width: int, height: int) -> np.ndarray:
    try:
        text = path.read_bytes().decode("ascii")
    except (OSError, UnicodeError) as exc:
        _fail("INTRINSICS_READ_FAILED", f"{path}:{exc}")
    tokens = text.split()
    if len(tokens) != 9 or any(NUMBER_RE.fullmatch(token) is None for token in tokens):
        _fail("INTRINSICS_TOKEN_CONTRACT_MISMATCH", str(path))
    matrix = np.asarray([float(token) for token in tokens], dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(matrix)):
        _fail("INTRINSICS_NONFINITE")
    if not np.allclose(matrix[2], [0.0, 0.0, 1.0], rtol=0.0, atol=1e-12):
        _fail("INTRINSICS_LAST_ROW_MISMATCH")
    if (
        matrix[0, 0] <= 0.0
        or matrix[1, 1] <= 0.0
        or abs(matrix[0, 1]) > 1e-12
        or abs(matrix[1, 0]) > 1e-12
    ):
        _fail("INTRINSICS_PINHOLE_FORM_MISMATCH")
    if not (0.0 <= matrix[0, 2] <= width - 1 and 0.0 <= matrix[1, 2] <= height - 1):
        _fail("INTRINSICS_PRINCIPAL_POINT_OUTSIDE_IMAGE")
    if not math.isfinite(float(np.linalg.det(matrix))) or abs(float(np.linalg.det(matrix))) <= 1e-12:
        _fail("INTRINSICS_SINGULAR")
    return matrix


def _normalization_transform(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = np.mean(points, axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    mean_distance = float(np.mean(distances))
    if not math.isfinite(mean_distance) or mean_distance <= np.finfo(np.float64).eps:
        _fail("HOMOGRAPHY_POINT_SET_DEGENERATE")
    scale = math.sqrt(2.0) / mean_distance
    transform = np.asarray(
        [[scale, 0.0, -scale * centroid[0]], [0.0, scale, -scale * centroid[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    normalized = (transform @ homogeneous.T).T[:, :2]
    return normalized, transform


def _homography_quad_to_unit(quad: np.ndarray) -> np.ndarray:
    unit = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    source, source_transform = _normalization_transform(quad)
    target, target_transform = _normalization_transform(unit)
    rows: list[list[float]] = []
    for (x, y), (u, v) in zip(source, target):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    design = np.asarray(rows, dtype=np.float64)
    _, singular_values, right = np.linalg.svd(design, full_matrices=True)
    tolerance = np.finfo(np.float64).eps * max(design.shape) * singular_values[0]
    if singular_values[-1] <= tolerance:
        _fail("QUAD_HOMOGRAPHY_RANK_DEFICIENT")
    normalized_h = right[-1].reshape(3, 3)
    homography = np.linalg.inv(target_transform) @ normalized_h @ source_transform
    magnitude = float(np.max(np.abs(homography)))
    if not math.isfinite(magnitude) or magnitude <= np.finfo(np.float64).eps:
        _fail("QUAD_HOMOGRAPHY_INVALID")
    homography /= magnitude
    pivot = homography.flat[int(np.argmax(np.abs(homography)))]
    if pivot < 0.0:
        homography = -homography
    if abs(float(np.linalg.det(homography))) <= np.finfo(np.float64).eps:
        _fail("QUAD_HOMOGRAPHY_SINGULAR")
    projected = _project_points(quad[:, 0], quad[:, 1], homography)
    if not np.allclose(projected, unit, rtol=0.0, atol=1e-8):
        _fail("QUAD_HOMOGRAPHY_CORNER_CHECK_FAILED")
    return homography


def _project_points(x: np.ndarray, y: np.ndarray, homography: np.ndarray) -> np.ndarray:
    pixels = np.column_stack((x, y, np.ones(len(x), dtype=np.float64)))
    projected = (homography @ pixels.T).T
    scale = np.max(np.abs(projected[:, :2]), axis=1) + 1.0
    if np.any(np.abs(projected[:, 2]) <= np.finfo(np.float64).eps * scale * 128.0):
        _fail("QUAD_HOMOGRAPHY_ZERO_DENOMINATOR")
    return projected[:, :2] / projected[:, 2, None]


def _region_pixels(width: int, height: int, quad: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    homography = _homography_quad_to_unit(quad)
    x_min = max(0, int(math.ceil(float(np.min(quad[:, 0])))))
    x_max = min(width - 1, int(math.floor(float(np.max(quad[:, 0])))))
    y_min = max(0, int(math.ceil(float(np.min(quad[:, 1])))))
    y_max = min(height - 1, int(math.floor(float(np.max(quad[:, 1])))))
    if x_min > x_max or y_min > y_max:
        return {
            "rim": (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)),
            "interior": (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)),
        }
    grid_x, grid_y = np.meshgrid(
        np.arange(x_min, x_max + 1, dtype=np.int64),
        np.arange(y_min, y_max + 1, dtype=np.int64),
    )
    flat_x = grid_x.ravel()
    flat_y = grid_y.ravel()
    normalized = _project_points(flat_x.astype(np.float64), flat_y.astype(np.float64), homography)
    tolerance = 1e-10
    inside = (
        (normalized[:, 0] >= -tolerance)
        & (normalized[:, 0] <= 1.0 + tolerance)
        & (normalized[:, 1] >= -tolerance)
        & (normalized[:, 1] <= 1.0 + tolerance)
    )
    x_inside = flat_x[inside]
    y_inside = flat_y[inside]
    unit_inside = np.clip(normalized[inside], 0.0, 1.0)
    central = (
        (unit_inside[:, 0] >= 0.15)
        & (unit_inside[:, 0] <= 0.85)
        & (unit_inside[:, 1] >= 0.15)
        & (unit_inside[:, 1] <= 0.85)
    )
    return {
        "rim": (x_inside[~central], y_inside[~central]),
        "interior": (x_inside[central], y_inside[central]),
    }


def _camera_rays(
    x: np.ndarray, y: np.ndarray, inverse_k: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.column_stack(
        (x.astype(np.float64), y.astype(np.float64), np.ones(len(x), dtype=np.float64))
    )
    rays = (inverse_k @ pixels.T).T
    valid = np.all(np.isfinite(rays), axis=1) & (rays[:, 2] > 1e-12)
    normalized = np.full_like(rays, np.nan)
    normalized[valid] = rays[valid] / rays[valid, 2, None]
    return normalized, valid


def _fit_tls_plane(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray] | None:
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        return None
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    if len(singular_values) < 2 or singular_values[0] <= 0.0:
        return None
    tolerance = np.finfo(np.float64).eps * max(centered.shape) * singular_values[0]
    if singular_values[1] <= tolerance:
        return None
    normal = right[-1].astype(np.float64)
    normal /= np.linalg.norm(normal)
    pivot = int(np.argmax(np.abs(normal)))
    if normal[pivot] < 0.0:
        normal = -normal
    offset = -float(np.dot(normal, centroid))
    return normal, offset, singular_values


def _linear_quantile(values: np.ndarray, quantile: float) -> float:
    if values.ndim != 1 or len(values) == 0 or not 0.0 <= quantile <= 1.0:
        _fail("QUANTILE_INPUT_INVALID")
    ordered = np.sort(values.astype(np.float64, copy=False))
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _sixty_percent(count: int, total: int) -> bool:
    return total > 0 and count * 5 >= total * 3


def _score_arrays(
    depth_m: np.ndarray,
    depth_valid: np.ndarray,
    intrinsics: np.ndarray,
    quad: np.ndarray,
) -> dict[str, Any]:
    height, width = depth_m.shape
    regions = _region_pixels(width, height, quad)
    inverse_k = np.linalg.inv(intrinsics)
    rim_x, rim_y = regions["rim"]
    interior_x, interior_y = regions["interior"]
    rim_total = len(rim_x)
    interior_total = len(interior_x)
    rim_source_mask = depth_valid[rim_y, rim_x] if rim_total else np.empty(0, dtype=bool)
    interior_source_mask = (
        depth_valid[interior_y, interior_x] if interior_total else np.empty(0, dtype=bool)
    )
    rim_source_count = int(np.count_nonzero(rim_source_mask))
    interior_source_count = int(np.count_nonzero(interior_source_mask))
    diagnostics: dict[str, Any] = {
        "regions": {
            "rim": {
                "pixels": rim_total,
                "source_depth_valid_pixels": rim_source_count,
                "source_depth_valid_fraction": rim_source_count / rim_total if rim_total else 0.0,
                "scoring_valid_pixels": 0,
                "scoring_valid_fraction": 0.0,
            },
            "interior": {
                "pixels": interior_total,
                "source_depth_valid_pixels": interior_source_count,
                "source_depth_valid_fraction": (
                    interior_source_count / interior_total if interior_total else 0.0
                ),
                "scoring_valid_pixels": 0,
                "scoring_valid_fraction": 0.0,
            },
        },
        "plane": None,
        "relief_q75": None,
        "rim_residual_ratio": None,
        "closure_score": None,
        "evaluable": False,
        "not_evaluable_reasons": [],
    }
    reasons: list[str] = diagnostics["not_evaluable_reasons"]
    if not _sixty_percent(rim_source_count, rim_total):
        reasons.append("RIM_VALID_FRACTION_BELOW_60_PERCENT")
    if not _sixty_percent(interior_source_count, interior_total):
        reasons.append("INTERIOR_VALID_FRACTION_BELOW_60_PERCENT")
    if reasons:
        return diagnostics

    rim_x_valid = rim_x[rim_source_mask]
    rim_y_valid = rim_y[rim_source_mask]
    rim_rays, rim_ray_valid = _camera_rays(rim_x_valid, rim_y_valid, inverse_k)
    rim_scoring_count = int(np.count_nonzero(rim_ray_valid))
    diagnostics["regions"]["rim"]["scoring_valid_pixels"] = rim_scoring_count
    diagnostics["regions"]["rim"]["scoring_valid_fraction"] = (
        rim_scoring_count / rim_total if rim_total else 0.0
    )
    if not _sixty_percent(rim_scoring_count, rim_total):
        reasons.append("RIM_BACKPROJECTABLE_FRACTION_BELOW_60_PERCENT")
        return diagnostics
    rim_z = depth_m[rim_y_valid, rim_x_valid][rim_ray_valid]
    rim_points = rim_rays[rim_ray_valid] * rim_z[:, None]
    plane = _fit_tls_plane(rim_points)
    if plane is None:
        reasons.append("RIM_TLS_PLANE_DEGENERATE")
        return diagnostics
    normal, offset, singular_values = plane
    residuals = np.abs(rim_points @ normal + offset)
    ranges = np.linalg.norm(rim_points, axis=1)
    median_range = float(np.median(ranges))
    if not math.isfinite(median_range) or median_range <= 0.0:
        reasons.append("RIM_MEDIAN_RANGE_INVALID")
        return diagnostics
    median_residual = float(np.median(residuals))
    residual_ratio = median_residual / median_range
    diagnostics["plane"] = {
        "fit": "UNWEIGHTED_TOTAL_LEAST_SQUARES",
        "point_count": len(rim_points),
        "unit_normal_xyz": [float(value) for value in normal],
        "offset_m": offset,
        "singular_values_m": [float(value) for value in singular_values],
        "median_absolute_point_to_plane_residual_m": median_residual,
        "median_euclidean_rim_range_m": median_range,
    }
    diagnostics["rim_residual_ratio"] = residual_ratio

    interior_x_valid = interior_x[interior_source_mask]
    interior_y_valid = interior_y[interior_source_mask]
    interior_rays, interior_ray_valid = _camera_rays(
        interior_x_valid, interior_y_valid, inverse_k
    )
    denominators = interior_rays @ normal
    expected_z = np.full(len(interior_rays), np.nan, dtype=np.float64)
    intersection_valid = interior_ray_valid & np.isfinite(denominators) & (np.abs(denominators) > 1e-12)
    expected_z[intersection_valid] = -offset / denominators[intersection_valid]
    intersection_valid &= np.isfinite(expected_z) & (expected_z > 0.0)
    interior_scoring_count = int(np.count_nonzero(intersection_valid))
    diagnostics["regions"]["interior"]["scoring_valid_pixels"] = interior_scoring_count
    diagnostics["regions"]["interior"]["scoring_valid_fraction"] = (
        interior_scoring_count / interior_total if interior_total else 0.0
    )
    if not _sixty_percent(interior_scoring_count, interior_total):
        reasons.append("INTERIOR_RAY_PLANE_VALID_FRACTION_BELOW_60_PERCENT")
        return diagnostics

    observed_z = depth_m[interior_y_valid, interior_x_valid][intersection_valid]
    expected = expected_z[intersection_valid]
    relief = np.clip((observed_z - expected) / np.maximum(expected, 0.1), 0.0, 1.0)
    relief_q75 = _linear_quantile(relief, 0.75)
    closure_score = math.exp(-residual_ratio) * math.exp(-relief_q75)
    if not all(math.isfinite(value) for value in (relief_q75, residual_ratio, closure_score)):
        reasons.append("NONFINITE_SCORE")
        return diagnostics
    diagnostics["relief_q75"] = relief_q75
    diagnostics["closure_score"] = closure_score
    diagnostics["evaluable"] = True
    return diagnostics


def _roc_auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    if not positive or not negative:
        _fail("ROC_AUC_DENOMINATOR_MISSING")
    wins = 0.0
    for positive_score in positive:
        for negative_score in negative:
            if positive_score > negative_score:
                wins += 1.0
            elif positive_score == negative_score:
                wins += 0.5
    return wins / (len(positive) * len(negative))


def _aggregate(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    evaluable_by_role = {
        role: sum(row["role"] == role and row["evaluable"] for row in rows)
        for role in ROLE_COUNTS
    }
    positive_rows = [row for row in rows if row["role"] == POSITIVE_ROLE]
    control_rows = [row for row in rows if row["role"] in CONTROL_ROLES]
    all_evaluable = len(rows) == 8 and all(row["evaluable"] for row in rows)
    minimum_positive: float | None = None
    maximum_control: float | None = None
    margin: float | None = None
    auc: float | None = None
    threshold: float | None = None
    balanced_accuracy: float | None = None
    strict_gap = False
    if all_evaluable:
        positive_scores = [float(row["closure_score"]) for row in positive_rows]
        control_scores = [float(row["closure_score"]) for row in control_rows]
        minimum_positive = min(positive_scores)
        maximum_control = max(control_scores)
        margin = minimum_positive - maximum_control
        auc = _roc_auc(positive_scores, control_scores)
        strict_gap = margin > 0.0
        if strict_gap:
            threshold = 0.5 * (minimum_positive + maximum_control)
            true_positive_rate = sum(score > threshold for score in positive_scores) / len(
                positive_scores
            )
            true_negative_rate = sum(score <= threshold for score in control_scores) / len(
                control_scores
            )
            balanced_accuracy = 0.5 * (true_positive_rate + true_negative_rate)
    metrics = {
        "frames": len(rows),
        "evaluable_frames": sum(row["evaluable"] for row in rows),
        "evaluable_frames_by_role": evaluable_by_role,
        "positive_scores": [
            {"index": row["index"], "frame_id": row["frame_id"], "score": row["closure_score"]}
            for row in positive_rows
        ],
        "control_scores": [
            {
                "index": row["index"],
                "frame_id": row["frame_id"],
                "role": row["role"],
                "score": row["closure_score"],
            }
            for row in control_rows
        ],
        "minimum_positive_score": minimum_positive,
        "maximum_control_score": maximum_control,
        "strict_separation_margin": margin,
        "roc_auc": auc,
        "roc_auc_definition": "PAIRWISE_MANN_WHITNEY_WITH_HALF_CREDIT_FOR_TIES",
        "midpoint_threshold": threshold,
        "balanced_accuracy_at_midpoint": balanced_accuracy,
        "aggregate_score_metrics_require_8_of_8_evaluable": True,
    }
    gate = {
        "eight_of_eight_evaluable": all_evaluable,
        "minimum_positive_strictly_greater_than_maximum_control": strict_gap,
        "privileged_mechanism_gate_met": all_evaluable and strict_gap,
        "outcome": "P0_PASS" if all_evaluable and strict_gap else "P0_FAIL",
        "decision": (
            "L10_PB11_METRIC_PORTAL_CLOSURE_P0_PRIVILEGED_GATE_MET"
            if all_evaluable and strict_gap
            else "L10_PB11_METRIC_PORTAL_CLOSURE_P0_PRIVILEGED_GATE_NOT_MET"
        ),
    }
    return metrics, gate


def _score_frame(root: Path, frame: dict[str, Any]) -> dict[str, Any]:
    paths: dict[str, Path] = {}
    receipts: dict[str, dict[str, Any]] = {}
    for kind in sorted(FILE_KINDS):
        path, receipt = _manifest_file(
            root, frame["files"][kind], f"FRAME_{frame['index']}_{kind.upper()}"
        )
        paths[kind] = path
        receipts[kind] = receipt
    width, height = _load_rgb_size(paths["rgb"])
    raw_depth = _load_depth_raw(paths["depth"], (width, height))
    if raw_depth.shape != (height, width):
        _fail(
            "RGB_DEPTH_SIZE_MISMATCH",
            f"frame={frame['index']}:rgb={width}x{height}:depth={raw_depth.shape[1]}x{raw_depth.shape[0]}",
        )
    intrinsics = _load_intrinsics(paths["intrinsics"], width, height)
    quad = _validate_quad(frame["aperture_quad_xy"], width, height, "APERTURE_QUAD")
    depth_m, valid, decode_stats = _decode_sunrgbd_depth(raw_depth)
    score = _score_arrays(depth_m, valid, intrinsics, quad)
    return {
        "index": frame["index"],
        "frame_id": frame["frame_id"],
        "capture_sequence_id": frame["capture_sequence_id"],
        "sensor_source_bucket": frame["sensor_source_bucket"],
        "role": frame["role"],
        "audit_note": frame["audit_note"],
        "canonical_source_path": frame["canonical_source_path"],
        "files": receipts,
        "image_size_wh": [width, height],
        "intrinsics_k": [[float(value) for value in row] for row in intrinsics],
        "aperture_quad_xy": [[float(value) for value in row] for row in quad],
        "depth_decode": decode_stats,
        **score,
    }


def _write_json_new(path: Path, value: dict[str, Any]) -> None:
    rendered = json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ) + "\n"
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError:
        _fail("OUTPUT_ALREADY_EXISTS", str(path))
    except OSError as exc:
        _fail("OUTPUT_WRITE_FAILED", str(exc))
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _run(
    protocol_argument: Path,
    cohort_argument: Path,
    extracted_root_argument: Path,
    depth_encoding: str,
    output_argument: Path,
) -> dict[str, Any]:
    _require_numpy()
    if depth_encoding != DEPTH_ENCODING:
        _fail("CLI_DEPTH_ENCODING_UNSUPPORTED", depth_encoding)
    protocol_path = _existing_file(protocol_argument, "PROTOCOL")
    cohort_path = _existing_file(cohort_argument, "COHORT")
    extracted_root = _existing_directory(extracted_root_argument, "EXTRACTED_ROOT")
    output_path = _output_path(output_argument)
    protocol_hash = _sha256(protocol_path)
    protocol = _read_json(protocol_path, "PROTOCOL")
    _validate_protocol(protocol, protocol_hash)
    cohort_hash = _sha256(cohort_path)
    cohort = _read_json(cohort_path, "COHORT")
    frames = _validate_cohort(cohort, protocol_hash)
    rows = [_score_frame(extracted_root, frame) for frame in frames]
    metrics, gate = _aggregate(rows)
    result = {
        "schema": RESULT_SCHEMA,
        "experiment": "L10-PB11 Metric Portal Closure",
        "arm": ARM,
        "inputs": {
            "protocol": {
                "path": str(protocol_path),
                "schema": PROTOCOL_SCHEMA,
                "sha256": protocol_hash,
            },
            "cohort": {
                "path": str(cohort_path),
                "schema": COHORT_SCHEMA,
                "sha256": cohort_hash,
                "status": COHORT_STATUS,
            },
            "extracted_root": str(extracted_root),
        },
        "scorer": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
            "learned_models_loaded": 0,
        },
        "contracts": {
            "depth": {
                **DEPTH_CONTRACT,
                "cli_selected_encoding": depth_encoding,
                "cohort_selected_encoding": cohort["depth_contract"]["encoding"],
                "decode_expression_uint16": "((raw >> 3) | (raw << 13)) & 0xffff",
            },
            "intrinsics": {
                **INTRINSICS_CONTRACT,
                "parse_expression": "numpy.reshape(9_ascii_tokens,(3,3),order='C')",
                "sunrgbd_matlab_equivalent": "reshape(fscanf(file),[3,3])'",
            },
            "pixel_coordinates": PIXEL_COORDINATE_CONTRACT,
            "quad": QUAD_CONTRACT,
            "regions": {
                "mapping": "PROJECTIVE_QUAD_TO_UNIT_SQUARE_FOR_MEMBERSHIP_ONLY",
                "rim": "unit-square points outside inclusive [0.15,0.85] x [0.15,0.85]",
                "interior": "inclusive [0.15,0.85] x [0.15,0.85]",
                "backprojection": "ORIGINAL_INTEGER_PIXEL_COORDINATES_WITH_UNSCALED_SOURCE_K",
            },
            "plane": "UNWEIGHTED_TOTAL_LEAST_SQUARES_ON_VALID_RIM_CAMERA_POINTS",
            "rim_range": "EUCLIDEAN_CAMERA_RANGE",
            "relief_quantile": "Q75_LINEAR_TYPE7",
            "validity_minimum": 0.6,
        },
        "frames": rows,
        "metrics": metrics,
        "gate": gate,
        "claim_boundary": protocol["claim_boundary"],
    }
    _write_json_new(output_path, result)
    return result


def _synthetic_geometry_self_test() -> dict[str, Any]:
    _require_numpy()
    width = height = 101
    intrinsics = np.asarray(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    quad = _validate_quad(
        [[10.0, 10.0], [90.0, 10.0], [90.0, 90.0], [10.0, 90.0]],
        width,
        height,
        "SYNTHETIC_QUAD",
    )
    valid = np.ones((height, width), dtype=bool)
    planar_depth = np.full((height, width), 2.0, dtype=np.float64)
    planar = _score_arrays(planar_depth, valid, intrinsics, quad)
    if not planar["evaluable"] or not math.isclose(
        planar["closure_score"], 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise AssertionError(f"planar closure self-test failed: {planar}")

    opening_depth = planar_depth.copy()
    interior_x, interior_y = _region_pixels(width, height, quad)["interior"]
    opening_depth[interior_y, interior_x] = 4.0
    opening = _score_arrays(opening_depth, valid, intrinsics, quad)
    if not opening["evaluable"] or not math.isclose(
        opening["closure_score"], math.exp(-1.0), rel_tol=0.0, abs_tol=1e-12
    ):
        raise AssertionError(f"opening relief self-test failed: {opening}")

    decoded_target = np.asarray([[0, 2000, 9000]], dtype=np.uint16)
    encoded = ((decoded_target.astype(np.uint32) << 3) | (decoded_target.astype(np.uint32) >> 13))
    depth_m, decoded_valid, stats = _decode_sunrgbd_depth(encoded.astype(np.uint16))
    if not np.array_equal(decoded_valid, [[False, True, True]]) or not np.allclose(
        depth_m, [[0.0, 2.0, 8.0]], rtol=0.0, atol=0.0
    ):
        raise AssertionError("SUN RGB-D rotate-right-3 decode self-test failed")

    synthetic_rows = [
        {
            "index": index + 1,
            "frame_id": f"p{index + 1}",
            "role": POSITIVE_ROLE,
            "evaluable": True,
            "closure_score": 0.9 + index * 0.01,
        }
        for index in range(4)
    ] + [
        {
            "index": index + 5,
            "frame_id": f"c{index + 1}",
            "role": "NO_PORTAL_NEGATIVE" if index < 2 else "LARGE_DOORLESS_OPENING_OOD",
            "evaluable": True,
            "closure_score": 0.1 + index * 0.01,
        }
        for index in range(4)
    ]
    metrics, gate = _aggregate(synthetic_rows)
    if metrics["roc_auc"] != 1.0 or metrics["balanced_accuracy_at_midpoint"] != 1.0:
        raise AssertionError("aggregate metric self-test failed")
    if not gate["privileged_mechanism_gate_met"]:
        raise AssertionError("8/8 strict-gap gate self-test failed")
    tied_rows = [dict(row) for row in synthetic_rows]
    tied_rows[4]["closure_score"] = min(
        row["closure_score"] for row in tied_rows[:4]
    )
    tied_metrics, tied_gate = _aggregate(tied_rows)
    if (
        tied_metrics["strict_separation_margin"] != 0.0
        or tied_metrics["midpoint_threshold"] is not None
        or tied_metrics["balanced_accuracy_at_midpoint"] is not None
        or tied_gate["privileged_mechanism_gate_met"]
    ):
        raise AssertionError("non-strict gap suppression self-test failed")
    if not _sixty_percent(3, 5) or _sixty_percent(2, 5):
        raise AssertionError("exact 60 percent validity self-test failed")
    return {
        "status": "PASS",
        "planar_closure_score": planar["closure_score"],
        "opening_closure_score": opening["closure_score"],
        "decode_stats": stats,
        "roc_auc": metrics["roc_auc"],
        "balanced_accuracy_at_midpoint": metrics["balanced_accuracy_at_midpoint"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score the frozen PB11 P0 SUN RGB-D cohort with pure metric geometry.",
        epilog=(
            "The cohort must use schema l10-named-poi-metric-portal-closure-cohort-v1 "
            "and manifest every RGB, depth, intrinsics, polygon, and scene-metadata file "
            "as an extracted-root-relative POSIX path plus lowercase SHA-256. No depth "
            "encoding is inferred."
        ),
    )
    parser.add_argument("--protocol", type=Path, required=True, help="Frozen PB11 protocol JSON.")
    parser.add_argument("--cohort", type=Path, required=True, help="Frozen eight-frame cohort JSON.")
    parser.add_argument(
        "--extracted-root",
        type=Path,
        required=True,
        help="Root containing only the selectively extracted manifest files.",
    )
    parser.add_argument(
        "--depth-encoding",
        choices=[DEPTH_ENCODING],
        required=True,
        help="Required explicit depth encoding; must also match cohort.depth_contract.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New stable JSON result path; an existing path is never overwritten.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = _run(
            args.protocol,
            args.cohort,
            args.extracted_root,
            args.depth_encoding,
            args.output,
        )
    except ContractError as exc:
        print(f"PB11_CONTRACT_ERROR:{exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "decision": result["gate"]["decision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
