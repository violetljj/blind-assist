"""Index and selectively extract SUN RGB-D sources for the PB11 source audit.

This utility is intentionally model-free.  ``audit`` classifies captures only by
the presence of an exact ``door`` object name in the official 2D annotation.
``extract`` copies the five source asset classes needed for a later, human-frozen
cohort.  Neither command assigns NONE/OOD roles or downloads any data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
import unicodedata
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = Path(__file__).with_name(
    "named_poi_metric_portal_closure_protocol_v1.json"
)
ARTIFACTS_LOCAL = ROOT / "artifacts.local"
DEFAULT_EXTRACT_ROOT = ARTIFACTS_LOCAL / "datasets" / "sunrgbd-metric-portal-closure"

SCHEMA = "l10-named-poi-sunrgbd-metric-closure-source-audit-v1"
EXPECTED_PROTOCOL_SCHEMA = "l10-named-poi-metric-portal-closure-protocol-v1"
EXPECTED_DATASET = "SUN RGB-D"
ANNOTATION_SUFFIX = "/annotation2Dfinal/index.json"
SCENE_SUFFIX = "/scene.txt"
INTRINSICS_SUFFIX = "/intrinsics.txt"
MAX_ANNOTATION_BYTES = 64 * 1024 * 1024
MAX_TEXT_BYTES = 64 * 1024
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
WINDOWS_INVALID_CHARS = re.compile(r'[<>:"|?*]')
RGBF_CAPTURE_RE = re.compile(
    r"^(?P<sequence>.+)_(?P<frame>rgbf[0-9]+(?:-resize)?)$", re.IGNORECASE
)
SUN3D_FRAME_RE = re.compile(r"^[0-9]{7}-[0-9]{12}$")
REALSENSE_SINGLE_CAPTURE_RE = re.compile(
    r"^[0-9]{4}_[0-9]{2}_[0-9]{2}-[0-9]{2}_[0-9]{2}_[0-9]{2}-[0-9]+$"
)
NEAR_LABEL_TERMS = ("doorway", "door frame", "arch", "opening", "passage", "hallway")


class SourceAuditError(ValueError):
    """Raised when source identity, structure, or content is not trustworthy."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        return os.path.commonpath((str(path), str(parent))) == str(parent)
    except ValueError:
        return False


def _json_object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SourceAuditError(f"JSON_DUPLICATE_KEY:{key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise SourceAuditError(f"JSON_NON_FINITE_NUMBER:{value}")


def _parse_json_bytes(payload: bytes, identity: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8-sig", errors="strict")
        parsed = json.loads(
            text,
            object_pairs_hook=_json_object_without_duplicates,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, SourceAuditError) as error:
        raise SourceAuditError(
            f"INVALID_JSON:{identity}:{type(error).__name__}:{error}"
        ) from error
    if not isinstance(parsed, dict):
        raise SourceAuditError(f"JSON_ROOT_NOT_OBJECT:{identity}")
    return parsed


def _load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise SourceAuditError(f"PROTOCOL_NOT_FOUND:{path}")
    payload = path.read_bytes()
    protocol = _parse_json_bytes(payload, str(path))
    if protocol.get("schema") != EXPECTED_PROTOCOL_SCHEMA:
        raise SourceAuditError(
            f"PROTOCOL_SCHEMA_MISMATCH:{protocol.get('schema')!r}"
        )
    source = protocol.get("source")
    if not isinstance(source, dict) or source.get("dataset") != EXPECTED_DATASET:
        raise SourceAuditError("PROTOCOL_SOURCE_IS_NOT_SUN_RGB_D")
    expected_bytes = source.get("expected_archive_bytes")
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
        raise SourceAuditError("PROTOCOL_EXPECTED_ARCHIVE_BYTES_INVALID")
    if expected_bytes <= 0:
        raise SourceAuditError("PROTOCOL_EXPECTED_ARCHIVE_BYTES_INVALID")
    return protocol, _sha256_bytes(payload)


def _validate_archive_identity(archive: Path, protocol: dict[str, Any]) -> int:
    if not archive.is_file():
        raise SourceAuditError(f"ARCHIVE_NOT_FOUND:{archive}")
    if archive.suffix.casefold() != ".zip":
        raise SourceAuditError(f"ARCHIVE_NOT_ZIP:{archive}")
    actual_bytes = archive.stat().st_size
    expected_bytes = int(protocol["source"]["expected_archive_bytes"])
    if actual_bytes != expected_bytes:
        raise SourceAuditError(
            f"ARCHIVE_SIZE_MISMATCH:actual={actual_bytes}:expected={expected_bytes}"
        )
    return actual_bytes


def _validate_member_name(raw_name: str, is_directory: bool) -> str:
    if not raw_name or "\x00" in raw_name or "\\" in raw_name:
        raise SourceAuditError(f"UNSAFE_ZIP_MEMBER_NAME:{raw_name!r}")
    if raw_name.startswith("/"):
        raise SourceAuditError(f"ABSOLUTE_ZIP_MEMBER_NAME:{raw_name!r}")
    comparable = raw_name[:-1] if is_directory and raw_name.endswith("/") else raw_name
    if not comparable:
        raise SourceAuditError(f"EMPTY_ZIP_MEMBER_NAME:{raw_name!r}")
    parts = comparable.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise SourceAuditError(f"NON_CANONICAL_ZIP_MEMBER_NAME:{raw_name!r}")
    for part in parts:
        if part != part.rstrip(" .") or WINDOWS_INVALID_CHARS.search(part):
            raise SourceAuditError(f"WINDOWS_UNSAFE_ZIP_MEMBER_NAME:{raw_name!r}")
        if part.split(".", 1)[0].upper() in WINDOWS_RESERVED_NAMES:
            raise SourceAuditError(f"WINDOWS_RESERVED_ZIP_MEMBER_NAME:{raw_name!r}")
    canonical = PurePosixPath(*parts).as_posix()
    expected = canonical + ("/" if is_directory else "")
    if expected != raw_name:
        raise SourceAuditError(f"NON_CANONICAL_ZIP_MEMBER_NAME:{raw_name!r}")
    return canonical


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = info.external_attr >> 16
    return info.create_system == 3 and stat.S_ISLNK(mode)


def _is_consumed_source_member(raw_name: str) -> bool:
    """Whitelist only the five asset classes this model-free audit can consume."""

    if raw_name.endswith((ANNOTATION_SUFFIX, SCENE_SUFFIX, INTRINSICS_SUFFIX)):
        return True
    parts = raw_name.split("/")
    if len(parts) < 3 or parts[-2] not in {"image", "depth"}:
        return False
    suffix = PurePosixPath(parts[-1]).suffix.casefold()
    return (parts[-2] == "image" and suffix == ".jpg") or (
        parts[-2] == "depth" and suffix == ".png"
    )


def _archive_file_index(archive_zip: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    files: dict[str, zipfile.ZipInfo] = {}
    casefold_names: dict[str, str] = {}
    for info in archive_zip.infolist():
        if info.is_dir() or not _is_consumed_source_member(info.filename):
            continue
        is_directory = info.is_dir()
        canonical = _validate_member_name(info.filename, is_directory)
        folded = canonical.casefold()
        previous_case = casefold_names.get(folded)
        if previous_case is not None:
            raise SourceAuditError(
                f"CASE_COLLIDING_ZIP_MEMBERS:{previous_case!r}:{canonical!r}"
            )
        casefold_names[folded] = canonical
        if _is_symlink(info):
            raise SourceAuditError(f"ZIP_SYMLINK_FORBIDDEN:{info.filename}")
        if info.flag_bits & 0x1:
            raise SourceAuditError(f"ENCRYPTED_ZIP_MEMBER_FORBIDDEN:{info.filename}")
        if info.file_size < 0 or info.compress_size < 0:
            raise SourceAuditError(f"INVALID_ZIP_MEMBER_SIZE:{info.filename}")
        if canonical in files:
            raise SourceAuditError(f"DUPLICATE_ZIP_MEMBER:{canonical}")
        files[canonical] = info
    if not files:
        raise SourceAuditError("ZIP_CONTAINS_NO_FILES")
    return files


def _discover_capture_roots(names: Iterable[str]) -> list[str]:
    roots: set[str] = set()
    for name in names:
        if name.endswith(ANNOTATION_SUFFIX):
            roots.add(name[: -len(ANNOTATION_SUFFIX)])
    if not roots:
        raise SourceAuditError("NO_SUNRGBD_CAPTURES_FOUND")
    return sorted(roots)


def _bucket_for_capture(capture_path: str) -> tuple[str, str, str]:
    parts = capture_path.split("/")
    positions = [index for index, value in enumerate(parts) if value == "SUNRGBD"]
    if len(positions) != 1:
        raise SourceAuditError(
            f"CAPTURE_PATH_MUST_HAVE_ONE_SUNRGBD_ROOT:{capture_path}"
        )
    root_index = positions[0]
    if len(parts) < root_index + 4:
        raise SourceAuditError(f"CAPTURE_PATH_TOO_SHALLOW:{capture_path}")
    sensor = parts[root_index + 1]
    source = parts[root_index + 2]
    return sensor, source, f"{sensor}/{source}"


def _capture_identity(capture_path: str) -> dict[str, Any]:
    """Resolve only path forms whose recording/frame boundary is explicit."""

    parts = capture_path.split("/")
    positions = [index for index, value in enumerate(parts) if value == "SUNRGBD"]
    if len(positions) != 1:
        raise SourceAuditError(
            f"CAPTURE_PATH_MUST_HAVE_ONE_SUNRGBD_ROOT:{capture_path}"
        )
    root_index = positions[0]
    sensor, source, _bucket = _bucket_for_capture(capture_path)
    leaf = parts[-1]

    rgbf = RGBF_CAPTURE_RE.fullmatch(leaf)
    if rgbf is not None:
        sequence_id = "/".join([*parts[:-1], rgbf.group("sequence")])
        return {
            "frame_id": rgbf.group("frame"),
            "capture_sequence_id": sequence_id,
            "capture_sequence_resolution": {
                "status": "RESOLVED",
                "rule": "REMOVE_EXPLICIT_RGBF_FRAME_SUFFIX",
                "evidence": leaf,
            },
        }

    tail = parts[root_index + 3 :]
    if (
        sensor == "xtion"
        and source == "sun3ddata"
        and len(tail) >= 3
        and SUN3D_FRAME_RE.fullmatch(leaf) is not None
    ):
        return {
            "frame_id": leaf,
            "capture_sequence_id": "/".join(parts[:-1]),
            "capture_sequence_resolution": {
                "status": "RESOLVED",
                "rule": "SUN3D_SEQUENCE_PARENT_PLUS_NUMERIC_FRAME_DIRECTORY",
                "evidence": "/".join(tail),
            },
        }

    if (
        sensor == "realsense"
        and len(tail) == 1
        and REALSENSE_SINGLE_CAPTURE_RE.fullmatch(leaf) is not None
    ):
        return {
            "frame_id": leaf,
            "capture_sequence_id": capture_path,
            "capture_sequence_resolution": {
                "status": "RESOLVED",
                "rule": "REALSENSE_TIMESTAMP_DEVICE_SINGLE_CAPTURE_DIRECTORY",
                "evidence": leaf,
            },
        }

    return {
        "frame_id": leaf,
        "capture_sequence_id": None,
        "capture_sequence_resolution": {
            "status": "UNRESOLVED",
            "rule": None,
            "evidence": (
                "No frozen path rule proves the recording/frame boundary; the "
                "capture directory is not treated as an independent sequence."
            ),
        },
    }


def _single_required_member(
    files: dict[str, zipfile.ZipInfo],
    capture_path: str,
    relative_path: str,
) -> zipfile.ZipInfo:
    name = f"{capture_path}/{relative_path}"
    info = files.get(name)
    if info is None:
        raise SourceAuditError(f"CAPTURE_REQUIRED_MEMBER_MISSING:{name}")
    return info


def _single_media_member(
    files: dict[str, zipfile.ZipInfo],
    capture_path: str,
    directory: str,
    extension: str,
) -> zipfile.ZipInfo:
    prefix = f"{capture_path}/{directory}/"
    matches = [
        info
        for name, info in files.items()
        if name.startswith(prefix) and "/" not in name[len(prefix) :]
    ]
    if len(matches) != 1:
        raise SourceAuditError(
            f"CAPTURE_{directory.upper()}_MEMBER_COUNT:{capture_path}:{len(matches)}"
        )
    if PurePosixPath(matches[0].filename).suffix.casefold() != extension:
        raise SourceAuditError(
            f"CAPTURE_{directory.upper()}_EXTENSION_INVALID:{matches[0].filename}"
        )
    return matches[0]


def _capture_members(
    files: dict[str, zipfile.ZipInfo], capture_path: str
) -> dict[str, zipfile.ZipInfo]:
    return {
        "annotation2d": _single_required_member(
            files, capture_path, "annotation2Dfinal/index.json"
        ),
        "scene": _single_required_member(files, capture_path, "scene.txt"),
        "intrinsics": _single_required_member(files, capture_path, "intrinsics.txt"),
        "rgb": _single_media_member(files, capture_path, "image", ".jpg"),
        "depth": _single_media_member(files, capture_path, "depth", ".png"),
    }


def _read_member_bytes(
    archive_zip: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    maximum_bytes: int,
) -> bytes:
    if info.file_size > maximum_bytes:
        raise SourceAuditError(
            f"ZIP_MEMBER_TOO_LARGE:{info.filename}:{info.file_size}:{maximum_bytes}"
        )
    try:
        payload = archive_zip.read(info)
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise SourceAuditError(
            f"ZIP_MEMBER_READ_FAILED:{info.filename}:{type(error).__name__}:{error}"
        ) from error
    if len(payload) != info.file_size:
        raise SourceAuditError(
            f"ZIP_MEMBER_SIZE_MISMATCH:{info.filename}:{len(payload)}:{info.file_size}"
        )
    return payload


def _parse_scene(payload: bytes, identity: str) -> str:
    if len(payload) > MAX_TEXT_BYTES:
        raise SourceAuditError(f"SCENE_TEXT_TOO_LARGE:{identity}")
    try:
        text = payload.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as error:
        raise SourceAuditError(f"SCENE_TEXT_INVALID_UTF8:{identity}") from error
    if "\x00" in text:
        raise SourceAuditError(f"SCENE_TEXT_CONTAINS_NUL:{identity}")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        raise SourceAuditError(f"SCENE_TEXT_LINE_COUNT:{identity}:{len(lines)}")
    return lines[0]


def _parse_intrinsics(payload: bytes, identity: str) -> list[list[float]]:
    if len(payload) > MAX_TEXT_BYTES:
        raise SourceAuditError(f"INTRINSICS_TOO_LARGE:{identity}")
    try:
        text = payload.decode("ascii", errors="strict")
        values = [float(token) for token in text.split()]
    except (UnicodeDecodeError, ValueError) as error:
        raise SourceAuditError(f"INTRINSICS_INVALID:{identity}:{error}") from error
    if len(values) != 9 or not all(math.isfinite(value) for value in values):
        raise SourceAuditError(f"INTRINSICS_NOT_FINITE_3X3:{identity}")
    return [values[0:3], values[3:6], values[6:9]]


def _valid_label(value: Any, identity: str, object_index: int) -> str:
    if not isinstance(value, str):
        raise SourceAuditError(
            f"OBJECT_NAME_NOT_STRING:{identity}:object={object_index}"
        )
    name = value.strip()
    if not name or any(ord(character) < 32 for character in name):
        raise SourceAuditError(
            f"OBJECT_NAME_INVALID:{identity}:object={object_index}"
        )
    try:
        name.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise SourceAuditError(
            f"OBJECT_NAME_INVALID_UNICODE:{identity}:object={object_index}"
        ) from error
    return name


def _coordinate_array(
    value: Any,
    axis: str,
    identity: str,
    polygon_index: int,
) -> list[int | float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise SourceAuditError(
                f"POLYGON_{axis.upper()}_NON_FINITE:{identity}:polygon={polygon_index}"
            )
        return [value]
    if not isinstance(value, list):
        raise SourceAuditError(
            f"POLYGON_{axis.upper()}_NOT_ARRAY:{identity}:polygon={polygon_index}"
        )
    result: list[int | float] = []
    for coordinate_index, coordinate in enumerate(value):
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise SourceAuditError(
                f"POLYGON_{axis.upper()}_NOT_NUMERIC:{identity}:"
                f"polygon={polygon_index}:coordinate={coordinate_index}"
            )
        if not math.isfinite(float(coordinate)):
            raise SourceAuditError(
                f"POLYGON_{axis.upper()}_NON_FINITE:{identity}:"
                f"polygon={polygon_index}:coordinate={coordinate_index}"
            )
        result.append(coordinate)
    return result


def _normalized_label(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    normalized = normalized.replace("_", " ").replace("-", " ")
    return " ".join(normalized.split())


def _near_label_matches(
    objects: list[dict[str, Any]], polygons: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    polygon_indices: defaultdict[int, list[int]] = defaultdict(list)
    for polygon in polygons:
        polygon_indices[int(polygon["object_index"])].append(
            int(polygon["polygon_index"])
        )
    matches: list[dict[str, Any]] = []
    for row in objects:
        if not isinstance(row.get("name"), str):
            continue
        normalized = _normalized_label(row["name"])
        compact = normalized.replace(" ", "")
        matched_terms = [
            term
            for term in NEAR_LABEL_TERMS
            if (term.replace(" ", "") in compact if term == "door frame" else term in normalized)
        ]
        if matched_terms:
            object_index = int(row["object_index"])
            matches.append(
                {
                    "object_index": object_index,
                    "name": row["name"],
                    "matched_terms": matched_terms,
                    "polygon_indices": polygon_indices[object_index],
                }
            )
    return matches


def _parse_annotation(
    annotation: dict[str, Any], identity: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int], dict[str, Any]]:
    raw_objects = annotation.get("objects")
    raw_frames = annotation.get("frames")
    if not isinstance(raw_objects, list):
        raise SourceAuditError(f"ANNOTATION_OBJECTS_NOT_ARRAY:{identity}")
    if not isinstance(raw_frames, list) or len(raw_frames) != 1:
        count = len(raw_frames) if isinstance(raw_frames, list) else "not-array"
        raise SourceAuditError(f"ANNOTATION_FRAME_COUNT:{identity}:{count}")

    objects: list[dict[str, Any]] = []
    for object_index, raw_object in enumerate(raw_objects):
        if raw_object is None or raw_object == []:
            objects.append(
                {
                    "object_index": object_index,
                    "name": None,
                    "annotation_status": "NULL_TOMBSTONE",
                }
            )
            continue
        if not isinstance(raw_object, dict):
            raise SourceAuditError(
                f"ANNOTATION_OBJECT_NOT_OBJECT:{identity}:object={object_index}"
            )
        objects.append(
            {
                "object_index": object_index,
                "name": _valid_label(raw_object.get("name"), identity, object_index),
                "annotation_status": "ACTIVE",
            }
        )

    frame = raw_frames[0]
    if not isinstance(frame, dict) or not isinstance(frame.get("polygon"), list):
        raise SourceAuditError(f"ANNOTATION_POLYGONS_NOT_ARRAY:{identity}")
    polygons: list[dict[str, Any]] = []
    polygons_by_object: defaultdict[int, int] = defaultdict(int)
    empty_polygon_placeholders = 0
    degenerate_polygon_placeholders = 0
    for polygon_index, raw_polygon in enumerate(frame["polygon"]):
        if not isinstance(raw_polygon, dict):
            raise SourceAuditError(
                f"ANNOTATION_POLYGON_NOT_OBJECT:{identity}:polygon={polygon_index}"
            )
        object_index = raw_polygon.get("object")
        if isinstance(object_index, bool) or not isinstance(object_index, int):
            raise SourceAuditError(
                f"POLYGON_OBJECT_INDEX_NOT_INTEGER:{identity}:polygon={polygon_index}"
            )
        if not 0 <= object_index < len(objects):
            raise SourceAuditError(
                f"POLYGON_OBJECT_INDEX_OUT_OF_RANGE:{identity}:"
                f"polygon={polygon_index}:object={object_index}:objects={len(objects)}"
            )
        x = _coordinate_array(raw_polygon.get("x"), "x", identity, polygon_index)
        y = _coordinate_array(raw_polygon.get("y"), "y", identity, polygon_index)
        if not x and not y:
            empty_polygon_placeholders += 1
            continue
        if len(x) != len(y):
            raise SourceAuditError(
                f"POLYGON_COORDINATE_COUNT_INVALID:{identity}:polygon={polygon_index}:"
                f"x={len(x)}:y={len(y)}"
            )
        if len(x) < 3:
            degenerate_polygon_placeholders += 1
            continue
        if objects[object_index]["annotation_status"] != "ACTIVE":
            raise SourceAuditError(
                f"POLYGON_REFERENCES_NULL_OBJECT:{identity}:"
                f"polygon={polygon_index}:object={object_index}"
            )
        polygons.append(
            {
                "polygon_index": polygon_index,
                "object_index": object_index,
                "object_name": objects[object_index]["name"],
                "x": x,
                "y": y,
            }
        )
        polygons_by_object[object_index] += 1

    exact_door_objects = [
        row["object_index"]
        for row in objects
        if isinstance(row["name"], str) and _normalized_label(row["name"]) == "door"
    ]
    exact_door_objects_without_valid_polygon = [
        object_index
        for object_index in exact_door_objects
        if polygons_by_object[object_index] == 0
    ]
    exact_door_polygons = [
        row["polygon_index"]
        for row in polygons
        if row["object_index"] in exact_door_objects
    ]
    return objects, polygons, exact_door_polygons, {
        "empty_polygon_placeholders_excluded": empty_polygon_placeholders,
        "fewer_than_three_point_polygons_excluded": degenerate_polygon_placeholders,
        "exact_door_object_indices_without_valid_polygon": (
            exact_door_objects_without_valid_polygon
        ),
    }


def _member_record(info: zipfile.ZipInfo, sha256: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": info.filename,
        "size_bytes": info.file_size,
        "compressed_size_bytes": info.compress_size,
        "crc32": f"{info.CRC:08x}",
    }
    if sha256 is not None:
        result["sha256"] = sha256
    return result


def _capture_record(
    archive_zip: zipfile.ZipFile,
    files: dict[str, zipfile.ZipInfo],
    capture_path: str,
) -> dict[str, Any]:
    sensor, source, bucket = _bucket_for_capture(capture_path)
    identity = _capture_identity(capture_path)
    members = _capture_members(files, capture_path)
    annotation_bytes = _read_member_bytes(
        archive_zip, members["annotation2d"], MAX_ANNOTATION_BYTES
    )
    scene_bytes = _read_member_bytes(archive_zip, members["scene"], MAX_TEXT_BYTES)
    intrinsics_bytes = _read_member_bytes(
        archive_zip, members["intrinsics"], MAX_TEXT_BYTES
    )
    annotation = _parse_json_bytes(annotation_bytes, members["annotation2d"].filename)
    objects, polygons, exact_door_polygon_indices, annotation_diagnostics = _parse_annotation(
        annotation, members["annotation2d"].filename
    )
    near_label_matches = _near_label_matches(objects, polygons)
    scene = _parse_scene(scene_bytes, members["scene"].filename)
    intrinsics = _parse_intrinsics(intrinsics_bytes, members["intrinsics"].filename)
    exact_door_object_indices = sorted(
        {
            row["object_index"]
            for row in polygons
            if row["polygon_index"] in exact_door_polygon_indices
        }
    )
    relative_files = {
        "rgb": members["rgb"].filename[len(capture_path) + 1 :],
        "depth": members["depth"].filename[len(capture_path) + 1 :],
        "intrinsics": members["intrinsics"].filename[len(capture_path) + 1 :],
        "polygon": members["annotation2d"].filename[len(capture_path) + 1 :],
        "scene_metadata": members["scene"].filename[len(capture_path) + 1 :],
    }
    return {
        "canonical_source_path": capture_path,
        **identity,
        "sensor": sensor,
        "source": source,
        "sensor_source_bucket": bucket,
        "scene": scene,
        "intrinsics_3x3": intrinsics,
        "relative_files": relative_files,
        "members": {
            "annotation2d": _member_record(
                members["annotation2d"], _sha256_bytes(annotation_bytes)
            ),
            "scene": _member_record(members["scene"], _sha256_bytes(scene_bytes)),
            "intrinsics": _member_record(
                members["intrinsics"], _sha256_bytes(intrinsics_bytes)
            ),
            "rgb": _member_record(members["rgb"]),
            "depth": _member_record(members["depth"]),
        },
        "official_object_names_source_order": [row["name"] for row in objects],
        "official_objects": objects,
        "official_polygons": polygons,
        "exact_door_object_indices": exact_door_object_indices,
        "exact_door_polygon_indices": exact_door_polygon_indices,
        "near_label_matches": near_label_matches,
        "annotation_geometry_diagnostics": annotation_diagnostics,
    }


def _compact_candidate_record(
    row: dict[str, Any], *, include_exact_door_polygons: bool
) -> dict[str, Any]:
    """Retain selection evidence without materializing every source polygon."""

    compact = {
        "canonical_source_path": row["canonical_source_path"],
        "capture_sequence_id": row["capture_sequence_id"],
        "capture_sequence_resolution": row["capture_sequence_resolution"],
        "frame_id": row["frame_id"],
        "sensor": row["sensor"],
        "source": row["source"],
        "sensor_source_bucket": row["sensor_source_bucket"],
        "scene": row["scene"],
        "intrinsics_3x3": row["intrinsics_3x3"],
        "relative_files": row["relative_files"],
        "members": row["members"],
        "official_object_names_source_order": row[
            "official_object_names_source_order"
        ],
        "near_label_matches": row["near_label_matches"],
        "annotation_geometry_diagnostics": row[
            "annotation_geometry_diagnostics"
        ],
        "annotation_prefilter": row["annotation_prefilter"],
    }
    if include_exact_door_polygons:
        exact_object_indices = set(row["exact_door_object_indices"])
        exact_polygon_indices = set(row["exact_door_polygon_indices"])
        compact["exact_door_objects"] = [
            source_object
            for source_object in row["official_objects"]
            if source_object["object_index"] in exact_object_indices
        ]
        compact["exact_door_polygons"] = [
            polygon
            for polygon in row["official_polygons"]
            if polygon["polygon_index"] in exact_polygon_indices
        ]
    return compact


def audit_archive(
    archive: Path,
    protocol_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    protocol, protocol_sha256 = _load_protocol(protocol_path)
    archive_size = _validate_archive_identity(archive, protocol)
    try:
        with zipfile.ZipFile(archive, mode="r", allowZip64=True) as archive_zip:
            files = _archive_file_index(archive_zip)
            capture_paths = _discover_capture_roots(files)
            buckets: defaultdict[str, dict[str, Any]] = defaultdict(dict)
            for capture_path in capture_paths:
                sensor, source, bucket_name = _bucket_for_capture(capture_path)
                bucket = buckets.get(bucket_name)
                if not bucket:
                    bucket = {
                        "sensor": sensor,
                        "source": source,
                        "capture_count": 0,
                        "rejected_capture_count": 0,
                        "resolved_sequence_frame_count": 0,
                        "unresolved_sequence_frame_count": 0,
                        "exact_door_candidate_count": 0,
                        "no_exact_door_candidate_count": 0,
                        "near_label_candidate_count": 0,
                        "door_label_without_valid_polygon_count": 0,
                        "exact_door_candidates": [],
                        "no_exact_door_candidates": [],
                        "near_label_candidates": [],
                        "door_label_without_valid_polygon_frames": [],
                        "rejected_captures": [],
                        "unresolved_sequence_frames": [],
                    }
                    buckets[bucket_name] = bucket
                bucket["capture_count"] += 1
                try:
                    row = _capture_record(archive_zip, files, capture_path)
                except SourceAuditError as error:
                    bucket["rejected_capture_count"] += 1
                    bucket["rejected_captures"].append(
                        {
                            "canonical_source_path": capture_path,
                            "reason": str(error),
                        }
                    )
                    continue
                row["annotation_prefilter"] = (
                    "EXACT_DOOR"
                    if row["exact_door_polygon_indices"]
                    else (
                        "DOOR_LABEL_WITHOUT_VALID_POLYGON_REJECT"
                        if row["annotation_geometry_diagnostics"][
                            "exact_door_object_indices_without_valid_polygon"
                        ]
                        else "NO_EXACT_DOOR"
                    )
                )
                if row["capture_sequence_id"] is None:
                    bucket["unresolved_sequence_frame_count"] += 1
                    bucket["unresolved_sequence_frames"].append(
                        _compact_candidate_record(
                            row, include_exact_door_polygons=False
                        )
                    )
                    continue
                bucket["resolved_sequence_frame_count"] += 1
                if row["exact_door_polygon_indices"]:
                    bucket["exact_door_candidate_count"] += 1
                    bucket["exact_door_candidates"].append(
                        _compact_candidate_record(
                            row, include_exact_door_polygons=True
                        )
                    )
                elif row["annotation_geometry_diagnostics"][
                    "exact_door_object_indices_without_valid_polygon"
                ]:
                    bucket["door_label_without_valid_polygon_count"] += 1
                    bucket["door_label_without_valid_polygon_frames"].append(
                        _compact_candidate_record(
                            row, include_exact_door_polygons=False
                        )
                    )
                else:
                    bucket["no_exact_door_candidate_count"] += 1
                    bucket["no_exact_door_candidates"].append(
                        _compact_candidate_record(
                            row, include_exact_door_polygons=False
                        )
                    )
                    if row["near_label_matches"]:
                        polygon_by_index = {
                            polygon["polygon_index"]: polygon
                            for polygon in row["official_polygons"]
                        }
                        matched_polygon_indices = sorted(
                            {
                                polygon_index
                                for match in row["near_label_matches"]
                                for polygon_index in match["polygon_indices"]
                            }
                        )
                        bucket["near_label_candidate_count"] += 1
                        bucket["near_label_candidates"].append(
                            {
                                "canonical_source_path": row[
                                    "canonical_source_path"
                                ],
                                "capture_sequence_id": row[
                                    "capture_sequence_id"
                                ],
                                "capture_sequence_resolution": row[
                                    "capture_sequence_resolution"
                                ],
                                "sensor_source_bucket": row[
                                    "sensor_source_bucket"
                                ],
                                "frame_id": row["frame_id"],
                                "scene": row["scene"],
                                "relative_files": row["relative_files"],
                                "near_label_matches": row["near_label_matches"],
                                "matched_official_polygons": [
                                    polygon_by_index[index]
                                    for index in matched_polygon_indices
                                ],
                                "automatic_ood_decision": False,
                            }
                        )
            member_count = len(files)
    except zipfile.BadZipFile as error:
        raise SourceAuditError(f"INVALID_ZIP:{archive}:{error}") from error

    ordered_buckets = {name: buckets[name] for name in sorted(buckets)}
    exact_count = sum(
        bucket["exact_door_candidate_count"] for bucket in ordered_buckets.values()
    )
    no_door_count = sum(
        bucket["no_exact_door_candidate_count"] for bucket in ordered_buckets.values()
    )
    near_label_count = sum(
        bucket["near_label_candidate_count"] for bucket in ordered_buckets.values()
    )
    invalid_door_polygon_count = sum(
        bucket["door_label_without_valid_polygon_count"]
        for bucket in ordered_buckets.values()
    )
    rejected_count = sum(
        bucket["rejected_capture_count"] for bucket in ordered_buckets.values()
    )
    unresolved_count = sum(
        bucket["unresolved_sequence_frame_count"] for bucket in ordered_buckets.values()
    )
    resolved_sequences = {
        row["capture_sequence_id"]
        for bucket in ordered_buckets.values()
        for candidate_key in ("exact_door_candidates", "no_exact_door_candidates")
        for row in bucket[candidate_key]
    }
    result = {
        "schema": SCHEMA,
        "stage": "L10-PB11-SUNRGBD-SOURCE-INDEX-ONLY",
        "protocol": {
            "path": str(protocol_path),
            "sha256": protocol_sha256,
            "schema": protocol["schema"],
            "status": protocol.get("status"),
        },
        "archive": {
            "path": str(archive),
            "size_bytes": archive_size,
            "expected_size_bytes": int(
                protocol["source"]["expected_archive_bytes"]
            ),
            "file_member_count": member_count,
            "full_archive_sha256_computed": False,
        },
        "selection_semantics": {
            "exact_door": (
                "At least one annotation2Dfinal object name equals 'door' after "
                "Unicode NFKC normalization, separator/whitespace normalization, "
                "and case folding; the complete normalized label must be the single "
                "token 'door', and that object has at least one valid polygon."
            ),
            "no_exact_door": (
                "No annotation2Dfinal object name meets the exact-door rule. This is "
                "an annotation prefilter only, not evidence that the RGB lacks a door "
                "or large opening."
            ),
            "near_label": (
                "A resolved no-exact-door frame whose normalized official object "
                "name contains doorway, door frame, arch, opening, passage, or "
                "hallway. This is only a human-audit shortlist and is never an OOD "
                "assignment."
            ),
            "capture_sequence": (
                "Only explicit _rgbf suffixes, SUN3D numeric frame directories, and "
                "the frozen RealSense timestamp-device single-capture directory "
                "contract are resolved. Other structures remain UNRESOLVED and are "
                "excluded from usable candidate arrays."
            ),
            "rejected_capture": (
                "A capture whose required asset or annotation cannot be parsed under "
                "the frozen contract is listed with its exact error and excluded; it "
                "is never silently reclassified as no-door."
            ),
            "automatic_none_or_ood_decisions": False,
        },
        "counts": {
            "sensor_source_buckets": len(ordered_buckets),
            "captures": len(capture_paths),
            "rejected_captures": rejected_count,
            "resolved_capture_sequences": len(resolved_sequences),
            "unresolved_sequence_frames": unresolved_count,
            "exact_door_candidates": exact_count,
            "no_exact_door_candidates": no_door_count,
            "near_label_candidates": near_label_count,
            "door_label_without_valid_polygon_frames": invalid_door_polygon_count,
        },
        "buckets": ordered_buckets,
        "next_step": (
            "Use this index only to choose frames for a pre-model human audit, then "
            "freeze roles, audit notes, source paths, hashes, capture identities, and "
            "aperture quadrilaterals under the protocol."
        ),
        "claim_boundary": (
            "Source indexing only. No model ran and no NONE/OOD, door-leaf presence, "
            "opening width, traversability, closure, guidance, product, user-benefit, "
            "or safety conclusion was made. "
            + str(protocol["claim_boundary"])
        ),
    }
    _write_json_exclusive(output_path, result)
    return result


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            created = True
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except Exception:
        if created and path.exists():
            path.unlink()
        raise


def _safe_extract_target(output_root: Path, capture_path: str) -> Path:
    artifacts_root = ARTIFACTS_LOCAL.resolve()
    resolved_root = output_root.resolve()
    if not _is_within(resolved_root, artifacts_root):
        raise SourceAuditError(
            f"EXTRACT_ROOT_MUST_BE_UNDER_ARTIFACTS_LOCAL:{resolved_root}"
        )
    target = (resolved_root / Path(*capture_path.split("/"))).resolve()
    if not _is_within(target, resolved_root):
        raise SourceAuditError(f"EXTRACT_TARGET_ESCAPES_ROOT:{target}")
    return target


def _copy_member(
    archive_zip: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    destination: Path,
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    written = 0
    try:
        with archive_zip.open(info, mode="r") as source, destination.open("xb") as sink:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                sink.write(chunk)
                digest.update(chunk)
                written += len(chunk)
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise SourceAuditError(
            f"ZIP_MEMBER_EXTRACTION_FAILED:{info.filename}:{type(error).__name__}:{error}"
        ) from error
    if written != info.file_size:
        raise SourceAuditError(
            f"EXTRACTED_SIZE_MISMATCH:{info.filename}:{written}:{info.file_size}"
        )
    return digest.hexdigest()


def extract_capture(
    archive: Path,
    protocol_path: Path,
    capture_path: str,
    output_root: Path,
) -> dict[str, Any]:
    protocol, protocol_sha256 = _load_protocol(protocol_path)
    _validate_archive_identity(archive, protocol)
    canonical_capture = _validate_member_name(capture_path, is_directory=False)
    _bucket_for_capture(canonical_capture)
    target = _safe_extract_target(output_root, canonical_capture)
    if target.exists():
        raise SourceAuditError(f"OUTPUT_CAPTURE_ALREADY_EXISTS:{target}")

    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".sunrgbd-extract-", dir=output_root))
    extracted: dict[str, dict[str, Any]] = {}
    try:
        try:
            with zipfile.ZipFile(archive, mode="r", allowZip64=True) as archive_zip:
                files = _archive_file_index(archive_zip)
                capture_paths = set(_discover_capture_roots(files))
                if canonical_capture not in capture_paths:
                    raise SourceAuditError(
                        f"CAPTURE_PATH_NOT_FOUND:{canonical_capture}"
                    )
                record = _capture_record(
                    archive_zip, files, canonical_capture
                )  # strict source validation before copying
                members = _capture_members(files, canonical_capture)
                for role in ("rgb", "depth", "intrinsics", "scene", "annotation2d"):
                    info = members[role]
                    relative = info.filename[len(canonical_capture) + 1 :]
                    destination = temporary / Path(*relative.split("/"))
                    digest = _copy_member(archive_zip, info, destination)
                    extracted[role] = {
                        "archive_member": info.filename,
                        "relative_path": relative,
                        "size_bytes": info.file_size,
                        "sha256": digest,
                    }
        except zipfile.BadZipFile as error:
            raise SourceAuditError(f"INVALID_ZIP:{archive}:{error}") from error

        target.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    return {
        "mode": "SELECTIVE_SOURCE_EXTRACTION_ONLY",
        "protocol_sha256": protocol_sha256,
        "canonical_source_path": canonical_capture,
        "capture_sequence_id": record["capture_sequence_id"],
        "capture_sequence_resolution": record["capture_sequence_resolution"],
        "frame_id": record["frame_id"],
        "output_capture_path": str(target),
        "sensor_source_bucket": record["sensor_source_bucket"],
        "scene": record["scene"],
        "has_exact_door_annotation": bool(record["exact_door_polygon_indices"]),
        "extracted": extracted,
        "automatic_none_or_ood_decisions": False,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Model-free SUN RGB-D source index and selective extractor for the "
            "L10 PB11 metric-closure protocol. No downloads or role decisions."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser(
        "audit",
        help="Index every capture and group exact-door/no-exact-door candidates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    audit.add_argument("--archive", type=Path, required=True, help="Official SUNRGBD.zip")
    audit.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    audit.add_argument("--output", type=Path, required=True, help="New JSON output path")

    extract = subparsers.add_parser(
        "extract",
        help="Extract only RGB/depth/intrinsics/scene/annotation2Dfinal for one capture.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    extract.add_argument("--archive", type=Path, required=True, help="Official SUNRGBD.zip")
    extract.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    extract.add_argument(
        "--capture-path",
        required=True,
        help="Exact canonical_source_path emitted by the audit JSON",
    )
    extract.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_EXTRACT_ROOT,
        help="Destination root; must resolve under repository artifacts.local",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        archive = _repo_path(args.archive)
        protocol = _repo_path(args.protocol)
        if args.command == "audit":
            output = _repo_path(args.output)
            result = audit_archive(archive, protocol, output)
            print(
                json.dumps(
                    {
                        "output": str(output),
                        "counts": result["counts"],
                        "automatic_none_or_ood_decisions": False,
                    },
                    indent=2,
                )
            )
        elif args.command == "extract":
            output_root = _repo_path(args.output_root)
            result = extract_capture(
                archive, protocol, args.capture_path, output_root
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:  # pragma: no cover - argparse enforces this branch is unreachable.
            parser.error(f"unsupported command: {args.command}")
    except (OSError, SourceAuditError, zipfile.BadZipFile) as error:
        print(f"ERROR:{type(error).__name__}:{error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
