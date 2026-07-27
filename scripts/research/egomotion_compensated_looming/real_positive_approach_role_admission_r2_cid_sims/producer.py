"""Geometry-only producer for the frozen CID-SIMS floor3_1 RGB-D archive.

The archive is never extracted. RGB pixels are never read. Only ZIP metadata,
the source-native pose table, and depth PNG members referenced by the first
half-open ten-second window are consumed.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from decimal import Decimal
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable, Mapping, Sequence
import zipfile

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")

import numpy as np
from PIL import Image

from scripts.research.egomotion_compensated_looming.pb_h1_role_proxy.geometry import (
    summarize_translation_induced_geometry,
    translation_induced_geometry,
)


PROTOCOL_ID = "RCLE_REAL_POSITIVE_APPROACH_ROLE_ADMISSION_R2_CID_SIMS"
CANDIDATE_ID = "CID_SIMS_V6_FLOOR3_1"
SEQUENCE_ID = "floor3_1"
WINDOW_SECONDS = 10.0
MAX_DT_SECONDS = 0.100
MAX_POSE_BRACKET_SECONDS = Decimal("0.100")
MIN_COVERAGE = 0.80
MIN_EVALUABLE_PAIRS = 8
MIN_SIGNED_RADIAL = 0.05
MIN_POSITIVE_FRACTION = 0.75
DEPTH_SAMPLE_STRIDE_PX = 8
DEPTH_UNITS_PER_METER = 1000.0
MINIMUM_RADIUS_PX = 8.0
ADMITTED = "REAL_POSITIVE_APPROACH_ROLE_ADMITTED / VALID"
HOLD = "HOLD_ALGORITHM_CANARY_APPROACH_ROLE_INCOMPLETE / VALID"
REQUIRED_BURNED_GROUPS = {
    "TUM_ALL_RCLE_ACCESSED_AND_DERIVED_FAMILIES",
    "BONN_ALL_RCLE_COHORTS_WINDOWS_AND_DERIVATIVES",
    "RCLE_PHASE_A_ALL_SYNTHETIC_GENERATOR_AND_SUPPORT_FAMILIES",
    "EVIMO2_V2_FLEA3_SANITY_LL_CAPTURE_FAMILY",
    "ALL_PREEXISTING_ARTIFACTS_LOCAL_PAYLOADS",
    "ETH3D_SLAM_SOFA_SCENE_CAPTURE_FAMILY",
}
EXPECTED_ANCESTRY = [
    "SCIENCEDB_OFFICIAL",
    "CID_SIMS_V6",
    "CID_SIMS_V6_OFFICE_BUILDING",
    "CID_SIMS_V6_OFFICE_BUILDING_FLOOR3",
    "CID_SIMS_V6_OFFICE_BUILDING_FLOOR3_1",
    PROTOCOL_ID,
]
INDEPENDENCE_GROUP = "CID_SIMS_V6_OFFICE_BUILDING_FLOOR3_THREE_RUN_CAPTURE_FAMILY"
REUSE_POLICY = (
    "GEOMETRY_SELECTED_REAL_APPROACH_ROLE_ADMISSION_OR_SOURCE_CHARACTERIZATION_"
    "COUNTEREXAMPLE_REGRESSION_ONLY; NEVER_CONFIRMATION"
)
INTRINSIC = np.asarray(
    (
        (386.52199190267083, 0.0, 326.5103569741365),
        (0.0, 387.32300428823663, 237.40293732598795),
        (0.0, 0.0, 1.0),
    ),
    dtype=np.float64,
)


@dataclass(frozen=True)
class AssociatedRow:
    rgb_timestamp: Decimal
    rgb_path: str
    depth_timestamp: Decimal
    depth_path: str


@dataclass(frozen=True)
class PoseRow:
    timestamp: Decimal
    center_world_m: np.ndarray
    quaternion_xyzw: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("R2_JSON_OBJECT_REQUIRED")
    return value


def _tokens(raw: bytes) -> Iterable[list[str]]:
    for line in raw.decode("utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            yield stripped.split()


def _safe_relative(value: str, expected_directory: str) -> str:
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or ".." in parsed.parts
        or "\\" in value
        or len(parsed.parts) != 2
        or parsed.parts[0] != expected_directory
    ):
        raise ValueError("R2_INDEX_UNSAFE_OR_WRONG_DIRECTORY")
    return parsed.as_posix()


def _parse_associated(raw: bytes) -> list[AssociatedRow]:
    rows: list[AssociatedRow] = []
    for values in _tokens(raw):
        if len(values) != 4:
            raise ValueError("R2_ASSOCIATED_COLUMNS")
        rows.append(
            AssociatedRow(
                Decimal(values[0]),
                _safe_relative(values[1], "rgb"),
                Decimal(values[2]),
                _safe_relative(values[3], "depth"),
            )
        )
    if len(rows) < 2:
        raise ValueError("R2_ASSOCIATED_TOO_SHORT")
    if any(
        left.depth_timestamp >= right.depth_timestamp
        for left, right in zip(rows, rows[1:])
    ):
        raise ValueError("R2_ASSOCIATED_DEPTH_NOT_MONOTONIC")
    if len({row.depth_path for row in rows}) != len(rows):
        raise ValueError("R2_ASSOCIATED_DUPLICATE_DEPTH")
    return rows


def _parse_depth_index(raw: bytes) -> list[tuple[Decimal, str]]:
    rows: list[tuple[Decimal, str]] = []
    for values in _tokens(raw):
        if len(values) != 2:
            raise ValueError("R2_DEPTH_INDEX_COLUMNS")
        rows.append((Decimal(values[0]), _safe_relative(values[1], "depth")))
    if not rows or any(left[0] >= right[0] for left, right in zip(rows, rows[1:])):
        raise ValueError("R2_DEPTH_INDEX_NOT_MONOTONIC")
    if len({path for _, path in rows}) != len(rows):
        raise ValueError("R2_DEPTH_INDEX_DUPLICATE_PATH")
    return rows


def _normalize_quaternion(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(result))
    if (
        result.shape != (4,)
        or not np.all(np.isfinite(result))
        or not math.isfinite(norm)
        or norm <= 0.0
    ):
        raise ValueError("R2_POSE_QUATERNION")
    return result / norm


def _parse_poses(raw: bytes) -> list[PoseRow]:
    result: list[PoseRow] = []
    for values in _tokens(raw):
        if len(values) != 8:
            raise ValueError("R2_GROUNDTRUTH_COLUMNS")
        numeric = np.asarray([float(item) for item in values[1:]], dtype=np.float64)
        if not np.all(np.isfinite(numeric)):
            raise ValueError("R2_GROUNDTRUTH_NONFINITE")
        result.append(
            PoseRow(
                Decimal(values[0]),
                numeric[:3],
                _normalize_quaternion(numeric[3:7]),
            )
        )
    if not result or any(
        left.timestamp >= right.timestamp for left, right in zip(result, result[1:])
    ):
        raise ValueError("R2_GROUNDTRUTH_NOT_MONOTONIC")
    return result


def _parse_intrinsic(raw: bytes) -> np.ndarray:
    values = [float(item) for item in raw.decode("utf-8").split()]
    if len(values) != 4 or not all(math.isfinite(item) for item in values):
        raise ValueError("R2_CALIBRATION_FORMAT")
    fx, fy, cx, cy = values
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("R2_CALIBRATION_FORMAT")
    return np.asarray(((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)))


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


def _interpolate_pose(
    poses: Sequence[PoseRow], timestamp: Decimal
) -> tuple[np.ndarray, np.ndarray]:
    timestamps = [row.timestamp for row in poses]
    index = bisect_left(timestamps, timestamp)
    if index < len(poses) and poses[index].timestamp == timestamp:
        return poses[index].center_world_m.copy(), poses[index].quaternion_xyzw.copy()
    if index == 0 or index == len(poses):
        raise ValueError("R2_POSE_NOT_BRACKETED")
    left, right = poses[index - 1], poses[index]
    span = right.timestamp - left.timestamp
    if span > MAX_POSE_BRACKET_SECONDS:
        raise ValueError("R2_POSE_BRACKET_TOO_WIDE")
    fraction = float((timestamp - left.timestamp) / span)
    return (
        left.center_world_m
        + fraction * (right.center_world_m - left.center_world_m),
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


def _relative_pose(
    previous: tuple[np.ndarray, np.ndarray],
    current: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    previous_center, previous_quaternion = previous
    current_center, current_quaternion = current
    previous_rotation = _rotation(previous_quaternion)
    current_rotation = _rotation(current_quaternion)
    return (
        current_rotation.T @ previous_rotation,
        current_rotation.T @ (previous_center - current_center),
    )


def _decode_depth(raw: bytes) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    try:
        with Image.open(BytesIO(raw)) as image:
            image.load()
            depth = np.asarray(image)
    except (OSError, ValueError) as error:
        raise ValueError("R2_DEPTH_PNG_INVALID") from error
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError("R2_DEPTH_PNG_FORMAT")
    height, width = depth.shape
    yy, xx = np.mgrid[
        0:height:DEPTH_SAMPLE_STRIDE_PX,
        0:width:DEPTH_SAMPLE_STRIDE_PX,
    ]
    sampled = depth[yy, xx].reshape(-1)
    valid = sampled > 0
    pixels = np.column_stack((xx.reshape(-1)[valid], yy.reshape(-1)[valid])).astype(
        np.float64
    )
    depth_m = sampled[valid].astype(np.float64) / DEPTH_UNITS_PER_METER
    return pixels, depth_m, (width, height)


def _pair_worker(
    task: tuple[
        dict[str, Any],
        bytes,
        np.ndarray,
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
        float,
    ]
) -> tuple[dict[str, Any], tuple[int, int] | None]:
    record, raw_depth, intrinsic, previous_pose, current_pose, dt = task
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - runtime fallback
        limits = None
    else:
        limits = threadpool_limits(limits=1)
        limits.__enter__()
    try:
        try:
            pixels, depth_m, current_size = _decode_depth(raw_depth)
        except ValueError as error:
            record.update(evaluable=False, reason=str(error))
            return record, None
        rotation, translation = _relative_pose(previous_pose, current_pose)
        summary = summarize_translation_induced_geometry(
            translation_induced_geometry(
                pixels,
                depth_m,
                intrinsic,
                rotation,
                translation,
                dt,
                image_size_wh=current_size,
                minimum_radius_px=MINIMUM_RADIUS_PX,
                zbuffer=True,
            )
        )
        record.update(summary)
        if not summary["evaluable"]:
            record["reason"] = "NO_VALID_GEOMETRY_SAMPLES"
        return record, current_size
    finally:
        if limits is not None:
            limits.__exit__(None, None, None)


class CidSimsArchive:
    """Read-fenced ZIP wrapper; color members have no read API."""

    def __init__(self, path: Path) -> None:
        self._archive = zipfile.ZipFile(path)
        infos = self._archive.infolist()
        names = [info.filename for info in infos]
        normalized = [PurePosixPath(name).as_posix() for name in names]
        if len(normalized) != len(set(normalized)):
            raise ValueError("R2_ARCHIVE_DUPLICATE_MEMBER")
        for info, name in zip(infos, names):
            parsed = PurePosixPath(name)
            if (
                parsed.is_absolute()
                or ".." in parsed.parts
                or "\\" in name
                or bool(re.match(r"^[A-Za-z]:", name))
                or info.flag_bits & 0x1
            ):
                raise ValueError("R2_ARCHIVE_UNSAFE_MEMBER")
        self.inventory = [
            {
                "name": info.filename,
                "crc32": f"{info.CRC:08x}",
                "compressed_bytes": info.compress_size,
                "uncompressed_bytes": info.file_size,
                "is_directory": info.is_dir(),
            }
            for info in infos
        ]
        self._files = {info.filename for info in infos if not info.is_dir()}
        required = {f"{SEQUENCE_ID}/pose.txt"}
        if not required.issubset(self._files):
            raise ValueError("R2_ARCHIVE_REQUIRED_MEMBER")
        roots = {PurePosixPath(name).parts[0] for name in self._files}
        if roots != {SEQUENCE_ID}:
            raise ValueError("R2_ARCHIVE_TOP_LEVEL_IDENTITY")
        if not any(name.startswith(f"{SEQUENCE_ID}/color/") for name in self._files):
            raise ValueError("R2_ARCHIVE_COLOR_DIRECTORY_IDENTITY")
        if not any(name.startswith(f"{SEQUENCE_ID}/depth/") for name in self._files):
            raise ValueError("R2_ARCHIVE_DEPTH_DIRECTORY_IDENTITY")
        self.depth_members_read: list[str] = []

    def read_control(self, name: str) -> bytes:
        if name != "pose.txt":
            raise ValueError("R2_CONTROL_READ_FORBIDDEN")
        return self._archive.read(f"{SEQUENCE_ID}/{name}")

    def read_depth(self, relative: str) -> bytes:
        safe = _safe_relative(relative, "depth")
        member = f"{SEQUENCE_ID}/{safe}"
        if member not in self._files:
            raise ValueError("R2_DEPTH_MEMBER_MISSING")
        self.depth_members_read.append(member)
        return self._archive.read(member)

    def member_exists(self, relative: str) -> bool:
        parsed = PurePosixPath(relative)
        if parsed.is_absolute() or ".." in parsed.parts or "\\" in relative:
            return False
        return f"{SEQUENCE_ID}/{parsed.as_posix()}" in self._files

    def close(self) -> None:
        self._archive.close()

    def __enter__(self) -> "CidSimsArchive":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _binding_state(bindings: Mapping[str, Path]) -> tuple[dict[str, str], dict[str, Any]]:
    required = {
        "archive",
        "acquisition_receipt",
        "contract",
        "claim",
        "source_authority",
        "burned_manifest",
        "implementation_lock",
    }
    if set(bindings) != required:
        raise ValueError("R2_BINDING_KEYS")
    hashes = {name: sha256_file(path) for name, path in bindings.items()}
    acquisition = _load_object(bindings["acquisition_receipt"])
    claim = _load_object(bindings["claim"])
    contract = _load_object(bindings["contract"])
    authority = _load_object(bindings["source_authority"])
    manifest = _load_object(bindings["burned_manifest"])
    if any(
        value.get("protocol_id") != PROTOCOL_ID
        for value in (acquisition, claim, contract, authority, manifest)
    ):
        raise ValueError("R2_BINDING_PROTOCOL")
    if acquisition.get("archive_sha256") != hashes["archive"]:
        raise ValueError("R2_ARCHIVE_HASH")
    if (
        acquisition.get("archive_bytes") != 2_211_008_069
        or acquisition.get("archive_md5")
        != "585d38855ad7d04817991cdbbb72016b"
        or acquisition.get("rgb_members_read") != 0
    ):
        raise ValueError("R2_ARCHIVE_OFFICIAL_IDENTITY")
    if acquisition.get("claim_sha256") != hashes["claim"]:
        raise ValueError("R2_ACQUISITION_CLAIM_HASH")
    for name in (
        "contract",
        "source_authority",
        "burned_manifest",
        "implementation_lock",
    ):
        if claim.get("bindings", {}).get(name, {}).get("sha256") != hashes[name]:
            raise ValueError(f"R2_CLAIM_BINDING:{name}")
        if acquisition.get("preaccess_document_sha256", {}).get(name) != hashes[name]:
            raise ValueError(f"R2_ACQUISITION_BINDING:{name}")
    if (
        acquisition.get("request_count") != 1
        or acquisition.get("retry_count") != 0
        or acquisition.get("fallback_count") != 0
        or acquisition.get("mirror_count") != 0
        or acquisition.get("head_request_count") != 0
        or acquisition.get("replacement_source_count") != 0
    ):
        raise ValueError("R2_ACQUISITION_NOT_ONE_SHOT")
    if (
        contract.get("source_selection", {}).get("candidate_id") != CANDIDATE_ID
        or contract.get("source_selection", {}).get("candidate_count") != 1
        or contract.get("source_selection", {}).get("no_replacement") is not True
    ):
        raise ValueError("R2_CONTRACT_CANDIDATE")
    gates = contract.get("geometry_rule", {}).get("gates", {})
    expected_gates = {
        "candidate_pair_coverage_min": MIN_COVERAGE,
        "median_signed_radial_expansion_per_s_min": MIN_SIGNED_RADIAL,
        "median_radial_expansion_positive_fraction_min": MIN_POSITIVE_FRACTION,
        "minimum_evaluable_pairs": MIN_EVALUABLE_PAIRS,
    }
    if any(float(gates.get(name, -1)) != float(value) for name, value in expected_gates.items()):
        raise ValueError("R2_CONTRACT_GATE_DRIFT")
    if authority.get("candidate_count") != 1:
        raise ValueError("R2_AUTHORITY_CANDIDATE_COUNT")
    ancestry = authority.get("ancestry_and_independence", {})
    if (
        ancestry.get("ancestry") != EXPECTED_ANCESTRY
        or ancestry.get("independence_group") != INDEPENDENCE_GROUP
        or ancestry.get("reuse_policy_after_access") != REUSE_POLICY
        or not REQUIRED_BURNED_GROUPS.issubset(set(ancestry.get("required_non_overlap", [])))
    ):
        raise ValueError("R2_AUTHORITY_ANCESTRY_OR_REUSE")
    if set(manifest.get("r2_candidate_must_not_descend_from", [])) != REQUIRED_BURNED_GROUPS:
        raise ValueError("R2_BURNED_MANIFEST_SCOPE")
    return hashes, {
        "acquisition": acquisition,
        "claim": claim,
        "contract": contract,
        "authority": authority,
        "manifest": manifest,
    }


def _evaluate(
    archive: CidSimsArchive,
    workers: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if workers < 1:
        raise ValueError("R2_WORKERS_MUST_BE_POSITIVE")
    intrinsic = INTRINSIC.copy()
    poses = _parse_poses(archive.read_control("pose.txt"))
    associated = [
        AssociatedRow(
            row.timestamp,
            f"color/{row.timestamp}.png",
            row.timestamp,
            f"depth/{row.timestamp}.png",
        )
        for row in poses
    ]
    depth_index = [(row.depth_timestamp, row.depth_path) for row in associated]
    start_row: AssociatedRow | None = None
    for row in associated:
        try:
            _interpolate_pose(poses, row.depth_timestamp)
        except ValueError as error:
            if str(error) not in {"R2_POSE_NOT_BRACKETED", "R2_POSE_BRACKET_TOO_WIDE"}:
                raise
        else:
            start_row = row
            break
    if start_row is None:
        raise ValueError("R2_NO_JOINT_RGBD_POSE_START")
    start = start_row.depth_timestamp
    end = start + Decimal(str(WINDOW_SECONDS))
    window_complete = (
        associated[-1].depth_timestamp >= end
        and poses[0].timestamp <= start
        and poses[-1].timestamp >= end
    )
    selected = [row for row in associated if start <= row.depth_timestamp < end]
    pair_records: list[dict[str, Any]] = []
    tasks: list[
        tuple[
            dict[str, Any],
            bytes,
            np.ndarray,
            tuple[np.ndarray, np.ndarray],
            tuple[np.ndarray, np.ndarray],
            float,
        ]
    ] = []
    image_size: tuple[int, int] | None = None
    for source_order, (previous, current) in enumerate(zip(selected, selected[1:])):
        dt = float(current.depth_timestamp - previous.depth_timestamp)
        record: dict[str, Any] = {
            "sequence_id": SEQUENCE_ID,
            "window_start_s": float(start),
            "window_end_s": float(end),
            "previous_depth_timestamp_s": float(previous.depth_timestamp),
            "current_depth_timestamp_s": float(current.depth_timestamp),
            "previous_depth_member": previous.depth_path,
            "current_depth_member": current.depth_path,
            "dt_s": dt,
            "_source_order": source_order,
        }
        if not 0.0 < dt <= MAX_DT_SECONDS:
            continue
        if (
            not archive.member_exists(previous.depth_path)
            or not archive.member_exists(previous.rgb_path)
            or not archive.member_exists(current.depth_path)
            or not archive.member_exists(current.rgb_path)
        ):
            record.update(evaluable=False, reason="ASSOCIATED_MEMBER_MISSING")
            pair_records.append(record)
            continue
        try:
            previous_pose = _interpolate_pose(poses, previous.depth_timestamp)
            current_pose = _interpolate_pose(poses, current.depth_timestamp)
        except ValueError as error:
            if str(error) not in {"R2_POSE_NOT_BRACKETED", "R2_POSE_BRACKET_TOO_WIDE"}:
                raise
            record.update(evaluable=False, reason=str(error))
            pair_records.append(record)
            continue
        try:
            raw_depth = archive.read_depth(previous.depth_path)
        except (KeyError, OSError, RuntimeError, ValueError, zipfile.BadZipFile):
            record.update(evaluable=False, reason="DEPTH_MEMBER_INVALID")
            pair_records.append(record)
            continue
        tasks.append((record, raw_depth, intrinsic, previous_pose, current_pose, dt))
    if workers == 1:
        evaluated = map(_pair_worker, tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        evaluated = executor.map(_pair_worker, tasks)
    try:
        for record, current_size in evaluated:
            if current_size is None:
                pair_records.append(record)
                continue
            if image_size is None:
                image_size = current_size
            elif image_size != current_size:
                raise ValueError("R2_DEPTH_IMAGE_SIZE_DRIFT")
            pair_records.append(record)
    finally:
        if workers != 1:
            executor.shutdown(wait=True, cancel_futures=True)
    pair_records.sort(key=lambda row: int(row["_source_order"]))
    for row in pair_records:
        del row["_source_order"]

    evaluable = [row for row in pair_records if row["evaluable"]]
    coverage = len(evaluable) / len(pair_records) if pair_records else 0.0

    def median(field: str) -> float | None:
        return (
            float(np.median([float(row[field]) for row in evaluable]))
            if evaluable
            else None
        )

    signed = median("median_signed_radial_expansion_per_s")
    positive = median("radial_expansion_positive_fraction")
    admitted = bool(
        coverage >= MIN_COVERAGE
        and window_complete
        and len(evaluable) >= MIN_EVALUABLE_PAIRS
        and signed is not None
        and signed >= MIN_SIGNED_RADIAL
        and positive is not None
        and positive >= MIN_POSITIVE_FRACTION
    )
    window = {
        "sequence_id": SEQUENCE_ID,
        "window_start_s": float(start),
        "window_end_s": float(end),
        "window_rule": "FIRST_SOURCE_NATIVE_HALF_OPEN_10_SECONDS_NO_SLIDING",
        "window_complete": window_complete,
        "candidate_pair_count": len(pair_records),
        "evaluable_pair_count": len(evaluable),
        "candidate_pair_coverage": coverage,
        "median_signed_radial_expansion_per_s": signed,
        "median_radial_expansion_positive_fraction": positive,
        "median_q90_time_normalized_parallax_rad_per_s": median(
            "q90_time_normalized_parallax_rad_per_s"
        ),
        "admitted": admitted,
    }
    source = {
        "pose_indexed_rgbd_row_count": len(associated),
        "depth_index_row_count": len(depth_index),
        "pose_row_count": len(poses),
        "image_size_wh": list(image_size) if image_size else None,
        "depth_members_read": list(archive.depth_members_read),
    }
    return window, pair_records, source


def _write_exclusive(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "wb", closefd=True) as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def run(
    output_dir: Path,
    bindings: Mapping[str, Path],
    *,
    workers: int = 1,
) -> dict[str, Any]:
    binding_hashes, bound = _binding_state(bindings)
    with CidSimsArchive(bindings["archive"]) as archive:
        if archive.inventory != bound["acquisition"].get("archive_member_inventory"):
            raise ValueError("R2_ARCHIVE_INVENTORY_MISMATCH")
        window, pairs, source_summary = _evaluate(archive, workers)
        inventory_hash = canonical_json_sha(archive.inventory)
    archive_sha = binding_hashes["archive"]
    window["content_identity"] = (
        f"sha256:{archive_sha}#{SEQUENCE_ID}"
        f"[{window['window_start_s']:.9f},{window['window_end_s']:.9f})"
    )
    terminal = ADMITTED if window["admitted"] else HOLD

    ledger_path = output_dir / "pair_ledger.jsonl"
    ledger_bytes = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in pairs
    ).encode("utf-8")
    _write_exclusive(ledger_path, ledger_bytes)

    result: dict[str, Any] = {
        "schema_version": "rcle.real_positive_approach_role.result.v2",
        "protocol_id": PROTOCOL_ID,
        "terminal": terminal,
        "authority": "DATA_ROLE_ONLY_RGB_ALGORITHM_NOT_AUTHORIZED",
        "source_identity": {
            "candidate_id": CANDIDATE_ID,
            "sequence_id": SEQUENCE_ID,
            "archive_bytes": bindings["archive"].stat().st_size,
            "archive_sha256": archive_sha,
            "archive_member_inventory_sha256": inventory_hash,
            **source_summary,
        },
        "access": {
            "read_kinds": [
                "ZIP_CENTRAL_DIRECTORY_METADATA",
                "pose.txt",
                "FIRST_WINDOW_REFERENCED_DEPTH_PNG",
            ],
            "archive_extracted": False,
            "rgb_index_or_pixel_content_read": False,
            "rgb_pixels_read": False,
            "events_read": False,
            "masks_read": False,
            "algorithm_outcome_read": False,
            "candidate_path_probe_before_claim": False,
            "replacement_source_count": 0,
            "network_request_count": 0,
        },
        "identity_and_independence": {
            "ancestry": EXPECTED_ANCESTRY,
            "independence_group": INDEPENDENCE_GROUP,
            "required_non_overlap": sorted(REQUIRED_BURNED_GROUPS),
            "reuse_policy": REUSE_POLICY,
            "future_confirmation_eligible": False,
            "future_confirmation_exclusion": INDEPENDENCE_GROUP,
        },
        "frozen_gates": {
            "window_seconds": WINDOW_SECONDS,
            "window_rule": "FIRST_SOURCE_NATIVE_HALF_OPEN_10_SECONDS_NO_SLIDING",
            "maximum_pair_dt_s": MAX_DT_SECONDS,
            "maximum_pose_bracket_s": float(MAX_POSE_BRACKET_SECONDS),
            "depth_sample_stride_px": DEPTH_SAMPLE_STRIDE_PX,
            "depth_units_per_meter": DEPTH_UNITS_PER_METER,
            "minimum_radius_px": MINIMUM_RADIUS_PX,
            "candidate_pair_coverage_min": MIN_COVERAGE,
            "minimum_evaluable_pairs": MIN_EVALUABLE_PAIRS,
            "median_signed_radial_expansion_per_s_min": MIN_SIGNED_RADIAL,
            "median_positive_fraction_min": MIN_POSITIVE_FRACTION,
        },
        "windows": [window],
        "admitted_window_count": int(window["admitted"]),
        "admitted_content_identities": (
            [window["content_identity"]] if window["admitted"] else []
        ),
        "pair_record_count": len(pairs),
        "pair_ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
        "bindings": binding_hashes,
        "algorithm_implementation_or_execution_authorized": False,
        "performance_qualification_authorized": False,
        "performance_qualification_task_may_be_created": terminal == ADMITTED,
        "worker_policy": {
            "workers": workers,
            "default_workers": 1,
            "pair_parallelism": True,
            "source_order_preserved": True,
            "worker_opencv_threads": 1,
            "worker_blas_threads": 1,
        },
    }
    result["result_payload_sha256"] = canonical_json_sha(result)
    result_path = output_dir / "result.json"
    _write_exclusive(result_path, _json_bytes(result))
    receipt = {
        "schema_version": "rcle.real_positive_approach_role.receipt.v2",
        "protocol_id": PROTOCOL_ID,
        "terminal": terminal,
        "result_sha256": sha256_file(result_path),
        "pair_ledger_sha256": sha256_file(ledger_path),
        "archive_sha256": archive_sha,
        "archive_member_inventory_sha256": inventory_hash,
        "algorithm_outcome_read": False,
        "rgb_pixels_read": False,
        "replacement_source_count": 0,
        "admitted_window_count": int(window["admitted"]),
        "performance_qualification_may_be_created": terminal == ADMITTED,
    }
    _write_exclusive(output_dir / "receipt.json", _json_bytes(receipt))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--acquisition-receipt", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--claim", required=True, type=Path)
    parser.add_argument("--source-authority", required=True, type=Path)
    parser.add_argument("--burned-manifest", required=True, type=Path)
    parser.add_argument("--implementation-lock", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    result = run(
        args.output_dir,
        {
            "archive": args.archive,
            "acquisition_receipt": args.acquisition_receipt,
            "contract": args.contract,
            "claim": args.claim,
            "source_authority": args.source_authority,
            "burned_manifest": args.burned_manifest,
            "implementation_lock": args.implementation_lock,
        },
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "terminal": result["terminal"],
                "admitted_window_count": result["admitted_window_count"],
                "pair_record_count": result["pair_record_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
