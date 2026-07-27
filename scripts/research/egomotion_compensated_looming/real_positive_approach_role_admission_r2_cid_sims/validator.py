"""Independent replay validator for CID-SIMS floor3_1 role-admission R2.

This module intentionally does not import the producer or its helpers.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
from dataclasses import dataclass
from decimal import Decimal
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import numpy as np
from PIL import Image


PROTOCOL_ID = "RCLE_REAL_POSITIVE_APPROACH_ROLE_ADMISSION_R2_CID_SIMS"
CANDIDATE_ID = "CID_SIMS_V6_FLOOR3_1"
SEQUENCE_ID = "floor3_1"
ADMITTED = "REAL_POSITIVE_APPROACH_ROLE_ADMITTED / VALID"
HOLD = "HOLD_ALGORITHM_CANARY_APPROACH_ROLE_INCOMPLETE / VALID"
WINDOW_SECONDS = Decimal("10.0")
MAX_DT = Decimal("0.100")
MAX_POSE_BRACKET = Decimal("0.100")
MIN_COVERAGE = 0.80
MIN_EVALUABLE = 8
MIN_SIGNED = 0.05
MIN_POSITIVE = 0.75
STRIDE = 8
DEPTH_SCALE = 1000.0
MIN_RADIUS = 8.0
INDEPENDENCE_GROUP = "CID_SIMS_V6_OFFICE_BUILDING_FLOOR3_THREE_RUN_CAPTURE_FAMILY"
ANCESTRY = [
    "SCIENCEDB_OFFICIAL",
    "CID_SIMS_V6",
    "CID_SIMS_V6_OFFICE_BUILDING",
    "CID_SIMS_V6_OFFICE_BUILDING_FLOOR3",
    "CID_SIMS_V6_OFFICE_BUILDING_FLOOR3_1",
    PROTOCOL_ID,
]
NON_OVERLAP = {
    "TUM_ALL_RCLE_ACCESSED_AND_DERIVED_FAMILIES",
    "BONN_ALL_RCLE_COHORTS_WINDOWS_AND_DERIVATIVES",
    "RCLE_PHASE_A_ALL_SYNTHETIC_GENERATOR_AND_SUPPORT_FAMILIES",
    "EVIMO2_V2_FLEA3_SANITY_LL_CAPTURE_FAMILY",
    "ALL_PREEXISTING_ARTIFACTS_LOCAL_PAYLOADS",
    "ETH3D_SLAM_SOFA_SCENE_CAPTURE_FAMILY",
}
REUSE = (
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
ACCESS = {
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
}


@dataclass(frozen=True)
class VAssociation:
    rgb_time: Decimal
    rgb_name: str
    depth_time: Decimal
    depth_name: str


@dataclass(frozen=True)
class VPose:
    time: Decimal
    center: np.ndarray
    quaternion: np.ndarray


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def object_from(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("R2_VALIDATOR_JSON_OBJECT")
    return value


def rows(raw: bytes) -> Iterable[list[str]]:
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            yield line.split()


def relative_name(value: str, directory: str) -> str:
    name = PurePosixPath(value)
    if (
        name.is_absolute()
        or ".." in name.parts
        or "\\" in value
        or len(name.parts) != 2
        or name.parts[0] != directory
    ):
        raise ValueError("R2_VALIDATOR_INDEX_NAME")
    return name.as_posix()


def associations(raw: bytes) -> list[VAssociation]:
    result: list[VAssociation] = []
    for tokens in rows(raw):
        if len(tokens) != 4:
            raise ValueError("R2_VALIDATOR_ASSOCIATED_COLUMNS")
        result.append(
            VAssociation(
                Decimal(tokens[0]),
                relative_name(tokens[1], "rgb"),
                Decimal(tokens[2]),
                relative_name(tokens[3], "depth"),
            )
        )
    if len(result) < 2 or any(
        a.depth_time >= b.depth_time for a, b in zip(result, result[1:])
    ):
        raise ValueError("R2_VALIDATOR_ASSOCIATED_ORDER")
    if len({item.depth_name for item in result}) != len(result):
        raise ValueError("R2_VALIDATOR_ASSOCIATED_DUPLICATE")
    return result


def depth_table(raw: bytes) -> list[tuple[Decimal, str]]:
    result: list[tuple[Decimal, str]] = []
    for tokens in rows(raw):
        if len(tokens) != 2:
            raise ValueError("R2_VALIDATOR_DEPTH_COLUMNS")
        result.append((Decimal(tokens[0]), relative_name(tokens[1], "depth")))
    if not result or any(a[0] >= b[0] for a, b in zip(result, result[1:])):
        raise ValueError("R2_VALIDATOR_DEPTH_ORDER")
    return result


def unit_quaternion(raw: np.ndarray) -> np.ndarray:
    value = np.array(raw, dtype=np.float64, copy=True)
    length = float(np.sqrt(np.dot(value, value)))
    if value.shape != (4,) or not np.all(np.isfinite(value)) or length <= 0.0:
        raise ValueError("R2_VALIDATOR_QUATERNION")
    return value / length


def pose_table(raw: bytes) -> list[VPose]:
    result: list[VPose] = []
    for tokens in rows(raw):
        if len(tokens) != 8:
            raise ValueError("R2_VALIDATOR_POSE_COLUMNS")
        numeric = np.array([float(value) for value in tokens[1:]], dtype=np.float64)
        if not np.all(np.isfinite(numeric)):
            raise ValueError("R2_VALIDATOR_POSE_FINITE")
        result.append(VPose(Decimal(tokens[0]), numeric[:3], unit_quaternion(numeric[3:])))
    if not result or any(a.time >= b.time for a, b in zip(result, result[1:])):
        raise ValueError("R2_VALIDATOR_POSE_ORDER")
    return result


def camera_matrix(raw: bytes) -> np.ndarray:
    values = [float(value) for value in raw.decode("utf-8").split()]
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("R2_VALIDATOR_CALIBRATION")
    fx, fy, cx, cy = values
    if min(fx, fy) <= 0.0:
        raise ValueError("R2_VALIDATOR_CALIBRATION")
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def interpolate(poses: Sequence[VPose], timestamp: Decimal) -> tuple[np.ndarray, np.ndarray]:
    times = [item.time for item in poses]
    at = bisect_left(times, timestamp)
    if at < len(poses) and poses[at].time == timestamp:
        return poses[at].center.copy(), poses[at].quaternion.copy()
    if at == 0 or at == len(poses):
        raise ValueError("R2_POSE_NOT_BRACKETED")
    before, after = poses[at - 1], poses[at]
    span = after.time - before.time
    if span > MAX_POSE_BRACKET:
        raise ValueError("R2_POSE_BRACKET_TOO_WIDE")
    fraction = float((timestamp - before.time) / span)
    q0, q1 = unit_quaternion(before.quaternion), unit_quaternion(after.quaternion)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = unit_quaternion(q0 + fraction * (q1 - q0))
    else:
        theta = math.acos(float(np.clip(dot, -1.0, 1.0)))
        q = unit_quaternion(
            math.sin((1.0 - fraction) * theta) / math.sin(theta) * q0
            + math.sin(fraction * theta) / math.sin(theta) * q1
        )
    return before.center + fraction * (after.center - before.center), q


def rotation(q: np.ndarray) -> np.ndarray:
    x, y, z, w = unit_quaternion(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def relative(before: tuple[np.ndarray, np.ndarray], after: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    before_center, before_q = before
    after_center, after_q = after
    before_r, after_r = rotation(before_q), rotation(after_q)
    return after_r.T.dot(before_r), after_r.T.dot(before_center - after_center)


def depth_samples(raw: bytes) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    try:
        with Image.open(BytesIO(raw)) as image:
            image.load()
            array = np.array(image, copy=True)
    except (OSError, ValueError) as error:
        raise ValueError("R2_VALIDATOR_DEPTH_PNG") from error
    if array.ndim != 2 or array.dtype != np.uint16:
        raise ValueError("R2_VALIDATOR_DEPTH_FORMAT")
    height, width = array.shape
    y_indices = np.arange(0, height, STRIDE, dtype=np.int64)
    x_indices = np.arange(0, width, STRIDE, dtype=np.int64)
    x_grid, y_grid = np.meshgrid(x_indices, y_indices)
    values = array[y_grid, x_grid].ravel()
    keep = values > 0
    pixels = np.stack((x_grid.ravel()[keep], y_grid.ravel()[keep]), axis=1).astype(
        np.float64
    )
    return pixels, values[keep].astype(np.float64) / DEPTH_SCALE, (width, height)


def independent_geometry(
    pixels: np.ndarray,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    relative_rotation: np.ndarray,
    relative_translation: np.ndarray,
    dt: float,
    size: tuple[int, int],
) -> dict[str, Any]:
    width, height = size
    homogeneous = np.concatenate(
        (pixels, np.ones((pixels.shape[0], 1), dtype=np.float64)), axis=1
    )
    previous_points = np.linalg.solve(intrinsic, homogeneous.T).T * depth[:, None]
    rotated = previous_points.dot(relative_rotation.T)
    moved = rotated + relative_translation
    projected_r_h = rotated.dot(intrinsic.T)
    projected_f_h = moved.dot(intrinsic.T)
    with np.errstate(divide="ignore", invalid="ignore"):
        projected_r = projected_r_h[:, :2] / projected_r_h[:, 2, None]
        projected_f = projected_f_h[:, :2] / projected_f_h[:, 2, None]
    center = np.array([intrinsic[0, 2], intrinsic[1, 2]])
    radius_r = np.sqrt(np.sum((projected_r - center) ** 2, axis=1))
    radius_f = np.sqrt(np.sum((projected_f - center) ** 2, axis=1))
    valid = (
        (rotated[:, 2] > 0)
        & (moved[:, 2] > 0)
        & (projected_r[:, 0] >= 0)
        & (projected_r[:, 0] < width)
        & (projected_r[:, 1] >= 0)
        & (projected_r[:, 1] < height)
        & (projected_f[:, 0] >= 0)
        & (projected_f[:, 0] < width)
        & (projected_f[:, 1] >= 0)
        & (projected_f[:, 1] < height)
        & np.isfinite(radius_r)
        & np.isfinite(radius_f)
        & (radius_r >= MIN_RADIUS)
        & (radius_f > 0)
    )
    winners: dict[tuple[int, int], int] = {}
    for index in np.flatnonzero(valid):
        destination = tuple(np.floor(projected_f[index] + 0.5).astype(np.int64))
        current = winners.get(destination)
        if current is None or moved[index, 2] < moved[current, 2]:
            winners[destination] = int(index)
    chosen = np.array(sorted(winners.values()), dtype=np.int64)
    base: dict[str, Any] = {
        "evaluable": bool(chosen.size),
        "source_count": int(pixels.shape[0]),
        "valid_count": int(chosen.size),
        "valid_fraction": float(chosen.size / pixels.shape[0] if pixels.shape[0] else 0),
        "raw_translation_speed_m_s": float(np.linalg.norm(relative_translation) / dt),
    }
    if chosen.size == 0:
        return base
    rotated_chosen, moved_chosen = rotated[chosen], moved[chosen]
    bearing_r = rotated_chosen / np.linalg.norm(rotated_chosen, axis=1)[:, None]
    bearing_f = moved_chosen / np.linalg.norm(moved_chosen, axis=1)[:, None]
    dot = np.sum(bearing_r * bearing_f, axis=1)
    cross = np.linalg.norm(np.cross(bearing_r, bearing_f), axis=1)
    parallax = np.arctan2(cross, np.clip(dot, -1.0, 1.0)) / dt
    radial = np.log(radius_f[chosen] / radius_r[chosen]) / dt
    base.update(
        {
            "median_signed_radial_expansion_per_s": float(np.median(radial)),
            "median_absolute_radial_expansion_per_s": float(np.median(np.abs(radial))),
            "radial_expansion_positive_fraction": float(np.count_nonzero(radial > 0) / radial.size),
            "q90_time_normalized_parallax_rad_per_s": float(np.quantile(parallax, 0.90)),
        }
    )
    return base


class ValidationArchive:
    def __init__(self, path: Path) -> None:
        self._zip = zipfile.ZipFile(path)
        infos = self._zip.infolist()
        names = [item.filename for item in infos]
        normalized = [PurePosixPath(name).as_posix() for name in names]
        if len(normalized) != len(set(normalized)):
            raise ValueError("R2_VALIDATOR_DUPLICATE_MEMBER")
        for item, name in zip(infos, names):
            parsed = PurePosixPath(name)
            if (
                parsed.is_absolute()
                or ".." in parsed.parts
                or "\\" in name
                or bool(re.match(r"^[A-Za-z]:", name))
                or item.flag_bits & 0x1
            ):
                raise ValueError("R2_VALIDATOR_UNSAFE_MEMBER")
        self.inventory = [
            {
                "name": item.filename,
                "crc32": f"{item.CRC:08x}",
                "compressed_bytes": item.compress_size,
                "uncompressed_bytes": item.file_size,
                "is_directory": item.is_dir(),
            }
            for item in infos
        ]
        self.files = {item.filename for item in infos if not item.is_dir()}
        required = {f"{SEQUENCE_ID}/pose.txt"}
        if not required.issubset(self.files):
            raise ValueError("R2_VALIDATOR_CONTROL_MEMBERS")
        if {PurePosixPath(name).parts[0] for name in self.files} != {SEQUENCE_ID}:
            raise ValueError("R2_VALIDATOR_ROOT")
        self.depth_reads: list[str] = []

    def control(self, name: str) -> bytes:
        if name != "pose.txt":
            raise ValueError("R2_VALIDATOR_CONTROL_READ")
        return self._zip.read(f"{SEQUENCE_ID}/{name}")

    def has(self, relative: str) -> bool:
        return f"{SEQUENCE_ID}/{relative}" in self.files

    def depth(self, relative: str) -> bytes:
        name = relative_name(relative, "depth")
        member = f"{SEQUENCE_ID}/{name}"
        if member not in self.files:
            raise ValueError("R2_VALIDATOR_DEPTH_MISSING")
        self.depth_reads.append(member)
        return self._zip.read(member)

    def __enter__(self) -> "ValidationArchive":
        return self

    def __exit__(self, *_: object) -> None:
        self._zip.close()


def replay(archive_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    with ValidationArchive(archive_path) as archive:
        intrinsic = INTRINSIC.copy()
        poses = pose_table(archive.control("pose.txt"))
        associated = [
            VAssociation(
                item.time,
                f"color/{item.time}.png",
                item.time,
                f"depth/{item.time}.png",
            )
            for item in poses
        ]
        indexed = [(item.depth_time, item.depth_name) for item in associated]
        start_item: VAssociation | None = None
        for item in associated:
            try:
                interpolate(poses, item.depth_time)
            except ValueError as error:
                if str(error) not in {
                    "R2_POSE_NOT_BRACKETED",
                    "R2_POSE_BRACKET_TOO_WIDE",
                }:
                    raise
            else:
                start_item = item
                break
        if start_item is None:
            raise ValueError("R2_VALIDATOR_NO_JOINT_RGBD_POSE_START")
        start = start_item.depth_time
        end = start + WINDOW_SECONDS
        window_complete = (
            associated[-1].depth_time >= end
            and poses[0].time <= start
            and poses[-1].time >= end
        )
        selected = [item for item in associated if start <= item.depth_time < end]
        pair_rows: list[dict[str, Any]] = []
        image_size: tuple[int, int] | None = None
        for before, after in zip(selected, selected[1:]):
            dt_decimal = after.depth_time - before.depth_time
            dt = float(dt_decimal)
            row: dict[str, Any] = {
                "sequence_id": SEQUENCE_ID,
                "window_start_s": float(start),
                "window_end_s": float(end),
                "previous_depth_timestamp_s": float(before.depth_time),
                "current_depth_timestamp_s": float(after.depth_time),
                "previous_depth_member": before.depth_name,
                "current_depth_member": after.depth_name,
                "dt_s": dt,
            }
            if not Decimal(0) < dt_decimal <= MAX_DT:
                continue
            if (
                not archive.has(before.depth_name)
                or not archive.has(before.rgb_name)
                or not archive.has(after.depth_name)
                or not archive.has(after.rgb_name)
            ):
                row.update(evaluable=False, reason="ASSOCIATED_MEMBER_MISSING")
                pair_rows.append(row)
                continue
            try:
                before_pose = interpolate(poses, before.depth_time)
                after_pose = interpolate(poses, after.depth_time)
            except ValueError as error:
                if str(error) not in {"R2_POSE_NOT_BRACKETED", "R2_POSE_BRACKET_TOO_WIDE"}:
                    raise
                row.update(evaluable=False, reason=str(error))
                pair_rows.append(row)
                continue
            try:
                pixels, depth, size = depth_samples(archive.depth(before.depth_name))
            except (KeyError, OSError, RuntimeError, ValueError, zipfile.BadZipFile):
                row.update(evaluable=False, reason="DEPTH_MEMBER_INVALID")
                pair_rows.append(row)
                continue
            if image_size is None:
                image_size = size
            elif image_size != size:
                raise ValueError("R2_VALIDATOR_IMAGE_SIZE")
            relative_r, relative_t = relative(before_pose, after_pose)
            summary = independent_geometry(
                pixels, depth, intrinsic, relative_r, relative_t, dt, size
            )
            row.update(summary)
            if not summary["evaluable"]:
                row["reason"] = "NO_VALID_GEOMETRY_SAMPLES"
            pair_rows.append(row)
        evaluable = [row for row in pair_rows if row["evaluable"]]
        coverage = len(evaluable) / len(pair_rows) if pair_rows else 0.0

        def median(field: str) -> float | None:
            return (
                float(np.median([float(row[field]) for row in evaluable]))
                if evaluable
                else None
            )

        signed = median("median_signed_radial_expansion_per_s")
        positive = median("radial_expansion_positive_fraction")
        is_admitted = bool(
            coverage >= MIN_COVERAGE
            and window_complete
            and len(evaluable) >= MIN_EVALUABLE
            and signed is not None
            and signed >= MIN_SIGNED
            and positive is not None
            and positive >= MIN_POSITIVE
        )
        window = {
            "sequence_id": SEQUENCE_ID,
            "window_start_s": float(start),
            "window_end_s": float(end),
            "window_rule": "FIRST_SOURCE_NATIVE_HALF_OPEN_10_SECONDS_NO_SLIDING",
            "window_complete": window_complete,
            "candidate_pair_count": len(pair_rows),
            "evaluable_pair_count": len(evaluable),
            "candidate_pair_coverage": coverage,
            "median_signed_radial_expansion_per_s": signed,
            "median_radial_expansion_positive_fraction": positive,
            "median_q90_time_normalized_parallax_rad_per_s": median(
                "q90_time_normalized_parallax_rad_per_s"
            ),
            "admitted": is_admitted,
        }
        source = {
            "pose_indexed_rgbd_row_count": len(associated),
            "depth_index_row_count": len(indexed),
            "pose_row_count": len(poses),
            "image_size_wh": list(image_size) if image_size else None,
            "depth_members_read": list(archive.depth_reads),
            "inventory": archive.inventory,
        }
        return window, pair_rows, source


def close_enough(left: Any, right: Any, tolerance: float = 1e-10) -> bool:
    if isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
        if left is None or right is None:
            return left is right
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            close_enough(left[key], right[key], tolerance) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            close_enough(a, b, tolerance) for a, b in zip(left, right)
        )
    return left == right


def _write_validation(path: Path, value: dict[str, Any]) -> None:
    encoded = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "wb", closefd=True) as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def validate_source_hold(
    formal_dir: Path,
    acquisition_receipt: Path,
    bindings: Mapping[str, Path],
) -> dict[str, Any]:
    """Independently validate a frozen one-shot source-access HOLD."""
    required = {
        "archive",
        "contract",
        "claim",
        "source_authority",
        "burned_manifest",
        "implementation_lock",
    }
    if set(bindings) != required:
        raise ValueError("R2_SOURCE_HOLD_BINDING_KEYS")
    acquisition = object_from(acquisition_receipt)
    claim = object_from(bindings["claim"])
    contract = object_from(bindings["contract"])
    authority = object_from(bindings["source_authority"])
    manifest = object_from(bindings["burned_manifest"])
    hashes = {name: file_hash(path) for name, path in bindings.items()}
    errors: list[str] = []
    if any(
        value.get("protocol_id") != PROTOCOL_ID
        for value in (acquisition, claim, contract, authority, manifest)
    ):
        errors.append("PROTOCOL_IDENTITY")
    if (
        acquisition.get("terminal") != HOLD
        or acquisition.get("source_access_complete") is not False
        or acquisition.get("validation_scope")
        != "ONE_SHOT_SOURCE_ACCESS_PROCEDURE_ONLY"
    ):
        errors.append("SOURCE_HOLD_TERMINAL")
    if (
        acquisition.get("request_count") != 1
        or acquisition.get("retry_count") != 0
        or acquisition.get("fallback_count") != 0
        or acquisition.get("mirror_count") != 0
        or acquisition.get("head_request_count") != 0
        or acquisition.get("replacement_source_count") != 0
        or acquisition.get("retry_authorized") is not False
        or acquisition.get("replacement_source_authorized") is not False
    ):
        errors.append("ONE_SHOT_ACQUISITION")
    if (
        acquisition.get("source_error_code")
        not in {
            "R2_OFFICIAL_SOURCE_ACCESS_FAILED",
            "R2_OFFICIAL_PAYLOAD_IDENTITY_MISMATCH",
            "R2_OFFICIAL_CONTAINER_UNREADABLE",
            "R2_OFFICIAL_CONTAINER_IDENTITY_INCOMPLETE",
        }
        or acquisition.get("response_artifact_bytes")
        != bindings["archive"].stat().st_size
        or acquisition.get("response_artifact_sha256") != hashes["archive"]
    ):
        errors.append("SOURCE_HOLD_PARTIAL_IDENTITY")
    allowed_urls = {
        "https://china.scidb.cn/download?fileId=c595882daafe788a29d687872cc1fc2a",
        "https://china.scidb.cn/download?fileId=c595882daafe788a29d687872cc1fc2a",
    }
    if (
        acquisition.get("request_method") != "GET"
        or acquisition.get("requested_url")
        != "https://china.scidb.cn/download?fileId=c595882daafe788a29d687872cc1fc2a"
        or acquisition.get("final_url") is not None
        and acquisition.get("final_url") not in allowed_urls
        or not isinstance(acquisition.get("redirect_chain"), list)
        or any(
            url not in allowed_urls
            for url in acquisition.get("redirect_chain", [])
        )
    ):
        errors.append("SOURCE_HOLD_REQUEST_IDENTITY")
    try:
        request_started = datetime.fromisoformat(
            str(acquisition["request_started_at_utc"])
        )
        claim_created = datetime.fromisoformat(str(claim["created_at_utc"]))
    except (KeyError, TypeError, ValueError):
        errors.append("CLAIM_BEFORE_REQUEST_TIME")
    else:
        if request_started < claim_created:
            errors.append("CLAIM_BEFORE_REQUEST_TIME")
    if acquisition.get("claim_sha256") != hashes["claim"]:
        errors.append("ACQUISITION_CLAIM_IDENTITY")
    for name in (
        "contract",
        "source_authority",
        "burned_manifest",
        "implementation_lock",
    ):
        if (
            claim.get("bindings", {}).get(name, {}).get("sha256") != hashes[name]
            or acquisition.get("preaccess_document_sha256", {}).get(name)
            != hashes[name]
        ):
            errors.append(f"PREACCESS_BINDING:{name}")
    if (
        claim.get("source_access_started_before_claim") is not False
        or claim.get("candidate_path_probe_started_before_claim") is not False
        or claim.get("request_count_before_claim") != 0
        or claim.get("payload_bytes_read_before_claim") != 0
    ):
        errors.append("PRECLAIM_ACCESS")
    if (
        contract.get("source_selection", {}).get("terminal_on_source_failure")
        != HOLD
        or contract.get("performance_qualification_gate", {}).get(
            "only_unlock_terminal"
        )
        != ADMITTED
    ):
        errors.append("CONTRACT_TERMINAL_DRIFT")
    validation = {
        "schema_version": "rcle.real_positive_approach_role.source_hold_validation.v1",
        "protocol_id": PROTOCOL_ID,
        "validation_terminal": "VALID" if not errors else "INVALID",
        "scientific_terminal": HOLD,
        "validation_scope": "ONE_SHOT_SOURCE_ACCESS_PROCEDURE_ONLY",
        "source_access_complete": False,
        "performance_qualification_may_be_created": False,
        "replacement_source_count": int(
            acquisition.get("replacement_source_count", -1)
        ),
        "errors": errors,
    }
    formal_dir.mkdir(parents=True, exist_ok=True)
    _write_validation(formal_dir / "validation.json", validation)
    return validation


def validate(formal_dir: Path, bindings: Mapping[str, Path]) -> dict[str, Any]:
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
        raise ValueError("R2_VALIDATOR_BINDING_KEYS")
    result_path = formal_dir / "result.json"
    receipt_path = formal_dir / "receipt.json"
    ledger_path = formal_dir / "pair_ledger.jsonl"
    result = object_from(result_path)
    receipt = object_from(receipt_path)
    acquisition = object_from(bindings["acquisition_receipt"])
    claim = object_from(bindings["claim"])
    contract = object_from(bindings["contract"])
    authority = object_from(bindings["source_authority"])
    manifest = object_from(bindings["burned_manifest"])
    produced_pairs = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    window, replay_pairs, source = replay(bindings["archive"])
    archive_sha = file_hash(bindings["archive"])
    window["content_identity"] = (
        f"sha256:{archive_sha}#{SEQUENCE_ID}"
        f"[{window['window_start_s']:.9f},{window['window_end_s']:.9f})"
    )
    expected_terminal = ADMITTED if window["admitted"] else HOLD
    hashes = {name: file_hash(path) for name, path in bindings.items()}
    errors: list[str] = []
    binding_mismatches: list[str] = []
    for name, digest in hashes.items():
        if result.get("bindings", {}).get(name) != digest:
            binding_mismatches.append(name)
    if binding_mismatches:
        errors.append("BINDING_MISMATCH")
    if any(
        value.get("protocol_id") != PROTOCOL_ID
        for value in (result, receipt, acquisition, claim, contract, authority, manifest)
    ):
        errors.append("PROTOCOL_IDENTITY")
    if acquisition.get("archive_sha256") != archive_sha:
        errors.append("ACQUISITION_ARCHIVE_IDENTITY")
    if acquisition.get("archive_member_inventory") != source["inventory"]:
        errors.append("ARCHIVE_MEMBER_IDENTITY")
    if acquisition.get("archive_member_inventory_sha256") != json_hash(source["inventory"]):
        errors.append("ARCHIVE_MEMBER_INVENTORY_HASH")
    result_source = result.get("source_identity", {})
    if (
        result_source.get("candidate_id") != CANDIDATE_ID
        or result_source.get("sequence_id") != SEQUENCE_ID
        or result_source.get("archive_sha256") != archive_sha
        or result_source.get("archive_bytes") != bindings["archive"].stat().st_size
        or result_source.get("archive_member_inventory_sha256")
        != json_hash(source["inventory"])
        or result_source.get("pose_indexed_rgbd_row_count")
        != source["pose_indexed_rgbd_row_count"]
        or result_source.get("depth_index_row_count") != source["depth_index_row_count"]
        or result_source.get("pose_row_count") != source["pose_row_count"]
        or result_source.get("image_size_wh") != source["image_size_wh"]
        or result_source.get("depth_members_read") != source["depth_members_read"]
    ):
        errors.append("RESULT_SOURCE_IDENTITY")
    if acquisition.get("claim_sha256") != hashes["claim"]:
        errors.append("ACQUISITION_CLAIM_IDENTITY")
    for name in (
        "contract",
        "source_authority",
        "burned_manifest",
        "implementation_lock",
    ):
        if (
            claim.get("bindings", {}).get(name, {}).get("sha256") != hashes[name]
            or acquisition.get("preaccess_document_sha256", {}).get(name) != hashes[name]
        ):
            errors.append(f"PREACCESS_BINDING:{name}")
    if (
        claim.get("source_access_started_before_claim") is not False
        or claim.get("candidate_path_probe_started_before_claim") is not False
        or claim.get("request_count_before_claim") != 0
        or claim.get("payload_bytes_read_before_claim") != 0
    ):
        errors.append("PRECLAIM_ACCESS")
    if (
        acquisition.get("request_count") != 1
        or acquisition.get("retry_count") != 0
        or acquisition.get("fallback_count") != 0
        or acquisition.get("mirror_count") != 0
        or acquisition.get("head_request_count") != 0
        or acquisition.get("replacement_source_count") != 0
    ):
        errors.append("ONE_SHOT_ACQUISITION")
    allowed_urls = {
        "https://china.scidb.cn/download?fileId=c595882daafe788a29d687872cc1fc2a",
        "https://china.scidb.cn/download?fileId=c595882daafe788a29d687872cc1fc2a",
    }
    if (
        acquisition.get("request_method") != "GET"
        or acquisition.get("requested_url") not in allowed_urls
        or acquisition.get("final_url") not in allowed_urls
        or acquisition.get("http_status") != 200
        or not isinstance(acquisition.get("redirect_chain"), list)
        or any(url not in allowed_urls for url in acquisition.get("redirect_chain", []))
        or acquisition.get("archive_bytes") != bindings["archive"].stat().st_size
        or acquisition.get("archive_bytes") != 2211008069
        or acquisition.get("archive_md5")
        != "585d38855ad7d04817991cdbbb72016b"
        or acquisition.get("rgb_members_read") != 0
    ):
        errors.append("ACQUISITION_REQUEST_IDENTITY")
    try:
        request_started = datetime.fromisoformat(
            str(acquisition["request_started_at_utc"])
        )
        claim_created = datetime.fromisoformat(str(claim["created_at_utc"]))
    except (KeyError, TypeError, ValueError):
        errors.append("CLAIM_BEFORE_REQUEST_TIME")
    else:
        if request_started < claim_created:
            errors.append("CLAIM_BEFORE_REQUEST_TIME")
    pair_mismatches = sum(
        not close_enough(produced, replayed)
        for produced, replayed in zip(produced_pairs, replay_pairs)
    ) + abs(len(produced_pairs) - len(replay_pairs))
    if pair_mismatches:
        errors.append("PAIR_REPLAY_MISMATCH")
    produced_windows = result.get("windows", [])
    window_mismatches = (
        int(len(produced_windows) != 1)
        if not produced_windows
        else int(not close_enough(produced_windows[0], window))
    )
    if window_mismatches:
        errors.append("WINDOW_REPLAY_MISMATCH")
    if result.get("terminal") != expected_terminal or receipt.get("terminal") != expected_terminal:
        errors.append("TERMINAL_MISMATCH")
    if receipt.get("result_sha256") != file_hash(result_path):
        errors.append("RECEIPT_RESULT_HASH")
    payload_without_hash = dict(result)
    recorded_payload_hash = payload_without_hash.pop("result_payload_sha256", None)
    if recorded_payload_hash != json_hash(payload_without_hash):
        errors.append("RESULT_PAYLOAD_HASH")
    if (
        receipt.get("pair_ledger_sha256") != file_hash(ledger_path)
        or result.get("pair_ledger_sha256") != file_hash(ledger_path)
    ):
        errors.append("PAIR_LEDGER_HASH")
    if result.get("access") != ACCESS:
        errors.append("FORBIDDEN_ACCESS_DECLARATION")
    expected_identity = {
        "ancestry": ANCESTRY,
        "independence_group": INDEPENDENCE_GROUP,
        "required_non_overlap": sorted(NON_OVERLAP),
        "reuse_policy": REUSE,
        "future_confirmation_eligible": False,
        "future_confirmation_exclusion": INDEPENDENCE_GROUP,
    }
    if result.get("identity_and_independence") != expected_identity:
        errors.append("RESULT_ANCESTRY_REUSE")
    source_authority = authority.get("ancestry_and_independence", {})
    if (
        source_authority.get("ancestry") != ANCESTRY
        or source_authority.get("independence_group") != INDEPENDENCE_GROUP
        or source_authority.get("reuse_policy_after_access") != REUSE
        or not NON_OVERLAP.issubset(set(source_authority.get("required_non_overlap", [])))
        or set(manifest.get("r2_candidate_must_not_descend_from", [])) != NON_OVERLAP
        or INDEPENDENCE_GROUP in NON_OVERLAP
    ):
        errors.append("AUTHORITY_ANCESTRY_REUSE")
    expected_gates = {
        "candidate_pair_coverage_min": MIN_COVERAGE,
        "median_signed_radial_expansion_per_s_min": MIN_SIGNED,
        "median_radial_expansion_positive_fraction_min": MIN_POSITIVE,
        "minimum_evaluable_pairs": MIN_EVALUABLE,
    }
    contract_gates = contract.get("geometry_rule", {}).get("gates", {})
    if any(
        float(contract_gates.get(name, -1)) != float(value)
        for name, value in expected_gates.items()
    ):
        errors.append("CONTRACT_GATE_DRIFT")
    if result.get("frozen_gates") != {
        "window_seconds": float(WINDOW_SECONDS),
        "window_rule": "FIRST_SOURCE_NATIVE_HALF_OPEN_10_SECONDS_NO_SLIDING",
        "maximum_pair_dt_s": float(MAX_DT),
        "maximum_pose_bracket_s": float(MAX_POSE_BRACKET),
        "depth_sample_stride_px": STRIDE,
        "depth_units_per_meter": DEPTH_SCALE,
        "minimum_radius_px": MIN_RADIUS,
        "candidate_pair_coverage_min": MIN_COVERAGE,
        "minimum_evaluable_pairs": MIN_EVALUABLE,
        "median_signed_radial_expansion_per_s_min": MIN_SIGNED,
        "median_positive_fraction_min": MIN_POSITIVE,
    }:
        errors.append("RESULT_GATE_DRIFT")
    if (
        result.get("algorithm_implementation_or_execution_authorized") is not False
        or result.get("performance_qualification_authorized") is not False
        or result.get("performance_qualification_task_may_be_created")
        != (expected_terminal == ADMITTED)
        or receipt.get("performance_qualification_may_be_created")
        != (expected_terminal == ADMITTED)
    ):
        errors.append("AUTHORITY_CEILING")
    validation = {
        "schema_version": "rcle.real_positive_approach_role.validation.v2",
        "protocol_id": PROTOCOL_ID,
        "validation_terminal": "VALID" if not errors else "INVALID",
        "scientific_terminal": expected_terminal,
        "producer_pair_record_count": len(produced_pairs),
        "validator_pair_record_count": len(replay_pairs),
        "pair_replay_mismatch_count": pair_mismatches,
        "window_replay_mismatch_count": window_mismatches,
        "binding_mismatches": binding_mismatches,
        "identity_mismatch_count": sum(
            error in {
                "ACQUISITION_ARCHIVE_IDENTITY",
                "ARCHIVE_MEMBER_IDENTITY",
                "ARCHIVE_MEMBER_INVENTORY_HASH",
                "RESULT_SOURCE_IDENTITY",
                "RESULT_ANCESTRY_REUSE",
                "AUTHORITY_ANCESTRY_REUSE",
            }
            for error in errors
        ),
        "ancestry_overlap_violation_count": int("AUTHORITY_ANCESTRY_REUSE" in errors),
        "forbidden_access_count": int("FORBIDDEN_ACCESS_DECLARATION" in errors),
        "algorithm_outcome_read_count": int(
            result.get("access", {}).get("algorithm_outcome_read") is not False
        ),
        "gate_drift_count": sum(
            error in {"CONTRACT_GATE_DRIFT", "RESULT_GATE_DRIFT"} for error in errors
        ),
        "replacement_source_count": int(
            acquisition.get("replacement_source_count", -1)
        ),
        "admitted_window_count": int(window["admitted"]),
        "archive_extracted": False,
        "rgb_index_or_pixel_content_read": False,
        "rgb_pixels_read": False,
        "producer_imported": False,
        "independent_geometry_implementation": "validator.independent_geometry",
        "performance_qualification_may_be_created": (
            expected_terminal == ADMITTED and not errors
        ),
        "errors": errors,
    }
    _write_validation(formal_dir / "validation.json", validation)
    return validation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-dir", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--acquisition-receipt", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--claim", required=True, type=Path)
    parser.add_argument("--source-authority", required=True, type=Path)
    parser.add_argument("--burned-manifest", required=True, type=Path)
    parser.add_argument("--implementation-lock", required=True, type=Path)
    args = parser.parse_args()
    validation = validate(
        args.formal_dir,
        {
            "archive": args.archive,
            "acquisition_receipt": args.acquisition_receipt,
            "contract": args.contract,
            "claim": args.claim,
            "source_authority": args.source_authority,
            "burned_manifest": args.burned_manifest,
            "implementation_lock": args.implementation_lock,
        },
    )
    print(json.dumps(validation, ensure_ascii=False, sort_keys=True))
    return 0 if validation["validation_terminal"] == "VALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
