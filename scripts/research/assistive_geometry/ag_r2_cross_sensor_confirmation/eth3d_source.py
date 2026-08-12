"""Fail-closed ETH3D archive and RGB-D source adapter for AG R2 F2.

The archive API deliberately has two stages.  ``verify_archive_binding`` only
opens the opaque file as bytes and returns a verification token.  ZIP member
enumeration is possible only by passing that token to ``preflight_archive``;
the latter rechecks the same open file descriptor before constructing a
``ZipFile``.  Importing this module performs no I/O.

The frame adapter exposes only the two payload surfaces required by the frozen
protocol: score-role RGB plus K for raw prediction, and role-matched metric
depth plus K for calibration/source scoring.  It has no model, metric, reducer,
training, or evidence-writing dependency.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import stat
import unicodedata
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from enum import Enum
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any, Self
from urllib.parse import unquote, urlsplit

import cv2
import numpy as np

from . import PROTOCOL_ID
from .contract import (
    ACCELEROMETER_SIGN_CONTRACT,
    CALIBRATION_ENCODING,
    CAMERA_FROM_IMU_DIRECTION,
    IMU_COLUMN_CONTRACT,
    IMU_FRAME_CONTRACT,
    ContractError,
    canonical_sha256,
    require,
    sha256_file,
)
from .control_format import parse_kalibr_camera_from_imu

_HEX_64 = re.compile(r"^[0-9A-Fa-f]{64}$")
_DECIMAL_TOKEN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_NANOSECOND_TOKEN = re.compile(r"^(?:0|[1-9]\d*)$")
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")
_VERIFICATION_SEAL = object()
NANOSECONDS_TO_SECONDS = Decimal("1e-9")


class SourcePhase(str, Enum):
    """Only phases that may touch an archive through this module."""

    ROSTER_METADATA = "ROSTER_METADATA"
    RAW_SCORE_PREDICTION = "RAW_SCORE_PREDICTION"
    CALIBRATION_CONTROL = "CALIBRATION_CONTROL"
    CALIBRATION_SOURCE = "CALIBRATION_SOURCE"
    SCORE_SOURCE = "SCORE_SOURCE"


@dataclass(frozen=True)
class ArchiveBudget:
    """Lock-bindable limits applied before any member is decompressed."""

    max_members: int = 100_000
    max_member_uncompressed_bytes: int = 1 << 30
    max_total_uncompressed_bytes: int = 32 << 30
    max_compression_ratio: float = 200.0
    max_metadata_bytes: int = 16 << 20

    def __post_init__(self) -> None:
        for name in (
            "max_members",
            "max_member_uncompressed_bytes",
            "max_total_uncompressed_bytes",
            "max_metadata_bytes",
        ):
            value = getattr(self, name)
            require(type(value) is int and value > 0, f"F2_ARCHIVE_BUDGET_{name.upper()}")
        require(
            type(self.max_compression_ratio) in {int, float}
            and math.isfinite(float(self.max_compression_ratio))
            and float(self.max_compression_ratio) >= 1.0,
            "F2_ARCHIVE_BUDGET_COMPRESSION_RATIO",
        )


@dataclass(frozen=True)
class ArchiveBinding:
    parent_id: str
    kind: str
    url: str
    filename: str
    bytes: int
    sha256: str

    @classmethod
    def from_manifest_row(cls, row: Mapping[str, Any]) -> ArchiveBinding:
        require(
            set(row) == {"parent_id", "kind", "url", "bytes", "sha256"},
            "F2_ARCHIVE_BINDING_KEY_SET",
        )
        parent_id = row["parent_id"]
        kind = row["kind"]
        url = row["url"]
        require(isinstance(parent_id, str) and parent_id and "\n" not in parent_id, "F2_ARCHIVE_PARENT_ID")
        require(
            kind in {
                "RGBD_TRAINING_ARCHIVE",
                "IMU_ARCHIVE",
                "CAMERA_IMU_CALIBRATION_ARCHIVE",
            },
            "F2_ARCHIVE_KIND",
        )
        require(isinstance(url, str), "F2_ARCHIVE_URL")
        parsed = urlsplit(url)
        require(
            parsed.scheme == "https"
            and parsed.hostname == "www.eth3d.net"
            and not parsed.query
            and not parsed.fragment,
            "F2_ARCHIVE_URL",
        )
        filename = unquote(PurePosixPath(parsed.path).name)
        require(
            filename
            and filename == Path(filename).name
            and filename.endswith(".zip")
            and "/" not in filename
            and "\\" not in filename,
            "F2_ARCHIVE_FILENAME",
        )
        if kind == "RGBD_TRAINING_ARCHIVE":
            require(filename == f"{parent_id}_rgbd.zip", "F2_ARCHIVE_FILENAME_IDENTITY")
        elif kind == "IMU_ARCHIVE":
            require(filename == f"{parent_id}_imu.zip", "F2_ARCHIVE_FILENAME_IDENTITY")
        else:
            require(
                parent_id == "ALL_THREE_SESSIONS" and filename == "camera_imu_calib_radtan.zip",
                "F2_ARCHIVE_FILENAME_IDENTITY",
            )
        byte_count = row["bytes"]
        sha256 = row["sha256"]
        require(type(byte_count) is int and byte_count > 0, "F2_ARCHIVE_BINDING_BYTES")
        require(isinstance(sha256, str) and _HEX_64.fullmatch(sha256) is not None, "F2_ARCHIVE_BINDING_SHA")
        return cls(parent_id, kind, url, filename, byte_count, sha256.upper())


@dataclass(frozen=True)
class VerifiedArchive:
    """Opaque-byte verification token; it contains no ZIP inventory."""

    binding: ArchiveBinding
    archive_root: Path
    path: Path
    _seal: object


@dataclass(frozen=True)
class ArchiveMember:
    name: str
    compressed_bytes: int
    uncompressed_bytes: int
    crc32: str
    is_directory: bool


@dataclass(frozen=True)
class ReadEvent:
    parent_id: str
    archive_kind: str
    phase: SourcePhase
    purpose: str
    member: str
    bytes: int


ReadObserver = Callable[[ReadEvent], None]


def _is_reparse_point(path: Path) -> bool:
    try:
        value = path.lstat()
    except OSError as error:
        raise ContractError("F2_ARCHIVE_PATH_STAT", str(error)) from error
    attributes = getattr(value, "st_file_attributes", 0)
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return path.is_symlink() or bool(attributes & flag)


def _checked_direct_child(archive_root: Path, filename: str) -> tuple[Path, Path]:
    root = Path(archive_root)
    require(root.is_absolute(), "F2_ARCHIVE_ROOT_NOT_ABSOLUTE")
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        raise ContractError("F2_ARCHIVE_ROOT_INVALID", str(error)) from error
    require(resolved_root.is_dir(), "F2_ARCHIVE_ROOT_NOT_DIRECTORY")
    candidate = root / filename
    require(candidate.parent == root and candidate.name == filename, "F2_ARCHIVE_NOT_DIRECT_CHILD")
    require(candidate.exists() and candidate.is_file(), "F2_ARCHIVE_FILE_MISSING")
    require(not _is_reparse_point(candidate), "F2_ARCHIVE_REPARSE_POINT")
    try:
        resolved_candidate = candidate.resolve(strict=True)
    except OSError as error:
        raise ContractError("F2_ARCHIVE_PATH_INVALID", str(error)) from error
    require(resolved_candidate.parent == resolved_root, "F2_ARCHIVE_DIRECTORY_ESCAPE")
    return resolved_root, resolved_candidate


def verify_archive_binding(archive_root: Path, binding: ArchiveBinding | Mapping[str, Any]) -> VerifiedArchive:
    """Verify direct-child identity, bytes, and SHA without opening the ZIP."""

    if not isinstance(binding, ArchiveBinding):
        require(isinstance(binding, Mapping), "F2_ARCHIVE_BINDING_TYPE")
        binding = ArchiveBinding.from_manifest_row(binding)
    resolved_root, path = _checked_direct_child(Path(archive_root), binding.filename)
    try:
        size = path.stat().st_size
    except OSError as error:
        raise ContractError("F2_ARCHIVE_PATH_STAT", str(error)) from error
    require(size == binding.bytes, "F2_ARCHIVE_BYTES_MISMATCH")
    require(sha256_file(path) == binding.sha256, "F2_ARCHIVE_SHA_MISMATCH")
    return VerifiedArchive(binding=binding, archive_root=resolved_root, path=path, _seal=_VERIFICATION_SEAL)


def _normalize_zip_member(name: str) -> str:
    require(isinstance(name, str) and name != "" and "\x00" not in name, "F2_ZIP_MEMBER_NAME")
    require("\\" not in name, "F2_ZIP_MEMBER_BACKSLASH")
    require(unicodedata.normalize("NFC", name) == name, "F2_ZIP_MEMBER_NOT_NFC")
    require(not name.startswith("/") and _WINDOWS_DRIVE.match(name) is None, "F2_ZIP_MEMBER_ABSOLUTE")
    stripped = name.removesuffix("/")
    require(stripped != "" and not stripped.endswith("/"), "F2_ZIP_MEMBER_NAME")
    parts = stripped.split("/")
    require(all(part not in {"", ".", ".."} for part in parts), "F2_ZIP_MEMBER_DIRECTORY_ESCAPE")
    parsed = PurePosixPath(stripped)
    require(not parsed.is_absolute(), "F2_ZIP_MEMBER_ABSOLUTE")
    normalized = parsed.as_posix()
    require(normalized == stripped, "F2_ZIP_MEMBER_NORMALIZATION")
    return normalized


def _validate_member_infos(
    infos: Sequence[zipfile.ZipInfo],
    budget: ArchiveBudget,
) -> tuple[tuple[ArchiveMember, ...], dict[str, zipfile.ZipInfo]]:
    require(len(infos) <= budget.max_members, "F2_ZIP_MEMBER_COUNT_BUDGET")
    seen: set[str] = set()
    records: list[ArchiveMember] = []
    by_name: dict[str, zipfile.ZipInfo] = {}
    total = 0
    for info in infos:
        # On Windows ``zipfile`` normalizes backslashes in ``filename``.  The
        # untouched central-directory spelling remains in ``orig_filename``
        # and must be checked or a forbidden name would become invisible.
        normalized = _normalize_zip_member(info.orig_filename)
        folded = normalized.casefold()
        require(folded not in seen, "F2_ZIP_MEMBER_CASEFOLD_DUPLICATE")
        seen.add(folded)
        require((info.flag_bits & 0x1) == 0, "F2_ZIP_MEMBER_ENCRYPTED")
        require(
            type(info.file_size) is int
            and type(info.compress_size) is int
            and info.file_size >= 0
            and info.compress_size >= 0,
            "F2_ZIP_MEMBER_SIZE_INVALID",
        )
        unix_mode = (info.external_attr >> 16) & 0xFFFF
        file_type = stat.S_IFMT(unix_mode)
        require(file_type not in {stat.S_IFLNK, stat.S_IFCHR, stat.S_IFBLK, stat.S_IFIFO, stat.S_IFSOCK}, "F2_ZIP_MEMBER_SPECIAL_FILE")
        if info.is_dir():
            require(info.file_size == 0, "F2_ZIP_DIRECTORY_NONEMPTY")
        else:
            require(info.file_size <= budget.max_member_uncompressed_bytes, "F2_ZIP_MEMBER_SIZE_BUDGET")
            total += info.file_size
            require(total <= budget.max_total_uncompressed_bytes, "F2_ZIP_TOTAL_SIZE_BUDGET")
            if info.file_size > 0:
                require(info.compress_size > 0, "F2_ZIP_COMPRESSION_BOMB")
                ratio = info.file_size / info.compress_size
                require(ratio <= float(budget.max_compression_ratio), "F2_ZIP_COMPRESSION_BOMB")
            by_name[normalized] = info
        records.append(
            ArchiveMember(
                name=normalized,
                compressed_bytes=info.compress_size,
                uncompressed_bytes=info.file_size,
                crc32=f"{info.CRC:08X}",
                is_directory=info.is_dir(),
            )
        )
    return tuple(records), by_name


class SafeZipArchive:
    """An already verified, preflighted ZIP held on the verified descriptor."""

    def __init__(
        self,
        verified: VerifiedArchive,
        raw_file: Any,
        archive: zipfile.ZipFile,
        members: tuple[ArchiveMember, ...],
        by_name: Mapping[str, zipfile.ZipInfo],
        budget: ArchiveBudget,
        observer: ReadObserver | None,
    ) -> None:
        self.verified = verified
        self.members = members
        self.budget = budget
        self._raw_file = raw_file
        self._archive = archive
        self._by_name = dict(by_name)
        self._observer = observer
        self._closed = False

    @property
    def file_names(self) -> frozenset[str]:
        return frozenset(self._by_name)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self._archive.close()
            finally:
                self._raw_file.close()

    def __enter__(self) -> Self:
        require(not self._closed, "F2_ZIP_ALREADY_CLOSED")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def require_member(self, name: str) -> ArchiveMember:
        normalized = _normalize_zip_member(name)
        require(not self._closed, "F2_ZIP_ALREADY_CLOSED")
        info = self._by_name.get(normalized)
        require(info is not None, "F2_ZIP_REQUIRED_MEMBER_MISSING", normalized)
        return ArchiveMember(
            name=normalized,
            compressed_bytes=info.compress_size,
            uncompressed_bytes=info.file_size,
            crc32=f"{info.CRC:08X}",
            is_directory=False,
        )

    def _require_read_authorized(
        self,
        name: str,
        phase: SourcePhase,
        purpose: str,
    ) -> None:
        binding = self.verified.binding
        role = {
            SourcePhase.CALIBRATION_SOURCE: "CALIBRATION",
            SourcePhase.SCORE_SOURCE: "SCORE",
        }.get(phase)
        if binding.kind == "RGBD_TRAINING_ARCHIVE":
            prefix = f"{binding.parent_id}/"
            allowed = (
                phase is SourcePhase.ROSTER_METADATA
                and purpose == "FREEZE_ASSOCIATED_ROSTER"
                and name == f"{prefix}associated.txt"
            ) or (
                phase is SourcePhase.RAW_SCORE_PREDICTION
                and (
                    (purpose == "RAW_SCORE_RGB" and name.startswith(f"{prefix}rgb/") and name.endswith(".png"))
                    or (purpose == "PINHOLE_INTRINSICS" and name == f"{prefix}calibration.txt")
                )
            ) or (
                role is not None
                and (
                    (purpose == f"{role}_SOURCE_DEPTH" and name.startswith(f"{prefix}depth/") and name.endswith(".png"))
                    or (purpose == f"{role}_CAMERA_TO_WORLD" and name == f"{prefix}groundtruth.txt")
                    or (purpose == "PINHOLE_INTRINSICS" and name == f"{prefix}calibration.txt")
                )
            )
        elif binding.kind == "IMU_ARCHIVE":
            prefix = f"{binding.parent_id}/"
            allowed = role is not None and (
                (purpose == f"{role}_IMU_GRAVITY" and name == f"{prefix}imu.txt")
                or (
                    purpose == f"{role}_SEQUENCE_CALIBRATION"
                    and name == f"{prefix}sequence_calibration.txt"
                )
            )
        else:
            allowed = (
                phase is SourcePhase.CALIBRATION_CONTROL
                and purpose == "DISCOVER_KALIBR_CAMERA_IMU_CONTROL"
            ) or (role is not None and purpose == f"{role}_CAMERA_IMU_CALIBRATION")
        require(allowed, "F2_SOURCE_MEMBER_PHASE_FIREWALL")

    def read_member_bytes(
        self,
        name: str,
        *,
        phase: SourcePhase,
        purpose: str,
        max_bytes: int,
    ) -> bytes:
        require(isinstance(phase, SourcePhase), "F2_SOURCE_PHASE_REQUIRED")
        require(isinstance(purpose, str) and purpose and "\n" not in purpose, "F2_SOURCE_READ_PURPOSE")
        require(type(max_bytes) is int and 0 < max_bytes <= self.budget.max_member_uncompressed_bytes, "F2_SOURCE_READ_BUDGET")
        normalized = _normalize_zip_member(name)
        self._require_read_authorized(normalized, phase, purpose)
        require(not self._closed, "F2_ZIP_ALREADY_CLOSED")
        info = self._by_name.get(normalized)
        require(info is not None, "F2_ZIP_REQUIRED_MEMBER_MISSING", normalized)
        require(info.file_size <= max_bytes, "F2_SOURCE_MEMBER_READ_BUDGET")
        try:
            with self._archive.open(info, "r") as stream:
                payload = stream.read(max_bytes + 1)
        except (OSError, RuntimeError, zipfile.BadZipFile) as error:
            raise ContractError("F2_ZIP_MEMBER_READ_FAILED", str(error)) from error
        require(len(payload) == info.file_size and len(payload) <= max_bytes, "F2_SOURCE_MEMBER_SIZE_DRIFT")
        if self._observer is not None:
            self._observer(
                ReadEvent(
                    parent_id=self.verified.binding.parent_id,
                    archive_kind=self.verified.binding.kind,
                    phase=phase,
                    purpose=purpose,
                    member=normalized,
                    bytes=len(payload),
                )
            )
        return payload


def preflight_archive(
    verified: VerifiedArchive,
    *,
    budget: ArchiveBudget,
    observer: ReadObserver | None = None,
) -> SafeZipArchive:
    """Reverify opaque bytes, then and only then enumerate safe ZIP metadata."""

    require(
        isinstance(verified, VerifiedArchive) and verified._seal is _VERIFICATION_SEAL,
        "F2_ARCHIVE_NOT_VERIFIED",
    )
    require(isinstance(budget, ArchiveBudget), "F2_ARCHIVE_BUDGET_TYPE")
    root, path = _checked_direct_child(verified.archive_root, verified.binding.filename)
    require(root == verified.archive_root and path == verified.path, "F2_ARCHIVE_PATH_DRIFT")
    raw_file = None
    archive = None
    try:
        raw_file = path.open("rb")
        descriptor_size = os.fstat(raw_file.fileno()).st_size
        require(descriptor_size == verified.binding.bytes, "F2_ARCHIVE_BYTES_MISMATCH")
        digest = hashlib.sha256()
        for chunk in iter(lambda: raw_file.read(1024 * 1024), b""):
            digest.update(chunk)
        require(digest.hexdigest().upper() == verified.binding.sha256, "F2_ARCHIVE_SHA_MISMATCH")
        raw_file.seek(0)
        archive = zipfile.ZipFile(raw_file, mode="r")
        infos = archive.infolist()
        members, by_name = _validate_member_infos(infos, budget)
        return SafeZipArchive(verified, raw_file, archive, members, by_name, budget, observer)
    except ContractError:
        if archive is not None:
            archive.close()
        if raw_file is not None:
            raw_file.close()
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        if archive is not None:
            archive.close()
        if raw_file is not None:
            raw_file.close()
        raise ContractError("F2_ARCHIVE_PREFLIGHT_FAILED", str(error)) from error


def validate_eth3d_member_contract(archive: SafeZipArchive) -> str | None:
    """Validate fixed per-parent roots and required control members."""

    binding = archive.verified.binding
    if binding.kind == "CAMERA_IMU_CALIBRATION_ARCHIVE":
        return None
    prefix = f"{binding.parent_id}/"
    require(
        archive.file_names and all(name.startswith(prefix) for name in archive.file_names),
        "F2_ZIP_TOP_LEVEL_IDENTITY",
    )
    if binding.kind == "RGBD_TRAINING_ARCHIVE":
        required = {
            f"{prefix}associated.txt",
            f"{prefix}rgb.txt",
            f"{prefix}depth.txt",
            f"{prefix}calibration.txt",
            f"{prefix}groundtruth.txt",
        }
        require(required.issubset(archive.file_names), "F2_RGBD_CONTROL_MEMBER_MISSING")
        require(
            any(name.startswith(f"{prefix}rgb/") and name.endswith(".png") for name in archive.file_names)
            and any(name.startswith(f"{prefix}depth/") and name.endswith(".png") for name in archive.file_names),
            "F2_RGBD_IMAGE_MEMBERS_MISSING",
        )
    else:
        require(
            {f"{prefix}imu.txt", f"{prefix}sequence_calibration.txt"}.issubset(archive.file_names),
            "F2_IMU_CONTROL_MEMBER_MISSING",
        )
    return prefix


def canonical_timestamp(token: str) -> str:
    """Canonical finite, nonnegative decimal spelling used by the rank token."""

    require(isinstance(token, str) and 0 < len(token) <= 128, "F2_ASSOCIATED_TIMESTAMP_TOKEN")
    require(_DECIMAL_TOKEN.fullmatch(token) is not None, "F2_ASSOCIATED_TIMESTAMP_TOKEN")
    try:
        value = Decimal(token)
    except InvalidOperation as error:
        raise ContractError("F2_ASSOCIATED_TIMESTAMP_TOKEN", str(error)) from error
    require(value.is_finite() and value >= 0, "F2_ASSOCIATED_TIMESTAMP_NONFINITE")
    if value == 0:
        return "0"
    canonical = format(value, "f")
    if "." in canonical:
        canonical = canonical.rstrip("0").rstrip(".")
    require(0 < len(canonical) <= 128, "F2_ASSOCIATED_TIMESTAMP_RANGE")
    return canonical


def camera_timestamp_nanoseconds_to_seconds(token: str) -> Decimal:
    """Convert a canonical camera-clock integer nanosecond token to seconds."""

    require(
        isinstance(token, str)
        and len(token) <= 32
        and _NANOSECOND_TOKEN.fullmatch(token) is not None,
        "F2_CAMERA_TIMESTAMP_NANOSECONDS",
    )
    result = Decimal(token) * NANOSECONDS_TO_SECONDS
    require(result.is_finite() and result >= 0, "F2_CAMERA_TIMESTAMP_SECONDS")
    return result


def _normalized_frame_path(value: str, expected_directory: str) -> str:
    normalized = _normalize_zip_member(value)
    parsed = PurePosixPath(normalized)
    require(
        len(parsed.parts) == 2
        and parsed.parts[0] == expected_directory
        and parsed.suffix == ".png"
        and parsed.name not in {"", ".png"},
        "F2_ASSOCIATED_FRAME_PATH",
    )
    return normalized


@dataclass(frozen=True)
class FrameIdentity:
    parent_id: str
    frame_id: str
    role: str
    rank_token: str
    rgb_timestamp: str
    rgb_relpath: str
    depth_timestamp: str
    depth_relpath: str
    rgb_member: str
    depth_member: str

    def as_dict(self) -> dict[str, str]:
        return {
            "parent_id": self.parent_id,
            "frame_id": self.frame_id,
            "role": self.role,
            "rank_token": self.rank_token,
            "rgb_timestamp": self.rgb_timestamp,
            "rgb_relpath": self.rgb_relpath,
            "depth_timestamp": self.depth_timestamp,
            "depth_relpath": self.depth_relpath,
            "rgb_member": self.rgb_member,
            "depth_member": self.depth_member,
        }


@dataclass(frozen=True)
class ParentRoster:
    protocol_id: str
    parent_id: str
    eligible_count: int
    calibration: tuple[FrameIdentity, ...]
    score: tuple[FrameIdentity, ...]
    content_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "parent_id": self.parent_id,
            "eligible_count": self.eligible_count,
            "calibration": [row.as_dict() for row in self.calibration],
            "score": [row.as_dict() for row in self.score],
            "content_sha256": self.content_sha256,
        }


def _parse_associated(
    raw: bytes,
    *,
    protocol_id: str,
    parent_id: str,
    prefix: str,
    archive: SafeZipArchive,
) -> list[FrameIdentity]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContractError("F2_ASSOCIATED_UTF8", str(error)) from error
    require("\x00" not in text, "F2_ASSOCIATED_NUL")
    require(protocol_id and parent_id and "\n" not in protocol_id and "\n" not in parent_id, "F2_ROSTER_IDENTITY")
    rows: list[FrameIdentity] = []
    rgb_seen: set[str] = set()
    depth_seen: set[str] = set()
    tokens: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        require(len(line) <= 4096, "F2_ASSOCIATED_LINE_TOO_LONG")
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        values = stripped.split()
        require(len(values) == 4, "F2_ASSOCIATED_COLUMN_COUNT", str(line_number))
        rgb_timestamp = canonical_timestamp(values[0])
        rgb_relpath = _normalized_frame_path(values[1], "rgb")
        depth_timestamp = canonical_timestamp(values[2])
        depth_relpath = _normalized_frame_path(values[3], "depth")
        camera_timestamp_nanoseconds_to_seconds(rgb_timestamp)
        camera_timestamp_nanoseconds_to_seconds(depth_timestamp)
        require(
            PurePosixPath(rgb_relpath).stem == PurePosixPath(depth_relpath).stem,
            "F2_ASSOCIATED_BASENAME_MISMATCH",
        )
        rgb_member = f"{prefix}{rgb_relpath}"
        depth_member = f"{prefix}{depth_relpath}"
        archive.require_member(rgb_member)
        archive.require_member(depth_member)
        require(rgb_member.casefold() not in rgb_seen, "F2_ASSOCIATED_DUPLICATE_RGB")
        require(depth_member.casefold() not in depth_seen, "F2_ASSOCIATED_DUPLICATE_DEPTH")
        rgb_seen.add(rgb_member.casefold())
        depth_seen.add(depth_member.casefold())
        material = (
            f"{protocol_id}\n{parent_id}\n{rgb_timestamp}\n{rgb_relpath}\n"
            f"{depth_timestamp}\n{depth_relpath}"
        ).encode()
        rank_token = hashlib.sha256(material).hexdigest().upper()
        require(rank_token not in tokens, "F2_ROSTER_RANK_TOKEN_DUPLICATE")
        tokens.add(rank_token)
        rows.append(
            FrameIdentity(
                parent_id=parent_id,
                frame_id=rank_token,
                role="ELIGIBLE",
                rank_token=rank_token,
                rgb_timestamp=rgb_timestamp,
                rgb_relpath=rgb_relpath,
                depth_timestamp=depth_timestamp,
                depth_relpath=depth_relpath,
                rgb_member=rgb_member,
                depth_member=depth_member,
            )
        )
    return rows


def freeze_parent_roster(
    archive: SafeZipArchive,
    *,
    protocol_id: str = PROTOCOL_ID,
    parent_id: str,
    calibration_count: int = 12,
    score_count: int = 12,
    minimum_eligible_count: int = 24,
) -> ParentRoster:
    """Freeze the F2 hash-ranked, disjoint calibration and score identities."""

    require(archive.verified.binding.kind == "RGBD_TRAINING_ARCHIVE", "F2_ROSTER_REQUIRES_RGBD_ARCHIVE")
    require(parent_id == archive.verified.binding.parent_id, "F2_ROSTER_PARENT_BINDING")
    require(
        calibration_count == 12 and score_count == 12 and minimum_eligible_count == 24,
        "F2_ROSTER_COUNTS_DRIFT",
    )
    prefix = validate_eth3d_member_contract(archive)
    require(prefix is not None, "F2_ROSTER_PREFIX")
    associated = archive.read_member_bytes(
        f"{prefix}associated.txt",
        phase=SourcePhase.ROSTER_METADATA,
        purpose="FREEZE_ASSOCIATED_ROSTER",
        max_bytes=archive.budget.max_metadata_bytes,
    )
    eligible = _parse_associated(
        associated,
        protocol_id=protocol_id,
        parent_id=parent_id,
        prefix=prefix,
        archive=archive,
    )
    require(len(eligible) >= minimum_eligible_count, "F2_ROSTER_INSUFFICIENT_ELIGIBLE")
    ordered = sorted(eligible, key=lambda row: (row.rank_token, row.rgb_timestamp, row.rgb_relpath))
    calibration = tuple(replace(row, role="CALIBRATION") for row in ordered[:calibration_count])
    score = tuple(
        replace(row, role="SCORE")
        for row in ordered[calibration_count : calibration_count + score_count]
    )
    require(len(calibration) == calibration_count and len(score) == score_count, "F2_ROSTER_COUNT")
    require(
        {row.frame_id for row in calibration}.isdisjoint({row.frame_id for row in score}),
        "F2_ROSTER_ROLE_OVERLAP",
    )
    payload = {
        "protocol_id": protocol_id,
        "parent_id": parent_id,
        "eligible_count": len(eligible),
        "calibration": [row.as_dict() for row in calibration],
        "score": [row.as_dict() for row in score],
    }
    return ParentRoster(
        protocol_id=protocol_id,
        parent_id=parent_id,
        eligible_count=len(eligible),
        calibration=calibration,
        score=score,
        content_sha256=canonical_sha256(payload),
    )


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _decode_rgb(raw: bytes) -> np.ndarray:
    encoded = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    require(bgr is not None and bgr.dtype == np.uint8 and bgr.ndim == 3 and bgr.shape[2] == 3, "F2_RGB_DECODE")
    return _readonly(np.ascontiguousarray(bgr[:, :, ::-1]))


def _decode_depth(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    encoded = np.frombuffer(raw, dtype=np.uint8)
    depth_u16 = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    require(depth_u16 is not None and depth_u16.dtype == np.uint16 and depth_u16.ndim == 2, "F2_DEPTH_DECODE")
    depth_m = depth_u16.astype(np.float32) / np.float32(5000.0)
    known = (depth_u16 > 0) & np.isfinite(depth_m) & (depth_m >= 0.25) & (depth_m <= 6.0)
    depth_m[~known] = np.nan
    return _readonly(depth_m), _readonly(np.asarray(known, dtype=np.bool_))


def _parse_intrinsics(raw: bytes) -> np.ndarray:
    try:
        values = [float(value) for value in raw.decode("utf-8").split()]
    except (UnicodeDecodeError, ValueError) as error:
        raise ContractError("F2_CALIBRATION_FORMAT", str(error)) from error
    require(
        len(values) == 4
        and all(math.isfinite(value) for value in values)
        and values[0] > 0.0
        and values[1] > 0.0,
        "F2_CALIBRATION_FORMAT",
    )
    fx, fy, cx, cy = values
    matrix = np.asarray(((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)), dtype=np.float64)
    return _readonly(matrix)


@dataclass(frozen=True)
class PredictionInput:
    parent_id: str
    frame_id: str
    rgb_timestamp: str
    rgb_relpath: str
    rgb_hwc_u8: np.ndarray
    K: np.ndarray


@dataclass(frozen=True)
class SourceArrays:
    parent_id: str
    frame_id: str
    role: str
    depth_timestamp: str
    depth_relpath: str
    depth_m_hw: np.ndarray
    depth_known_hw: np.ndarray
    K: np.ndarray


@dataclass(frozen=True)
class CalibrationMemberBinding:
    """Execution-lock binding for the otherwise unfrozen calibration layout.

    The future execution lock must bind the exact Kalibr YAML member, camera
    node, matrix key, encoding, transform direction, and IMU convention.  The
    parser accepts no alternate inline-text or generic YAML shape.
    """

    member: str
    camera_node_key: str
    camera_from_imu_key: str
    calibration_encoding: str
    camera_from_imu_transform_direction: str
    mocap_time_scale_key: str
    mocap_time_anchor_seconds_key: str
    mocap_time_offset_seconds_key: str
    camera_timestamp_to_seconds: str
    imu_timestamp_to_seconds: str
    imu_clock_domain: str
    groundtruth_timestamp_unit: str
    imu_delimiter_and_column_order: str
    imu_axis_and_frame_mapping: str
    accelerometer_specific_force_sign: str
    maximum_pose_bracket_seconds: Decimal
    imu_half_window_seconds: Decimal
    minimum_imu_samples: int

    def __post_init__(self) -> None:
        normalized = _normalize_zip_member(self.member)
        require(normalized == self.member and not self.member.endswith("/"), "F2_IMU_CALIBRATION_MEMBER_BINDING")
        require(
            isinstance(self.camera_node_key, str)
            and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", self.camera_node_key) is not None,
            "F2_IMU_CALIBRATION_CAMERA_NODE_BINDING",
        )
        require(
            isinstance(self.camera_from_imu_key, str)
            and self.camera_from_imu_key == "T_cam_imu",
            "F2_IMU_CALIBRATION_KEY_BINDING",
        )
        require(
            self.calibration_encoding == CALIBRATION_ENCODING
            and self.camera_from_imu_transform_direction == CAMERA_FROM_IMU_DIRECTION,
            "F2_IMU_CALIBRATION_FORMAT_BINDING",
        )
        time_keys = (
            self.mocap_time_scale_key,
            self.mocap_time_anchor_seconds_key,
            self.mocap_time_offset_seconds_key,
        )
        require(
            time_keys == (
                "mocap_timescaling_camera",
                "timescaling_anchor",
                "mocap_timeoffset_camera",
            ),
            "F2_MOCAP_TIME_KEY_BINDING",
        )
        require(
            self.camera_timestamp_to_seconds == "INTEGER_NANOSECONDS_TIMES_1E_MINUS_9"
            and self.imu_timestamp_to_seconds == "INTEGER_NANOSECONDS_TIMES_1E_MINUS_9"
            and self.imu_clock_domain == "CAMERA_CLOCK_NO_MOCAP_TRANSFORM"
            and self.groundtruth_timestamp_unit == "SECONDS",
            "F2_TIME_DOMAIN_BINDING",
        )
        require(
            self.imu_delimiter_and_column_order == IMU_COLUMN_CONTRACT
            and self.imu_axis_and_frame_mapping == IMU_FRAME_CONTRACT
            and self.accelerometer_specific_force_sign == ACCELEROMETER_SIGN_CONTRACT,
            "F2_IMU_CONVENTION_BINDING",
        )
        require(
            isinstance(self.maximum_pose_bracket_seconds, Decimal)
            and self.maximum_pose_bracket_seconds.is_finite()
            and self.maximum_pose_bracket_seconds > 0,
            "F2_POSE_BRACKET_BINDING",
        )
        require(
            isinstance(self.imu_half_window_seconds, Decimal)
            and self.imu_half_window_seconds.is_finite()
            and self.imu_half_window_seconds > 0,
            "F2_IMU_WINDOW_BINDING",
        )
        require(
            type(self.minimum_imu_samples) is int and self.minimum_imu_samples >= 5,
            "F2_IMU_MINIMUM_SAMPLE_BINDING",
        )


@dataclass(frozen=True)
class PoseGravity:
    parent_id: str
    frame_id: str
    role: str
    camera_timestamp_nanoseconds: str
    camera_timestamp_seconds: str
    mocap_timestamp_seconds: str
    camera_to_world: np.ndarray
    gravity_up_camera_xyz: np.ndarray
    imu_sample_count: int


@dataclass(frozen=True)
class MocapTimeTransform:
    scale: Decimal
    anchor_seconds: Decimal
    offset_seconds: Decimal

    def camera_seconds_to_mocap_seconds(self, camera_seconds: Decimal) -> Decimal:
        result = (
            self.scale * (camera_seconds - self.anchor_seconds)
            + self.anchor_seconds
            + self.offset_seconds
        )
        require(result.is_finite() and result >= 0, "F2_MOCAP_TIMESTAMP_SECONDS")
        return result


@dataclass(frozen=True)
class _PoseSample:
    timestamp: Decimal
    translation: np.ndarray
    quaternion_xyzw: np.ndarray


@dataclass(frozen=True)
class _ImuSample:
    timestamp: Decimal
    acceleration_imu_xyz: np.ndarray


def _text_rows(raw: bytes, code: str) -> list[list[str]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContractError(f"{code}_UTF8", str(error)) from error
    require("\x00" not in text, f"{code}_NUL")
    result: list[list[str]] = []
    for line in text.splitlines():
        require(len(line) <= 4096, f"{code}_LINE_TOO_LONG")
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            result.append(stripped.split())
    return result


def _finite_floats(values: Sequence[str], count: int, code: str) -> np.ndarray:
    require(len(values) == count, code)
    try:
        result = np.asarray([float(value) for value in values], dtype=np.float64)
    except ValueError as error:
        raise ContractError(code, str(error)) from error
    require(bool(np.all(np.isfinite(result))), code)
    return result


def _normalize_quaternion_xyzw(value: np.ndarray, code: str) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    require(math.isfinite(norm) and 0.999 <= norm <= 1.001, code)
    return value / norm


def _parse_groundtruth(raw: bytes) -> tuple[_PoseSample, ...]:
    samples: list[_PoseSample] = []
    for values in _text_rows(raw, "F2_GROUNDTRUTH"):
        require(len(values) == 8, "F2_GROUNDTRUTH_COLUMNS")
        timestamp = Decimal(canonical_timestamp(values[0]))
        numeric = _finite_floats(values[1:], 7, "F2_GROUNDTRUTH_NUMERIC")
        samples.append(
            _PoseSample(
                timestamp=timestamp,
                translation=numeric[:3],
                quaternion_xyzw=_normalize_quaternion_xyzw(numeric[3:], "F2_GROUNDTRUTH_QUATERNION"),
            )
        )
    require(len(samples) >= 2, "F2_GROUNDTRUTH_TOO_SHORT")
    require(
        all(left.timestamp < right.timestamp for left, right in pairwise(samples)),
        "F2_GROUNDTRUTH_NOT_STRICTLY_MONOTONIC",
    )
    return tuple(samples)


def _parse_imu(raw: bytes) -> tuple[_ImuSample, ...]:
    samples: list[_ImuSample] = []
    for values in _text_rows(raw, "F2_IMU"):
        # ETH3D-style IMU rows are timestamp, angular velocity xyz, then
        # accelerometer xyz.  The parser accepts no alternative column layout.
        require(len(values) == 7, "F2_IMU_COLUMNS")
        timestamp = camera_timestamp_nanoseconds_to_seconds(canonical_timestamp(values[0]))
        numeric = _finite_floats(values[1:], 6, "F2_IMU_NUMERIC")
        samples.append(_ImuSample(timestamp=timestamp, acceleration_imu_xyz=numeric[3:6]))
    require(len(samples) >= 2, "F2_IMU_TOO_SHORT")
    require(
        all(left.timestamp < right.timestamp for left, right in pairwise(samples)),
        "F2_IMU_NOT_STRICTLY_MONOTONIC",
    )
    return tuple(samples)


def _parse_camera_from_imu(raw: bytes, binding: CalibrationMemberBinding) -> np.ndarray:
    return parse_kalibr_camera_from_imu(
        raw,
        camera_node_key=binding.camera_node_key,
        matrix_key=binding.camera_from_imu_key,
    )


def _parse_mocap_time_transform(
    raw: bytes,
    binding: CalibrationMemberBinding,
) -> MocapTimeTransform:
    expected = {
        binding.mocap_time_scale_key: "scale",
        binding.mocap_time_anchor_seconds_key: "anchor_seconds",
        binding.mocap_time_offset_seconds_key: "offset_seconds",
    }
    matches: dict[str, list[Decimal]] = {name: [] for name in expected.values()}
    for values in _text_rows(raw, "F2_SEQUENCE_CALIBRATION"):
        target = expected.get(values[0])
        if target is not None:
            require(
                len(values) == 2 and _DECIMAL_TOKEN.fullmatch(values[1]) is not None,
                "F2_MOCAP_TIME_VALUE_FORMAT",
            )
            try:
                matches[target].append(Decimal(values[1]))
            except InvalidOperation as error:
                raise ContractError("F2_MOCAP_TIME_VALUE_FORMAT", str(error)) from error
    require(
        all(len(values) == 1 for values in matches.values()),
        "F2_MOCAP_TIME_KEY_AMBIGUOUS_OR_MISSING",
    )
    scale = matches["scale"][0]
    anchor = matches["anchor_seconds"][0]
    offset = matches["offset_seconds"][0]
    require(
        scale.is_finite()
        and scale > 0
        and anchor.is_finite()
        and anchor >= 0
        and offset.is_finite(),
        "F2_MOCAP_TIME_TRANSFORM_INVALID",
    )
    return MocapTimeTransform(scale=scale, anchor_seconds=anchor, offset_seconds=offset)


def _rotation_from_quaternion_xyzw(value: np.ndarray) -> np.ndarray:
    x, y, z, w = value
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _slerp(left: np.ndarray, right: np.ndarray, fraction: float) -> np.ndarray:
    dot = float(np.dot(left, right))
    if dot < 0.0:
        right = -right
        dot = -dot
    dot = min(max(dot, -1.0), 1.0)
    if dot > 0.9995:
        return _normalize_quaternion_xyzw(left + fraction * (right - left), "F2_POSE_INTERPOLATION_QUATERNION")
    angle = math.acos(dot)
    denominator = math.sin(angle)
    require(abs(denominator) > 1e-12, "F2_POSE_INTERPOLATION_QUATERNION")
    return _normalize_quaternion_xyzw(
        math.sin((1.0 - fraction) * angle) / denominator * left
        + math.sin(fraction * angle) / denominator * right,
        "F2_POSE_INTERPOLATION_QUATERNION",
    )


def _camera_to_world(
    samples: Sequence[_PoseSample],
    timestamp: Decimal,
    maximum_pose_bracket_seconds: Decimal,
) -> np.ndarray:
    exact = [sample for sample in samples if sample.timestamp == timestamp]
    if exact:
        require(len(exact) == 1, "F2_POSE_TIMESTAMP_AMBIGUOUS")
        translation = exact[0].translation
        quaternion = exact[0].quaternion_xyzw
    else:
        bracket = [
            (left, right)
            for left, right in pairwise(samples)
            if left.timestamp < timestamp < right.timestamp
        ]
        require(len(bracket) == 1, "F2_POSE_NOT_UNIQUELY_BRACKETED")
        left, right = bracket[0]
        denominator = right.timestamp - left.timestamp
        require(
            denominator > 0 and denominator <= maximum_pose_bracket_seconds,
            "F2_POSE_INTERPOLATION_DENOMINATOR",
        )
        fraction = float((timestamp - left.timestamp) / denominator)
        require(0.0 < fraction < 1.0 and math.isfinite(fraction), "F2_POSE_INTERPOLATION_FRACTION")
        translation = left.translation + fraction * (right.translation - left.translation)
        quaternion = _slerp(left.quaternion_xyzw, right.quaternion_xyzw, fraction)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _rotation_from_quaternion_xyzw(quaternion)
    matrix[:3, 3] = translation
    return _readonly(matrix)


def _gravity_up_camera(
    samples: Sequence[_ImuSample],
    timestamp: Decimal,
    camera_from_imu: np.ndarray,
    *,
    half_window_seconds: Decimal,
    minimum_samples: int,
) -> tuple[np.ndarray, int]:
    require(half_window_seconds > 0 and type(minimum_samples) is int and minimum_samples >= 5, "F2_IMU_WINDOW_CONTRACT")
    selected = [sample.acceleration_imu_xyz for sample in samples if abs(sample.timestamp - timestamp) <= half_window_seconds]
    require(len(selected) >= minimum_samples, "F2_IMU_INSUFFICIENT_ASSOCIATED_SAMPLES")
    acceleration = np.median(np.stack(selected, axis=0), axis=0)
    norm = float(np.linalg.norm(acceleration))
    require(math.isfinite(norm) and norm > 1e-9, "F2_IMU_GRAVITY_DENOMINATOR")
    # Accelerometers at rest report specific force opposite gravity, hence the
    # normalized reading is the up direction.  Dynamic cancellation is rejected
    # by the nonzero denominator; higher-level session consistency remains a
    # source gate rather than an alternative sign convention here.
    up_imu = acceleration / norm
    up_camera = camera_from_imu[:3, :3] @ up_imu
    camera_norm = float(np.linalg.norm(up_camera))
    require(math.isfinite(camera_norm) and camera_norm > 1e-9, "F2_IMU_CAMERA_GRAVITY_DENOMINATOR")
    return _readonly(np.asarray(up_camera / camera_norm, dtype=np.float64)), len(selected)


class Eth3dParentSource:
    """Role-aware, read-only RGB-D surface for one preflighted parent."""

    def __init__(
        self,
        archive: SafeZipArchive,
        *,
        parent_id: str,
        protocol_id: str = PROTOCOL_ID,
    ) -> None:
        self.archive = archive
        self.roster = freeze_parent_roster(
            archive,
            protocol_id=protocol_id,
            parent_id=parent_id,
        )
        self._calibration = {row.frame_id: row for row in self.roster.calibration}
        self._score = {row.frame_id: row for row in self.roster.score}
        self._prefix = f"{parent_id}/"
        self._K: np.ndarray | None = None

    def _intrinsics(self, phase: SourcePhase) -> np.ndarray:
        if self._K is None:
            raw = self.archive.read_member_bytes(
                f"{self._prefix}calibration.txt",
                phase=phase,
                purpose="PINHOLE_INTRINSICS",
                max_bytes=self.archive.budget.max_metadata_bytes,
            )
            self._K = _parse_intrinsics(raw)
        return self._K

    def read_prediction_input(self, frame_id: str, *, phase: SourcePhase) -> PredictionInput:
        """Read score-role RGB+K only; no source outcome is reachable here."""

        require(phase is SourcePhase.RAW_SCORE_PREDICTION, "F2_PREDICTION_PHASE_FORBIDDEN")
        require(isinstance(frame_id, str), "F2_FRAME_ID_TYPE")
        frame = self._score.get(frame_id)
        require(frame is not None, "F2_PREDICTION_FRAME_NOT_SCORE_ROLE")
        raw = self.archive.read_member_bytes(
            frame.rgb_member,
            phase=phase,
            purpose="RAW_SCORE_RGB",
            max_bytes=self.archive.budget.max_member_uncompressed_bytes,
        )
        return PredictionInput(
            parent_id=frame.parent_id,
            frame_id=frame.frame_id,
            rgb_timestamp=frame.rgb_timestamp,
            rgb_relpath=frame.rgb_relpath,
            rgb_hwc_u8=_decode_rgb(raw),
            K=self._intrinsics(phase),
        )

    def read_source_arrays(self, frame_id: str, *, phase: SourcePhase) -> SourceArrays:
        """Read depth only for the role exactly corresponding to ``phase``."""

        require(isinstance(frame_id, str), "F2_FRAME_ID_TYPE")
        if phase is SourcePhase.CALIBRATION_SOURCE:
            frame = self._calibration.get(frame_id)
            role = "CALIBRATION"
        elif phase is SourcePhase.SCORE_SOURCE:
            frame = self._score.get(frame_id)
            role = "SCORE"
        else:
            raise ContractError("F2_SOURCE_ARRAY_PHASE_FORBIDDEN")
        require(frame is not None, "F2_SOURCE_FRAME_ROLE_MISMATCH")
        raw = self.archive.read_member_bytes(
            frame.depth_member,
            phase=phase,
            purpose=f"{role}_SOURCE_DEPTH",
            max_bytes=self.archive.budget.max_member_uncompressed_bytes,
        )
        depth_m, known = _decode_depth(raw)
        return SourceArrays(
            parent_id=frame.parent_id,
            frame_id=frame.frame_id,
            role=role,
            depth_timestamp=frame.depth_timestamp,
            depth_relpath=frame.depth_relpath,
            depth_m_hw=depth_m,
            depth_known_hw=known,
            K=self._intrinsics(phase),
        )

    def read_pose_and_gravity(
        self,
        frame_id: str,
        *,
        phase: SourcePhase,
        imu_archive: SafeZipArchive,
        calibration_archive: SafeZipArchive,
        calibration_binding: CalibrationMemberBinding,
    ) -> PoseGravity:
        """Read role-matched pose and IMU-derived up direction fail-closed.

        The calibration member and transform-direction key must be supplied by
        the future execution lock.  No default member/key is guessed here.
        """

        require(isinstance(frame_id, str), "F2_FRAME_ID_TYPE")
        if phase is SourcePhase.CALIBRATION_SOURCE:
            frame = self._calibration.get(frame_id)
            role = "CALIBRATION"
        elif phase is SourcePhase.SCORE_SOURCE:
            frame = self._score.get(frame_id)
            role = "SCORE"
        else:
            raise ContractError("F2_POSE_GRAVITY_PHASE_FORBIDDEN")
        require(frame is not None, "F2_SOURCE_FRAME_ROLE_MISMATCH")
        require(
            imu_archive.verified.binding.kind == "IMU_ARCHIVE"
            and imu_archive.verified.binding.parent_id == self.roster.parent_id,
            "F2_IMU_ARCHIVE_PARENT_BINDING",
        )
        require(
            calibration_archive.verified.binding.kind == "CAMERA_IMU_CALIBRATION_ARCHIVE",
            "F2_IMU_CALIBRATION_ARCHIVE_BINDING",
        )
        imu_prefix = validate_eth3d_member_contract(imu_archive)
        require(imu_prefix == self._prefix, "F2_IMU_ARCHIVE_PARENT_BINDING")
        require(isinstance(calibration_binding, CalibrationMemberBinding), "F2_IMU_CALIBRATION_BINDING_TYPE")
        groundtruth_raw = self.archive.read_member_bytes(
            f"{self._prefix}groundtruth.txt",
            phase=phase,
            purpose=f"{role}_CAMERA_TO_WORLD",
            max_bytes=self.archive.budget.max_metadata_bytes,
        )
        imu_raw = imu_archive.read_member_bytes(
            f"{self._prefix}imu.txt",
            phase=phase,
            purpose=f"{role}_IMU_GRAVITY",
            max_bytes=imu_archive.budget.max_metadata_bytes,
        )
        # This control is opened and bound explicitly even though its unknown
        # official semantics are not interpreted by the adapter.  Empty or
        # ambiguous content cannot silently disappear from the source chain.
        sequence_calibration = imu_archive.read_member_bytes(
            f"{self._prefix}sequence_calibration.txt",
            phase=phase,
            purpose=f"{role}_SEQUENCE_CALIBRATION",
            max_bytes=imu_archive.budget.max_metadata_bytes,
        )
        mocap_transform = _parse_mocap_time_transform(
            sequence_calibration,
            calibration_binding,
        )
        calibration_raw = calibration_archive.read_member_bytes(
            calibration_binding.member,
            phase=phase,
            purpose=f"{role}_CAMERA_IMU_CALIBRATION",
            max_bytes=calibration_archive.budget.max_metadata_bytes,
        )
        camera_timestamp_seconds = camera_timestamp_nanoseconds_to_seconds(frame.rgb_timestamp)
        mocap_timestamp_seconds = mocap_transform.camera_seconds_to_mocap_seconds(
            camera_timestamp_seconds
        )
        camera_to_world = _camera_to_world(
            _parse_groundtruth(groundtruth_raw),
            mocap_timestamp_seconds,
            calibration_binding.maximum_pose_bracket_seconds,
        )
        camera_from_imu = _parse_camera_from_imu(calibration_raw, calibration_binding)
        gravity_up, sample_count = _gravity_up_camera(
            _parse_imu(imu_raw),
            camera_timestamp_seconds,
            camera_from_imu,
            half_window_seconds=calibration_binding.imu_half_window_seconds,
            minimum_samples=calibration_binding.minimum_imu_samples,
        )
        return PoseGravity(
            parent_id=frame.parent_id,
            frame_id=frame.frame_id,
            role=role,
            camera_timestamp_nanoseconds=frame.rgb_timestamp,
            camera_timestamp_seconds=canonical_timestamp(str(camera_timestamp_seconds)),
            mocap_timestamp_seconds=canonical_timestamp(str(mocap_timestamp_seconds)),
            camera_to_world=camera_to_world,
            gravity_up_camera_xyz=gravity_up,
            imu_sample_count=sample_count,
        )


__all__ = [
    "ArchiveBinding",
    "ArchiveBudget",
    "ArchiveMember",
    "CalibrationMemberBinding",
    "Eth3dParentSource",
    "FrameIdentity",
    "MocapTimeTransform",
    "ParentRoster",
    "PoseGravity",
    "PredictionInput",
    "ReadEvent",
    "SafeZipArchive",
    "SourceArrays",
    "SourcePhase",
    "VerifiedArchive",
    "camera_timestamp_nanoseconds_to_seconds",
    "canonical_timestamp",
    "freeze_parent_roster",
    "preflight_archive",
    "validate_eth3d_member_contract",
    "verify_archive_binding",
]
