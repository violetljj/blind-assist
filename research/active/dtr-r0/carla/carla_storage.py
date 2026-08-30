"""Bounded CARLA experiment storage accounting and exact-file deduplication.

The storage guard counts each NTFS inode once, so hard-linked model/evaluator
views do not consume quota twice.  The deduplicator is deliberately narrower:
it only merges regular files that have the same size, SHA-256, modification
time, and permission mode.  Paths and file contents remain unchanged.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Sequence


GIB = 1024**3
FILE_ATTRIBUTE_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
DEFAULT_POLICY = Path(__file__).with_name("carla_storage_policy.json")
MAINTENANCE_LOCK_NAME = ".carla-storage-dedupe.lock"
COORDINATION_LOCK_NAME = ".carla-storage-coordinate.lock"
LEASE_DIRECTORY_NAME = ".carla-storage-leases"


class StorageError(RuntimeError):
    """Raised when a storage operation cannot preserve its safety contract."""


def _os_path(path: str | os.PathLike[str]) -> str:
    """Return an absolute path that remains usable beyond MAX_PATH on Windows."""

    value = os.path.abspath(os.fspath(path))
    if os.name != "nt" or value.startswith("\\\\?\\"):
        return value
    if value.startswith("\\\\"):
        return "\\\\?\\UNC\\" + value[2:]
    return "\\\\?\\" + value


def _path_exists(path: Path) -> bool:
    return os.path.exists(_os_path(path))


def _regular_file_exists(path: Path) -> bool:
    try:
        return stat.S_ISREG(os.stat(_os_path(path), follow_symlinks=False).st_mode)
    except FileNotFoundError:
        return False


def _named_streams(path: Path) -> tuple[tuple[str, int], ...]:
    """Return NTFS named streams; the unnamed default data stream is omitted."""

    if os.name != "nt":
        return ()

    find_first, find_next, find_close, data_type = _stream_api()
    data = data_type()
    handle = find_first(_os_path(path), 0, ctypes.byref(data), 0)
    invalid_handle = ctypes.c_void_p(-1).value
    if handle == invalid_handle:
        error = ctypes.get_last_error()
        if error in (2, 38):
            return ()
        raise StorageError(f"cannot enumerate named streams for {path}: win32={error}")
    values: list[tuple[str, int]] = []
    try:
        while True:
            name = str(data.stream_name)
            if name and name != "::$DATA":
                values.append((name, int(data.stream_size)))
            if not find_next(handle, ctypes.byref(data)):
                error = ctypes.get_last_error()
                if error == 38:
                    break
                raise StorageError(
                    f"cannot continue named-stream enumeration for {path}: win32={error}"
                )
    finally:
        find_close(handle)
    return tuple(sorted(values))


@lru_cache(maxsize=1)
def _stream_api() -> tuple[Any, Any, Any, type[ctypes.Structure]]:
    class Win32FindStreamData(ctypes.Structure):
        _fields_ = [
            ("stream_size", ctypes.c_longlong),
            ("stream_name", ctypes.c_wchar * 296),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    find_first = kernel32.FindFirstStreamW
    find_first.argtypes = [ctypes.c_wchar_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
    find_first.restype = ctypes.c_void_p
    find_next = kernel32.FindNextStreamW
    find_next.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    find_next.restype = ctypes.c_int
    find_close = kernel32.FindClose
    find_close.argtypes = [ctypes.c_void_p]
    find_close.restype = ctypes.c_int
    return find_first, find_next, find_close, Win32FindStreamData


def _security_descriptor_sha256(path: Path) -> str:
    """Hash owner, group, and DACL metadata that a hard link would share."""

    if os.name != "nt":
        return "NON_WINDOWS"
    get_security = _security_api()
    security_information = 0x00000001 | 0x00000002 | 0x00000004
    needed = ctypes.c_uint(0)
    get_security(_os_path(path), security_information, None, 0, ctypes.byref(needed))
    error = ctypes.get_last_error()
    if needed.value == 0 or error not in (0, 122):
        raise StorageError(f"cannot size security descriptor for {path}: win32={error}")
    buffer = ctypes.create_string_buffer(needed.value)
    if not get_security(
        _os_path(path),
        security_information,
        buffer,
        needed.value,
        ctypes.byref(needed),
    ):
        raise StorageError(
            f"cannot read security descriptor for {path}: win32={ctypes.get_last_error()}"
        )
    return hashlib.sha256(buffer.raw[: needed.value]).hexdigest().upper()


@lru_cache(maxsize=1)
def _security_api() -> Any:
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    get_security = advapi32.GetFileSecurityW
    get_security.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_uint),
    ]
    get_security.restype = ctypes.c_int
    return get_security


def _safe_relative_path(value: str, label: str) -> PurePosixPath:
    normalized = value.replace("\\", "/")
    relative = PurePosixPath(normalized)
    if (
        not normalized
        or relative.is_absolute()
        or any(part in ("", ".", "..") for part in relative.parts)
        or re.match(r"^[A-Za-z]:", normalized)
    ):
        raise StorageError(f"unsafe {label} path: {value!r}")
    return relative


def _contained_plan_path(root: Path, value: str, label: str) -> Path:
    relative = _safe_relative_path(value, label)
    candidate = root.joinpath(*relative.parts)
    if candidate == root or root not in candidate.parents:
        raise StorageError(f"{label} escapes storage root: {value!r}")
    return candidate


@dataclass(frozen=True)
class FileRecord:
    path: Path
    size: int
    identity: tuple[int, int] | tuple[str, str]
    link_count: int
    mtime_ns: int
    mode: int
    file_attributes: int
    security_sha256: str
    sealed_sha256: str | None = None


@dataclass(frozen=True)
class SealIndex:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    result_path: Path
    result_sha256: str
    members: dict[str, tuple[int, str]]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def sha256_file(path: Path, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(_os_path(path), "rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest().upper()


def _write_json_atomic(path: Path, value: object) -> None:
    os.makedirs(_os_path(path.parent), exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(_os_path(temporary), "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(_os_path(temporary), _os_path(path))
    finally:
        if _path_exists(temporary):
            os.unlink(_os_path(temporary))


def load_policy(path: Path = DEFAULT_POLICY) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise StorageError(f"storage policy must be a JSON object: {path}")
    expected_schema = "blindassist-carla-storage-policy-v1"
    if value.get("schema_version") != expected_schema:
        raise StorageError(f"unexpected storage policy schema: {value.get('schema_version')}")
    for key in (
        "maximum_experiment_unique_bytes",
        "minimum_volume_free_bytes",
        "default_run_reservation_bytes",
        "multimap_run_reservation_bytes",
        "dedupe_minimum_age_seconds",
    ):
        if not isinstance(value.get(key), int) or int(value[key]) < 0:
            raise StorageError(f"storage policy {key} must be a non-negative integer")
    extensions = value.get("dedupe_extensions")
    if not isinstance(extensions, list) or not extensions:
        raise StorageError("storage policy dedupe_extensions must be a non-empty list")
    if value.get("dedupe_require_sealed_ancestor") is not True:
        raise StorageError("dedupe_require_sealed_ancestor must remain enabled")
    if value.get("automatic_payload_deletion") is not False:
        raise StorageError("automatic payload deletion must remain disabled")
    if value.get("overflow_action") != "REFUSE_NEW_RUN":
        raise StorageError("overflow_action must be REFUSE_NEW_RUN")
    value["policy_path"] = str(path.resolve())
    value["policy_sha256"] = sha256_file(path)
    return value


def _is_reparse(stat_result: os.stat_result) -> bool:
    return bool(getattr(stat_result, "st_file_attributes", 0) & FILE_ATTRIBUTE_REPARSE_POINT)


def _file_identity(path: Path, stat_result: os.stat_result) -> tuple[int, int] | tuple[str, str]:
    inode = int(getattr(stat_result, "st_ino", 0))
    device = int(getattr(stat_result, "st_dev", 0))
    if inode:
        return (device, inode)
    return ("path", os.path.normcase(os.path.abspath(os.fspath(path))))


def iter_regular_files(root: Path) -> Iterator[tuple[Path, os.stat_result]]:
    physical_root = root.resolve(strict=True)
    if not physical_root.is_dir():
        raise StorageError(f"storage root is not a directory: {physical_root}")
    stack = [physical_root]
    while stack:
        current = stack.pop()
        with os.scandir(_os_path(current)) as entries:
            for entry in entries:
                path = current / entry.name
                try:
                    # DirEntry.stat() can expose zeroed inode/link fields on
                    # Windows.  os.stat() opens the file and returns the NTFS
                    # file identity needed for hardlink-aware accounting.
                    entry_stat = os.stat(_os_path(path), follow_symlinks=False)
                except FileNotFoundError:
                    raise StorageError(f"file changed during storage scan: {path}")
                if _is_reparse(entry_stat) or entry.is_symlink():
                    continue
                if stat.S_ISDIR(entry_stat.st_mode):
                    stack.append(path)
                elif stat.S_ISREG(entry_stat.st_mode):
                    yield path, entry_stat


def storage_accounting(root: Path) -> dict[str, Any]:
    physical_root = root.resolve(strict=True)
    identities: dict[tuple[int, int] | tuple[str, str], int] = {}
    streams_by_identity: dict[
        tuple[int, int] | tuple[str, str], tuple[tuple[str, int], ...]
    ] = {}
    logical_bytes = 0
    file_count = 0
    hardlink_entries = 0
    latest_mtime_ns = 0
    named_stream_file_count = 0
    named_stream_logical_bytes = 0
    for path, path_stat in iter_regular_files(physical_root):
        relative = path.relative_to(physical_root)
        if (
            relative.parts
            and relative.parts[0] == LEASE_DIRECTORY_NAME
        ) or (
            path.parent == physical_root
            and path.name in {MAINTENANCE_LOCK_NAME, COORDINATION_LOCK_NAME}
        ):
            continue
        file_count += 1
        identity = _file_identity(path, path_stat)
        streams = streams_by_identity.get(identity)
        if streams is None:
            streams = _named_streams(path)
            streams_by_identity[identity] = streams
        stream_bytes = sum(size for _, size in streams)
        if streams:
            named_stream_file_count += 1
            named_stream_logical_bytes += stream_bytes
        file_bytes = int(path_stat.st_size) + stream_bytes
        logical_bytes += file_bytes
        latest_mtime_ns = max(latest_mtime_ns, int(path_stat.st_mtime_ns))
        identities.setdefault(identity, file_bytes)
        if int(getattr(path_stat, "st_nlink", 1)) > 1:
            hardlink_entries += 1
    unique_bytes = sum(identities.values())
    disk = shutil.disk_usage(physical_root)
    return {
        "schema_version": "blindassist-carla-storage-accounting-v1",
        "root": str(physical_root),
        "file_count": file_count,
        "unique_file_count": len(identities),
        "logical_bytes": logical_bytes,
        "unique_bytes": unique_bytes,
        "hardlink_savings_bytes": logical_bytes - unique_bytes,
        "hardlink_entry_count": hardlink_entries,
        "named_stream_file_count": named_stream_file_count,
        "named_stream_logical_bytes": named_stream_logical_bytes,
        "latest_mtime_ns": latest_mtime_ns,
        "volume_total_bytes": disk.total,
        "volume_used_bytes": disk.used,
        "volume_free_bytes": disk.free,
    }


def guard_storage(root: Path, policy: dict[str, Any], reservation_bytes: int) -> dict[str, Any]:
    if reservation_bytes < 0:
        raise StorageError("storage reservation must be non-negative")
    accounting = storage_accounting(root)
    active_leases = _load_active_leases(Path(str(accounting["root"])))
    active_reservation = sum(int(value["reservation_bytes"]) for value in active_leases)
    projected_unique = (
        int(accounting["unique_bytes"]) + active_reservation + reservation_bytes
    )
    projected_free = (
        int(accounting["volume_free_bytes"]) - active_reservation - reservation_bytes
    )
    reasons: list[str] = []
    if (Path(str(accounting["root"])) / MAINTENANCE_LOCK_NAME).exists():
        reasons.append("STORAGE_MAINTENANCE_LOCK")
    if projected_unique > int(policy["maximum_experiment_unique_bytes"]):
        reasons.append("EXPERIMENT_UNIQUE_BYTE_CAP")
    if projected_free < int(policy["minimum_volume_free_bytes"]):
        reasons.append("VOLUME_FREE_BYTE_FLOOR")
    return {
        "schema_version": "blindassist-carla-storage-guard-v1",
        "status": "PASS" if not reasons else "REFUSE_NEW_RUN",
        "reasons": reasons,
        "reservation_bytes": reservation_bytes,
        "active_lease_count": len(active_leases),
        "active_reservation_bytes": active_reservation,
        "projected_unique_bytes": projected_unique,
        "projected_volume_free_bytes": projected_free,
        "policy": {
            "path": policy["policy_path"],
            "sha256": policy["policy_sha256"],
            "maximum_experiment_unique_bytes": policy[
                "maximum_experiment_unique_bytes"
            ],
            "minimum_volume_free_bytes": policy["minimum_volume_free_bytes"],
            "overflow_action": policy["overflow_action"],
            "automatic_payload_deletion": policy["automatic_payload_deletion"],
        },
        "accounting": accounting,
    }


def _lease_directory(root: Path) -> Path:
    return root / LEASE_DIRECTORY_NAME


def _load_active_leases(root: Path) -> list[dict[str, Any]]:
    directory = _lease_directory(root)
    if not _path_exists(directory):
        return []
    values: list[dict[str, Any]] = []
    with os.scandir(_os_path(directory)) as entries:
        for entry in entries:
            path = directory / entry.name
            path_stat = os.stat(_os_path(path), follow_symlinks=False)
            if _is_reparse(path_stat) or not stat.S_ISREG(path_stat.st_mode):
                raise StorageError(f"invalid CARLA storage lease entry: {path}")
            try:
                with open(_os_path(path), "r", encoding="utf-8-sig") as handle:
                    value = json.load(handle)
            except (OSError, ValueError) as exc:
                raise StorageError(f"invalid CARLA storage lease: {path}: {exc}") from exc
            if (
                not isinstance(value, dict)
                or value.get("schema_version") != "blindassist-carla-storage-lease-v1"
                or not re.fullmatch(r"[0-9a-f]{32}", str(value.get("lease_token", "")))
                or not isinstance(value.get("reservation_bytes"), int)
                or int(value["reservation_bytes"]) < 0
            ):
                raise StorageError(f"invalid CARLA storage lease payload: {path}")
            if entry.name != f"{value['lease_token']}.json":
                raise StorageError(f"CARLA storage lease filename mismatch: {path}")
            values.append(value)
    return values


@contextmanager
def storage_coordination_lock(root: Path) -> Iterator[None]:
    physical_root = root.resolve(strict=True)
    lock_path = physical_root / COORDINATION_LOCK_NAME
    descriptor: int | None = None
    try:
        descriptor = os.open(
            _os_path(lock_path),
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        )
        os.write(
            descriptor,
            _canonical_json(
                {
                    "schema_version": "blindassist-carla-storage-coordinate-lock-v1",
                    "pid": os.getpid(),
                    "created_ns": time.time_ns(),
                }
            )
            + b"\n",
        )
        os.fsync(descriptor)
        yield
    except FileExistsError as exc:
        raise StorageError(f"CARLA storage coordination lock exists: {lock_path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
            try:
                os.unlink(_os_path(lock_path))
            except FileNotFoundError:
                pass


def _resolve_governed_output(root: Path, output_root: Path) -> Path:
    resolved = output_root.resolve(strict=False)
    if resolved != root and root not in resolved.parents:
        raise StorageError(
            f"CARLA output root is outside governed experiments root: {resolved}"
        )
    return resolved


def acquire_storage_lease(
    root: Path,
    policy: dict[str, Any],
    reservation_bytes: int,
    owner_pid: int,
    label: str,
    output_root: Path,
) -> dict[str, Any]:
    if reservation_bytes < 0:
        raise StorageError("storage reservation must be non-negative")
    if owner_pid <= 0:
        raise StorageError("storage lease owner PID must be positive")
    if not label.strip() or len(label) > 256:
        raise StorageError("storage lease label must contain 1 to 256 characters")
    physical_root = root.resolve(strict=True)
    governed_output = _resolve_governed_output(physical_root, output_root)
    with storage_coordination_lock(physical_root):
        if _path_exists(physical_root / MAINTENANCE_LOCK_NAME):
            raise StorageError("CARLA storage maintenance is active")
        accounting = storage_accounting(physical_root)
        active_leases = _load_active_leases(physical_root)
        active_reservation = sum(
            int(value["reservation_bytes"]) for value in active_leases
        )
        projected_unique = (
            int(accounting["unique_bytes"]) + active_reservation + reservation_bytes
        )
        projected_free = (
            int(accounting["volume_free_bytes"]) - active_reservation - reservation_bytes
        )
        reasons: list[str] = []
        if projected_unique > int(policy["maximum_experiment_unique_bytes"]):
            reasons.append("EXPERIMENT_UNIQUE_BYTE_CAP")
        if projected_free < int(policy["minimum_volume_free_bytes"]):
            reasons.append("VOLUME_FREE_BYTE_FLOOR")
        if reasons:
            raise StorageError(
                "storage lease refused: "
                f"reasons={','.join(reasons)} unique={accounting['unique_bytes']} "
                f"active_reservation={active_reservation} requested={reservation_bytes} "
                f"cap={policy['maximum_experiment_unique_bytes']} "
                f"free={accounting['volume_free_bytes']} "
                f"free_floor={policy['minimum_volume_free_bytes']}"
            )
        token = uuid.uuid4().hex
        lease = {
            "schema_version": "blindassist-carla-storage-lease-v1",
            "lease_token": token,
            "label": label,
            "owner_pid": owner_pid,
            "created_ns": time.time_ns(),
            "reservation_bytes": reservation_bytes,
            "baseline_unique_bytes": int(accounting["unique_bytes"]),
            "governed_output_root": str(governed_output),
            "policy_sha256": str(policy["policy_sha256"]),
        }
        lease_directory = _lease_directory(physical_root)
        os.makedirs(_os_path(lease_directory), exist_ok=True)
        _write_json_atomic(lease_directory / f"{token}.json", lease)
    return {
        "schema_version": "blindassist-carla-storage-lease-acquire-result-v1",
        "status": "ACQUIRED",
        "lease_token": token,
        "reservation_bytes": reservation_bytes,
        "active_lease_count": len(active_leases) + 1,
        "projected_unique_bytes": projected_unique,
        "projected_volume_free_bytes": projected_free,
        "governed_output_root": str(governed_output),
    }


def check_storage_lease(
    root: Path,
    policy: dict[str, Any],
    lease_token: str,
    output_root: Path | None = None,
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{32}", lease_token):
        raise StorageError("invalid CARLA storage lease token")
    physical_root = root.resolve(strict=True)
    leases = _load_active_leases(physical_root)
    lease = next(
        (value for value in leases if value["lease_token"] == lease_token),
        None,
    )
    if lease is None:
        raise StorageError("CARLA storage lease is missing")
    if str(lease.get("policy_sha256", "")) != str(policy["policy_sha256"]):
        raise StorageError("CARLA storage policy changed after lease acquisition")
    governed_output = Path(str(lease["governed_output_root"])).resolve(strict=False)
    if output_root is not None:
        checked_output = _resolve_governed_output(physical_root, output_root)
        if checked_output != governed_output and governed_output not in checked_output.parents:
            raise StorageError(
                "CARLA output root is outside the acquired lease scope: "
                f"output={checked_output} lease_scope={governed_output}"
            )
    if _path_exists(physical_root / MAINTENANCE_LOCK_NAME):
        raise StorageError("CARLA storage maintenance started during a run")
    accounting = storage_accounting(physical_root)
    other_reservation = sum(
        int(value["reservation_bytes"])
        for value in leases
        if value["lease_token"] != lease_token
    )
    projected_unique = int(accounting["unique_bytes"]) + other_reservation
    projected_free = int(accounting["volume_free_bytes"]) - other_reservation
    reasons: list[str] = []
    if projected_unique > int(policy["maximum_experiment_unique_bytes"]):
        reasons.append("EXPERIMENT_UNIQUE_BYTE_CAP")
    if projected_free < int(policy["minimum_volume_free_bytes"]):
        reasons.append("VOLUME_FREE_BYTE_FLOOR")
    if reasons:
        raise StorageError(
            "storage lease checkpoint refused: "
            f"reasons={','.join(reasons)} unique={accounting['unique_bytes']} "
            f"other_reservation={other_reservation} "
            f"cap={policy['maximum_experiment_unique_bytes']} "
            f"free={accounting['volume_free_bytes']} "
            f"free_floor={policy['minimum_volume_free_bytes']}"
        )
    return {
        "schema_version": "blindassist-carla-storage-lease-check-result-v1",
        "status": "PASS",
        "lease_token": lease_token,
        "unique_bytes": int(accounting["unique_bytes"]),
        "volume_free_bytes": int(accounting["volume_free_bytes"]),
        "other_active_reservation_bytes": other_reservation,
        "projected_unique_bytes": projected_unique,
        "projected_volume_free_bytes": projected_free,
    }


def release_storage_lease(root: Path, lease_token: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{32}", lease_token):
        raise StorageError("invalid CARLA storage lease token")
    physical_root = root.resolve(strict=True)
    lease_path = _lease_directory(physical_root) / f"{lease_token}.json"
    with storage_coordination_lock(physical_root):
        if not _path_exists(lease_path):
            status = "ALREADY_RELEASED"
        else:
            leases = _load_active_leases(physical_root)
            if not any(value["lease_token"] == lease_token for value in leases):
                raise StorageError("CARLA storage lease token mismatch")
            os.unlink(_os_path(lease_path))
            try:
                os.rmdir(_os_path(_lease_directory(physical_root)))
            except OSError:
                # Other leases, or a concurrent read handle, can keep the
                # control directory alive. Its contents remain authoritative.
                pass
            status = "RELEASED"
    return {
        "schema_version": "blindassist-carla-storage-lease-release-result-v1",
        "status": status,
        "lease_token": lease_token,
    }


def _normalize_extensions(values: Iterable[str]) -> frozenset[str]:
    normalized = set()
    for value in values:
        extension = str(value).strip().lower()
        if not extension:
            continue
        if not extension.startswith("."):
            extension = "." + extension
        normalized.add(extension)
    if not normalized:
        raise StorageError("at least one dedupe extension is required")
    return frozenset(normalized)


def _path_preference(path: Path, root: Path) -> tuple[int, int, str]:
    relative = path.relative_to(root).as_posix()
    parts = {value.lower() for value in path.parts}
    if "shards" in parts and "payload" in parts:
        rank = 0
    elif "child-evidence" in parts:
        rank = 1
    elif "model" in parts or "evaluator" in parts:
        rank = 2
    elif "final-package" in parts:
        rank = 3
    else:
        rank = 4
    return rank, len(relative), relative.lower()


def _load_seal_index(seal_root: Path) -> SealIndex:
    manifest_path = seal_root / "sealed_evidence_manifest.json"
    result_path = seal_root / "result.json"
    if not _regular_file_exists(result_path):
        raise StorageError(f"sealed scope has no terminal result.json: {seal_root}")
    try:
        with open(_os_path(result_path), "r", encoding="utf-8-sig") as handle:
            result_value = json.load(handle)
    except (OSError, ValueError) as exc:
        raise StorageError(f"invalid sealed result JSON: {result_path}: {exc}") from exc
    if (
        not isinstance(result_value, dict)
        or not isinstance(result_value.get("status"), str)
        or not str(result_value["status"]).strip()
    ):
        raise StorageError(f"sealed result has no terminal status: {result_path}")

    try:
        with open(_os_path(manifest_path), "r", encoding="utf-8-sig") as handle:
            manifest_value = json.load(handle)
    except (OSError, ValueError) as exc:
        raise StorageError(f"invalid sealed evidence manifest: {manifest_path}: {exc}") from exc
    if isinstance(manifest_value, list):
        rows = manifest_value
    elif isinstance(manifest_value, dict) and isinstance(manifest_value.get("files"), list):
        rows = manifest_value["files"]
        declared_count = manifest_value.get("file_count")
        if declared_count is not None and declared_count != len(rows):
            raise StorageError(f"sealed manifest file_count mismatch: {manifest_path}")
    else:
        raise StorageError(f"sealed manifest has no files array: {manifest_path}")

    members: dict[str, tuple[int, str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise StorageError(f"sealed manifest row is not an object: {manifest_path}")
        relative = _safe_relative_path(str(row.get("path", "")), "sealed manifest")
        size = row.get("bytes")
        digest = str(row.get("sha256", "")).upper()
        if not isinstance(size, int) or size < 0 or not re.fullmatch(r"[0-9A-F]{64}", digest):
            raise StorageError(f"invalid sealed manifest row: {manifest_path}: {relative}")
        key = relative.as_posix()
        value = (size, digest)
        if key in members and members[key] != value:
            raise StorageError(f"conflicting sealed manifest row: {manifest_path}: {key}")
        members[key] = value
    return SealIndex(
        root=seal_root,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        result_path=result_path,
        result_sha256=sha256_file(result_path),
        members=members,
    )


def _collect_dedupe_records(
    root: Path,
    extensions: frozenset[str],
    minimum_age_seconds: int,
    require_sealed_ancestor: bool,
) -> tuple[list[FileRecord], dict[str, int], list[dict[str, str]]]:
    if minimum_age_seconds < 0:
        raise StorageError("dedupe minimum age must be non-negative")
    cutoff_ns = time.time_ns() - minimum_age_seconds * 1_000_000_000
    values: list[FileRecord] = []
    skipped_young = 0
    skipped_unsealed = 0
    skipped_unlisted = 0
    skipped_named_streams = 0
    scanned_files = 0
    scanned_bytes = 0
    sealed_cache: dict[Path, SealIndex | None] = {}
    seal_indexes: dict[Path, SealIndex] = {}

    def find_seal_index(directory: Path) -> SealIndex | None:
        trail: list[Path] = []
        current = directory
        while True:
            if current in sealed_cache:
                result = sealed_cache[current]
                break
            trail.append(current)
            if _regular_file_exists(current / "sealed_evidence_manifest.json"):
                result = seal_indexes.get(current)
                if result is None:
                    result = _load_seal_index(current)
                    seal_indexes[current] = result
                break
            if current == root:
                result = None
                break
            parent = current.parent
            if parent == current or root not in parent.parents and parent != root:
                result = None
                break
            current = parent
        for value in trail:
            sealed_cache[value] = result
        return result

    for path, path_stat in iter_regular_files(root):
        if path.name == MAINTENANCE_LOCK_NAME:
            continue
        scanned_files += 1
        if path.suffix.lower() not in extensions:
            continue
        if int(path_stat.st_mtime_ns) > cutoff_ns:
            skipped_young += 1
            continue
        sealed_sha256: str | None = None
        if require_sealed_ancestor:
            seal = find_seal_index(path.parent)
            if seal is None:
                skipped_unsealed += 1
                continue
            relative_to_seal = path.relative_to(seal.root).as_posix()
            membership = seal.members.get(relative_to_seal)
            if membership is None:
                skipped_unlisted += 1
                continue
            expected_size, sealed_sha256 = membership
            if expected_size != int(path_stat.st_size):
                raise StorageError(f"sealed candidate size mismatch: {path}")
        if _named_streams(path):
            skipped_named_streams += 1
            continue
        scanned_bytes += int(path_stat.st_size)
        values.append(
            FileRecord(
                path=path,
                size=int(path_stat.st_size),
                identity=_file_identity(path, path_stat),
                link_count=int(getattr(path_stat, "st_nlink", 1)),
                mtime_ns=int(path_stat.st_mtime_ns),
                mode=stat.S_IMODE(path_stat.st_mode),
                file_attributes=int(getattr(path_stat, "st_file_attributes", 0)),
                security_sha256=_security_descriptor_sha256(path),
                sealed_sha256=sealed_sha256,
            )
        )
    seal_artifacts = [
        {
            "manifest": value.manifest_path.relative_to(root).as_posix(),
            "manifest_sha256": value.manifest_sha256,
            "result": value.result_path.relative_to(root).as_posix(),
            "result_sha256": value.result_sha256,
        }
        for value in sorted(seal_indexes.values(), key=lambda item: item.root.as_posix().lower())
    ]
    return values, {
        "scanned_file_count": scanned_files,
        "eligible_file_count": len(values),
        "eligible_logical_bytes": scanned_bytes,
        "skipped_young_file_count": skipped_young,
        "skipped_unsealed_file_count": skipped_unsealed,
        "skipped_unlisted_file_count": skipped_unlisted,
        "skipped_named_stream_file_count": skipped_named_streams,
    }, seal_artifacts


def build_dedupe_plan(
    root: Path,
    extensions: frozenset[str],
    minimum_age_seconds: int,
    require_sealed_ancestor: bool = True,
    progress: bool = False,
) -> dict[str, Any]:
    physical_root = root.resolve(strict=True)
    records, scan, seal_artifacts = _collect_dedupe_records(
        physical_root,
        extensions,
        minimum_age_seconds,
        require_sealed_ancestor,
    )
    by_size: dict[int, list[FileRecord]] = defaultdict(list)
    for record in records:
        by_size[record.size].append(record)

    hash_cache: dict[tuple[int, int] | tuple[str, str], str] = {}
    exact: dict[tuple[int, str, int, int, int, str], list[FileRecord]] = defaultdict(list)
    candidates = [
        group
        for group in by_size.values()
        if len({record.identity for record in group}) > 1
    ]
    unique_candidates: dict[tuple[int, int] | tuple[str, str], FileRecord] = {}
    for size_group in candidates:
        for record in size_group:
            unique_candidates.setdefault(record.identity, record)
    ordered_candidates = sorted(
        unique_candidates.values(),
        key=lambda value: value.path.as_posix().lower(),
    )
    hashed_files = 0
    hashed_bytes = 0

    def hash_record(record: FileRecord) -> tuple[FileRecord, str]:
        return record, sha256_file(record.path)

    with ThreadPoolExecutor(max_workers=min(4, max(1, len(ordered_candidates)))) as pool:
        for offset in range(0, len(ordered_candidates), 512):
            batch = ordered_candidates[offset : offset + 512]
            for record, digest in pool.map(hash_record, batch):
                hash_cache[record.identity] = digest
                hashed_files += 1
                hashed_bytes += record.size
                if progress and hashed_files % 5000 == 0:
                    print(
                        f"HASHED files={hashed_files} gib={hashed_bytes / GIB:.3f}",
                        file=sys.stderr,
                        flush=True,
                    )
    for size_group in candidates:
        for record in size_group:
            digest = hash_cache[record.identity]
            if record.sealed_sha256 is not None and digest != record.sealed_sha256:
                raise StorageError(f"sealed candidate hash mismatch: {record.path}")
            exact[
                (
                    record.size,
                    digest,
                    record.mtime_ns,
                    record.mode,
                    record.file_attributes,
                    record.security_sha256,
                )
            ].append(record)

    groups: list[dict[str, Any]] = []
    action_path_count = 0
    duplicate_identity_count = 0
    reclaimable_bytes = 0
    for (
        size,
        digest,
        mtime_ns,
        mode,
        file_attributes,
        security_sha256,
    ), group in exact.items():
        by_identity: dict[tuple[int, int] | tuple[str, str], list[FileRecord]] = defaultdict(list)
        for record in group:
            by_identity[record.identity].append(record)
        if len(by_identity) < 2:
            continue
        canonical_identity, canonical_records = min(
            by_identity.items(),
            key=lambda item: (
                -len(item[1]),
                min(_path_preference(value.path, physical_root) for value in item[1]),
            ),
        )
        canonical = min(
            (value.path for value in canonical_records),
            key=lambda value: _path_preference(value, physical_root),
        )
        duplicate_sets: list[dict[str, Any]] = []
        for identity, identity_records in sorted(
            by_identity.items(),
            key=lambda item: min(
                _path_preference(value.path, physical_root) for value in item[1]
            ),
        ):
            if identity == canonical_identity:
                continue
            paths = sorted(
                (
                    value.path.relative_to(physical_root).as_posix()
                    for value in identity_records
                ),
                key=str.lower,
            )
            observed_links = max(value.link_count for value in identity_records)
            fully_observed = len(identity_records) >= observed_links
            duplicate_sets.append(
                {
                    "paths": paths,
                    "observed_link_count": observed_links,
                    "fully_observed": fully_observed,
                }
            )
            action_path_count += len(paths)
            duplicate_identity_count += 1
            if fully_observed:
                reclaimable_bytes += size
        groups.append(
            {
                "bytes": size,
                "sha256": digest,
                "mtime_ns": mtime_ns,
                "mode": mode,
                "file_attributes": file_attributes,
                "security_sha256": security_sha256,
                "canonical": canonical.relative_to(physical_root).as_posix(),
                "duplicate_sets": duplicate_sets,
            }
        )
    groups.sort(key=lambda value: (str(value["sha256"]), str(value["canonical"])))
    plan_core = {
        "schema_version": "blindassist-carla-hardlink-dedupe-plan-v1",
        "root": str(physical_root),
        "extensions": sorted(extensions),
        "minimum_age_seconds": minimum_age_seconds,
        "require_sealed_ancestor": require_sealed_ancestor,
        "sealed_artifacts": seal_artifacts,
        "scan": scan,
        "hashed_file_count": hashed_files,
        "hashed_bytes": hashed_bytes,
        "duplicate_group_count": len(groups),
        "duplicate_identity_count": duplicate_identity_count,
        "action_path_count": action_path_count,
        "reclaimable_bytes": reclaimable_bytes,
        "groups": groups,
    }
    plan_core["plan_sha256"] = _sha256_bytes(_canonical_json(plan_core))
    return plan_core


def _current_record(path: Path) -> FileRecord:
    path_stat = os.stat(_os_path(path), follow_symlinks=False)
    if _is_reparse(path_stat) or not stat.S_ISREG(path_stat.st_mode):
        raise StorageError(f"dedupe path is no longer a regular file: {path}")
    if _named_streams(path):
        raise StorageError(f"dedupe path acquired a named stream: {path}")
    return FileRecord(
        path=path,
        size=int(path_stat.st_size),
        identity=_file_identity(path, path_stat),
        link_count=int(getattr(path_stat, "st_nlink", 1)),
        mtime_ns=int(path_stat.st_mtime_ns),
        mode=stat.S_IMODE(path_stat.st_mode),
        file_attributes=int(getattr(path_stat, "st_file_attributes", 0)),
        security_sha256=_security_descriptor_sha256(path),
    )


def _assert_record_matches(
    record: FileRecord,
    expected_size: int,
    expected_mtime_ns: int,
    expected_mode: int,
    expected_file_attributes: int,
    expected_security_sha256: str,
) -> None:
    if (
        record.size != expected_size
        or record.mtime_ns != expected_mtime_ns
        or record.mode != expected_mode
        or record.file_attributes != expected_file_attributes
        or record.security_sha256 != expected_security_sha256
    ):
        raise StorageError(f"dedupe candidate changed after planning: {record.path}")


def _replace_with_hardlink(canonical: Path, duplicate: Path, token: str) -> None:
    temporary = duplicate.with_name(
        f".{duplicate.name}.carla-dedupe-{token}-{uuid.uuid4().hex}.tmp"
    )
    if _path_exists(temporary):
        raise StorageError(f"dedupe temporary path already exists: {temporary}")
    try:
        os.link(_os_path(canonical), _os_path(temporary))
        os.replace(_os_path(temporary), _os_path(duplicate))
    finally:
        if _path_exists(temporary):
            os.unlink(_os_path(temporary))


@contextmanager
def storage_maintenance_lock(root: Path) -> Iterator[None]:
    physical_root = root.resolve(strict=True)
    lock_path = physical_root / MAINTENANCE_LOCK_NAME
    descriptor: int | None = None
    try:
        with storage_coordination_lock(physical_root):
            active_leases = _load_active_leases(physical_root)
            if active_leases:
                raise StorageError(
                    "CARLA storage maintenance refused while run leases are active: "
                    + ",".join(str(value["lease_token"]) for value in active_leases)
                )
            descriptor = os.open(
                _os_path(lock_path),
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
            payload = _canonical_json(
                {
                    "schema_version": "blindassist-carla-storage-lock-v1",
                    "pid": os.getpid(),
                    "created_ns": time.time_ns(),
                }
            )
            os.write(descriptor, payload + b"\n")
            os.fsync(descriptor)
        yield
    except FileExistsError as exc:
        raise StorageError(f"CARLA storage maintenance lock already exists: {lock_path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
            try:
                os.unlink(_os_path(lock_path))
            except FileNotFoundError:
                pass


def apply_dedupe_plan(
    plan: dict[str, Any],
    receipt_dir: Path,
    expected_plan_sha256: str,
    progress: bool = False,
) -> dict[str, Any]:
    if plan.get("schema_version") != "blindassist-carla-hardlink-dedupe-plan-v1":
        raise StorageError(f"unexpected dedupe plan schema: {plan.get('schema_version')}")
    plan_core = {key: value for key, value in plan.items() if key != "plan_sha256"}
    actual_plan_sha256 = _sha256_bytes(_canonical_json(plan_core))
    embedded_plan_sha256 = str(plan.get("plan_sha256", "")).upper()
    if embedded_plan_sha256 != actual_plan_sha256:
        raise StorageError(
            "dedupe plan content does not match its embedded plan_sha256: "
            f"embedded={embedded_plan_sha256} computed={actual_plan_sha256}"
        )
    if actual_plan_sha256 != expected_plan_sha256.upper():
        raise StorageError(
            f"dedupe plan changed: expected {expected_plan_sha256}, got {actual_plan_sha256}"
        )
    root = Path(str(plan["root"])).resolve(strict=True)
    seen_action_paths: set[str] = set()
    for artifact in plan.get("sealed_artifacts", []):
        manifest = _contained_plan_path(
            root,
            str(artifact["manifest"]),
            "sealed manifest",
        )
        result_path = _contained_plan_path(
            root,
            str(artifact["result"]),
            "sealed result",
        )
        if sha256_file(manifest) != str(artifact["manifest_sha256"]).upper():
            raise StorageError(f"sealed manifest changed after planning: {manifest}")
        if sha256_file(result_path) != str(artifact["result_sha256"]).upper():
            raise StorageError(f"sealed result changed after planning: {result_path}")
    for group in plan.get("groups", []):
        canonical_value = str(group["canonical"])
        _contained_plan_path(root, canonical_value, "canonical")
        for duplicate_set in group.get("duplicate_sets", []):
            for value in duplicate_set.get("paths", []):
                relative = _safe_relative_path(str(value), "duplicate").as_posix()
                _contained_plan_path(root, relative, "duplicate")
                if relative == canonical_value or relative in seen_action_paths:
                    raise StorageError(f"duplicate plan path is repeated: {relative}")
                seen_action_paths.add(relative)
    receipt_dir.mkdir(parents=True, exist_ok=False)
    _write_json_atomic(receipt_dir / "dedupe-plan.json", plan)
    events_path = receipt_dir / "dedupe-events.jsonl"
    token = uuid.uuid4().hex[:12]
    linked_paths = 0
    merged_identities = 0
    reclaimed_bytes = 0
    started_ns = time.time_ns()
    free_before = shutil.disk_usage(root).free
    with events_path.open("w", encoding="utf-8", newline="\n") as events:
        for group in plan["groups"]:
            canonical = _contained_plan_path(
                root,
                str(group["canonical"]),
                "canonical",
            )
            expected_size = int(group["bytes"])
            expected_mtime_ns = int(group["mtime_ns"])
            expected_mode = int(group["mode"])
            expected_file_attributes = int(group["file_attributes"])
            expected_security_sha256 = str(group["security_sha256"])
            canonical_record = _current_record(canonical)
            _assert_record_matches(
                canonical_record,
                expected_size,
                expected_mtime_ns,
                expected_mode,
                expected_file_attributes,
                expected_security_sha256,
            )
            if sha256_file(canonical) != str(group["sha256"]):
                raise StorageError(f"canonical file hash changed after planning: {canonical}")
            for duplicate_set in group["duplicate_sets"]:
                duplicate_paths = [
                    _contained_plan_path(root, str(value), "duplicate")
                    for value in duplicate_set["paths"]
                ]
                before_records = [_current_record(value) for value in duplicate_paths]
                for record in before_records:
                    _assert_record_matches(
                        record,
                        expected_size,
                        expected_mtime_ns,
                        expected_mode,
                        expected_file_attributes,
                        expected_security_sha256,
                    )
                if len({record.identity for record in before_records}) != 1:
                    raise StorageError(
                        f"planned hardlink set changed before apply: {duplicate_paths[0]}"
                    )
                if sha256_file(duplicate_paths[0]) != str(group["sha256"]):
                    raise StorageError(f"duplicate file hash changed: {duplicate_paths[0]}")
                event = {
                    "status": "STARTED",
                    "canonical": str(group["canonical"]),
                    "paths": list(duplicate_set["paths"]),
                    "bytes": expected_size,
                    "sha256": str(group["sha256"]),
                }
                events.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
                events.flush()
                for duplicate in duplicate_paths:
                    _replace_with_hardlink(canonical, duplicate, token)
                    if not os.path.samefile(_os_path(canonical), _os_path(duplicate)):
                        raise StorageError(f"hardlink verification failed: {duplicate}")
                    linked_paths += 1
                merged_identities += 1
                if bool(duplicate_set["fully_observed"]):
                    reclaimed_bytes += expected_size
                event["status"] = "COMPLETE"
                events.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
                if merged_identities % 256 == 0:
                    events.flush()
                    os.fsync(events.fileno())
                if progress and merged_identities % 2000 == 0:
                    print(
                        "LINKED "
                        f"identities={merged_identities} paths={linked_paths} "
                        f"gib={reclaimed_bytes / GIB:.3f}",
                        file=sys.stderr,
                        flush=True,
                    )
        events.flush()
        os.fsync(events.fileno())
    free_after = shutil.disk_usage(root).free
    result = {
        "schema_version": "blindassist-carla-hardlink-dedupe-result-v1",
        "status": "COMPLETE",
        "root": str(root),
        "plan_sha256": actual_plan_sha256,
        "linked_path_count": linked_paths,
        "merged_identity_count": merged_identities,
        "expected_reclaimed_bytes": reclaimed_bytes,
        "volume_free_before_bytes": free_before,
        "volume_free_after_bytes": free_after,
        "observed_volume_free_delta_bytes": free_after - free_before,
        "started_ns": started_ns,
        "completed_ns": time.time_ns(),
        "content_or_path_deleted": False,
    }
    _write_json_atomic(receipt_dir / "dedupe-result.json", result)
    return result


def _copy_regular_file_exclusive(source: Path, destination: Path) -> None:
    """Copy one file without ever opening an existing destination for write."""

    created = False
    try:
        with open(_os_path(source), "rb") as source_handle:
            destination_handle = open(_os_path(destination), "xb")
            created = True
            with destination_handle:
                shutil.copyfileobj(source_handle, destination_handle, 4 * 1024 * 1024)
                destination_handle.flush()
                os.fsync(destination_handle.fileno())
        shutil.copystat(
            _os_path(source),
            _os_path(destination),
            follow_symlinks=False,
        )
        if sha256_file(source) != sha256_file(destination):
            raise StorageError(f"clone copy verification failed: {destination}")
    except BaseException:
        if created and _path_exists(destination):
            os.unlink(_os_path(destination))
        raise


def clone_tree(source: Path, destination: Path) -> dict[str, Any]:
    source = source.resolve(strict=True)
    destination = Path(os.path.abspath(os.fspath(destination)))
    if not source.is_dir():
        raise StorageError(f"clone source is not a directory: {source}")
    if destination == source or source in destination.parents:
        raise StorageError(f"clone destination must be outside its source: {destination}")
    if _path_exists(destination):
        raise StorageError(f"clone destination already exists: {destination}")
    linked_files = 0
    copied_files = 0
    linked_bytes = 0

    os.makedirs(_os_path(destination.parent), exist_ok=True)
    try:
        os.mkdir(_os_path(destination))
    except FileExistsError as exc:
        raise StorageError(f"clone destination already exists: {destination}") from exc
    try:
        pending = [(source, destination)]
        while pending:
            source_directory, destination_directory = pending.pop()
            with os.scandir(_os_path(source_directory)) as entries:
                for entry in entries:
                    source_path = source_directory / entry.name
                    destination_path = destination_directory / entry.name
                    entry_stat = os.stat(_os_path(source_path), follow_symlinks=False)
                    if _is_reparse(entry_stat) or entry.is_symlink():
                        raise StorageError(
                            f"clone source contains a reparse point: {source_path}"
                        )
                    if stat.S_ISDIR(entry_stat.st_mode):
                        os.mkdir(_os_path(destination_path))
                        pending.append((source_path, destination_path))
                        continue
                    if not stat.S_ISREG(entry_stat.st_mode):
                        raise StorageError(
                            f"clone source contains a non-regular file: {source_path}"
                        )
                    if _named_streams(source_path):
                        raise StorageError(
                            f"clone source contains a named stream: {source_path}"
                        )
                    try:
                        os.link(_os_path(source_path), _os_path(destination_path))
                        linked_files += 1
                        linked_bytes += int(entry_stat.st_size)
                    except OSError:
                        _copy_regular_file_exclusive(source_path, destination_path)
                        copied_files += 1
    except BaseException:
        if _path_exists(destination):
            shutil.rmtree(_os_path(destination))
        raise
    return {
        "schema_version": "blindassist-carla-tree-clone-v1",
        "source": str(source),
        "destination": str(destination),
        "linked_file_count": linked_files,
        "copied_file_count": copied_files,
        "linked_bytes": linked_bytes,
    }


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="count logical and unique storage")
    audit.add_argument("--root", type=Path, required=True)

    guard = subparsers.add_parser("guard", help="refuse a run above the frozen cap")
    guard.add_argument("--root", type=Path, required=True)
    guard.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    guard.add_argument("--reservation-bytes", type=int)

    acquire = subparsers.add_parser(
        "lease-acquire",
        help="atomically reserve capacity for an official runner",
    )
    acquire.add_argument("--root", type=Path, required=True)
    acquire.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    acquire.add_argument("--reservation-bytes", type=int, required=True)
    acquire.add_argument("--owner-pid", type=int, required=True)
    acquire.add_argument("--label", required=True)
    acquire.add_argument("--output-root", type=Path, required=True)

    check = subparsers.add_parser(
        "lease-check",
        help="fail closed when an active runner crosses a storage boundary",
    )
    check.add_argument("--root", type=Path, required=True)
    check.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    check.add_argument("--lease-token", required=True)
    check.add_argument("--output-root", type=Path)

    release = subparsers.add_parser(
        "lease-release",
        help="release one exact runner capacity lease",
    )
    release.add_argument("--root", type=Path, required=True)
    release.add_argument("--lease-token", required=True)

    dedupe = subparsers.add_parser("dedupe", help="plan or apply exact hardlink dedupe")
    dedupe.add_argument("--root", type=Path, required=True)
    dedupe.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    dedupe.add_argument("--extension", action="append", dest="extensions")
    dedupe.add_argument("--minimum-age-seconds", type=int)
    dedupe.add_argument("--receipt-dir", type=Path)
    dedupe.add_argument("--expect-plan-sha256")
    dedupe.add_argument("--apply", action="store_true")
    dedupe.add_argument("--progress", action="store_true")

    clone = subparsers.add_parser("clone-tree", help="clone a tree with hardlinks")
    clone.add_argument("--source", type=Path, required=True)
    clone.add_argument("--destination", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "audit":
            _print_json(storage_accounting(args.root))
            return 0
        if args.command == "guard":
            policy = load_policy(args.policy)
            reservation = (
                int(args.reservation_bytes)
                if args.reservation_bytes is not None
                else int(policy["default_run_reservation_bytes"])
            )
            result = guard_storage(args.root, policy, reservation)
            _print_json(result)
            return 0 if result["status"] == "PASS" else 2
        if args.command == "lease-acquire":
            policy = load_policy(args.policy)
            _print_json(
                acquire_storage_lease(
                    args.root,
                    policy,
                    int(args.reservation_bytes),
                    int(args.owner_pid),
                    str(args.label),
                    args.output_root,
                )
            )
            return 0
        if args.command == "lease-check":
            policy = load_policy(args.policy)
            _print_json(
                check_storage_lease(
                    args.root,
                    policy,
                    str(args.lease_token),
                    args.output_root,
                )
            )
            return 0
        if args.command == "lease-release":
            _print_json(release_storage_lease(args.root, str(args.lease_token)))
            return 0
        if args.command == "dedupe":
            policy = load_policy(args.policy)
            extensions = _normalize_extensions(
                args.extensions or policy["dedupe_extensions"]
            )
            minimum_age = (
                int(args.minimum_age_seconds)
                if args.minimum_age_seconds is not None
                else int(policy["dedupe_minimum_age_seconds"])
            )
            if not args.apply:
                plan = build_dedupe_plan(
                    args.root,
                    extensions,
                    minimum_age,
                    require_sealed_ancestor=bool(
                        policy["dedupe_require_sealed_ancestor"]
                    ),
                    progress=bool(args.progress),
                )
                if args.receipt_dir is not None:
                    args.receipt_dir.mkdir(parents=True, exist_ok=False)
                    _write_json_atomic(args.receipt_dir / "dedupe-plan.json", plan)
                _print_json({key: value for key, value in plan.items() if key != "groups"})
                return 0
            if args.receipt_dir is None or not args.expect_plan_sha256:
                raise StorageError(
                    "--apply requires a new --receipt-dir and --expect-plan-sha256"
                )
            with storage_maintenance_lock(args.root):
                plan = build_dedupe_plan(
                    args.root,
                    extensions,
                    minimum_age,
                    require_sealed_ancestor=bool(
                        policy["dedupe_require_sealed_ancestor"]
                    ),
                    progress=bool(args.progress),
                )
                result = apply_dedupe_plan(
                    plan,
                    args.receipt_dir,
                    args.expect_plan_sha256,
                    progress=bool(args.progress),
                )
            _print_json(result)
            return 0
        if args.command == "clone-tree":
            _print_json(clone_tree(args.source, args.destination))
            return 0
    except (OSError, ValueError, StorageError) as exc:
        print(f"CARLA_STORAGE_ERROR: {exc}", file=sys.stderr)
        return 1
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
