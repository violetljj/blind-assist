from __future__ import annotations

import argparse
import base64
import bz2
from collections.abc import Iterable
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import http.client
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import struct
import sys
import time
from typing import Any
import urllib.request
import zipfile
import zlib


OPENLORIS_BUDGET = 3_947_000_000
DLR_BUDGET = 1_073_741_824
NETWORK_CHUNK = 8 << 20
DLR_START_NS = 1_634_201_323_995_618_343
DLR_END_NS = 1_634_201_333_995_618_343


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class ByteBudget:
    limit: int
    consumed: int = 0

    def reserve(self, amount: int) -> None:
        if amount < 0:
            raise ValueError("negative byte reservation")
        if self.consumed + amount > self.limit:
            raise BudgetExceeded(
                f"BYTE_BUDGET_EXCEEDED:{self.consumed}+{amount}>{self.limit}"
            )
        self.consumed += amount

    def release(self, amount: int) -> None:
        if amount < 0 or amount > self.consumed:
            raise ValueError("invalid byte release")
        self.consumed -= amount


class BudgetedRemoteRange(io.RawIOBase):
    def __init__(
        self, url: str, length: int, budget: int, progress_path: Path
    ) -> None:
        self.url = url
        self.length = length
        self.position = 0
        self.budget = ByteBudget(budget)
        self.requests: list[dict[str, Any]] = []
        self.progress_path = progress_path
        self.started_at = time.monotonic()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.length + offset
        else:
            raise ValueError(f"unsupported whence: {whence}")
        if not 0 <= position <= self.length:
            raise ValueError(f"seek outside object: {position}")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.length - self.position
        size = min(size, self.length - self.position, NETWORK_CHUNK)
        if size <= 0:
            return b""
        start = self.position
        end = start + size - 1
        request = urllib.request.Request(
            self.url,
            headers={
                "Range": f"bytes={start}-{end}",
                "User-Agent": "BlindAssist-RCLE-RGB-Segment-Confirmation-R1",
                "Connection": "close",
            },
        )
        body = None
        for attempt in range(1, 4):
            # Authorize the largest body we will allow the socket to yield before
            # opening it. The extra byte detects a server that ignores the range.
            read_cap = size + 1
            self.budget.reserve(read_cap)
            received_bytes = read_cap
            status: int | None = None
            content_range: str | None = None
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    status = response.status
                    content_range = response.headers.get("Content-Range")
                    body = response.read(read_cap)
                    received_bytes = len(body)
                self.budget.release(read_cap - received_bytes)
                expected_content_range = (
                    f"bytes {start}-{end}/{self.length}"
                )
                if (
                    status != 206
                    or len(body) != size
                    or content_range != expected_content_range
                ):
                    raise RuntimeError(
                        "RANGE_CONTRACT:"
                        f"{status}:{len(body)}:{size}:"
                        f"{content_range!r}:{expected_content_range!r}"
                    )
                self.requests.append(
                    {
                        "start": start,
                        "end": end,
                        "attempt": attempt,
                        "status": "PASS",
                        "http_status": status,
                        "content_range": content_range,
                        "bytes": received_bytes,
                        "sha256": hashlib.sha256(body).hexdigest(),
                    }
                )
                self._write_progress()
                break
            except (OSError, http.client.IncompleteRead, RuntimeError) as error:
                # If the read failed mid-stream, keep the full preauthorization:
                # Python may not expose every byte already transferred.
                accounted_bytes = (
                    received_bytes
                    if isinstance(error, RuntimeError)
                    else read_cap
                )
                self.requests.append(
                    {
                        "start": start,
                        "end": end,
                        "attempt": attempt,
                        "status": "RETRYABLE_FAILURE",
                        "http_status": status,
                        "content_range": content_range,
                        "accounted_bytes": accounted_bytes,
                        "error_type": type(error).__name__,
                    }
                )
                self._write_progress()
                if attempt == 3:
                    raise
                time.sleep(0.25 * attempt)
        assert body is not None
        self.position += len(body)

        return body

    def _write_progress(self) -> None:
        elapsed = max(time.monotonic() - self.started_at, 1e-9)
        throughput = self.budget.consumed / elapsed
        remaining = self.budget.limit - self.budget.consumed
        write_json_atomic(
            self.progress_path,
            {
                "phase": "OPAQUE_RANGE_ACQUISITION",
                "completed_units": self.budget.consumed,
                "total_units": self.budget.limit,
                "throughput": throughput,
                "eta_seconds": (
                    remaining / throughput if throughput > 0 else None
                ),
                "last_progress_at": datetime.now(timezone.utc).isoformat(),
                "status": "RUNNING",
                "completed_bytes": self.budget.consumed,
                "total_budget_bytes": self.budget.limit,
                "fraction_of_budget": self.budget.consumed / self.budget.limit,
                "request_count": len(self.requests),
                "terminal": "RUNNING",
            },
        )


def parse_fields(raw: bytes) -> dict[str, bytes]:
    fields: dict[str, bytes] = {}
    offset = 0
    while offset < len(raw):
        if offset + 4 > len(raw):
            raise ValueError("FIELD_LENGTH_TRUNCATED")
        length = struct.unpack_from("<I", raw, offset)[0]
        offset += 4
        if offset + length > len(raw):
            raise ValueError("FIELD_TRUNCATED")
        field = raw[offset : offset + length]
        offset += length
        key, separator, value = field.partition(b"=")
        if not separator:
            raise ValueError("FIELD_KEY_ABSENT")
        fields[key.decode("ascii")] = value
    return fields


def records(raw: bytes) -> Iterable[tuple[dict[str, bytes], bytes]]:
    offset = 0
    while offset < len(raw):
        if offset + 4 > len(raw):
            raise EOFError("RECORD_HEADER_LENGTH_TRUNCATED")
        header_length = struct.unpack_from("<I", raw, offset)[0]
        offset += 4
        if offset + header_length + 4 > len(raw):
            raise EOFError("RECORD_HEADER_TRUNCATED")
        header = parse_fields(raw[offset : offset + header_length])
        offset += header_length
        data_length = struct.unpack_from("<I", raw, offset)[0]
        offset += 4
        if offset + data_length > len(raw):
            raise EOFError("RECORD_DATA_TRUNCATED")
        data = raw[offset : offset + data_length]
        offset += data_length
        yield header, data


def u32(value: bytes) -> int:
    if len(value) != 4:
        raise ValueError("U32_SIZE")
    return struct.unpack("<I", value)[0]


def parse_ros_string(raw: bytes, offset: int) -> tuple[str, int]:
    if offset + 4 > len(raw):
        raise ValueError("ROS_STRING_LENGTH")
    length = struct.unpack_from("<I", raw, offset)[0]
    offset += 4
    if offset + length > len(raw):
        raise ValueError("ROS_STRING_DATA")
    return raw[offset : offset + length].decode("utf-8"), offset + length


def parse_sensor_image(raw: bytes) -> dict[str, Any]:
    if len(raw) < 12:
        raise ValueError("IMAGE_HEADER")
    offset = 4
    sec, nsec = struct.unpack_from("<II", raw, offset)
    offset += 8
    frame_id, offset = parse_ros_string(raw, offset)
    if offset + 8 > len(raw):
        raise ValueError("IMAGE_SHAPE")
    height, width = struct.unpack_from("<II", raw, offset)
    offset += 8
    encoding, offset = parse_ros_string(raw, offset)
    if offset + 9 > len(raw):
        raise ValueError("IMAGE_LAYOUT")
    is_bigendian = raw[offset]
    offset += 1
    step, data_length = struct.unpack_from("<II", raw, offset)
    offset += 8
    if offset + data_length != len(raw):
        raise ValueError("IMAGE_PAYLOAD_LENGTH")
    payload = raw[offset:]
    return {
        "header_timestamp_ns": sec * 1_000_000_000 + nsec,
        "frame_id": frame_id,
        "height": height,
        "width": width,
        "encoding": encoding,
        "is_bigendian": is_bigendian,
        "step": step,
        "payload": payload,
    }


@dataclass
class RosbagImageCollector:
    start_ns: int
    end_ns: int
    buffer: bytearray = field(default_factory=bytearray)
    connections: dict[int, dict[str, str]] = field(default_factory=dict)
    before: dict[str, Any] | None = None
    selected: list[dict[str, Any]] = field(default_factory=list)
    after: dict[str, Any] | None = None
    color_connection: int | None = None

    @property
    def complete(self) -> bool:
        return self.before is not None and bool(self.selected) and self.after is not None

    def feed(self, data: bytes) -> None:
        self.buffer.extend(data)
        if self.buffer.startswith(b"#ROSBAG V2.0\n"):
            del self.buffer[:13]
        while True:
            try:
                consumed = self._consume_one(bytes(self.buffer))
            except EOFError:
                return
            if consumed == 0:
                return
            del self.buffer[:consumed]

    def _consume_one(self, raw: bytes) -> int:
        if len(raw) < 4:
            raise EOFError
        header_length = struct.unpack_from("<I", raw, 0)[0]
        if len(raw) < 4 + header_length + 4:
            raise EOFError
        header = parse_fields(raw[4 : 4 + header_length])
        data_length = struct.unpack_from("<I", raw, 4 + header_length)[0]
        total = 8 + header_length + data_length
        if len(raw) < total:
            raise EOFError
        data = raw[8 + header_length : total]
        operation = header.get("op", b"\x00")[0]
        if operation == 0x07:
            connection_id = u32(header["conn"])
            connection_data = parse_fields(data)
            topic = header.get("topic", b"").decode("utf-8")
            message_type = connection_data.get("type", b"").decode("utf-8")
            self.connections[connection_id] = {
                "topic": topic,
                "type": message_type,
            }
            if (
                message_type == "sensor_msgs/Image"
                and "/color/" in topic
                and "depth" not in topic
            ):
                if self.color_connection is not None and self.color_connection != connection_id:
                    raise ValueError("MULTIPLE_COLOR_IMAGE_CONNECTIONS")
                self.color_connection = connection_id
        elif operation == 0x05:
            compression = header["compression"].decode("ascii")
            if compression == "none":
                chunk = data
            elif compression == "bz2":
                chunk = bz2.decompress(data)
            else:
                raise ValueError(f"UNSUPPORTED_ROSBAG_COMPRESSION:{compression}")
            for inner_header, inner_data in records(chunk):
                if inner_header.get("op", b"\x00")[0] != 0x02:
                    continue
                connection_id = u32(inner_header["conn"])
                if connection_id != self.color_connection:
                    continue
                bag_time = struct.unpack("<II", inner_header["time"])
                bag_timestamp_ns = bag_time[0] * 1_000_000_000 + bag_time[1]
                image = parse_sensor_image(inner_data)
                row = {
                    "bag_timestamp_ns": bag_timestamp_ns,
                    "header_timestamp_ns": image["header_timestamp_ns"],
                    "frame_id": image["frame_id"],
                    "height": image["height"],
                    "width": image["width"],
                    "encoding": image["encoding"],
                    "is_bigendian": image["is_bigendian"],
                    "step": image["step"],
                    "serialized_message_sha256": hashlib.sha256(inner_data).hexdigest(),
                    "payload_sha256": hashlib.sha256(image["payload"]).hexdigest(),
                    "payload_bytes": len(image["payload"]),
                    "payload": image["payload"],
                }
                timestamp = image["header_timestamp_ns"]
                if timestamp < self.start_ns:
                    self.before = row
                elif timestamp < self.end_ns:
                    self.selected.append(row)
                elif self.after is None:
                    self.after = row
        return total


def exclusive_claim(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_runtime_lock(repo: Path, runtime_path: Path) -> dict[str, Any]:
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    python = runtime.get("python", {})
    executable = Path(sys.executable).resolve()
    if str(executable).casefold() != str(
        Path(python.get("venv_executable", "")).resolve()
    ).casefold():
        raise ValueError("OPAQUE_IDENTITY_RUNTIME_EXECUTABLE")
    if sha256_file(executable) != python.get("venv_executable_sha256"):
        raise ValueError("OPAQUE_IDENTITY_RUNTIME_EXECUTABLE_SHA256")
    if (
        ".".join(map(str, sys.version_info[:3]))
        != python.get("version")
    ):
        raise ValueError("OPAQUE_IDENTITY_RUNTIME_PYTHON_VERSION")
    base_executable = Path(
        getattr(sys, "_base_executable", sys.executable)
    ).resolve()
    if sha256_file(base_executable) != python.get("base_executable_sha256"):
        raise ValueError("OPAQUE_IDENTITY_RUNTIME_BASE_SHA256")
    wheel_cache = repo / runtime.get("wheel_cache", "")
    verified_files: dict[str, set[Path]] = {}
    for name, expected in runtime.get("distributions", {}).items():
        distribution = importlib.metadata.distribution(name)
        if distribution.version != expected.get("version"):
            raise ValueError(f"OPAQUE_IDENTITY_RUNTIME_VERSION:{name}")
        wheel = wheel_cache / expected.get("wheel", "")
        if (
            not wheel.is_file()
            or sha256_file(wheel) != expected.get("sha256")
        ):
            raise ValueError(f"OPAQUE_IDENTITY_RUNTIME_WHEEL:{name}")
        with zipfile.ZipFile(wheel) as archive:
            record_names = [
                item
                for item in archive.namelist()
                if item.endswith(".dist-info/RECORD")
            ]
            if len(record_names) != 1:
                raise ValueError(f"OPAQUE_IDENTITY_RUNTIME_RECORD:{name}")
            rows = csv.reader(
                archive.read(record_names[0]).decode("utf-8").splitlines()
            )
            installed: set[Path] = set()
            for relative, digest, size in rows:
                if not digest:
                    continue
                algorithm, separator, expected_digest = digest.partition("=")
                if algorithm != "sha256" or not separator:
                    raise ValueError(
                        f"OPAQUE_IDENTITY_RUNTIME_RECORD_HASH:{name}"
                    )
                installed_path = Path(
                    distribution.locate_file(relative)
                ).resolve()
                if not installed_path.is_file():
                    raise ValueError(
                        f"OPAQUE_IDENTITY_RUNTIME_INSTALLED_FILE:{name}"
                    )
                payload = installed_path.read_bytes()
                actual_digest = (
                    base64.urlsafe_b64encode(
                        hashlib.sha256(payload).digest()
                    )
                    .rstrip(b"=")
                    .decode("ascii")
                )
                if actual_digest != expected_digest or (
                    size and len(payload) != int(size)
                ):
                    raise ValueError(
                        f"OPAQUE_IDENTITY_RUNTIME_INSTALLED_HASH:{name}"
                    )
                installed.add(installed_path)
            verified_files[name.casefold().replace("_", "-")] = installed
    if os.environ.get("PYTHONPATH"):
        raise ValueError("OPAQUE_IDENTITY_RUNTIME_PYTHONPATH")
    original_sys_path = list(sys.path)
    venv_root = executable.parents[1]
    base_root = Path(sys.base_prefix).resolve()
    try:
        sys.path[:] = [
            entry
            for entry in sys.path
            if entry
            and any(
                _is_relative_to(Path(entry).resolve(), root)
                for root in (venv_root, base_root)
            )
        ]
        py7zr = importlib.import_module("py7zr")
    finally:
        sys.path[:] = original_sys_path
    py7zr_file = Path(py7zr.__file__).resolve()
    if py7zr_file not in verified_files.get("py7zr", set()):
        raise ValueError("OPAQUE_IDENTITY_RUNTIME_PY7ZR_ORIGIN")
    return runtime


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def validate_activation(
    lock_path: Path, lock: dict[str, Any], signature_path: Path
) -> dict[str, Any]:
    signature = json.loads(signature_path.read_text(encoding="utf-8"))
    if signature.get("decision") != "OPAQUE_IDENTITY_EXTRACTION_ACTIVATED":
        raise ValueError("OPAQUE_IDENTITY_ACTIVATION_DECISION")
    bindings = signature.get("bindings", {})
    if bindings.get("preaccess_lock", {}).get("sha256") != sha256_file(lock_path):
        raise ValueError("OPAQUE_IDENTITY_ACTIVATION_LOCK")
    transport_path = Path(__file__).resolve()
    if bindings.get("opaque_transport", {}).get("sha256") != sha256_file(
        transport_path
    ):
        raise ValueError("OPAQUE_IDENTITY_ACTIVATION_TRANSPORT")
    contract_binding = bindings.get("candidate_lock", {})
    contract_path = lock_path.parents[3] / contract_binding.get("path", "")
    if (
        not contract_path.is_file()
        or sha256_file(contract_path) != contract_binding.get("sha256")
    ):
        raise ValueError("OPAQUE_IDENTITY_ACTIVATION_CONTRACT")
    review_binding = bindings.get("independent_review", {})
    review_path = lock_path.parents[3] / review_binding.get("path", "")
    if (
        not review_path.is_file()
        or sha256_file(review_path) != review_binding.get("sha256")
    ):
        raise ValueError("OPAQUE_IDENTITY_ACTIVATION_REVIEW_BINDING")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if (
        review.get("decision")
        != "PREACCESS_IDENTITY_LOCK_INDEPENDENT_REVIEW_PASS"
    ):
        raise ValueError("OPAQUE_IDENTITY_ACTIVATION_REVIEW_DECISION")
    review_bindings = review.get("bindings", {})
    required_review_bindings = {
        "preaccess_lock": {
            "path": bindings.get("preaccess_lock", {}).get("path"),
            "sha256": sha256_file(lock_path),
        },
        "candidate_lock": contract_binding,
        "opaque_transport": bindings.get("opaque_transport", {}),
        "runtime_lock": bindings.get("runtime_lock", {}),
    }
    for name, expected in required_review_bindings.items():
        if review_bindings.get(name) != expected:
            raise ValueError(
                f"OPAQUE_IDENTITY_ACTIVATION_REVIEW_{name.upper()}"
            )
    validation_binding = review_bindings.get("machine_validation", {})
    validation_path = (
        lock_path.parents[3] / validation_binding.get("path", "")
    )
    if (
        not validation_path.is_file()
        or sha256_file(validation_path) != validation_binding.get("sha256")
    ):
        raise ValueError("OPAQUE_IDENTITY_ACTIVATION_VALIDATION_BINDING")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if (
        validation.get("decision")
        != "PREACCESS_LOCK_MACHINE_VALIDATION_PASS"
        or validation.get("errors") != []
    ):
        raise ValueError("OPAQUE_IDENTITY_ACTIVATION_VALIDATION_DECISION")
    runtime_binding = bindings.get("runtime_lock", {})
    runtime_path = lock_path.parents[3] / runtime_binding.get("path", "")
    if (
        not runtime_path.is_file()
        or sha256_file(runtime_path) != runtime_binding.get("sha256")
    ):
        raise ValueError("OPAQUE_IDENTITY_ACTIVATION_RUNTIME_BINDING")
    validate_runtime_lock(lock_path.parents[3], runtime_path)
    authority = signature.get("authority", {})
    if authority != {
        "opaque_identity_extraction": True,
        "rgb_algorithm": False,
        "android": False,
    }:
        raise ValueError("OPAQUE_IDENTITY_ACTIVATION_AUTHORITY")
    if lock.get("execution_authority") != {
        "opaque_identity_extraction": False,
        "rgb_algorithm": False,
        "android": False,
    }:
        raise ValueError("PREACCESS_LOCK_AUTHORITY_MUTATED")
    return signature


def openloris_extract(
    lock: dict[str, Any], output: Path, claim: Path, progress: Path
) -> dict[str, Any]:
    import py7zr

    segment = lock["segments"][0]
    targets = [
        segment["guard_before"]["path"],
        *[row["rgb_member_path"] for row in segment["frame_inventory"]],
        segment["guard_after"]["path"],
    ]
    exclusive_claim(
        claim,
        {
            "protocol_id": lock["protocol_id"],
            "source": "OPENLORIS_CORRIDOR",
            "targets_sha256": hashlib.sha256("\n".join(targets).encode()).hexdigest(),
            "budget": OPENLORIS_BUDGET,
        },
    )
    remote = BudgetedRemoteRange(
        segment["source_url"],
        int(segment["container"]["length"]),
        OPENLORIS_BUDGET,
        progress,
    )
    output.mkdir(parents=True, exist_ok=False)
    try:
        with py7zr.SevenZipFile(remote, mode="r") as archive:
            archive.extract(path=output, targets=targets)
        actual_paths = sorted(
            path.relative_to(output).as_posix()
            for path in output.rglob("*")
            if path.is_file()
        )
        if actual_paths != sorted(targets):
            raise ValueError("OPENLORIS_TARGET_SET_MISMATCH")
        rows = []
        for relative in targets:
            path = output / Path(relative)
            rows.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        return {
            "decision": "OPENLORIS_OPAQUE_IDENTITY_EXTRACTED",
            "remote_bytes": remote.budget.consumed,
            "budget": OPENLORIS_BUDGET,
            "members": rows,
            "range_ledger": remote.requests,
            "pixel_decode_calls": 0,
            "rgb_algorithm_calls": 0,
        }
    except BudgetExceeded:
        return {
            "decision": "SEGMENT_IDENTITY_NOT_EVALUABLE",
            "reason": "OPENLORIS_BYTE_BUDGET_EXHAUSTED",
            "remote_bytes": remote.budget.consumed,
            "budget": OPENLORIS_BUDGET,
            "range_ledger": remote.requests,
            "pixel_decode_calls": 0,
            "rgb_algorithm_calls": 0,
        }


def dlr_extract(
    lock: dict[str, Any], output: Path, claim: Path, progress: Path
) -> dict[str, Any]:
    segment = lock["segments"][1]
    member = segment["zip_member"]
    exclusive_claim(
        claim,
        {
            "protocol_id": lock["protocol_id"],
            "source": "DLR_RGBD_VICON",
            "member": member["name"],
            "budget": DLR_BUDGET,
        },
    )
    remote = BudgetedRemoteRange(
        segment["outer_url"],
        int(segment["outer_object_bytes"]),
        DLR_BUDGET,
        progress,
    )
    remote.seek(int(member["local_header_offset"]))
    fixed = remote.read(30)
    if len(fixed) != 30 or fixed[:4] != b"PK\x03\x04":
        raise ValueError("DLR_LOCAL_HEADER")
    (
        _signature,
        _version,
        _flags,
        method,
        _mtime,
        _mdate,
        _crc,
        _compressed,
        _uncompressed,
        name_length,
        extra_length,
    ) = struct.unpack("<IHHHHHIIIHH", fixed)
    if method != 8:
        raise ValueError(f"DLR_ZIP_METHOD:{method}")
    variable = remote.read(name_length + extra_length)
    name = variable[:name_length].decode("utf-8")
    if name != member["name"]:
        raise ValueError("DLR_ZIP_MEMBER_IDENTITY")
    decompressor = zlib.decompressobj(-15)
    collector = RosbagImageCollector(DLR_START_NS, DLR_END_NS)
    remaining = int(member["compressed_bytes"])
    try:
        while remaining > 0 and not collector.complete:
            size = min(NETWORK_CHUNK, remaining)
            compressed = remote.read(size)
            if not compressed:
                break
            remaining -= len(compressed)
            collector.feed(decompressor.decompress(compressed))
    except BudgetExceeded:
        pass
    if not collector.complete:
        return {
            "decision": "SEGMENT_IDENTITY_NOT_EVALUABLE",
            "reason": "DLR_BYTE_BUDGET_EXHAUSTED_OR_RGB_GUARD_ABSENT",
            "remote_bytes": remote.budget.consumed,
            "budget": DLR_BUDGET,
            "color_connection": collector.color_connection,
            "selected_frame_count": len(collector.selected),
            "pixel_decode_calls": 0,
            "rgb_algorithm_calls": 0,
            "range_ledger": remote.requests,
        }
    output.mkdir(parents=True, exist_ok=False)
    all_rows = [collector.before, *collector.selected, collector.after]
    ledger = []
    for index, row in enumerate(all_rows):
        assert row is not None
        payload = row.pop("payload")
        payload_path = output / f"{index:04d}.rgb.payload"
        payload_path.write_bytes(payload)
        ledger.append({**row, "payload_path": payload_path.name})
    return {
        "decision": "DLR_OPAQUE_IDENTITY_EXTRACTED",
        "remote_bytes": remote.budget.consumed,
        "budget": DLR_BUDGET,
        "rgb_topic_connection_id": collector.color_connection,
        "guard_before_count": 1,
        "selected_frame_count": len(collector.selected),
        "guard_after_count": 1,
        "frames": ledger,
        "pixel_decode_calls": 0,
        "rgb_algorithm_calls": 0,
        "range_ledger": remote.requests,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--signature", required=True, type=Path)
    parser.add_argument("--source", required=True, choices=["openloris", "dlr"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--claim", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--failure-receipt", required=True, type=Path)
    parser.add_argument("--progress", required=True, type=Path)
    args = parser.parse_args()
    try:
        lock = json.loads(args.lock.read_text(encoding="utf-8"))
        validate_activation(
            args.lock.resolve(), lock, args.signature.resolve()
        )
        result = (
            openloris_extract(lock, args.output, args.claim, args.progress)
            if args.source == "openloris"
            else dlr_extract(lock, args.output, args.claim, args.progress)
        )
        exit_code = 0
    except Exception as error:
        result = {
            "decision": "INVALID_IDENTITY_EXTRACTION_CLOSE_ATTEMPT",
            "error_type": type(error).__name__,
            "pixel_decode_calls": 0,
            "rgb_algorithm_calls": 0,
        }
        exit_code = 1
    terminal_path = args.receipt if exit_code == 0 else args.failure_receipt
    write_json_atomic(terminal_path, result)
    write_json_atomic(
        args.progress,
        {
            "phase": "OPAQUE_RANGE_ACQUISITION",
            "completed_units": result.get("remote_bytes", 0),
            "total_units": result.get("budget", 1),
            "throughput": 0,
            "eta_seconds": 0,
            "last_progress_at": datetime.now(timezone.utc).isoformat(),
            "status": result["decision"],
            "terminal": result["decision"],
            "completed": True,
        },
    )
    print(json.dumps({k: v for k, v in result.items() if k != "range_ledger"}))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
