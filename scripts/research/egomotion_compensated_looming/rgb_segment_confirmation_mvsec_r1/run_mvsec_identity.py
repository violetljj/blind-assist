from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
from typing import Any
import urllib.request

from mvsec_target_chunks import (
    MAGIC,
    OP_CHUNK,
    decompress_chunk,
    fields,
    image_messages_from_chunk,
    pair_window,
    parse_final_index,
    record,
    select_target_chunks,
    u32,
    u64,
)


CANDIDATE_PATH = (
    "artifacts.local/evidence/rcle_rgb_segment_confirmation_mvsec_r1/"
    "candidate_lock.v1.json"
)
REVIEW_PATH = (
    "artifacts.local/evidence/rcle_rgb_segment_confirmation_mvsec_r1/"
    "candidate_independent_review.v1.json"
)
ACTIVATION_PATH = (
    "artifacts.local/evidence/rcle_rgb_segment_confirmation_mvsec_r1/"
    "identity_activation.v1.json"
)
IMAGE_TOPIC = "/davis/left/image_raw"
IMAGE_TYPE = "sensor_msgs/Image"
IMAGE_MD5 = "060021388200f6f0f447d0fcd9c64743"


class IdentityFailure(RuntimeError):
    pass


class TransportFailure(RuntimeError):
    pass


def terminal_decision(error: BaseException) -> str:
    return (
        "INVALID_SOURCE_OR_CAPTURE_IDENTITY"
        if isinstance(error, IdentityFailure)
        else "MVSEC_RGB_IDENTITY_NOT_EVALUABLE"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        raise


class RangeLedger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError("RANGE_LEDGER_EXISTS")
        self.path = path
        self.previous: str | None = None
        self.rows = 0
        self.bytes = 0

    def append(self, row: dict[str, Any]) -> None:
        payload = {
            **row,
            "sequence": self.rows,
            "previous_row_sha256": self.previous,
        }
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        payload["row_sha256"] = hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        self.previous = payload["row_sha256"]
        self.rows += 1
        self.bytes += int(row["accounted_bytes"])

    def binding(self, repo: Path) -> dict[str, Any]:
        return {
            "path": self.path.relative_to(repo).as_posix(),
            "sha256": sha256_file(self.path),
            "rows": self.rows,
            "head_sha256": self.previous,
            "accounted_bytes": self.bytes,
            "retry_count": 0,
        }


class ExactRangeReader:
    def __init__(
        self,
        *,
        url: str,
        expected_bytes: int,
        expected_etag: str,
        maximum_bytes: int,
        ledger: RangeLedger,
        opener=urllib.request.urlopen,
    ) -> None:
        self.url = url
        self.expected_bytes = expected_bytes
        self.expected_etag = expected_etag
        self.maximum_bytes = maximum_bytes
        self.ledger = ledger
        self.opener = opener
        self.request_count = 0

    def head(self) -> dict[str, Any]:
        request = urllib.request.Request(
            self.url,
            method="HEAD",
            headers={
                "Accept-Encoding": "identity",
                "User-Agent": "BlindAssist-RCLE-MVSEC-RGB-Identity-R1",
            },
        )
        try:
            with self.opener(request, timeout=30) as response:
                status = int(response.status)
                final_url = response.geturl()
                headers = response.headers
        except BaseException as error:
            raise TransportFailure("HEAD_TRANSPORT") from error
        if (
            status != 200
            or final_url != self.url
            or int(headers["Content-Length"]) != self.expected_bytes
            or headers.get("ETag") != self.expected_etag
            or headers.get("Accept-Ranges") != "bytes"
            or headers.get("Content-Encoding") not in (None, "", "identity")
        ):
            raise IdentityFailure("HEAD_IDENTITY")
        return {
            "status": status,
            "content_length": self.expected_bytes,
            "etag": headers.get("ETag"),
            "last_modified": headers.get("Last-Modified"),
            "accept_ranges": headers.get("Accept-Ranges"),
        }

    def fetch(self, start: int, end: int, label: str) -> bytes:
        requested = end - start + 1
        if (
            requested <= 0
            or start < 0
            or end >= self.expected_bytes
        ):
            raise ValueError("RANGE_BOUNDS")
        # Pre-authorize the one-byte sentinel used to prove that the server did
        # not return more than the exact requested range.
        if self.ledger.bytes + requested + 1 > self.maximum_bytes:
            raise TransportFailure("REMOTE_BYTE_CAP")
        request = urllib.request.Request(
            self.url,
            headers={
                "Range": f"bytes={start}-{end}",
                "Accept-Encoding": "identity",
                "Connection": "close",
                "User-Agent": "BlindAssist-RCLE-MVSEC-RGB-Identity-R1",
            },
        )
        self.request_count += 1
        try:
            with self.opener(request, timeout=120) as response:
                status = int(response.status)
                final_url = response.geturl()
                content_range = response.headers.get("Content-Range")
                content_encoding = response.headers.get("Content-Encoding")
                body = response.read(requested + 1)
        except BaseException as error:
            self.ledger.append(
                {
                    "label": label,
                    "start": start,
                    "end": end,
                    "requested_bytes": requested,
                    "accounted_bytes": requested,
                    "status": "TERMINAL_TRANSPORT_FAILURE",
                    "attempt": 1,
                    "error_type": type(error).__name__,
                }
            )
            raise TransportFailure("RANGE_TRANSPORT_NO_RETRY") from error
        expected_range = f"bytes {start}-{end}/{self.expected_bytes}"
        passed = (
            status == 206
            and final_url == self.url
            and content_range == expected_range
            and content_encoding in (None, "", "identity")
            and len(body) == requested
        )
        self.ledger.append(
            {
                "label": label,
                "start": start,
                "end": end,
                "requested_bytes": requested,
                "accounted_bytes": len(body),
                "status": "PASS" if passed else "TERMINAL_RANGE_IDENTITY_FAILURE",
                "attempt": 1,
                "http_status": status,
                "content_range": content_range,
                "body_sha256": hashlib.sha256(body).hexdigest(),
            }
        )
        if not passed:
            raise IdentityFailure("RANGE_IDENTITY")
        return body


def load_bound(repo: Path, binding: dict[str, str]) -> Any:
    path = repo / binding["path"]
    if not path.is_file() or sha256_file(path) != binding["sha256"]:
        raise IdentityFailure("LOCAL_BINDING")
    return json.loads(path.read_text(encoding="utf-8"))


def geometry_timestamps(
    geometry_source: dict[str, Any],
    *,
    start_ns: int,
    end_ns: int,
) -> list[int]:
    values = [
        int(row["timestamp_ns"])
        for row in geometry_source["depth"]["frames"]
        if start_ns <= int(row["timestamp_ns"]) < end_ns
    ]
    if not values or any(left >= right for left, right in zip(values, values[1:])):
        raise IdentityFailure("GEOMETRY_TIMESTAMP_IDENTITY")
    return values


def validate_geometry_window(
    result: dict[str, Any],
    *,
    window: dict[str, Any],
) -> None:
    matching = [
        row
        for row in result["window_summaries"]
        if row["window_id"] == window["window_id"]
    ]
    if len(matching) != 1:
        raise IdentityFailure("GEOMETRY_WINDOW_MISSING")
    row = matching[0]
    if (
        row["role"] != window["role"]
        or int(Decimal(row["start_timestamp_s"]) * Decimal(1_000_000_000))
        != window["start_ns"]
        or int(Decimal(row["end_timestamp_s"]) * Decimal(1_000_000_000))
        != window["end_ns"]
    ):
        raise IdentityFailure("GEOMETRY_WINDOW_IDENTITY")


def validate_authority(
    repo: Path,
    config_path: Path,
    activation_path: Path,
) -> dict[str, Any]:
    candidate_path = repo / CANDIDATE_PATH
    review_path = repo / REVIEW_PATH
    if not candidate_path.is_file() or not review_path.is_file():
        raise IdentityFailure("REVIEW_GATE_MISSING")
    candidate_sha = sha256_file(candidate_path)
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    runner_path = Path(__file__).resolve()
    expected_candidate_runtime_bindings = {
        "config": {
            "path": config_path.relative_to(repo).as_posix(),
            "sha256": sha256_file(config_path),
        },
        "runner": {
            "path": runner_path.relative_to(repo).as_posix(),
            "sha256": sha256_file(runner_path),
        },
    }
    for name, expected in expected_candidate_runtime_bindings.items():
        if candidate.get("bindings", {}).get(name) != expected:
            raise IdentityFailure("CANDIDATE_RUNTIME_BINDING")
    for name in ("parser", "tests"):
        binding = candidate.get("bindings", {}).get(name)
        if (
            not isinstance(binding, dict)
            or not (repo / binding.get("path", "")).is_file()
            or sha256_file(repo / binding["path"]) != binding.get("sha256")
        ):
            raise IdentityFailure("CANDIDATE_RUNTIME_BINDING")
    if (
        candidate.get("decision") != "IDENTITY_EXTRACTION_NOT_AUTHORIZED"
        or candidate.get("execution_authority") is not False
        or review.get("decision")
        != "MVSEC_RGB_IDENTITY_CANDIDATE_REVIEW_PASS"
        or review.get("candidate_sha256") != candidate_sha
        or review.get("execution_authority") is not False
    ):
        raise IdentityFailure("REVIEW_GATE")
    expected = {
        "schema_version": "rcle_mvsec_rgb_identity_activation.v1",
        "decision": "MVSEC_RGB_IDENTITY_ONE_SHOT_AUTHORIZED",
        "execution_authority": True,
        "bindings": {
            "candidate": {
                "path": CANDIDATE_PATH,
                "sha256": candidate_sha,
            },
            "review": {
                "path": REVIEW_PATH,
                "sha256": sha256_file(review_path),
            },
            "config": {
                "path": config_path.relative_to(repo).as_posix(),
                "sha256": sha256_file(config_path),
            },
            "runner": {
                "path": runner_path.relative_to(repo).as_posix(),
                "sha256": sha256_file(runner_path),
            },
            "parser": candidate["bindings"]["parser"],
            "tests": candidate["bindings"]["tests"],
        },
        "authority": {
            "bag_index_and_target_chunks": True,
            "selected_mono8_payload_materialization": True,
            "image_decode": False,
            "rectification": False,
            "rgb_algorithm": False,
            "android": False,
        },
    }
    activation = json.loads(activation_path.read_text(encoding="utf-8"))
    if activation != expected:
        raise IdentityFailure("ACTIVATION_IDENTITY")
    return candidate


def inspect_capture(
    repo: Path,
    namespace: Path,
    capture: dict[str, Any],
    geometry_source: dict[str, Any],
) -> dict[str, Any]:
    capture_root = namespace / capture["capture_id"]
    ledger = RangeLedger(capture_root / "range_ledger.jsonl")
    reader = ExactRangeReader(
        url=capture["data_bag"]["url"],
        expected_bytes=capture["data_bag"]["bytes"],
        expected_etag=capture["data_bag"]["etag"],
        maximum_bytes=capture["transport"]["maximum_remote_bytes"],
        ledger=ledger,
    )
    head = reader.head()
    opening = reader.fetch(0, 8191, "opening")
    if opening[: len(MAGIC)] != MAGIC:
        raise IdentityFailure("ROSBAG_MAGIC")
    bag_header, _, _ = record(opening, len(MAGIC))
    index_pos = u64(bag_header["index_pos"])
    connection_count = u32(bag_header["conn_count"])
    chunk_count = u32(bag_header["chunk_count"])
    final_index_bytes = capture["data_bag"]["bytes"] - index_pos
    if (
        index_pos <= 0
        or final_index_bytes <= 0
        or final_index_bytes > capture["transport"]["maximum_index_bytes"]
    ):
        raise IdentityFailure("FINAL_INDEX_BOUND")
    final_index = reader.fetch(
        index_pos,
        capture["data_bag"]["bytes"] - 1,
        "final-index",
    )
    connections, chunks = parse_final_index(
        final_index,
        expected_connection_count=connection_count,
        expected_chunk_count=chunk_count,
    )
    matching = [
        row for row in connections.values() if row["topic"] == IMAGE_TOPIC
    ]
    if (
        len(matching) != 1
        or matching[0]["type"] != IMAGE_TYPE
        or matching[0]["md5sum"] != IMAGE_MD5
    ):
        raise IdentityFailure("IMAGE_CONNECTION_IDENTITY")
    connection = matching[0]
    selected_chunks = select_target_chunks(
        chunks,
        connection_id=connection["connection_id"],
        window_start_ns=capture["window"]["start_ns"],
        window_end_ns=capture["window"]["end_ns"],
    )
    messages = []
    chunk_rows = []
    for selected in selected_chunks:
        position = selected["chunk_pos"]
        prefix = reader.fetch(position, position + 3, "chunk-header-length")
        header_size = struct.unpack("<I", prefix)[0]
        header_and_size = reader.fetch(
            position + 4,
            position + 4 + header_size + 3,
            "chunk-header",
        )
        chunk_header = fields(header_and_size[:header_size])
        compressed_size = struct.unpack_from(
            "<I", header_and_size, header_size
        )[0]
        if chunk_header["op"][0] != OP_CHUNK:
            raise IdentityFailure("CHUNK_POSITION")
        data_start = position + 4 + header_size + 4
        compressed = reader.fetch(
            data_start,
            data_start + compressed_size - 1,
            "chunk-payload",
        )
        uncompressed = decompress_chunk(
            compression=chunk_header["compression"].decode("ascii"),
            compressed=compressed,
            declared_uncompressed_bytes=u32(chunk_header["size"]),
        )
        chunk_messages = image_messages_from_chunk(
            uncompressed,
            connection_id=connection["connection_id"],
        )
        messages.extend(chunk_messages)
        chunk_rows.append(
            {
                "chunk_index": selected["chunk_index"],
                "chunk_pos": position,
                "start_ns": selected["start_ns"],
                "end_ns": selected["end_ns"],
                "compression": chunk_header["compression"].decode("ascii"),
                "compressed_bytes": compressed_size,
                "uncompressed_bytes": len(uncompressed),
                "image_message_count": len(chunk_messages),
            }
        )
    unique = {
        (message.header_timestamp_ns, message.serialized_sha256): message
        for message in messages
    }
    ordered = [unique[key] for key in sorted(unique)]
    geometry = geometry_timestamps(
        geometry_source,
        start_ns=capture["window"]["start_ns"],
        end_ns=capture["window"]["end_ns"],
    )
    paired = pair_window(
        ordered,
        geometry_timestamps_ns=geometry,
        maximum_delta_ns=capture["pairing"]["maximum_abs_delta_ns"],
    )
    retained = paired["retained"]
    frame_ids = {message.frame_id for message in retained}
    encodings = {message.encoding for message in retained}
    layouts = {
        (
            message.height,
            message.width,
            message.step,
            message.is_bigendian,
            len(message.payload),
        )
        for message in retained
    }
    if (
        len(frame_ids) != 1
        or encodings != {"mono8"}
        or layouts != {(260, 346, 346, 0, 89_960)}
    ):
        raise IdentityFailure("IMAGE_LAYOUT_IDENTITY")
    frame_root = capture_root / "frames"
    frame_root.mkdir(parents=True, exist_ok=False)
    rows = []
    selected_set = {message.header_timestamp_ns for message in paired["selected"]}
    for message in retained:
        if message.header_timestamp_ns == paired["before"].header_timestamp_ns:
            role = "GUARD_BEFORE"
        elif message.header_timestamp_ns == paired["after"].header_timestamp_ns:
            role = "GUARD_AFTER"
        elif message.header_timestamp_ns in selected_set:
            role = "SELECTED"
        else:
            raise AssertionError("UNREACHABLE_FRAME_ROLE")
        relative = (
            Path(capture["capture_id"])
            / "frames"
            / f"{message.header_timestamp_ns}.mono8"
        )
        output = namespace / relative
        descriptor = os.open(
            output,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(message.payload)
            stream.flush()
            os.fsync(stream.fileno())
        rows.append(
            {
                "role": role,
                "header_timestamp_ns": message.header_timestamp_ns,
                "bag_timestamp_ns": message.bag_timestamp_ns,
                "sequence": message.sequence,
                "frame_id": message.frame_id,
                "encoding": message.encoding,
                "height": message.height,
                "width": message.width,
                "step": message.step,
                "serialized_sha256": message.serialized_sha256,
                "payload_path": relative.as_posix(),
                "payload_sha256": message.payload_sha256,
                "payload_bytes": len(message.payload),
            }
        )
    ledger_path = capture_root / "frame_ledger.jsonl"
    with ledger_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "capture_id": capture["capture_id"],
        "window": capture["window"],
        "head": head,
        "bag": {
            "index_pos": index_pos,
            "connection_count": connection_count,
            "chunk_count": chunk_count,
            "final_index_bytes": final_index_bytes,
        },
        "image_connection": connection,
        "selected_chunks": chunk_rows,
        "geometry_timestamp_count": len(geometry),
        "selected_frame_count": len(paired["selected"]),
        "guard_frame_count": 2,
        "maximum_abs_pair_delta_ns": paired["maximum_abs_delta_ns"],
        "frame_id": next(iter(frame_ids)),
        "encoding": "mono8",
        "layout": {
            "height": 260,
            "width": 346,
            "step": 346,
            "payload_bytes": 89_960,
        },
        "frame_ledger": {
            "path": ledger_path.relative_to(repo).as_posix(),
            "sha256": sha256_file(ledger_path),
            "rows": len(rows),
        },
        "range_ledger": ledger.binding(repo),
        "materialized_payload_files": len(rows),
        "materialized_payload_bytes": sum(row["payload_bytes"] for row in rows),
        "image_decode_calls": 0,
        "rectification_calls": 0,
        "rgb_algorithm_calls": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--activation", required=True, type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = (
        args.config.resolve()
        if args.config.is_absolute()
        else (repo / args.config).resolve()
    )
    activation_path = (
        args.activation.resolve()
        if args.activation.is_absolute()
        else (repo / args.activation).resolve()
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config.get("schema_version") != "rcle_mvsec_rgb_identity_config.v1"
        or config.get("protocol_id")
        != "RCLE_RGB_SEGMENT_CONFIRMATION_MVSEC_R1"
        or config.get("execution_authority") is not False
        or config.get("claim_namespace")
        != "artifacts.local/evidence/rcle_rgb_segment_confirmation_mvsec_r1/identity_run_v1"
    ):
        raise IdentityFailure("CONFIG_IDENTITY")
    calibration_path = repo / config["camera_relation"]["calibration_path"]
    if (
        not calibration_path.is_file()
        or sha256_file(calibration_path)
        != config["camera_relation"]["calibration_sha256"]
    ):
        raise IdentityFailure("CALIBRATION_BINDING")
    validate_authority(repo, config_path, activation_path)
    namespace = repo / config["claim_namespace"]
    if namespace.exists():
        raise FileExistsError("CLAIM_NAMESPACE_EXISTS")
    write_json_exclusive(
        namespace / "claim.json",
        {
            "schema_version": "rcle_mvsec_rgb_identity_claim.v1",
            "config_sha256": sha256_file(config_path),
            "activation_sha256": sha256_file(activation_path),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "retry_or_resume_authority": False,
        },
    )
    try:
        captures = []
        for capture in config["captures"]:
            geometry_source = load_bound(repo, capture["geometry_source"])
            geometry_result = load_bound(repo, capture["geometry_result"])
            validate_geometry_window(
                geometry_result,
                window=capture["window"],
            )
            captures.append(
                inspect_capture(repo, namespace, capture, geometry_source)
            )
        terminal = {
            "schema_version": "rcle_mvsec_rgb_identity_terminal.v1",
            "decision": "MVSEC_RGB_EXACT_CAPTURE_IDENTITY_COMPLETE",
            "captures": captures,
            "camera_relation": config["camera_relation"],
            "adapter": config["adapter"],
            "payload_scope": {
                "selected_windows": 2,
                "guard_frames_per_window": 2,
                "full_bag_downloaded": False,
                "non_target_payload_files": 0,
            },
            "image_decode_calls": 0,
            "rectification_calls": 0,
            "rgb_algorithm_calls": 0,
            "algorithm_execution_authority": False,
            "android_authority": False,
        }
        write_json_exclusive(namespace / "TERMINAL.json", terminal)
        print(json.dumps({"decision": terminal["decision"]}))
        return 0
    except BaseException as error:
        terminal = {
            "schema_version": "rcle_mvsec_rgb_identity_failure.v1",
            "decision": terminal_decision(error),
            "error_type": type(error).__name__,
            "claim_sha256": sha256_file(namespace / "claim.json"),
            "retry_or_resume_authority": False,
            "algorithm_execution_authority": False,
            "android_authority": False,
        }
        write_json_exclusive(namespace / "FAILURE.json", terminal)
        print(json.dumps(terminal))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
