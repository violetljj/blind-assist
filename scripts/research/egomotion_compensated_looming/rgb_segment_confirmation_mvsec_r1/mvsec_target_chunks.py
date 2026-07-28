from __future__ import annotations

import bz2
from dataclasses import dataclass
import hashlib
import struct
from typing import Any, Iterable


MAGIC = b"#ROSBAG V2.0\n"
OP_MSG_DATA = 0x02
OP_CHUNK = 0x05
OP_CHUNK_INFO = 0x06
OP_CONNECTION = 0x07
MAX_CHUNK_UNCOMPRESSED_BYTES = 64 * (1 << 20)


def fields(data: bytes) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    cursor = 0
    while cursor < len(data):
        if cursor + 4 > len(data):
            raise ValueError("FIELD_LENGTH_TRUNCATED")
        size = struct.unpack_from("<I", data, cursor)[0]
        cursor += 4
        value = data[cursor : cursor + size]
        cursor += size
        if len(value) != size or b"=" not in value:
            raise ValueError("FIELD_MALFORMED")
        key, payload = value.split(b"=", 1)
        result[key.decode("ascii")] = payload
    return result


def record(data: bytes, cursor: int) -> tuple[dict[str, bytes], bytes, int]:
    if cursor + 4 > len(data):
        raise ValueError("RECORD_HEADER_LENGTH_TRUNCATED")
    header_size = struct.unpack_from("<I", data, cursor)[0]
    header_start = cursor + 4
    header_end = header_start + header_size
    if header_end + 4 > len(data):
        raise ValueError("RECORD_HEADER_TRUNCATED")
    header = fields(data[header_start:header_end])
    data_size = struct.unpack_from("<I", data, header_end)[0]
    data_start = header_end + 4
    data_end = data_start + data_size
    if data_end > len(data):
        raise ValueError("RECORD_DATA_TRUNCATED")
    return header, data[data_start:data_end], data_end


def u32(value: bytes) -> int:
    if len(value) != 4:
        raise ValueError("U32_SIZE")
    return struct.unpack("<I", value)[0]


def u64(value: bytes) -> int:
    if len(value) != 8:
        raise ValueError("U64_SIZE")
    return struct.unpack("<Q", value)[0]


def time_ns(value: bytes) -> int:
    if len(value) != 8:
        raise ValueError("TIME_SIZE")
    sec, nsec = struct.unpack("<II", value)
    return sec * 1_000_000_000 + nsec


def parse_final_index(
    payload: bytes,
    *,
    expected_connection_count: int,
    expected_chunk_count: int,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    connections: dict[int, dict[str, Any]] = {}
    chunks: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(payload):
        header, data, cursor = record(payload, cursor)
        operation = header["op"][0]
        if operation == OP_CONNECTION:
            connection_id = u32(header["conn"])
            detail = fields(data)
            connections[connection_id] = {
                "connection_id": connection_id,
                "topic": header["topic"].decode("utf-8"),
                "type": detail.get("type", b"").decode("utf-8"),
                "md5sum": detail.get("md5sum", b"").decode("ascii"),
                "message_definition_sha256": hashlib.sha256(
                    detail.get("message_definition", b"")
                ).hexdigest(),
            }
        elif operation == OP_CHUNK_INFO:
            count = u32(header["count"])
            if len(data) != count * 8:
                raise ValueError("CHUNK_INFO_COUNT_MISMATCH")
            counts = {
                struct.unpack_from("<I", data, offset)[0]:
                struct.unpack_from("<I", data, offset + 4)[0]
                for offset in range(0, len(data), 8)
            }
            chunks.append(
                {
                    "chunk_pos": u64(header["chunk_pos"]),
                    "start_ns": time_ns(header["start_time"]),
                    "end_ns": time_ns(header["end_time"]),
                    "counts": counts,
                }
            )
    if len(connections) != expected_connection_count:
        raise ValueError("CONNECTION_COUNT_MISMATCH")
    if len(chunks) != expected_chunk_count:
        raise ValueError("CHUNK_COUNT_MISMATCH")
    chunks.sort(key=lambda row: row["chunk_pos"])
    return connections, chunks


def select_target_chunks(
    chunks: list[dict[str, Any]],
    *,
    connection_id: int,
    window_start_ns: int,
    window_end_ns: int,
) -> list[dict[str, Any]]:
    containing = [
        index
        for index, chunk in enumerate(chunks)
        if chunk["counts"].get(connection_id, 0)
        and chunk["start_ns"] < window_end_ns
        and chunk["end_ns"] >= window_start_ns
    ]
    if not containing:
        raise ValueError("NO_TARGET_CHUNK")
    selected = set(containing)
    first = min(containing)
    last = max(containing)
    for index in range(first - 1, -1, -1):
        if chunks[index]["counts"].get(connection_id, 0):
            selected.add(index)
            break
    for index in range(last + 1, len(chunks)):
        if chunks[index]["counts"].get(connection_id, 0):
            selected.add(index)
            break
    return [
        {"chunk_index": index, **chunks[index]}
        for index in sorted(selected)
    ]


def decompress_chunk(
    *,
    compression: str,
    compressed: bytes,
    declared_uncompressed_bytes: int,
) -> bytes:
    if declared_uncompressed_bytes > MAX_CHUNK_UNCOMPRESSED_BYTES:
        raise ValueError("CHUNK_UNCOMPRESSED_LIMIT")
    if compression == "none":
        output = compressed
    elif compression == "bz2":
        output = bz2.decompress(compressed)
    else:
        raise ValueError("UNSUPPORTED_CHUNK_COMPRESSION")
    if len(output) != declared_uncompressed_bytes:
        raise ValueError("CHUNK_UNCOMPRESSED_SIZE")
    return output


@dataclass(frozen=True)
class ImageMessage:
    bag_timestamp_ns: int
    header_timestamp_ns: int
    sequence: int
    frame_id: str
    height: int
    width: int
    encoding: str
    is_bigendian: int
    step: int
    serialized_sha256: str
    payload_sha256: str
    payload: bytes


def _string(data: bytes, cursor: int) -> tuple[str, int]:
    if cursor + 4 > len(data):
        raise ValueError("STRING_LENGTH_TRUNCATED")
    size = struct.unpack_from("<I", data, cursor)[0]
    cursor += 4
    end = cursor + size
    if end > len(data):
        raise ValueError("STRING_TRUNCATED")
    return data[cursor:end].decode("utf-8"), end


def parse_sensor_image(
    serialized: bytes,
    *,
    bag_timestamp_ns: int,
) -> ImageMessage:
    if len(serialized) < 16:
        raise ValueError("IMAGE_MESSAGE_TRUNCATED")
    cursor = 0
    sequence, sec, nsec = struct.unpack_from("<III", serialized, cursor)
    cursor += 12
    frame_id, cursor = _string(serialized, cursor)
    if cursor + 8 > len(serialized):
        raise ValueError("IMAGE_DIMENSIONS_TRUNCATED")
    height, width = struct.unpack_from("<II", serialized, cursor)
    cursor += 8
    encoding, cursor = _string(serialized, cursor)
    if cursor + 9 > len(serialized):
        raise ValueError("IMAGE_LAYOUT_TRUNCATED")
    is_bigendian = serialized[cursor]
    cursor += 1
    step, data_size = struct.unpack_from("<II", serialized, cursor)
    cursor += 8
    end = cursor + data_size
    if end != len(serialized):
        raise ValueError("IMAGE_PAYLOAD_SIZE")
    payload = serialized[cursor:end]
    return ImageMessage(
        bag_timestamp_ns=bag_timestamp_ns,
        header_timestamp_ns=sec * 1_000_000_000 + nsec,
        sequence=sequence,
        frame_id=frame_id,
        height=height,
        width=width,
        encoding=encoding,
        is_bigendian=is_bigendian,
        step=step,
        serialized_sha256=hashlib.sha256(serialized).hexdigest(),
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        payload=payload,
    )


def image_messages_from_chunk(
    uncompressed: bytes,
    *,
    connection_id: int,
) -> list[ImageMessage]:
    result: list[ImageMessage] = []
    cursor = 0
    while cursor < len(uncompressed):
        header, data, cursor = record(uncompressed, cursor)
        if header["op"][0] != OP_MSG_DATA:
            continue
        if u32(header["conn"]) != connection_id:
            continue
        result.append(
            parse_sensor_image(
                data,
                bag_timestamp_ns=time_ns(header["time"]),
            )
        )
    return result


def pair_window(
    messages: Iterable[ImageMessage],
    *,
    geometry_timestamps_ns: list[int],
    maximum_delta_ns: int,
) -> dict[str, Any]:
    ordered = sorted(messages, key=lambda row: row.header_timestamp_ns)
    if any(
        left.header_timestamp_ns >= right.header_timestamp_ns
        for left, right in zip(ordered, ordered[1:])
    ):
        raise ValueError("IMAGE_TIMESTAMPS_NOT_STRICT")
    selected: list[ImageMessage] = []
    used: set[int] = set()
    for geometry_timestamp_ns in geometry_timestamps_ns:
        candidates = sorted(
            (
                (
                    abs(message.header_timestamp_ns - geometry_timestamp_ns),
                    index,
                    message,
                )
                for index, message in enumerate(ordered)
            ),
            key=lambda row: (row[0], row[2].header_timestamp_ns),
        )
        if not candidates or candidates[0][0] > maximum_delta_ns:
            raise ValueError("IMAGE_GEOMETRY_PAIRING")
        minimum_delta_ns = candidates[0][0]
        nearest = [
            candidate
            for candidate in candidates
            if candidate[0] == minimum_delta_ns
        ]
        if len(nearest) != 1:
            raise ValueError("IMAGE_GEOMETRY_PAIRING_TIE")
        _, index, message = nearest[0]
        if index in used:
            raise ValueError("IMAGE_GEOMETRY_PAIRING_REUSE")
        used.add(index)
        selected.append(message)
    selected_indices = [ordered.index(message) for message in selected]
    if selected_indices != sorted(selected_indices):
        raise ValueError("IMAGE_PAIRING_NOT_MONOTONIC")
    before_index = selected_indices[0] - 1
    after_index = selected_indices[-1] + 1
    if before_index < 0 or after_index >= len(ordered):
        raise ValueError("IMAGE_GUARD_MISSING")
    retained = [ordered[before_index], *selected, ordered[after_index]]
    return {
        "before": ordered[before_index],
        "selected": selected,
        "after": ordered[after_index],
        "retained": retained,
        "maximum_abs_delta_ns": max(
            abs(message.header_timestamp_ns - geometry_timestamp_ns)
            for message, geometry_timestamp_ns in zip(
                selected, geometry_timestamps_ns
            )
        ),
    }
