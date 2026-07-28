from __future__ import annotations

import bz2
from dataclasses import dataclass, field
import struct
from typing import Any, Callable


ROSBAG_MAGIC = b"#ROSBAG V2.0\n"
MAX_HEADER_BYTES = 1 << 20
MAX_CONNECTION_DATA_BYTES = 1 << 20
MAX_SELECTED_ROWS_PER_CONNECTION = 1_000_000


def parse_fields(raw: bytes) -> dict[str, bytes]:
    fields: dict[str, bytes] = {}
    offset = 0
    while offset < len(raw):
        if offset + 4 > len(raw):
            raise ValueError("FIELD_LENGTH_TRUNCATED")
        length = struct.unpack_from("<I", raw, offset)[0]
        offset += 4
        if offset + length > len(raw):
            raise ValueError("FIELD_DATA_TRUNCATED")
        field = raw[offset : offset + length]
        offset += length
        key, separator, value = field.partition(b"=")
        if not separator or not key:
            raise ValueError("FIELD_FORMAT")
        fields[key.decode("ascii")] = value
    return fields


def u32(value: bytes) -> int:
    if len(value) != 4:
        raise ValueError("U32_SIZE")
    return struct.unpack("<I", value)[0]


def time_ns(value: bytes) -> int:
    if len(value) != 8:
        raise ValueError("TIME_SIZE")
    sec, nsec = struct.unpack("<II", value)
    return sec * 1_000_000_000 + nsec


@dataclass
class ConnectionWindow:
    connection_id: int
    count: int = 0
    minimum_bag_timestamp_ns: int | None = None
    maximum_bag_timestamp_ns: int | None = None
    before: dict[str, int] | None = None
    selected: list[dict[str, int]] = field(default_factory=list)
    after: dict[str, int] | None = None

    def observe(
        self,
        *,
        bag_timestamp_ns: int,
        chunk_logical_offset: int,
        inner_record_offset: int,
        serialized_bytes: int,
        start_ns: int,
        end_ns: int,
    ) -> None:
        self.count += 1
        self.minimum_bag_timestamp_ns = (
            bag_timestamp_ns
            if self.minimum_bag_timestamp_ns is None
            else min(self.minimum_bag_timestamp_ns, bag_timestamp_ns)
        )
        self.maximum_bag_timestamp_ns = (
            bag_timestamp_ns
            if self.maximum_bag_timestamp_ns is None
            else max(self.maximum_bag_timestamp_ns, bag_timestamp_ns)
        )
        row = {
            "connection_id": self.connection_id,
            "bag_timestamp_ns": bag_timestamp_ns,
            "chunk_logical_offset": chunk_logical_offset,
            "inner_record_offset": inner_record_offset,
            "serialized_bytes": serialized_bytes,
        }
        if bag_timestamp_ns < start_ns:
            self.before = row
        elif bag_timestamp_ns < end_ns:
            if len(self.selected) >= MAX_SELECTED_ROWS_PER_CONNECTION:
                raise ValueError("SELECTED_ROW_LIMIT")
            self.selected.append(row)
        elif self.after is None:
            self.after = row


class InnerMessageParser:
    def __init__(
        self,
        *,
        chunk_logical_offset: int,
        on_message: Callable[..., None],
    ) -> None:
        self.chunk_logical_offset = chunk_logical_offset
        self.on_message = on_message
        self.buffer = bytearray()
        self.logical_offset = 0
        self.current: dict[str, Any] | None = None
        self.maximum_buffer_observed = 0
        self.last_complete_record_offset: int | None = None
        self.message_count = 0
        self.minimum_bag_timestamp_ns: int | None = None
        self.maximum_bag_timestamp_ns: int | None = None

    def feed(self, data: bytes) -> None:
        self.buffer.extend(data)
        self.maximum_buffer_observed = max(
            self.maximum_buffer_observed, len(self.buffer)
        )
        while True:
            if self.current is None:
                if len(self.buffer) < 4:
                    return
                header_length = struct.unpack_from("<I", self.buffer, 0)[0]
                if header_length > MAX_HEADER_BYTES:
                    raise ValueError("INNER_HEADER_LIMIT")
                if len(self.buffer) < 4 + header_length + 4:
                    return
                record_start = self.logical_offset
                header = parse_fields(bytes(self.buffer[4 : 4 + header_length]))
                data_length = struct.unpack_from(
                    "<I", self.buffer, 4 + header_length
                )[0]
                prefix = 8 + header_length
                del self.buffer[:prefix]
                self.logical_offset += prefix
                operation = header.get("op", b"\x00")[0]
                self.current = {
                    "record_start": record_start,
                    "header": header,
                    "operation": operation,
                    "data_length": data_length,
                    "remaining": data_length,
                }
            current = self.current
            assert current is not None
            if not self.buffer and current["remaining"]:
                return
            consumed = min(len(self.buffer), current["remaining"])
            del self.buffer[:consumed]
            self.logical_offset += consumed
            current["remaining"] -= consumed
            if current["remaining"]:
                return
            if current["operation"] == 0x02:
                header = current["header"]
                timestamp = time_ns(header["time"])
                self.message_count += 1
                self.minimum_bag_timestamp_ns = (
                    timestamp
                    if self.minimum_bag_timestamp_ns is None
                    else min(self.minimum_bag_timestamp_ns, timestamp)
                )
                self.maximum_bag_timestamp_ns = (
                    timestamp
                    if self.maximum_bag_timestamp_ns is None
                    else max(self.maximum_bag_timestamp_ns, timestamp)
                )
                self.on_message(
                    connection_id=u32(header["conn"]),
                    bag_timestamp_ns=timestamp,
                    chunk_logical_offset=self.chunk_logical_offset,
                    inner_record_offset=current["record_start"],
                    serialized_bytes=current["data_length"],
                )
            self.last_complete_record_offset = current["record_start"]
            self.current = None

    @property
    def partial_bytes(self) -> int:
        remaining = self.current["remaining"] if self.current else 0
        return len(self.buffer) + remaining


class ChunkConsumer:
    def __init__(
        self,
        *,
        compression: str,
        chunk_logical_offset: int,
        on_message: Callable[..., None],
    ) -> None:
        self.compression = compression
        self.inner = InnerMessageParser(
            chunk_logical_offset=chunk_logical_offset,
            on_message=on_message,
        )
        if compression == "bz2":
            self.decompressor: bz2.BZ2Decompressor | None = bz2.BZ2Decompressor()
        elif compression == "none":
            self.decompressor = None
        else:
            raise ValueError("UNSUPPORTED_CHUNK_COMPRESSION")
        self.uncompressed_bytes = 0

    def feed(self, data: bytes) -> None:
        if self.decompressor is None:
            self.uncompressed_bytes += len(data)
            self.inner.feed(data)
            return
        offset = 0
        while offset < len(data):
            piece = data[offset : offset + 65_536]
            offset += len(piece)
            output = self.decompressor.decompress(piece, max_length=1 << 20)
            while output:
                self.uncompressed_bytes += len(output)
                self.inner.feed(output)
                if self.decompressor.eof or self.decompressor.needs_input:
                    break
                output = self.decompressor.decompress(b"", max_length=1 << 20)

    def finish(self) -> None:
        if self.decompressor is not None and not self.decompressor.eof:
            raise ValueError("CHUNK_BZ2_NOT_EOF")
        if self.inner.partial_bytes != 0:
            raise ValueError("INNER_RECORD_PARTIAL")


class StreamingBagIndexer:
    def __init__(self, *, start_ns: int, end_ns: int) -> None:
        self.start_ns = start_ns
        self.end_ns = end_ns
        self.buffer = bytearray()
        self.magic_complete = False
        self.logical_offset = 0
        self.current: dict[str, Any] | None = None
        self.top_level_records: list[dict[str, Any]] = []
        self.connections: dict[int, dict[str, Any]] = {}
        self.windows: dict[int, ConnectionWindow] = {}
        self.operation_counts: dict[int, int] = {}
        self.chunk_count = 0
        self.chunk_records: list[dict[str, Any]] = []
        self.highest_observed_timestamp_ns: int | None = None
        self.last_complete_record: dict[str, Any] | None = None
        self.maximum_buffer_observed = 0
        self.maximum_inner_buffer_observed = 0

    def _on_message(self, **row: int) -> None:
        connection_id = row["connection_id"]
        timestamp = row["bag_timestamp_ns"]
        self.highest_observed_timestamp_ns = (
            timestamp
            if self.highest_observed_timestamp_ns is None
            else max(self.highest_observed_timestamp_ns, timestamp)
        )
        window = self.windows.setdefault(
            connection_id, ConnectionWindow(connection_id)
        )
        observation = {
            key: value for key, value in row.items() if key != "connection_id"
        }
        window.observe(
            **observation,
            start_ns=self.start_ns,
            end_ns=self.end_ns,
        )

    def feed(self, data: bytes) -> None:
        self.buffer.extend(data)
        self.maximum_buffer_observed = max(
            self.maximum_buffer_observed, len(self.buffer)
        )
        while True:
            if not self.magic_complete:
                if len(self.buffer) < len(ROSBAG_MAGIC):
                    return
                if bytes(self.buffer[: len(ROSBAG_MAGIC)]) != ROSBAG_MAGIC:
                    raise ValueError("ROSBAG_MAGIC")
                del self.buffer[: len(ROSBAG_MAGIC)]
                self.logical_offset += len(ROSBAG_MAGIC)
                self.magic_complete = True
            if self.current is None:
                if len(self.buffer) < 4:
                    return
                header_length = struct.unpack_from("<I", self.buffer, 0)[0]
                if header_length > MAX_HEADER_BYTES:
                    raise ValueError("TOP_HEADER_LIMIT")
                if len(self.buffer) < 4 + header_length + 4:
                    return
                record_start = self.logical_offset
                header = parse_fields(bytes(self.buffer[4 : 4 + header_length]))
                data_length = struct.unpack_from(
                    "<I", self.buffer, 4 + header_length
                )[0]
                prefix = 8 + header_length
                del self.buffer[:prefix]
                self.logical_offset += prefix
                operation = header.get("op", b"\x00")[0]
                consumer = None
                retain = operation == 0x07
                if operation == 0x05:
                    compression = header["compression"].decode("ascii")
                    consumer = ChunkConsumer(
                        compression=compression,
                        chunk_logical_offset=record_start,
                        on_message=self._on_message,
                    )
                    self.chunk_count += 1
                if retain and data_length > MAX_CONNECTION_DATA_BYTES:
                    raise ValueError("CONNECTION_DATA_LIMIT")
                self.current = {
                    "record_start": record_start,
                    "header": header,
                    "operation": operation,
                    "data_length": data_length,
                    "remaining": data_length,
                    "retained": bytearray(),
                    "consumer": consumer,
                }
                self.operation_counts[operation] = (
                    self.operation_counts.get(operation, 0) + 1
                )
            current = self.current
            assert current is not None
            if not self.buffer and current["remaining"]:
                return
            consumed = min(len(self.buffer), current["remaining"])
            piece = bytes(self.buffer[:consumed])
            del self.buffer[:consumed]
            self.logical_offset += consumed
            current["remaining"] -= consumed
            if current["operation"] == 0x07:
                current["retained"].extend(piece)
            elif current["consumer"] is not None:
                current["consumer"].feed(piece)
            if current["remaining"]:
                return
            consumer = current["consumer"]
            if consumer is not None:
                consumer.finish()
                self.maximum_inner_buffer_observed = max(
                    self.maximum_inner_buffer_observed,
                    consumer.inner.maximum_buffer_observed,
                )
                self.chunk_records.append(
                    {
                        "chunk_logical_offset": current["record_start"],
                        "compression": consumer.compression,
                        "compressed_data_length": current["data_length"],
                        "declared_uncompressed_length": u32(
                            current["header"]["size"]
                        ),
                        "actual_uncompressed_length": consumer.uncompressed_bytes,
                        "message_count": consumer.inner.message_count,
                        "minimum_bag_timestamp_ns": (
                            consumer.inner.minimum_bag_timestamp_ns
                        ),
                        "maximum_bag_timestamp_ns": (
                            consumer.inner.maximum_bag_timestamp_ns
                        ),
                    }
                )
            if current["operation"] == 0x07:
                header = current["header"]
                data_fields = parse_fields(bytes(current["retained"]))
                connection_id = u32(header["conn"])
                self.connections[connection_id] = {
                    "connection_id": connection_id,
                    "topic": header.get("topic", b"").decode("utf-8"),
                    "type": data_fields.get("type", b"").decode("utf-8"),
                    "md5sum": data_fields.get("md5sum", b"").decode("ascii"),
                    "record_logical_offset": current["record_start"],
                }
            record_row = {
                "record_logical_offset": current["record_start"],
                "operation": current["operation"],
                "data_length": current["data_length"],
            }
            self.top_level_records.append(record_row)
            self.last_complete_record = record_row
            self.current = None

    @property
    def partial_buffer_length(self) -> int:
        remaining = self.current["remaining"] if self.current else 0
        return len(self.buffer) + remaining

    def terminal(self) -> dict[str, Any]:
        candidates = []
        for connection_id, connection in sorted(self.connections.items()):
            topic = connection["topic"]
            message_type = connection["type"]
            if (
                message_type == "sensor_msgs/Image"
                and "/color/" in topic
                and "depth" not in topic
            ):
                window = self.windows.get(
                    connection_id, ConnectionWindow(connection_id)
                )
                candidates.append(
                    {
                        **connection,
                        "message_count": window.count,
                        "minimum_bag_timestamp_ns": window.minimum_bag_timestamp_ns,
                        "maximum_bag_timestamp_ns": window.maximum_bag_timestamp_ns,
                        "before": window.before,
                        "selected": window.selected,
                        "after": window.after,
                    }
                )
        return {
            "magic_complete": self.magic_complete,
            "partial_buffer_length": self.partial_buffer_length,
            "last_complete_record": self.last_complete_record,
            "highest_observed_timestamp_ns": self.highest_observed_timestamp_ns,
            "top_level_record_count": len(self.top_level_records),
            "chunk_count": self.chunk_count,
            "connection_count": len(self.connections),
            "operation_counts": {
                str(key): value for key, value in sorted(self.operation_counts.items())
            },
            "candidate_color_connections": candidates,
            "maximum_buffer_observed": self.maximum_buffer_observed,
            "maximum_inner_buffer_observed": self.maximum_inner_buffer_observed,
            "pixel_firewall": {
                "rgb_payload_files_written": 0,
                "rgb_payload_bytes_retained": 0,
                "rgb_per_frame_payload_hash_calls": 0,
                "image_decode_calls": 0,
                "image_visualization_calls": 0,
                "rgb_algorithm_calls": 0,
            },
            "resumability": {
                "mode": "SEQUENTIAL_ONLY",
                "random_access": False,
                "restartable_checkpoint_count": 0,
            },
        }
