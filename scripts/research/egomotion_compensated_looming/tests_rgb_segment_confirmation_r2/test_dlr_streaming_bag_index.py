from __future__ import annotations

import bz2
import importlib.util
from pathlib import Path
import struct
import sys
import unittest


REPO = Path(__file__).resolve().parents[4]
MODULE_PATH = (
    REPO
    / "scripts/research/egomotion_compensated_looming"
    / "rgb_segment_confirmation_r2"
    / "dlr_streaming_bag_index.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("dlr_streaming_bag_index", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def field(key: str, value: bytes) -> bytes:
    payload = key.encode("ascii") + b"=" + value
    return struct.pack("<I", len(payload)) + payload


def header(**fields: bytes) -> bytes:
    return b"".join(field(key, value) for key, value in fields.items())


def record(header_bytes: bytes, data: bytes) -> bytes:
    return (
        struct.pack("<I", len(header_bytes))
        + header_bytes
        + struct.pack("<I", len(data))
        + data
    )


def time_value(timestamp_ns: int) -> bytes:
    return struct.pack(
        "<II",
        timestamp_ns // 1_000_000_000,
        timestamp_ns % 1_000_000_000,
    )


def build_bag(*, compression: str, truncate: int = 0) -> bytes:
    inner = b""
    for timestamp, payload in (
        (900, b"a" * 100_000),
        (1_100, b"b" * 200_000),
        (1_500, b"c" * 200_000),
        (2_100, b"d" * 100_000),
    ):
        inner += record(
            header(
                op=b"\x02",
                conn=struct.pack("<I", 7),
                time=time_value(timestamp),
            ),
            payload,
        )
    chunk_data = bz2.compress(inner) if compression == "bz2" else inner
    chunk = record(
        header(
            op=b"\x05",
            compression=compression.encode("ascii"),
            size=struct.pack("<I", len(inner)),
        ),
        chunk_data,
    )
    connection = record(
        header(
            op=b"\x07",
            conn=struct.pack("<I", 7),
            topic=b"/camera/color/image_raw",
        ),
        header(
            type=b"sensor_msgs/Image",
            md5sum=b"060021388200f6f0f447d0fcd9c64743",
        ),
    )
    bag_header = record(header(op=b"\x03"), b"")
    result = b"#ROSBAG V2.0\n" + bag_header + chunk + connection
    return result[:-truncate] if truncate else result


class DlrStreamingBagIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def run_fixture(self, compression: str):
        indexer = self.module.StreamingBagIndexer(start_ns=1_000, end_ns=2_000)
        raw = build_bag(compression=compression)
        for offset in range(0, len(raw), 257):
            indexer.feed(raw[offset : offset + 257])
        return indexer, indexer.terminal()

    def test_none_chunk_indexes_time_without_retaining_payload(self) -> None:
        indexer, terminal = self.run_fixture("none")
        self.assertEqual(terminal["partial_buffer_length"], 0)
        self.assertEqual(len(terminal["candidate_color_connections"]), 1)
        candidate = terminal["candidate_color_connections"][0]
        self.assertEqual(candidate["before"]["bag_timestamp_ns"], 900)
        self.assertEqual(
            [row["bag_timestamp_ns"] for row in candidate["selected"]],
            [1_100, 1_500],
        )
        self.assertEqual(candidate["after"]["bag_timestamp_ns"], 2_100)
        self.assertTrue(
            all(value == 0 for value in terminal["pixel_firewall"].values())
        )
        self.assertEqual(len(indexer.chunk_records), 1)
        chunk = indexer.chunk_records[0]
        self.assertEqual(chunk["compression"], "none")
        self.assertEqual(chunk["message_count"], 4)
        self.assertEqual(chunk["minimum_bag_timestamp_ns"], 900)
        self.assertEqual(chunk["maximum_bag_timestamp_ns"], 2_100)
        self.assertEqual(
            chunk["declared_uncompressed_length"],
            chunk["actual_uncompressed_length"],
        )
        self.assertLess(indexer.maximum_buffer_observed, 2_000)

    def test_bz2_chunk_streams_and_indexes_same_rows(self) -> None:
        indexer, terminal = self.run_fixture("bz2")
        candidate = terminal["candidate_color_connections"][0]
        self.assertEqual(candidate["message_count"], 4)
        self.assertEqual(len(candidate["selected"]), 2)
        self.assertEqual(terminal["resumability"]["restartable_checkpoint_count"], 0)
        self.assertFalse(terminal["resumability"]["random_access"])
        self.assertEqual(indexer.chunk_records[0]["compression"], "bz2")
        self.assertEqual(indexer.chunk_records[0]["message_count"], 4)

    def test_connection_after_chunk_is_resolved(self) -> None:
        _indexer, terminal = self.run_fixture("bz2")
        self.assertEqual(terminal["connection_count"], 1)
        self.assertEqual(
            terminal["candidate_color_connections"][0]["connection_id"], 7
        )

    def test_truncated_stream_remains_partial(self) -> None:
        raw = build_bag(compression="none", truncate=17)
        indexer = self.module.StreamingBagIndexer(start_ns=1_000, end_ns=2_000)
        for offset in range(0, len(raw), 113):
            indexer.feed(raw[offset : offset + 113])
        self.assertGreater(indexer.terminal()["partial_buffer_length"], 0)

    def test_unsupported_chunk_compression_fails_closed(self) -> None:
        chunk = record(
            header(
                op=b"\x05",
                compression=b"lz4",
                size=struct.pack("<I", 0),
            ),
            b"",
        )
        raw = b"#ROSBAG V2.0\n" + chunk
        indexer = self.module.StreamingBagIndexer(start_ns=1_000, end_ns=2_000)
        with self.assertRaises(ValueError):
            indexer.feed(raw)


if __name__ == "__main__":
    unittest.main()
