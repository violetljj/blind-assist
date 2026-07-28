from __future__ import annotations

import bz2
import importlib.util
from pathlib import Path
import struct
import sys
import unittest


MODULE = (
    Path(__file__).resolve().parents[1]
    / "rgb_segment_confirmation_mvsec_r1"
    / "mvsec_target_chunks.py"
)
spec = importlib.util.spec_from_file_location("mvsec_target_chunks", MODULE)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def field(key: str, value: bytes) -> bytes:
    payload = key.encode("ascii") + b"=" + value
    return struct.pack("<I", len(payload)) + payload


def header(**values: bytes) -> bytes:
    return b"".join(field(key, value) for key, value in values.items())


def record(header_bytes: bytes, data: bytes) -> bytes:
    return (
        struct.pack("<I", len(header_bytes))
        + header_bytes
        + struct.pack("<I", len(data))
        + data
    )


def time_value(value: int) -> bytes:
    return struct.pack("<II", value // 1_000_000_000, value % 1_000_000_000)


def image_message(timestamp_ns: int, value: int) -> bytes:
    frame = bytes([value]) * 12
    frame_id = b"davis_left_rect"
    return (
        struct.pack(
            "<III",
            value,
            timestamp_ns // 1_000_000_000,
            timestamp_ns % 1_000_000_000,
        )
        + struct.pack("<I", len(frame_id))
        + frame_id
        + struct.pack("<II", 3, 4)
        + struct.pack("<I", 5)
        + b"mono8"
        + struct.pack("<BII", 0, 4, len(frame))
        + frame
    )


class MvsecTargetChunkTests(unittest.TestCase):
    def test_bz2_chunk_parses_mono8_without_decode(self) -> None:
        raw = b"".join(
            record(
                header(
                    op=b"\x02",
                    conn=struct.pack("<I", 7),
                    time=time_value(timestamp),
                ),
                image_message(timestamp, value),
            )
            for timestamp, value in (
                (900_000_000, 1),
                (1_000_000_000, 2),
                (1_050_000_000, 3),
                (1_100_000_000, 4),
                (1_200_000_000, 5),
            )
        )
        compressed = bz2.compress(raw)
        uncompressed = module.decompress_chunk(
            compression="bz2",
            compressed=compressed,
            declared_uncompressed_bytes=len(raw),
        )
        messages = module.image_messages_from_chunk(
            uncompressed,
            connection_id=7,
        )
        self.assertEqual(len(messages), 5)
        self.assertEqual(messages[1].encoding, "mono8")
        self.assertEqual(messages[1].height, 3)
        self.assertEqual(messages[1].width, 4)
        self.assertEqual(messages[1].step, 4)
        self.assertEqual(len(messages[1].payload), 12)

    def test_unique_pairing_retains_only_one_guard_each_side(self) -> None:
        messages = [
            module.parse_sensor_image(
                image_message(timestamp, index),
                bag_timestamp_ns=timestamp,
            )
            for index, timestamp in enumerate(
                (
                    900_000_000,
                    1_000_000_000,
                    1_050_000_000,
                    1_100_000_000,
                    1_200_000_000,
                ),
                1,
            )
        ]
        paired = module.pair_window(
            messages,
            geometry_timestamps_ns=[
                1_001_000_000,
                1_049_000_000,
                1_101_000_000,
            ],
            maximum_delta_ns=2_000_000,
        )
        self.assertEqual(paired["before"].header_timestamp_ns, 900_000_000)
        self.assertEqual(paired["after"].header_timestamp_ns, 1_200_000_000)
        self.assertEqual(len(paired["selected"]), 3)
        self.assertEqual(len(paired["retained"]), 5)
        self.assertEqual(paired["maximum_abs_delta_ns"], 1_000_000)

    def test_chunk_plan_adds_adjacent_image_chunks_only(self) -> None:
        chunks = [
            {"chunk_pos": 10, "start_ns": 0, "end_ns": 9, "counts": {7: 2}},
            {"chunk_pos": 20, "start_ns": 10, "end_ns": 19, "counts": {8: 2}},
            {"chunk_pos": 30, "start_ns": 20, "end_ns": 29, "counts": {7: 2}},
            {"chunk_pos": 40, "start_ns": 30, "end_ns": 39, "counts": {7: 2}},
            {"chunk_pos": 50, "start_ns": 40, "end_ns": 49, "counts": {8: 2}},
            {"chunk_pos": 60, "start_ns": 50, "end_ns": 59, "counts": {7: 2}},
        ]
        selected = module.select_target_chunks(
            chunks,
            connection_id=7,
            window_start_ns=31,
            window_end_ns=39,
        )
        self.assertEqual(
            [row["chunk_pos"] for row in selected],
            [30, 40, 60],
        )

    def test_pairing_fails_closed_without_guard_or_tolerance(self) -> None:
        messages = [
            module.parse_sensor_image(
                image_message(timestamp, index),
                bag_timestamp_ns=timestamp,
            )
            for index, timestamp in enumerate(
                (1_000_000_000, 1_050_000_000, 1_100_000_000),
                1,
            )
        ]
        with self.assertRaisesRegex(ValueError, "IMAGE_GUARD_MISSING"):
            module.pair_window(
                messages,
                geometry_timestamps_ns=[
                    1_000_000_000,
                    1_050_000_000,
                    1_100_000_000,
                ],
                maximum_delta_ns=1,
            )
        with self.assertRaisesRegex(ValueError, "IMAGE_GEOMETRY_PAIRING"):
            module.pair_window(
                messages,
                geometry_timestamps_ns=[1_010_000_000],
                maximum_delta_ns=1_000_000,
            )

    def test_pairing_fails_closed_on_equal_distance_nearest_images(
        self,
    ) -> None:
        messages = [
            module.parse_sensor_image(
                image_message(timestamp, index),
                bag_timestamp_ns=timestamp,
            )
            for index, timestamp in enumerate(
                (
                    80_000_000,
                    90_000_000,
                    110_000_000,
                    120_000_000,
                ),
                1,
            )
        ]
        with self.assertRaisesRegex(
            ValueError,
            "IMAGE_GEOMETRY_PAIRING_TIE",
        ):
            module.pair_window(
                messages,
                geometry_timestamps_ns=[100_000_000],
                maximum_delta_ns=10_000_000,
            )

    def test_pairing_fails_closed_when_nearest_image_would_be_reused(
        self,
    ) -> None:
        messages = [
            module.parse_sensor_image(
                image_message(timestamp, index),
                bag_timestamp_ns=timestamp,
            )
            for index, timestamp in enumerate(
                (
                    80_000_000,
                    100_000_000,
                    120_000_000,
                ),
                1,
            )
        ]
        with self.assertRaisesRegex(
            ValueError,
            "IMAGE_GEOMETRY_PAIRING_REUSE",
        ):
            module.pair_window(
                messages,
                geometry_timestamps_ns=[99_000_000, 101_000_000],
                maximum_delta_ns=2_000_000,
            )


if __name__ == "__main__":
    unittest.main()
