from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import struct
import sys
import tempfile
import unittest


BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "rgb_segment_confirmation_mvsec_r1"))
sys.path.insert(0, str(BASE / "rgb_segment_confirmation_mvsec_r2"))
core = importlib.import_module("run_mvsec_identity")
capture_r2 = importlib.import_module("mvsec_identity_capture_r2")


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
    payload = bytes([value]) * 89_960
    frame_id = b"davis_left"
    return (
        struct.pack(
            "<III",
            value,
            timestamp_ns // 1_000_000_000,
            timestamp_ns % 1_000_000_000,
        )
        + struct.pack("<I", len(frame_id))
        + frame_id
        + struct.pack("<II", 260, 346)
        + struct.pack("<I", 5)
        + b"mono8"
        + struct.pack("<BII", 0, 346, len(payload))
        + payload
    )


def image_record(timestamp_ns: int, value: int) -> bytes:
    return record(
        header(
            op=b"\x02",
            conn=struct.pack("<I", 7),
            time=time_value(timestamp_ns),
        ),
        image_message(timestamp_ns, value),
    )


def fixture_bag(
    timestamps: list[int],
    *,
    raw_suffix: bytes = b"",
) -> bytes:
    chunk_pos = 9_000
    index_pos = 500_000
    raw = b"".join(
        image_record(timestamp_ns, value)
        for value, timestamp_ns in enumerate(timestamps, 1)
    ) + raw_suffix
    chunk_header = header(
        op=b"\x05",
        compression=b"none",
        size=struct.pack("<I", len(raw)),
    )
    chunk = (
        struct.pack("<I", len(chunk_header))
        + chunk_header
        + struct.pack("<I", len(raw))
        + raw
    )
    connection = record(
        header(
            op=b"\x07",
            conn=struct.pack("<I", 7),
            topic=b"/davis/left/image_raw",
        ),
        header(
            type=b"sensor_msgs/Image",
            md5sum=b"060021388200f6f0f447d0fcd9c64743",
            message_definition=b"fixture",
        ),
    )
    chunk_info = record(
        header(
            op=b"\x06",
            chunk_pos=struct.pack("<Q", chunk_pos),
            start_time=time_value(min(timestamps)),
            end_time=time_value(max(timestamps)),
            count=struct.pack("<I", 1),
        ),
        struct.pack("<II", 7, len(timestamps)),
    )
    final_index = connection + chunk_info
    bag_size = index_pos + len(final_index)
    blob = bytearray(bag_size)
    bag_header = record(
        header(
            op=b"\x03",
            index_pos=struct.pack("<Q", index_pos),
            conn_count=struct.pack("<I", 1),
            chunk_count=struct.pack("<I", 1),
        ),
        b"",
    )
    opening = b"#ROSBAG V2.0\n" + bag_header
    blob[: len(opening)] = opening
    blob[chunk_pos : chunk_pos + len(chunk)] = chunk
    blob[index_pos:] = final_index
    return bytes(blob)


class CaptureR2Tests(unittest.TestCase):
    def run_fixture(
        self,
        directory: str,
        *,
        timestamps: list[int],
        geometry_timestamp_ns: int,
        raw_suffix: bytes = b"",
        expected_geometry_count: int = 1,
    ):
        blob = fixture_bag(timestamps, raw_suffix=raw_suffix)
        original_reader = core.ExactRangeReader

        class FixtureReader:
            def __init__(
                self,
                *,
                url,
                expected_bytes,
                expected_etag,
                maximum_bytes,
                ledger,
            ):
                self.url = url
                self.expected_bytes = expected_bytes
                self.expected_etag = expected_etag
                self.maximum_bytes = maximum_bytes
                self.ledger = ledger

            def head(self):
                return {
                    "status": 200,
                    "content_length": len(blob),
                    "etag": '"fixture"',
                    "last_modified": "fixed",
                    "accept_ranges": "bytes",
                }

            def fetch(self, start, end, label):
                body = blob[start : end + 1]
                if len(body) != end - start + 1:
                    raise core.IdentityFailure("RANGE_IDENTITY")
                self.ledger.append(
                    {
                        "label": label,
                        "start": start,
                        "end": end,
                        "requested_bytes": len(body),
                        "accounted_bytes": len(body),
                        "status": "PASS",
                        "attempt": 1,
                        "http_status": 206,
                        "content_range": (
                            f"bytes {start}-{end}/{len(blob)}"
                        ),
                        "body_sha256": hashlib.sha256(body).hexdigest(),
                    }
                )
                return body

        repo = Path(directory)
        namespace = repo / "claim"
        capture = {
            "capture_id": "frozen",
            "window": {
                "window_id": "frozen:w001",
                "role": "POSITIVE_APPROACH_WINDOW",
                "start_ns": geometry_timestamp_ns,
                "end_ns": geometry_timestamp_ns + 1,
                "geometry_frame_count": expected_geometry_count,
            },
            "data_bag": {
                "url": "https://example.invalid/fixture.bag",
                "bytes": len(blob),
                "etag": '"fixture"',
            },
            "transport": {
                "maximum_remote_bytes": len(blob) * 2,
                "maximum_index_bytes": 64 * (1 << 20),
            },
            "pairing": {"maximum_abs_delta_ns": 20},
        }
        geometry_source = {
            "depth": {
                "frames": [{"timestamp_ns": geometry_timestamp_ns}]
            }
        }
        core.ExactRangeReader = FixtureReader
        try:
            return capture_r2.inspect_capture_r2(
                core,
                repo=repo,
                namespace=namespace,
                capture=capture,
                geometry_source=geometry_source,
            )
        finally:
            core.ExactRangeReader = original_reader

    def test_direct_r2_capture_success_without_monkeypatching_core_flow(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_fixture(
                directory,
                timestamps=[90, 100, 110],
                geometry_timestamp_ns=100,
            )
            self.assertEqual(result["selected_frame_count"], 1)
            self.assertEqual(result["guard_frame_count"], 2)
            self.assertEqual(result["materialized_payload_files"], 3)
            self.assertEqual(result["image_decode_calls"], 0)
            self.assertEqual(result["rectification_calls"], 0)
            self.assertEqual(result["rgb_algorithm_calls"], 0)
            self.assertEqual(result["image_metadata_ledger"]["rows"], 3)
            self.assertEqual(result["pairing_diagnostic_ledger"]["rows"], 1)
            self.assertTrue(
                (
                    Path(directory)
                    / "claim"
                    / "frozen"
                    / "target_chunk_plan.json"
                ).is_file()
            )

    def test_direct_r2_pairing_failure_keeps_metadata_not_pixels(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ValueError,
                "IMAGE_GEOMETRY_PAIRING_TIE",
            ):
                self.run_fixture(
                    directory,
                    timestamps=[80, 90, 110, 120],
                    geometry_timestamp_ns=100,
                )
            capture = Path(directory) / "claim" / "frozen"
            self.assertEqual(
                len(
                    (capture / "image_metadata_ledger.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ),
                4,
            )
            self.assertFalse((capture / "frames").exists())
            stage = (capture / "stage_ledger.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertIn("IMAGE_GEOMETRY_PAIRING_TIE", stage)
            self.assertIn('"stage":"PAIRING"', stage)

    def test_direct_r2_record_failure_persists_partial_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ValueError,
                "RECORD_HEADER_LENGTH_TRUNCATED",
            ):
                self.run_fixture(
                    directory,
                    timestamps=[90],
                    geometry_timestamp_ns=90,
                    raw_suffix=b"\x01\x02",
                )
            capture = Path(directory) / "claim" / "frozen"
            metadata_rows = (
                capture / "image_metadata_ledger.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(metadata_rows), 1)
            self.assertFalse((capture / "frames").exists())
            stage = (capture / "stage_ledger.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertIn("RECORD_HEADER_LENGTH_TRUNCATED", stage)
            self.assertIn('"parsed_image_count":1', stage)

    def test_direct_r2_rejects_geometry_count_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                core.IdentityFailure,
                "GEOMETRY_FRAME_COUNT_IDENTITY",
            ):
                self.run_fixture(
                    directory,
                    timestamps=[80, 90, 100],
                    geometry_timestamp_ns=90,
                    expected_geometry_count=2,
                )
            capture = Path(directory) / "claim" / "frozen"
            self.assertFalse((capture / "frames").exists())
            stage = (capture / "stage_ledger.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertIn("GEOMETRY_FRAME_COUNT_IDENTITY", stage)


if __name__ == "__main__":
    unittest.main()
