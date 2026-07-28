from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import sys
import tempfile
import unittest
import zlib


MODULE_DIR = (
    Path(__file__).resolve().parents[1]
    / "rgb_segment_confirmation_r2"
)
sys.path.insert(0, str(MODULE_DIR))
runner = importlib.import_module("run_dlr_sequential_index")


class MemoryRemote:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.position = 0

    def read(self, size: int) -> bytes:
        result = self.payload[self.position : self.position + size]
        self.position += len(result)
        return result


class HashingSink:
    def __init__(self) -> None:
        self.sha256 = hashlib.sha256()
        self.count = 0

    def feed(self, data: bytes) -> None:
        self.sha256.update(data)
        self.count += len(data)


def raw_deflate(payload: bytes) -> bytes:
    compressor = zlib.compressobj(level=6, wbits=-15)
    return compressor.compress(payload) + compressor.flush()


class DlrSequentialRunnerTests(unittest.TestCase):
    def test_streams_complete_raw_deflate_without_payload_file(self) -> None:
        payload = (b"bounded-rosbag-stream-" * 200_000) + b"terminal"
        compressed = raw_deflate(payload)
        remote = MemoryRemote(compressed)
        sink = HashingSink()
        result = runner.stream_deflate_member(
            remote,
            compressed_bytes=len(compressed),
            network_chunk_bytes=317,
            parser=sink,
        )
        self.assertTrue(result["decompressor"].eof)
        self.assertEqual(result["compressed_count"], len(compressed))
        self.assertEqual(result["uncompressed_count"], len(payload))
        self.assertEqual(result["bag_sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(sink.count, len(payload))
        self.assertGreater(len(result["observations"]), 1)
        self.assertTrue(all(not row["restartable"] for row in result["observations"]))

    def test_truncated_member_fails_closed(self) -> None:
        compressed = raw_deflate(b"x" * 100_000)
        remote = MemoryRemote(compressed[:-1])
        with self.assertRaisesRegex(
            runner.ActivationPreflightFailure,
            "COMPRESSED_MEMBER_TRUNCATED",
        ):
            runner.stream_deflate_member(
                remote,
                compressed_bytes=len(compressed),
                network_chunk_bytes=101,
                parser=HashingSink(),
            )

    def test_range_summary_closes_attempts_retries_and_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "range.jsonl"
            ledger = runner.AppendOnlyLedger(path)
            ledger.append(
                {
                    "logical_request_id": 1,
                    "attempt": 1,
                    "accounted_bytes": 31,
                }
            )
            ledger.append(
                {
                    "logical_request_id": 1,
                    "attempt": 2,
                    "accounted_bytes": 31,
                }
            )
            ledger.append(
                {
                    "logical_request_id": 2,
                    "attempt": 1,
                    "accounted_bytes": 101,
                }
            )
            summary = runner.summarize_append_only_ledger(path)
            self.assertEqual(summary["rows"], 3)
            self.assertEqual(summary["logical_request_count"], 2)
            self.assertEqual(summary["attempt_count"], 3)
            self.assertEqual(summary["retry_count"], 1)
            self.assertEqual(summary["accounted_byte_sum"], 163)
            self.assertIsNotNone(summary["head_sha256"])

    def test_identity_failure_has_legal_terminal(self) -> None:
        self.assertEqual(
            runner.failure_decision(
                runner.SourceMemberIdentityFailure("ZIP_MEMBER_NAME")
            ),
            "INVALID_SOURCE_OR_MEMBER_IDENTITY",
        )
        self.assertEqual(
            runner.failure_decision(ValueError("ROSBAG_PARSE")),
            "DLR_INDEX_NOT_EVALUABLE",
        )


if __name__ == "__main__":
    unittest.main()
