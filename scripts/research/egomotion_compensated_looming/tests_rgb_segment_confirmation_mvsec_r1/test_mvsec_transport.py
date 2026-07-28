from __future__ import annotations

import importlib
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_DIR = (
    Path(__file__).resolve().parents[1]
    / "rgb_segment_confirmation_mvsec_r1"
)
sys.path.insert(0, str(MODULE_DIR))
runner = importlib.import_module("run_mvsec_identity")


class Headers(dict):
    pass


class Response:
    def __init__(
        self,
        *,
        status: int,
        url: str,
        headers: dict[str, str],
        body: bytes = b"",
    ) -> None:
        self.status = status
        self.url = url
        self.headers = Headers(headers)
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def geturl(self) -> str:
        return self.url

    def read(self, size: int) -> bytes:
        return self.body[:size]


class Opener:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, _request, timeout: int):
        self.calls += 1
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class MvsecTransportTests(unittest.TestCase):
    def make_reader(self, directory: str, opener, maximum: int = 100):
        ledger = runner.RangeLedger(Path(directory) / "range.jsonl")
        reader = runner.ExactRangeReader(
            url="https://example.invalid/bag",
            expected_bytes=10,
            expected_etag='"identity"',
            maximum_bytes=maximum,
            ledger=ledger,
            opener=opener,
        )
        return reader, ledger

    def test_exact_range_success_is_one_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            opener = Opener(
                [
                    Response(
                        status=206,
                        url="https://example.invalid/bag",
                        headers={
                            "Content-Range": "bytes 2-5/10",
                            "Content-Encoding": "identity",
                        },
                        body=b"2345",
                    )
                ]
            )
            reader, ledger = self.make_reader(directory, opener)
            self.assertEqual(reader.fetch(2, 5, "test"), b"2345")
            self.assertEqual(opener.calls, 1)
            self.assertEqual(ledger.rows, 1)
            self.assertEqual(ledger.bytes, 4)

    def test_transport_failure_never_retries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            opener = Opener([OSError("first"), AssertionError("retry")])
            reader, ledger = self.make_reader(directory, opener)
            with self.assertRaisesRegex(
                runner.TransportFailure,
                "RANGE_TRANSPORT_NO_RETRY",
            ):
                reader.fetch(2, 5, "test")
            self.assertEqual(opener.calls, 1)
            self.assertEqual(ledger.rows, 1)
            self.assertEqual(ledger.bytes, 4)

    def test_budget_stops_before_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            opener = Opener([AssertionError("network")])
            reader, ledger = self.make_reader(directory, opener, maximum=4)
            with self.assertRaisesRegex(
                runner.TransportFailure,
                "REMOTE_BYTE_CAP",
            ):
                reader.fetch(2, 5, "test")
            self.assertEqual(opener.calls, 0)
            self.assertEqual(ledger.rows, 0)

    def test_oversized_response_is_accounted_within_preauthorized_cap(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            opener = Opener(
                [
                    Response(
                        status=206,
                        url="https://example.invalid/bag",
                        headers={
                            "Content-Range": "bytes 2-5/10",
                            "Content-Encoding": "identity",
                        },
                        body=b"23456",
                    )
                ]
            )
            reader, ledger = self.make_reader(directory, opener, maximum=5)
            with self.assertRaisesRegex(
                runner.IdentityFailure,
                "RANGE_IDENTITY",
            ):
                reader.fetch(2, 5, "test")
            self.assertEqual(opener.calls, 1)
            self.assertEqual(ledger.rows, 1)
            self.assertEqual(ledger.bytes, 5)
            self.assertLessEqual(ledger.bytes, reader.maximum_bytes)

    def test_head_binds_size_etag_range_and_url(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            opener = Opener(
                [
                    Response(
                        status=200,
                        url="https://example.invalid/bag",
                        headers={
                            "Content-Length": "10",
                            "ETag": '"identity"',
                            "Accept-Ranges": "bytes",
                            "Last-Modified": "fixed",
                        },
                    )
                ]
            )
            reader, _ledger = self.make_reader(directory, opener)
            result = reader.head()
            self.assertEqual(result["content_length"], 10)
            self.assertEqual(result["etag"], '"identity"')
            self.assertEqual(opener.calls, 1)

    def test_identity_and_transport_have_distinct_legal_terminals(self) -> None:
        self.assertEqual(
            runner.terminal_decision(runner.IdentityFailure("identity")),
            "INVALID_SOURCE_OR_CAPTURE_IDENTITY",
        )
        self.assertEqual(
            runner.terminal_decision(runner.TransportFailure("transport")),
            "MVSEC_RGB_IDENTITY_NOT_EVALUABLE",
        )


if __name__ == "__main__":
    unittest.main()
