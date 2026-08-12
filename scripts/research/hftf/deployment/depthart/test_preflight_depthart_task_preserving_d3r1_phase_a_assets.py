import io
import tempfile
import unittest
import urllib.error
from email.message import Message
from pathlib import Path

from scripts.research.hftf.deployment.depthart.preflight_depthart_task_preserving_d3r1_phase_a_assets import (
    ASSETS,
    PASS_TERMINAL,
    disposition,
    head,
    requests_for,
    reserve_fresh_output_root,
    reserve_output,
    roster_rows,
    write_exclusive,
)


def fixture_roster() -> dict:
    pairs = [f"{100000 + index}/{20000000 + index}" for index in range(127)]
    import hashlib

    digest = hashlib.sha256(("\n".join(pairs) + "\n").encode("utf-8")).hexdigest().upper()
    return {
        "schema": "blindassist_depthart_task_preserving_d3r1_fresh_metadata_roster_lock_v1",
        "status": "D3R1_FRESH_METADATA_POOL_127_LOCKED_MEDIA_UNOPENED",
        "pool": pairs,
        "pool_pairs_sha256": digest,
    }


class Response:
    def __init__(self, url: str) -> None:
        self.status = 200
        self.headers = Message()
        self.headers["Content-Length"] = "123"
        self.headers["ETag"] = '"etag"'
        self.headers["Last-Modified"] = "Wed, 12 Aug 2026 00:00:00 GMT"
        self._url = url

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def geturl(self) -> str:
        return self._url

    def read(self, *args):
        raise AssertionError("HEAD response body read")

    def readinto(self, *args):
        raise AssertionError("HEAD response body readinto")

    def peek(self, *args):
        raise AssertionError("HEAD response peek")


class D3R1PhaseAHeadPreflightTest(unittest.TestCase):
    def test_request_plan_is_exact(self) -> None:
        rows = roster_rows(fixture_roster())
        self.assertEqual(127, len(rows))
        requests = requests_for(fixture_roster(), "https://example.invalid/raw")
        self.assertEqual(254, len(requests))
        self.assertEqual(list(ASSETS), [row["asset"] for row in requests[:2]])
        self.assertEqual(254, len({row["url"] for row in requests}))

    def test_success_never_reads_response_body(self) -> None:
        row = requests_for(fixture_roster(), "https://example.invalid/raw")[0]
        result = head(row, 1.0, 3, opener=lambda request, timeout: Response(row["url"]))
        self.assertEqual(200, result["http_status"])
        self.assertEqual(0, result["response_body_bytes_read"])
        self.assertEqual(["HEAD"], [item["method"] for item in result["attempt_history"]])

    def test_redirect_is_not_followed_or_retried(self) -> None:
        row = requests_for(fixture_roster(), "https://example.invalid/raw")[0]
        calls = []

        def opener(request, timeout):
            calls.append(request.get_method())
            headers = Message()
            headers["Location"] = "https://redirect.invalid/body"
            raise urllib.error.HTTPError(row["url"], 302, "Found", headers, io.BytesIO())

        result = head(row, 1.0, 3, opener=opener)
        self.assertEqual(["HEAD"], calls)
        self.assertTrue(result["redirected"])
        self.assertTrue(result["unresolved_error"])

    def test_transient_error_recovers_with_head_only(self) -> None:
        row = requests_for(fixture_roster(), "https://example.invalid/raw")[0]
        calls = []

        def opener(request, timeout):
            calls.append(request.get_method())
            if len(calls) == 1:
                raise TimeoutError("transient")
            return Response(row["url"])

        result = head(row, 1.0, 3, opener=opener)
        self.assertEqual(["HEAD", "HEAD"], calls)
        self.assertTrue(result["recovered_error"])
        self.assertFalse(result["unresolved_error"])

    def test_permanent_404_does_not_retry(self) -> None:
        row = requests_for(fixture_roster(), "https://example.invalid/raw")[0]
        calls = []

        def opener(request, timeout):
            calls.append(request.get_method())
            raise urllib.error.HTTPError(row["url"], 404, "Not Found", Message(), io.BytesIO())

        result = head(row, 1.0, 3, opener=opener)
        self.assertEqual(["HEAD"], calls)
        self.assertEqual(404, result["http_status"])
        self.assertTrue(result["unresolved_error"])

    def test_malformed_content_length_is_terminal_not_retried(self) -> None:
        row = requests_for(fixture_roster(), "https://example.invalid/raw")[0]
        calls = []

        def opener(request, timeout):
            calls.append(request.get_method())
            response = Response(row["url"])
            response.headers.replace_header("Content-Length", "not-an-integer")
            return response

        result = head(row, 1.0, 3, opener=opener)
        self.assertEqual(["HEAD"], calls)
        self.assertIsNone(result["content_length_bytes"])
        self.assertNotEqual(PASS_TERMINAL, disposition([result] * 254))

    def test_disposition_requires_three_headers_and_zero_body(self) -> None:
        base = {
            "http_status": 200,
            "url": "https://example.invalid/a",
            "final_url": "https://example.invalid/a",
            "redirected": False,
            "content_length_bytes": 1,
            "etag": '"e"',
            "last_modified": "date",
            "response_body_bytes_read": 0,
            "unresolved_error": False,
        }
        rows = [dict(base) for _ in range(254)]
        self.assertEqual(PASS_TERMINAL, disposition(rows))
        for key in ("etag", "last_modified", "content_length_bytes"):
            broken = [dict(row) for row in rows]
            broken[0][key] = None
            self.assertNotEqual(PASS_TERMINAL, disposition(broken))

    def test_exclusive_output_rejects_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_exclusive(path, b"first")
            with self.assertRaises(FileExistsError):
                write_exclusive(path, b"second")
            self.assertEqual(b"first", path.read_bytes())

    def test_output_reservation_rejects_before_transport(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_bytes(b"frozen")
            with self.assertRaises(FileExistsError):
                reserve_output(path)
            self.assertEqual(b"frozen", path.read_bytes())

    def test_fresh_attempt_root_rejects_existing_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing-attempt" / "result.json"
            path.parent.mkdir()
            with self.assertRaises(FileExistsError):
                reserve_fresh_output_root(path)


if __name__ == "__main__":
    unittest.main()
