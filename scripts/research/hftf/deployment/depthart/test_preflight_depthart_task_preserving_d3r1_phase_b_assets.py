import copy
import hashlib
import io
import tempfile
import unittest
import urllib.error
from email.message import Message
from pathlib import Path

from scripts.research.hftf.deployment.depthart.preflight_depthart_task_preserving_d3r1_phase_b_assets import (
    ASSETS,
    INCOMPLETE_TERMINAL,
    PASS_TERMINAL,
    UNAVAILABLE_TERMINAL,
    disposition,
    head,
    request_plan_sha256,
    requests_for,
    reserve_fresh_output_root,
    selection_rows,
    write_exclusive,
)


def fixtures() -> tuple[dict, dict]:
    identities = [
        {
            "selection_order": index,
            "pool_order": index + 4,
            "visit_id": str(100000 + index),
            "video_id": str(20000000 + index),
        }
        for index in range(1, 33)
    ]
    encoded = "\n".join(
        f"{row['selection_order']}/{row['pool_order']}/{row['visit_id']}/{row['video_id']}"
        for row in identities
    ) + "\n"
    scope = {
        "schema": (
            "blindassist_depthart_task_preserving_d3r1_phase_b_depth_confidence_"
            "source_scope_receipt_v1"
        ),
        "status": "D3R1_EXACT_32_PHASE_B_DEPTH_CONFIDENCE_SOURCE_SCOPE_REGISTERED_MEDIA_UNOPENED",
        "next_gate": "EXPLICIT_D3R1_PHASE_B_DEPTH_CONFIDENCE_HEAD_ONLY_PREFLIGHT_ACTIVATION",
        "exact_phase_a_selection": {
            "identities": identities,
            "selection_sha256": hashlib.sha256(encoded.encode("ascii")).hexdigest().upper(),
        },
        "registered_future_asset_scope": {
            "assets": list(ASSETS),
            "future_head_request_count": 64,
            "future_body_asset_count": 64,
            "phase_c_rgb_registered": False,
        },
    }
    result = {
        "schema": "blindassist_depthart_task_preserving_d3r1_phase_a_governed_result_v1",
        "terminal": "D3R1_PHASE_A_PORTRAIT_POSE_CONTINUITY_PASS_32_IDENTITIES_LOCKED",
        "selected_phase_a": copy.deepcopy(identities),
    }
    return scope, result


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


class D3R1PhaseBHeadPreflightTest(unittest.TestCase):
    def test_request_plan_is_exact(self) -> None:
        scope, result = fixtures()
        self.assertEqual(32, len(selection_rows(scope, result)))
        requests = requests_for(scope, result, "https://example.invalid/raw")
        self.assertEqual(64, len(requests))
        self.assertEqual(list(ASSETS), [row["asset"] for row in requests[:2]])
        self.assertEqual(64, len({row["url"] for row in requests}))
        self.assertEqual(64, len(request_plan_sha256(requests)))

    def test_selection_drift_is_rejected(self) -> None:
        scope, result = fixtures()
        result["selected_phase_a"][0] = dict(result["selected_phase_a"][0])
        result["selected_phase_a"][0]["video_id"] = "99999999"
        with self.assertRaisesRegex(ValueError, "selection mismatch"):
            requests_for(scope, result, "https://example.invalid/raw")

    def test_success_never_reads_response_body(self) -> None:
        scope, result = fixtures()
        row = requests_for(scope, result, "https://example.invalid/raw")[0]
        observed = head(row, 1.0, 3, "test-agent", lambda request, timeout: Response(row["url"]))
        self.assertEqual(200, observed["http_status"])
        self.assertEqual(0, observed["response_body_bytes_read"])
        self.assertEqual(["HEAD"], [item["method"] for item in observed["attempt_history"]])

    def test_redirect_is_not_followed_or_retried(self) -> None:
        scope, result = fixtures()
        row = requests_for(scope, result, "https://example.invalid/raw")[0]
        calls = []

        def opener(request, timeout):
            calls.append(request.get_method())
            headers = Message()
            headers["Location"] = "https://redirect.invalid/body"
            raise urllib.error.HTTPError(row["url"], 302, "Found", headers, io.BytesIO())

        observed = head(row, 1.0, 3, "test-agent", opener)
        self.assertEqual(["HEAD"], calls)
        self.assertEqual(1, observed["redirect_count"])
        self.assertTrue(observed["unresolved_error"])

    def test_transient_error_recovers_with_head_only(self) -> None:
        scope, result = fixtures()
        row = requests_for(scope, result, "https://example.invalid/raw")[0]
        calls = []

        def opener(request, timeout):
            calls.append(request.get_method())
            if len(calls) == 1:
                raise TimeoutError("transient")
            return Response(row["url"])

        observed = head(row, 1.0, 3, "test-agent", opener)
        self.assertEqual(["HEAD", "HEAD"], calls)
        self.assertTrue(observed["recovered_error"])
        self.assertFalse(observed["unresolved_error"])

    def test_permanent_404_does_not_retry(self) -> None:
        scope, result = fixtures()
        row = requests_for(scope, result, "https://example.invalid/raw")[0]
        calls = []

        def opener(request, timeout):
            calls.append(request.get_method())
            raise urllib.error.HTTPError(row["url"], 404, "Not Found", Message(), io.BytesIO())

        observed = head(row, 1.0, 3, "test-agent", opener)
        self.assertEqual(["HEAD"], calls)
        self.assertEqual(404, observed["http_status"])
        self.assertTrue(observed["unresolved_error"])

    def test_missing_required_header_is_unavailable(self) -> None:
        scope, result = fixtures()
        row = requests_for(scope, result, "https://example.invalid/raw")[0]

        def opener(request, timeout):
            response = Response(row["url"])
            del response.headers["ETag"]
            return response

        observed = head(row, 1.0, 3, "test-agent", opener)
        self.assertTrue(observed["unresolved_error"])
        self.assertEqual(UNAVAILABLE_TERMINAL, disposition([observed] * 64))

    def test_invalid_length_is_not_retried(self) -> None:
        scope, result = fixtures()
        row = requests_for(scope, result, "https://example.invalid/raw")[0]
        for invalid in ("0", "not-an-integer"):
            calls = []

            def opener(request, timeout):
                calls.append(request.get_method())
                response = Response(row["url"])
                response.headers.replace_header("Content-Length", invalid)
                return response

            observed = head(row, 1.0, 3, "test-agent", opener)
            self.assertEqual(["HEAD"], calls)
            self.assertTrue(observed["unresolved_error"])

    def test_final_transport_error_after_http_500_is_incomplete(self) -> None:
        scope, result = fixtures()
        row = requests_for(scope, result, "https://example.invalid/raw")[0]
        calls = []

        def opener(request, timeout):
            calls.append(request.get_method())
            if len(calls) == 1:
                raise urllib.error.HTTPError(row["url"], 500, "Server Error", Message(), io.BytesIO())
            raise TimeoutError("transport incomplete")

        observed = head(row, 1.0, 3, "test-agent", opener)
        self.assertEqual(3, observed["attempts"])
        self.assertEqual(500, observed["http_status"])
        self.assertIsNone(observed["attempt_history"][-1]["http_status"])
        self.assertEqual(INCOMPLETE_TERMINAL, disposition([observed] * 64))

    def test_disposition_requires_all_64_rows(self) -> None:
        base = {
            "http_status": 200,
            "url": "https://example.invalid/a",
            "final_url": "https://example.invalid/a",
            "redirect_count": 0,
            "content_length_bytes": 1,
            "etag": '"e"',
            "last_modified": "date",
            "response_body_bytes_read": 0,
            "attempt_history": [
                {"attempt": 1, "method": "HEAD", "http_status": 200, "error": None}
            ],
            "unresolved_error": False,
        }
        self.assertEqual(PASS_TERMINAL, disposition([dict(base) for _ in range(64)]))
        with self.assertRaisesRegex(ValueError, "row count"):
            disposition([dict(base) for _ in range(63)])

    def test_exclusive_output_and_fresh_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_exclusive(path, b"first")
            with self.assertRaises(FileExistsError):
                write_exclusive(path, b"second")
            attempt = Path(directory) / "attempt" / "result.json"
            attempt.parent.mkdir()
            with self.assertRaises(FileExistsError):
                reserve_fresh_output_root(attempt)


if __name__ == "__main__":
    unittest.main()
