from __future__ import annotations

import importlib.util
import http.client
import json
from pathlib import Path
import socket
import ssl
import sys
import tempfile
import unittest
from urllib.error import HTTPError, URLError


REPO = Path(__file__).resolve().parents[4]
MODULE_PATH = (
    REPO
    / "scripts"
    / "research"
    / "egomotion_compensated_looming"
    / "rgb_segment_confirmation_r2"
    / "diagnostic_transport.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("diagnostic_transport", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(self, body: bytes, content_range: str, status: int = 206) -> None:
        self.body = body
        self.status = status
        self.headers = {"Content-Range": content_range}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, cap: int) -> bytes:
        return self.body[:cap]


class DiagnosticTransportTests(unittest.TestCase):
    def make_transport(self, root: Path, opener, budget: int = 100):
        module = load_module()
        transport = module.DiagnosticRemoteRange(
            url="https://example.invalid/object?secret=do-not-log",
            length=10,
            budget=budget,
            ledger_path=root / "ledger.jsonl",
            progress_path=root / "progress.json",
            failure_receipt_path=root / "failure.json",
            phase="SOLID_FOLDER_PACK",
            user_agent="R2-test",
            opener=opener,
            sleep=lambda _delay: None,
        )
        return module, transport

    def test_success_persists_exact_attempt_without_url(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _module, transport = self.make_transport(
                root, lambda *_args, **_kwargs: FakeResponse(b"abcd", "bytes 0-3/10")
            )
            self.assertEqual(transport.read(4), b"abcd")
            row = json.loads((root / "ledger.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(row["status"], "PASS")
            self.assertEqual(row["body_sha256"], "88d4266fd4e6338d13b845fcf289579d209c897823b9217da3e161936f031589")
            self.assertNotIn("secret", json.dumps(row))

    def test_three_timeouts_raise_structured_failure_and_preserve_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def timeout(*_args, **_kwargs):
                raise URLError(TimeoutError("contains-secret-text"))

            module, transport = self.make_transport(root, timeout, budget=15)
            with self.assertRaises(module.TransportFailure) as caught:
                transport.read(4)
            self.assertEqual(caught.exception.category, "TIMEOUT")
            rows = [
                json.loads(line)
                for line in (root / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 3)
            self.assertEqual([row["attempt"] for row in rows], [1, 2, 3])
            self.assertTrue(all(row["accounted_bytes"] == 5 for row in rows))
            rendered = json.dumps(rows)
            self.assertNotIn("contains-secret-text", rendered)
            self.assertNotIn("example.invalid", rendered)
            progress = json.loads((root / "progress.json").read_text(encoding="utf-8"))
            self.assertEqual(progress["budget_consumed"], 15)
            self.assertEqual(progress["request_attempt_count"], 3)
            self.assertEqual(progress["status"], "TERMINAL_NOT_EVALUABLE")
            self.assertEqual(rows[-1]["status"], "TERMINAL_TRANSPORT_FAILURE")
            failure = json.loads((root / "failure.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["terminal_category"], "TIMEOUT")
            self.assertEqual(failure["last_request"]["attempt"], 3)

    def test_budget_stops_before_network(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calls = 0

            def opener(*_args, **_kwargs):
                nonlocal calls
                calls += 1
                return FakeResponse(b"abcd", "bytes 0-3/10")

            module, transport = self.make_transport(root, opener, budget=4)
            with self.assertRaises(module.BudgetExceeded):
                transport.read(4)
            self.assertEqual(calls, 0)
            row = json.loads((root / "ledger.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(row["status"], "TERMINAL_BUDGET_STOP_BEFORE_REQUEST")
            self.assertEqual(row["accounted_bytes"], 0)
            self.assertTrue((root / "failure.json").is_file())

    def test_sanitized_exception_matrix(self) -> None:
        module = load_module()
        cases = [
            (URLError(socket.gaierror(11001, "private-host")), "DNS"),
            (URLError(ssl.SSLError("private-cert-detail")), "TLS"),
            (URLError(ConnectionResetError("private-reset-detail")), "CONNECTION_RESET"),
            (
                HTTPError(
                    "https://secret.invalid/?token=hidden",
                    407,
                    "proxy secret",
                    {},
                    None,
                ),
                "PROXY_AUTH",
            ),
            (
                HTTPError(
                    "https://secret.invalid/?token=hidden",
                    503,
                    "server secret",
                    {},
                    None,
                ),
                "HTTP_ERROR",
            ),
            (http.client.IncompleteRead(b"x", 4), "INCOMPLETE_READ"),
            (module.RangeContractError("private range detail"), "RANGE_CONTRACT"),
        ]
        for error, expected in cases:
            with self.subTest(expected=expected):
                result = module.sanitized_error(error)
                self.assertEqual(result["category"], expected)
                rendered = json.dumps(result)
                self.assertNotIn("private", rendered)
                self.assertNotIn("token", rendered)
                self.assertNotIn("secret.invalid", rendered)

    def test_wrong_content_range_retries_three_times_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            module, transport = self.make_transport(
                root,
                lambda *_args, **_kwargs: FakeResponse(
                    b"abcd", "bytes 1-4/10"
                ),
                budget=15,
            )
            with self.assertRaises(module.TransportFailure) as caught:
                transport.read(4)
            self.assertEqual(caught.exception.category, "RANGE_CONTRACT")
            rows = [
                json.loads(line)
                for line in (root / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["attempt"] for row in rows], [1, 2, 3])
            self.assertEqual(
                [row["status"] for row in rows],
                [
                    "RETRYABLE_FAILURE",
                    "RETRYABLE_FAILURE",
                    "TERMINAL_TRANSPORT_FAILURE",
                ],
            )
            self.assertEqual(transport.snapshot()["budget_consumed"], 12)

    def test_retry_budget_exhaustion_is_logged_before_second_request(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calls = 0

            def timeout(*_args, **_kwargs):
                nonlocal calls
                calls += 1
                raise URLError(TimeoutError())

            module, transport = self.make_transport(root, timeout, budget=9)
            with self.assertRaises(module.BudgetExceeded):
                transport.read(4)
            self.assertEqual(calls, 1)
            rows = [
                json.loads(line)
                for line in (root / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [row["status"] for row in rows],
                [
                    "RETRYABLE_FAILURE",
                    "TERMINAL_BUDGET_STOP_BEFORE_REQUEST",
                ],
            )
            self.assertEqual(transport.snapshot()["request_attempt_count"], 1)

    def test_evidence_write_failure_never_retries_network(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calls = 0

            def opener(*_args, **_kwargs):
                nonlocal calls
                calls += 1
                return FakeResponse(b"abcd", "bytes 0-3/10")

            module, transport = self.make_transport(root, opener)

            def fail_append(_row):
                raise module.EvidencePersistenceFailure("injected")

            transport.ledger.append = fail_append
            with self.assertRaises(module.EvidencePersistenceFailure):
                transport.read(4)
            self.assertEqual(calls, 1)
            self.assertTrue(transport.failed)
            with self.assertRaises(RuntimeError):
                transport.read(4)

    def test_new_namespace_rejects_existing_or_partial_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "ledger.jsonl").write_text('{"partial":', encoding="utf-8")
            with self.assertRaises(FileExistsError):
                self.make_transport(
                    root,
                    lambda *_args, **_kwargs: FakeResponse(
                        b"abcd", "bytes 0-3/10"
                    ),
                )

    def test_content_range_is_sanitized_before_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            module, transport = self.make_transport(
                root,
                lambda *_args, **_kwargs: FakeResponse(
                    b"abcd", "bytes 0-3/10\r\nX-Secret: hidden"
                ),
                budget=15,
            )
            with self.assertRaises(module.TransportFailure):
                transport.read(4)
            rows = [
                json.loads(line)
                for line in (root / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(all(row["content_range"] is None for row in rows))
            self.assertNotIn("hidden", json.dumps(rows))


if __name__ == "__main__":
    unittest.main()
