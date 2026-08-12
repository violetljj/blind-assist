from __future__ import annotations

import copy
import unittest
from pathlib import Path

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o1r_r11_abstention_runtime import fresh_pool
from scripts.research.taro_o1r_r11_abstention_runtime import run_pool_head as runner


ROOT = Path(__file__).resolve().parents[3]


def response(*, status: int | None = 200, length: int | None = 1, redirects: list[str] | None = None, errors: list[str] | None = None) -> dict[str, object]:
    return {
        "http_status": status,
        "content_length_bytes": length,
        "etag": "E",
        "last_modified": "L",
        "redirect_chain": redirects or [],
        "transport_errors": errors or [],
    }


def reseal(value: dict[str, object]) -> dict[str, object]:
    clone = copy.deepcopy(value)
    clone.pop("content_sha256", None)
    clone["content_sha256"] = adapter.canonical_sha256(clone)
    return clone


class PoolHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = fresh_pool.build_pool(ROOT)

    def build(self, head_fn, *, attempts: int = 2) -> dict[str, object]:
        return runner.build_head_receipt(
            self.plan,
            lock_sha256="A" * 64,
            protocol_sha256="B" * 64,
            authorization_sha256="C" * 64,
            head_fn=head_fn,
            timeout_seconds=1,
            maximum_attempts=attempts,
            maximum_bytes=runner.EXPECTED_BUDGET["maximum_compressed_source_bytes"],
        )

    def test_exact_144_zero_body_success(self) -> None:
        receipt = self.build(lambda _request, _timeout: response())
        self.assertTrue(receipt["passed"])
        self.assertEqual(receipt["asset_count"], 144)
        self.assertEqual(receipt["available_asset_count"], 144)
        self.assertEqual(receipt["request_attempt_count"], 144)
        self.assertEqual(receipt["response_body_bytes_read"], 0)

    def test_transient_failure_retries_then_succeeds(self) -> None:
        calls: dict[str, int] = {}

        def fake(request: dict[str, str], _timeout: float) -> dict[str, object]:
            key = request["url"]
            calls[key] = calls.get(key, 0) + 1
            return response(status=None, length=None, errors=["TIMEOUT"]) if calls[key] == 1 else response(length=2)

        receipt = self.build(fake)
        self.assertTrue(receipt["passed"])
        self.assertEqual(receipt["request_attempt_count"], 288)
        self.assertTrue(all(row["attempt_count"] == 2 for row in receipt["assets"]))

    def test_redirect_or_byte_ceiling_is_valid_negative_terminal(self) -> None:
        redirected = self.build(lambda _request, _timeout: response(redirects=["https://redirect.invalid"]), attempts=1)
        self.assertFalse(redirected["passed"])
        self.assertEqual(redirected["terminal"], runner.UNAVAILABLE_TERMINAL)
        per_asset_over_budget = runner.EXPECTED_BUDGET["maximum_compressed_source_bytes"] // runner.ASSET_COUNT + 1
        over_budget = self.build(lambda _request, _timeout: response(length=per_asset_over_budget), attempts=1)
        self.assertFalse(over_budget["passed"])
        self.assertEqual(over_budget["available_asset_count"], 144)

    def test_resealed_attempt_and_final_mismatch_fails_closed(self) -> None:
        receipt = self.build(lambda _request, _timeout: response())
        receipt["assets"][0]["http_status"] = 503
        with self.assertRaisesRegex(runner.PoolHeadError, "final fields"):
            runner.validate_head_receipt(self.plan, reseal(receipt), maximum_attempts=2)
        receipt = self.build(lambda _request, _timeout: response())
        receipt["assets"][0]["attempts"][0]["attempt"] = 2
        with self.assertRaisesRegex(runner.PoolHeadError, "attempt index"):
            runner.validate_head_receipt(self.plan, reseal(receipt), maximum_attempts=2)

    def test_exact_authorization_receipt_and_mutation(self) -> None:
        receipt = runner._read_json(ROOT / runner.AUTHORIZATION_RELATIVE)
        runner.validate_authorization_receipt(receipt)
        receipt["scope_binding"]["pool_parent_count"] = 47
        with self.assertRaisesRegex(runner.PoolHeadError, "scope drift"):
            runner.validate_authorization_receipt(reseal(receipt))

    def test_resealed_authorization_context_or_authority_mutation_fails_closed(self) -> None:
        receipt = runner._read_json(ROOT / runner.AUTHORIZATION_RELATIVE)
        receipt["authorization_request_verbatim"] = "完全不同且未授权的上下文"
        with self.assertRaisesRegex(runner.PoolHeadError, "identity drift"):
            runner.validate_authorization_receipt(reseal(receipt))
        receipt = runner._read_json(ROOT / runner.AUTHORIZATION_RELATIVE)
        receipt["authority"] = "UNBOUNDED"
        with self.assertRaisesRegex(runner.PoolHeadError, "authority text drift"):
            runner.validate_authorization_receipt(reseal(receipt))


if __name__ == "__main__":
    unittest.main()
