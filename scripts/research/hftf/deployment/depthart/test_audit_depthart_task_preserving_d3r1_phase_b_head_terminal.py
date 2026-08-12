import copy
import unittest

from scripts.research.hftf.deployment.depthart.audit_depthart_task_preserving_d3r1_phase_b_head_terminal import (
    audit_result,
)
from scripts.research.hftf.deployment.depthart.preflight_depthart_task_preserving_d3r1_phase_b_assets import (
    INCOMPLETE_TERMINAL,
    UNAVAILABLE_TERMINAL,
)
from scripts.research.hftf.deployment.depthart.test_validate_depthart_task_preserving_d3r1_phase_b_head import (
    fixture_result,
)


class D3R1PhaseBHeadPostResultAuditTest(unittest.TestCase):
    def test_exact_pass_fixture(self) -> None:
        result, protocol, scope, phase_a = fixture_result()
        summary = audit_result(result, protocol, scope, phase_a)
        self.assertEqual(64, summary["available_asset_count"])
        self.assertEqual(0, summary["row_history_status_mismatch_count"])

    def test_row_history_status_contradiction_fails(self) -> None:
        result, protocol, scope, phase_a = fixture_result()
        result["assets"][0]["attempt_history"][0].update({"http_status": 404, "error": None})
        with self.assertRaisesRegex(ValueError, "row/history status mismatch"):
            audit_result(result, protocol, scope, phase_a)

    def test_redirect_contradiction_fails(self) -> None:
        result, protocol, scope, phase_a = fixture_result()
        result["assets"][0]["redirect_count"] = 1
        with self.assertRaisesRegex(ValueError, "redirect evidence mismatch"):
            audit_result(result, protocol, scope, phase_a)

    def test_unresolved_flag_cannot_self_define_availability(self) -> None:
        result, protocol, scope, phase_a = fixture_result()
        result["assets"][0]["unresolved_error"] = True
        result["available_asset_count"] = 63
        result["terminal"] = UNAVAILABLE_TERMINAL
        result["next_gate"] = None
        with self.assertRaisesRegex(ValueError, "availability flag mismatch"):
            audit_result(result, protocol, scope, phase_a)

    def test_recovered_flag_contradiction_fails(self) -> None:
        result, protocol, scope, phase_a = fixture_result()
        result["assets"][0]["recovered_error"] = True
        with self.assertRaisesRegex(ValueError, "recovered-error flag mismatch"):
            audit_result(result, protocol, scope, phase_a)

    def test_non_transient_retry_fails(self) -> None:
        result, protocol, scope, phase_a = fixture_result()
        row = result["assets"][0]
        row["attempts"] = 2
        row["attempt_history"] = [
            {"attempt": 1, "method": "HEAD", "http_status": 404, "error": "not found"},
            {"attempt": 2, "method": "HEAD", "http_status": 200, "error": None},
        ]
        with self.assertRaisesRegex(ValueError, "non-transient attempt was retried"):
            audit_result(result, protocol, scope, phase_a)

    def test_legitimate_incomplete_terminal(self) -> None:
        result, protocol, scope, phase_a = fixture_result()
        row = result["assets"][0]
        row.update(
            {
                "attempts": 3,
                "http_status": None,
                "final_url": None,
                "content_length_bytes": None,
                "etag": None,
                "last_modified": None,
                "attempt_history": [
                    {"attempt": 1, "method": "HEAD", "http_status": None, "error": "timeout"},
                    {"attempt": 2, "method": "HEAD", "http_status": None, "error": "timeout"},
                    {"attempt": 3, "method": "HEAD", "http_status": None, "error": "timeout"},
                ],
                "unresolved_error": True,
            }
        )
        result["available_asset_count"] = 63
        result["total_content_length_bytes"] -= 123
        result["terminal"] = INCOMPLETE_TERMINAL
        result["next_gate"] = None
        summary = audit_result(result, protocol, scope, phase_a)
        self.assertEqual(INCOMPLETE_TERMINAL, summary["scientific_terminal"])


if __name__ == "__main__":
    unittest.main()
