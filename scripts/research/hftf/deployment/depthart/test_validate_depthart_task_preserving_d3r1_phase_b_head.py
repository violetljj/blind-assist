import copy
import unittest

from scripts.research.hftf.deployment.depthart.preflight_depthart_task_preserving_d3r1_phase_b_assets import (
    ASSETS,
    INCOMPLETE_TERMINAL,
    PASS_SUCCESSOR,
    PASS_TERMINAL,
    PROTOCOL_ID,
    RESULT_SCHEMA,
    UNAVAILABLE_TERMINAL,
    request_plan_sha256,
    requests_for,
)
from scripts.research.hftf.deployment.depthart.test_preflight_depthart_task_preserving_d3r1_phase_b_assets import (
    fixtures,
)
from scripts.research.hftf.deployment.depthart.validate_depthart_task_preserving_d3r1_phase_b_head import (
    validate_result,
)


def fixture_result() -> tuple[dict, dict, dict, dict]:
    scope, phase_a_result = fixtures()
    protocol = {"base_url": "https://example.invalid/raw", "max_attempts": 3}
    plan = requests_for(scope, phase_a_result, protocol["base_url"])
    assets = []
    for row in plan:
        assets.append(
            row
            | {
                "attempts": 1,
                "http_status": 200,
                "final_url": row["url"],
                "redirect_count": 0,
                "content_length_bytes": 123,
                "etag": '"etag"',
                "last_modified": "date",
                "response_body_bytes_read": 0,
                "attempt_history": [
                    {"attempt": 1, "method": "HEAD", "http_status": 200, "error": None}
                ],
                "recovered_error": False,
                "unresolved_error": False,
            }
        )
    result = {
        "schema": RESULT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "terminal": PASS_TERMINAL,
        "next_gate": PASS_SUCCESSOR,
        "request_plan_sha256": request_plan_sha256(plan),
        "request_method": "HEAD",
        "expected_request_count": 64,
        "asset_count": 64,
        "available_asset_count": 64,
        "response_body_bytes_read": 0,
        "media_body_bytes_read": 0,
        "archive_member_read": False,
        "depth_confidence_decoded": False,
        "source_truth_support_read": False,
        "truth_or_model_output_read": False,
        "phase_b_selection_made": False,
        "role_assignment_made": False,
        "r2_cohort_access": "NONE",
        "total_content_length_bytes": 64 * 123,
        "authority": "HEAD availability and declared-size evidence only; no body authority.",
        "assets": assets,
    }
    return result, protocol, scope, phase_a_result


class D3R1PhaseBHeadValidatorTest(unittest.TestCase):
    def test_exact_pass_fixture(self) -> None:
        result, protocol, scope, phase_a = fixture_result()
        summary = validate_result(result, protocol, scope, phase_a)
        self.assertEqual(PASS_TERMINAL, summary["scientific_terminal"])
        self.assertEqual({32}, {entry["count"] for entry in summary["by_asset"].values()})
        self.assertEqual(set(ASSETS), set(summary["by_asset"]))

    def test_legitimate_unavailable_terminal_validates(self) -> None:
        result, protocol, scope, phase_a = fixture_result()
        result["assets"][0]["etag"] = None
        result["assets"][0]["unresolved_error"] = True
        result["available_asset_count"] = 63
        result["terminal"] = UNAVAILABLE_TERMINAL
        result["next_gate"] = None
        summary = validate_result(result, protocol, scope, phase_a)
        self.assertEqual(UNAVAILABLE_TERMINAL, summary["scientific_terminal"])
        self.assertEqual(1, summary["unresolved_error_row_count"])

    def test_legitimate_incomplete_terminal_validates(self) -> None:
        result, protocol, scope, phase_a = fixture_result()
        row = result["assets"][0]
        row.update(
            {
                "attempts": 3,
                "http_status": 500,
                "final_url": None,
                "content_length_bytes": None,
                "etag": None,
                "last_modified": None,
                "attempt_history": [
                    {"attempt": 1, "method": "HEAD", "http_status": 500, "error": "HTTPError"},
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
        summary = validate_result(result, protocol, scope, phase_a)
        self.assertEqual(INCOMPLETE_TERMINAL, summary["scientific_terminal"])

    def test_non_transient_retry_is_rejected(self) -> None:
        result, protocol, scope, phase_a = fixture_result()
        row = result["assets"][0]
        row["attempts"] = 2
        row["attempt_history"] = [
            {"attempt": 1, "method": "HEAD", "http_status": 404, "error": "not found"},
            {"attempt": 2, "method": "HEAD", "http_status": 200, "error": None},
        ]
        with self.assertRaisesRegex(ValueError, "non-transient attempt was retried"):
            validate_result(result, protocol, scope, phase_a)

    def test_request_order_drift_fails(self) -> None:
        result, protocol, scope, phase_a = fixture_result()
        mutated = copy.deepcopy(result)
        mutated["assets"][0], mutated["assets"][1] = mutated["assets"][1], mutated["assets"][0]
        with self.assertRaisesRegex(ValueError, "request plan drift"):
            validate_result(mutated, protocol, scope, phase_a)

    def test_get_attempt_fails(self) -> None:
        result, protocol, scope, phase_a = fixture_result()
        result["assets"][0]["attempt_history"][0]["method"] = "GET"
        with self.assertRaisesRegex(ValueError, "attempt history drift"):
            validate_result(result, protocol, scope, phase_a)

    def test_scientific_terminal_mismatch_fails(self) -> None:
        result, protocol, scope, phase_a = fixture_result()
        result["terminal"] = UNAVAILABLE_TERMINAL
        result["next_gate"] = None
        with self.assertRaisesRegex(ValueError, "scientific terminal drift"):
            validate_result(result, protocol, scope, phase_a)


if __name__ == "__main__":
    unittest.main()
