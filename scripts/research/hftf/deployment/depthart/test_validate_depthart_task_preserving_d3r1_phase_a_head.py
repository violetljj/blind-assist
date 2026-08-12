import copy
import unittest

from scripts.research.hftf.deployment.depthart.preflight_depthart_task_preserving_d3r1_phase_a_assets import (
    ASSETS,
    PASS_TERMINAL,
    PASS_SUCCESSOR,
    PROTOCOL_ID,
    requests_for,
)
from scripts.research.hftf.deployment.depthart.test_preflight_depthart_task_preserving_d3r1_phase_a_assets import (
    fixture_roster,
)
from scripts.research.hftf.deployment.depthart.validate_depthart_task_preserving_d3r1_phase_a_head import (
    validate_result,
)


def fixture_result() -> tuple[dict, dict, dict]:
    roster = fixture_roster()
    protocol = {"base_url": "https://example.invalid/raw", "max_attempts": 3}
    assets = []
    for row in requests_for(roster, protocol["base_url"]):
        assets.append(
            row
            | {
                "attempts": 1,
                "http_status": 200,
                "final_url": row["url"],
                "redirected": False,
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
        "schema": "blindassist_depthart_task_preserving_d3r1_phase_a_asset_header_preflight_v1",
        "protocol_id": PROTOCOL_ID,
        "terminal": PASS_TERMINAL,
        "next_gate": PASS_SUCCESSOR,
        "request_method": "HEAD",
        "expected_request_count": 254,
        "asset_count": 254,
        "available_asset_count": 254,
        "response_body_bytes_read": 0,
        "media_body_bytes_read": 0,
        "archive_member_read": False,
        "pose_content_read": False,
        "truth_or_model_output_read": False,
        "role_assignment_made": False,
        "r2_cohort_access": "NONE",
        "total_content_length_bytes": 254 * 123,
        "authority": "HEAD availability and declared-size evidence only; no body authority.",
        "assets": assets,
    }
    return result, protocol, roster


class D3R1PhaseAHeadValidatorTest(unittest.TestCase):
    def test_exact_fixture_passes(self) -> None:
        result, protocol, roster = fixture_result()
        summary = validate_result(result, protocol, roster)
        self.assertEqual(254, summary["etag_present_count"])
        self.assertEqual(0, summary["redirect_count"])
        self.assertEqual({127}, {entry["count"] for entry in summary["by_asset"].values()})
        self.assertEqual(set(ASSETS), set(summary["by_asset"]))

    def test_request_order_drift_fails(self) -> None:
        result, protocol, roster = fixture_result()
        mutated = copy.deepcopy(result)
        mutated["assets"][0], mutated["assets"][1] = mutated["assets"][1], mutated["assets"][0]
        with self.assertRaisesRegex(ValueError, "request plan drift"):
            validate_result(mutated, protocol, roster)

    def test_get_attempt_fails(self) -> None:
        result, protocol, roster = fixture_result()
        result["assets"][0]["attempt_history"][0]["method"] = "GET"
        with self.assertRaisesRegex(ValueError, "attempt history drift"):
            validate_result(result, protocol, roster)


if __name__ == "__main__":
    unittest.main()
