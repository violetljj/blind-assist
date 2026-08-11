import unittest

from scripts.research.hftf.deployment.depthart.preflight_depthart_task_preserving_d2_phase_c_rgb_assets import (
    disposition,
    requests_for,
)


def fixtures() -> tuple[dict, dict]:
    scope_rows = [
        {"role": "D2_TRAIN" if index < 4 else "D2_DEVELOPMENT_SEALED", "visit_id": f"v{index}", "video_id": f"s{index}"}
        for index in range(8)
    ]
    result_rows = [row | {"role_order": index % 4 + 1} for index, row in enumerate(scope_rows)]
    return {"identity_scope": scope_rows}, {"role_assignments": result_rows}


class D2PhaseCRgbHeadTest(unittest.TestCase):
    def test_exact_eight_requests(self) -> None:
        scope, result = fixtures()
        rows = requests_for(scope, result, "https://example.invalid/raw")
        self.assertEqual(8, len(rows))
        self.assertTrue(all(row["asset"] == "lowres_wide.zip" for row in rows))

    def test_result_binding_drift_fails(self) -> None:
        scope, result = fixtures()
        result["role_assignments"][0]["video_id"] = "changed"
        with self.assertRaisesRegex(ValueError, "role binding drift"):
            requests_for(scope, result, "https://example.invalid/raw")

    def test_disposition_fails_closed(self) -> None:
        rows = [{"http_status": 200, "content_length_bytes": 1} for _ in range(8)]
        self.assertEqual("D2_PHASE_C_RGB_HEADERS_AVAILABLE_BODY_UNOPENED", disposition(rows))
        rows[0]["http_status"] = 404
        self.assertEqual("D2_PHASE_C_RGB_NOT_AVAILABLE_BODY_UNOPENED", disposition(rows))


if __name__ == "__main__":
    unittest.main()
