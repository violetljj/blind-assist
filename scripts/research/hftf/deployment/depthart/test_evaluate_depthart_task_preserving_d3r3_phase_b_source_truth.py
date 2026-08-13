from __future__ import annotations

import unittest

from scripts.research.hftf.deployment.depthart import (
    evaluate_depthart_task_preserving_d3r3_phase_b_source_truth as subject,
)


class D3R3SourceTruthUnknownTest(unittest.TestCase):
    def test_missing_modalities_become_unknown_without_substitution(self) -> None:
        result = subject.resolve_frame_availability(
            ["a", "b", "c", "d"], {"a", "c", "d"}, {"a", "b", "d"}
        )
        self.assertEqual(["a", "d"], result["available"])
        self.assertEqual(["b", "c"], result["source_unavailable"])
        self.assertEqual(["b"], result["depth_missing"])
        self.assertEqual(["c"], result["confidence_missing"])

    def test_selection_keeps_phase_a_order(self) -> None:
        processed = [
            {
                "selection_order": order,
                "pool_order": order + 10,
                "visit_id": str(order),
                "video_id": str(order + 100),
                "selected_frame_plan_sha256": "A" * 64,
                "source_unavailable_frame_count": int(order == 2),
                "source_truth_support_qualified": order != 3,
                "strict_complete_case_qualified": order not in {2, 3},
            }
            for order in range(1, 20)
        ]
        primary = subject.select_first(processed, "source_truth_support_qualified")
        strict = subject.select_first(processed, "strict_complete_case_qualified")
        self.assertEqual(
            [1, 2, *range(4, 18)],
            [row["phase_a_selection_order"] for row in primary],
        )
        self.assertEqual(16, len(primary))
        self.assertEqual(16, len(strict))
        self.assertNotEqual(
            [row["video_id"] for row in primary], [row["video_id"] for row in strict]
        )


if __name__ == "__main__":
    unittest.main()
