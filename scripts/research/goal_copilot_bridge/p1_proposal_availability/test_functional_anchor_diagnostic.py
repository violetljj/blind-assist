from __future__ import annotations

import unittest

from scripts.research.goal_copilot_bridge.p1_proposal_availability.evaluate_functional_anchor_diagnostic import anchor_diagnostic


class FunctionalAnchorDiagnosticTest(unittest.TestCase):
    def test_reports_first_center_containing_candidate_without_relabeling_iou(self) -> None:
        result = anchor_diagnostic([10, 10, 30, 30], [
            {"rank": 1, "bbox_xyxy": [0, 0, 5, 5]},
            {"rank": 2, "bbox_xyxy": [15, 15, 25, 25]},
        ])
        self.assertEqual(2, result["first_containing_rank"])
        self.assertEqual(0.25, result["target_area_coverage"])
        self.assertEqual(0.25, result["candidate_to_target_area_ratio"])
        self.assertEqual(0.25, result["iou"])

    def test_reports_absent_anchor(self) -> None:
        result = anchor_diagnostic([10, 10, 30, 30], [{"rank": 1, "bbox_xyxy": [0, 0, 5, 5]}])
        self.assertIsNone(result["first_containing_rank"])
        self.assertEqual(0.0, result["target_area_coverage"])


if __name__ == "__main__":
    unittest.main()
