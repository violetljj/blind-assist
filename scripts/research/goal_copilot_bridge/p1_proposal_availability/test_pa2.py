from __future__ import annotations

import unittest

import numpy as np

from scripts.research.goal_copilot_bridge.p1_proposal_availability.run_pa2_oracle_representation_audit import (
    expand_box,
    prepare_arm,
    summarize,
)


class Pa2MechanicsTest(unittest.TestCase):
    def test_expand_box_clips_to_image(self) -> None:
        self.assertEqual([0.0, 0.0, 30.0, 30.0], expand_box([0, 0, 20, 20], 2.0, 100, 100))

    def test_exact_and_roi_crops_remap_truth(self) -> None:
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        target = [80.5, 40.5, 100.5, 60.5]
        exact, exact_crop, exact_target = prepare_arm(image, target, "exact_target_target_only")
        roi, roi_crop, roi_target = prepare_arm(image, target, "oracle_roi_target_only")
        self.assertEqual((21, 21), exact.shape[:2])
        self.assertEqual([80, 40, 101, 61], exact_crop)
        self.assertEqual([0.5, 0.5, 20.5, 20.5], exact_target)
        self.assertEqual((61, 61), roi.shape[:2])
        self.assertEqual([60, 20, 121, 81], roi_crop)
        self.assertEqual([20.5, 20.5, 40.5, 40.5], roi_target)

    def test_summary_uses_full_rank_and_k10_separately(self) -> None:
        cases = []
        for case_index in range(2):
            rows = {}
            for arm in ("exact_target_target_only", "oracle_roi_target_only", "oracle_roi_target_plus_context"):
                rows[arm] = {
                    "candidate_count": 12,
                    "latency_ms": 2.0 + case_index,
                    "first_correct_rank": {"0.1": 11 if case_index == 0 else None, "0.3": None, "0.5": None},
                }
            cases.append({"arms": rows})
        result = summarize(cases)
        self.assertEqual(0.5, result["exact_target_target_only"]["recall_full_rank"]["0.1"])
        self.assertEqual(0.0, result["exact_target_target_only"]["recall_at_10"]["0.1"])


if __name__ == "__main__":
    unittest.main()
