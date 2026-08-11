import unittest

from scripts.research.hftf.deployment.depthart.materialize_depthart_task_preserving_d2_phase_b import (
    add_counts,
    qualifies,
    summarize_truth,
)


class D2PhaseBMaterializerTest(unittest.TestCase):
    def test_truth_summary_counts_known_clear_and_occupied(self) -> None:
        truth = {
            "bands": {
                "left": {"clearance_m": 0.8, "occupied_by_horizon": {"1.0": True, "1.5": True, "2.0": True}},
                "center": {"clearance_m": None, "occupied_by_horizon": {"1.0": False, "1.5": None, "2.0": None}},
                "right": {"clearance_m": None, "occupied_by_horizon": {"1.0": None, "1.5": None, "2.0": None}},
            }
        }
        result = summarize_truth(truth)
        self.assertEqual(4, result["known_cells"])
        self.assertEqual(1, result["clear_cells"])
        self.assertEqual(3, result["occupied_cells"])
        self.assertEqual(1, result["valid_band_clearances"])

    def test_add_counts_accumulates_grid(self) -> None:
        frame = summarize_truth({"bands": {}})
        total = summarize_truth({"bands": {}})
        frame["known_cells"] = 2
        frame["known_by_grid"]["left@1.0m"] = 2
        add_counts(total, frame)
        self.assertEqual(2, total["known_cells"])
        self.assertEqual(2, total["known_by_grid"]["left@1.0m"])

    def test_qualification_fails_every_missing_denominator(self) -> None:
        counts = summarize_truth({"bands": {}})
        thresholds = {
            "minimum_truth_known_cells_per_identity": 1,
            "minimum_truth_clear_cells_per_identity": 1,
            "minimum_truth_occupied_cells_per_identity": 1,
            "minimum_valid_band_clearances_per_identity": 1,
            "minimum_truth_known_cells_per_band_horizon": 1,
        }
        passed, failures = qualifies(counts, thresholds)
        self.assertFalse(passed)
        self.assertGreaterEqual(len(failures), 13)


if __name__ == "__main__":
    unittest.main()
