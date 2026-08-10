from __future__ import annotations

import unittest

from scripts.research.taro_o0r_factor_headroom_runtime.factor_evaluator import frozen_query_strata


class FactorEvaluatorTests(unittest.TestCase):
    def test_frozen_query_strata_are_source_and_truth_derived(self) -> None:
        strata = frozen_query_strata(
            {"value_m": 0.08, "knownness": {"known": True}},
            {"confidence_value": 2, "range_m": 1.4},
        )
        self.assertEqual(strata["orientation"], "LANDSCAPE")
        self.assertTrue(strata["near_decision_boundary_0p10m"])
        self.assertEqual(strata["appledepth_confidence"], 2)
        self.assertEqual(strata["range_band_m"], "RANGE_1_TO_2_M")

    def test_unknown_truth_is_not_near_boundary(self) -> None:
        strata = frozen_query_strata(
            {"value_m": None, "knownness": {"known": False}},
            {"confidence_value": 1, "range_m": 0.8},
        )
        self.assertFalse(strata["near_decision_boundary_0p10m"])


if __name__ == "__main__":
    unittest.main()
