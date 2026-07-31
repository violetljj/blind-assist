from __future__ import annotations

import unittest

from .evaluate import _event_score, _select_ring, _state, _wrong_signed


class EvaluateTests(unittest.TestCase):
    def test_event_median_and_pair_coverage_boundaries(self):
        score, evaluable, coverage = _event_score([0.01, 0.03, 0.02], 5)
        self.assertTrue(evaluable)
        self.assertAlmostEqual(score, 0.02)
        self.assertAlmostEqual(coverage, 0.6)
        score, evaluable, coverage = _event_score([0.01, 0.03], 5)
        self.assertFalse(evaluable)
        self.assertIsNone(score)
        self.assertAlmostEqual(coverage, 0.4)

    def test_deadband_edges_are_quasi_static(self):
        self.assertEqual(_state(0.02), "quasi-static")
        self.assertEqual(_state(-0.02), "quasi-static")
        self.assertEqual(_state(0.020001), "approach")
        self.assertEqual(_state(-0.020001), "receding")

    def test_wrong_signed_excludes_abstention(self):
        self.assertFalse(_wrong_signed("approach", None))
        self.assertTrue(_wrong_signed("approach", "receding"))
        self.assertTrue(_wrong_signed("quasi-static", "approach"))

    def test_selection_uses_fixed_tie_break(self):
        base = {
            "metrics": {"paired_event_gain_count": 2, "residual_coverage": 0.8},
            "by_session": {
                "S1": {"paired_event_gain_count": 1, "residual_evaluable_event_count": 1, "residual_coverage": 0.8},
                "S2": {"paired_event_gain_count": 1, "residual_evaluable_event_count": 1, "residual_coverage": 0.8},
            },
            "median_ring_area_px": 100,
        }
        other = {**base, "median_ring_area_px": 120}
        selected, status = _select_ring({"R1": base, "R2": other}, ["S1", "S2"])
        self.assertEqual(selected, "R1")
        self.assertEqual(status, "UNIQUE_SELECTION")


if __name__ == "__main__":
    unittest.main()
