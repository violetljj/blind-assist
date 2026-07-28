from __future__ import annotations

import unittest

import numpy as np

from scripts.research.egomotion_compensated_looming.degradation_flow_quality_diagnostic_r0.analyze import (
    gated_triggers,
)
from scripts.research.egomotion_compensated_looming.degradation_flow_quality_diagnostic_r0.extract import (
    centered_rolling_median,
    gate_reasons,
    rank_average,
)


class DiagnosticTest(unittest.TestCase):
    def test_rank_average_is_deterministic_with_ties(self) -> None:
        actual = rank_average(np.asarray([3.0, 1.0, 1.0, 2.0]))
        np.testing.assert_allclose(actual, [1.0, 1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0])

    def test_centered_rolling_median_uses_bounded_edges(self) -> None:
        actual = centered_rolling_median(
            np.asarray([0.0, 100.0, 2.0, 3.0, 4.0]), 3
        )
        np.testing.assert_allclose(actual, [50.0, 2.0, 3.0, 3.0, 3.5])

    def test_gate_is_mechanical(self) -> None:
        good = {
            "detected_feature_count": 120,
            "forward_backward_consistent_count": 90,
            "forward_backward_consistent_fraction": 0.75,
            "median_forward_backward_error_px": 0.2,
            "occupied_grid_cells": 9,
        }
        self.assertEqual(gate_reasons(good, 0.75), [])
        bad = dict(good)
        bad["forward_backward_consistent_count"] = 59
        bad["occupied_grid_cells"] = 4
        self.assertEqual(
            gate_reasons(bad, 0.75),
            ["FB_TRACKS_LT_60", "OCCUPIED_CELLS_LT_5"],
        )

    def test_gate_abstention_resets_unchanged_three_pair_rule(self) -> None:
        rows = [
            {
                "evaluable": True,
                "compensated_expansion_median_per_s": 0.02,
            }
            for _ in range(7)
        ]
        accepted = np.asarray(
            [True, True, True, False, True, True, True], dtype=bool
        )
        np.testing.assert_array_equal(
            gated_triggers(rows, accepted),
            [False, False, True, False, False, False, True],
        )

    def test_threshold_remains_strict(self) -> None:
        rows = [
            {
                "evaluable": True,
                "compensated_expansion_median_per_s": value,
            }
            for value in (0.01, 0.02, 0.02, 0.02)
        ]
        np.testing.assert_array_equal(
            gated_triggers(rows, np.ones(4, dtype=bool)),
            [False, False, False, True],
        )


if __name__ == "__main__":
    unittest.main()
