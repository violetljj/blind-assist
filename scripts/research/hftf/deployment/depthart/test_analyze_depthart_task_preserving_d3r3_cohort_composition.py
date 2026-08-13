from __future__ import annotations

import unittest

from scripts.research.hftf.deployment.depthart import (
    analyze_depthart_task_preserving_d3r3_cohort_composition as subject,
)


def row(order: int, clear: int, occupied: int) -> dict:
    grid = {"center@1.0m": clear}
    occupied_grid = {"center@1.0m": occupied}
    return {
        "selection_order": order,
        "pool_order": order,
        "visit_id": str(order),
        "video_id": str(order + 100),
        "source_unavailable_frame_count": 0,
        "coverage_evaluable": True,
        "truth_support": {
            "known_cells": 2700,
            "clear_cells": max(clear, 270),
            "occupied_cells": max(occupied, 900),
            "valid_band_clearances": 900,
            "clear_by_grid": grid,
            "occupied_by_grid": occupied_grid,
        },
    }


class CohortCompositionTest(unittest.TestCase):
    def test_solver_builds_disjoint_complementary_roles(self) -> None:
        rows = [row(1, 40, 1000), row(2, 35, 1100), row(3, 50, 1200), row(4, 45, 1300)]
        train, development, optimum = subject.solve_role_split(rows, role_size=1)
        self.assertGreaterEqual(optimum, 35)
        self.assertEqual(1, train["identity_count"])
        self.assertEqual(1, development["identity_count"])
        self.assertTrue(
            set(train["phase_a_selection_orders"]).isdisjoint(
                development["phase_a_selection_orders"]
            )
        )


if __name__ == "__main__":
    unittest.main()
