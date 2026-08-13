from __future__ import annotations

import unittest

from scripts.research.hftf.deployment.depthart import (
    analyze_depthart_d3r4_selective_horizon_certificate as subject,
)


def row(order: int, clear: int, occupied: int) -> dict:
    clear_grid = {f"{band}@{horizon:.1f}m": (clear if horizon == 1.0 else 0)
                  for band in ("left", "center", "right") for horizon in (1.0, 1.5, 2.0)}
    occupied_grid = {key: occupied for key in clear_grid}
    return {
        "selection_order": order,
        "pool_order": order,
        "visit_id": str(order),
        "video_id": str(order + 100),
        "source_unavailable_frame_count": 0,
        "coverage_evaluable": True,
        "truth_support": {
            "known_cells": 2700,
            "valid_band_clearances": 900,
            "clear_by_grid": clear_grid,
            "occupied_by_grid": occupied_grid,
        },
    }


class SelectiveHorizonCertificateTest(unittest.TestCase):
    def test_both_roles_have_parent_diverse_active_support(self) -> None:
        rows = [row(order, 40 + order, 100 + order) for order in range(1, 9)]
        train, development, optimum = subject.solve_selective_roles(rows, role_size=2)
        self.assertGreaterEqual(optimum, 80)
        self.assertGreaterEqual(train["minimum_parent_contributors"], 2)
        self.assertGreaterEqual(development["minimum_parent_contributors"], 2)


if __name__ == "__main__":
    unittest.main()
