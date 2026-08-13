from __future__ import annotations

import unittest

from scripts.research.hftf.deployment.depthart.confirm_depthart_d3r5_parent_relative_veto import (
    select_fresh_rows,
)


def row(pool_order: int, video_id: str, observable: bool = True) -> dict:
    return {
        "pool_order": pool_order,
        "video_id": video_id,
        "coverage_evaluable": True,
        "truth_support": {
            "known_cells": 1800 if observable else 1799,
            "valid_band_clearances": 450,
        },
    }


class FreshConfirmationPlanTest(unittest.TestCase):
    def test_selects_pool_ordered_observable_unused_parents(self) -> None:
        rows = [
            row(4, "used"),
            row(3, "low-support", observable=False),
            row(2, "fresh-b"),
            row(1, "fresh-a"),
        ]
        selected = select_fresh_rows(rows, {"used"}, count=2)
        self.assertEqual([item["video_id"] for item in selected], ["fresh-a", "fresh-b"])


if __name__ == "__main__":
    unittest.main()
