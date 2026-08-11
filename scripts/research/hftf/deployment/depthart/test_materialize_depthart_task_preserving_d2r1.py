import unittest

from scripts.research.hftf.deployment.depthart.materialize_depthart_task_preserving_d2r1 import (
    earliest_qualified_window,
    role_assignments,
    split_continuous_portrait_runs,
)


def thresholds() -> dict:
    return {
        "minimum_truth_known_cells_per_identity": 3,
        "minimum_truth_clear_cells_per_identity": 1,
        "minimum_truth_occupied_cells_per_identity": 2,
        "minimum_valid_band_clearances_per_identity": 1,
        "minimum_truth_known_cells_per_band_horizon": 0,
    }


def counts(clear: int, occupied: int) -> dict:
    return {
        "known_cells": clear + occupied,
        "clear_cells": clear,
        "occupied_cells": occupied,
        "valid_band_clearances": 1,
        "known_by_grid": {
            f"{band}@{horizon:.1f}m": 0
            for band in ("left", "center", "right")
            for horizon in (1.0, 1.5, 2.0)
        },
    }


class D2R1MaterializerTest(unittest.TestCase):
    def test_nonportrait_and_gap_break_runs(self) -> None:
        rows = [
            {"stem": "a", "timestamp": 0.0, "portrait": True},
            {"stem": "b", "timestamp": 0.1, "portrait": True},
            {"stem": "c", "timestamp": 0.2, "portrait": False},
            {"stem": "d", "timestamp": 0.3, "portrait": True},
            {"stem": "e", "timestamp": 1.0, "portrait": True},
        ]
        self.assertEqual([["a", "b"], ["d"], ["e"]], [[row["stem"] for row in run] for run in split_continuous_portrait_runs(rows, 0.5)])

    def test_prefix_scan_selects_earliest_passing_window(self) -> None:
        run = [[{"stem": str(index), "timestamp": float(index), "portrait": True} for index in range(4)]]
        by_stem = {"0": counts(0, 1), "1": counts(0, 1), "2": counts(1, 0), "3": counts(1, 0)}
        result = earliest_qualified_window(run, lambda row: by_stem[row["stem"]], 3, thresholds())
        self.assertTrue(result["qualified"])
        self.assertEqual(["0", "1", "2"], result["selected_frame_stems"])
        self.assertEqual(1, result["windows_tested"])

    def test_roles_fail_closed_and_lock_first_eight(self) -> None:
        rows = [
            {"phase_a_order": i + 1, "pool_order": i + 10, "visit_id": f"v{i}", "video_id": f"s{i}", "selected_frame_stems": ["x"]}
            for i in range(9)
        ]
        self.assertEqual([], role_assignments(rows[:7]))
        roles = role_assignments(rows)
        self.assertEqual(8, len(roles))
        self.assertEqual(["D2_TRAIN"] * 4 + ["D2_DEVELOPMENT_SEALED"] * 4, [row["role"] for row in roles])
        self.assertEqual("s7", roles[-1]["video_id"])


if __name__ == "__main__":
    unittest.main()
