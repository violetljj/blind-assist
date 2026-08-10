from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_ag_st_student_checkpoint import (  # noqa: E402
    _checkpoint_parent_ids,
    _core_factor_names,
    _diagnostic_parent_split,
    _improvements,
)


class AgStFreshZeroShotEvaluatorTest(unittest.TestCase):
    def test_depth_support_checkpoint_excludes_obstacle_and_boundary_from_core(self) -> None:
        self.assertEqual(
            ("depth_mae", "support_bce"),
            _core_factor_names("depth_support"),
        )

    def test_checkpoint_parent_firewall_includes_every_previous_role(self) -> None:
        payload = {
            "split": {
                "train_parents": ["a", "b"],
                "selection_parents": ["c"],
                "canary_parents": ["d"],
            }
        }
        self.assertEqual({"a", "b", "c", "d"}, _checkpoint_parent_ids(payload))

    def test_metric_improvement_uses_parent_macro_and_preserves_regression_sign(self) -> None:
        before = {
            "parent_macro": {
                "depth_mae_m": 2.0,
                "support_bce": 1.0,
                "boundary_bce": 0.1,
                "boundary_soft_bce": 0.2,
                "boundary_distance_mae_px": 4.0,
                "obstacle_bce": 0.8,
            }
        }
        after = {
            "parent_macro": {
                "depth_mae_m": 1.0,
                "support_bce": 0.5,
                "boundary_bce": 0.2,
                "boundary_soft_bce": 0.1,
                "boundary_distance_mae_px": 2.0,
                "obstacle_bce": 0.4,
            }
        }
        values = _improvements(before, after)
        self.assertEqual(0.5, values["depth_mae"])
        self.assertEqual(0.5, values["support_bce"])
        self.assertEqual(-1.0, values["boundary_bce"])
        self.assertEqual(0.5, values["boundary_soft_bce"])

    def test_diagnostic_split_is_deterministic_and_disjoint(self) -> None:
        parents = {f"p{index:02d}" for index in range(16)}
        first = _diagnostic_parent_split(parents)
        second = _diagnostic_parent_split(set(reversed(sorted(parents))))
        self.assertEqual(first, second)
        fit, selection, canary, receipt = first
        self.assertEqual((12, 2, 2), (len(fit), len(selection), len(canary)))
        self.assertEqual(16, len(set(fit) | set(selection) | set(canary)))
        self.assertIn("DIAGNOSTIC_ONLY", receipt["method"])


if __name__ == "__main__":
    unittest.main()
