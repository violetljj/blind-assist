import unittest

import numpy as np

from evaluate_collision_risk_field_a1_incremental_ablation import evaluate_arrays
from evaluate_motion_conditioned_occupancy_a0 import FEATURE_NAMES


class CollisionRiskFieldA1IncrementalAblationTest(unittest.TestCase):
    def test_reports_consumed_terminal_without_changing_a1(self) -> None:
        rng = np.random.default_rng(7)
        rows = 120
        x = rng.normal(size=(rows, len(FEATURE_NAMES)))
        groups = np.asarray([f"w{index // 20}" for index in range(rows)])
        y = (x[:, 0] + 1.5 * x[:, 8] > 0).astype(np.float64)
        result = evaluate_arrays(x, y, groups)
        self.assertEqual(result["opportunities"], rows)
        self.assertEqual(result["windows"], 6)
        self.assertEqual(
            result["preserved_terminal"], "COLLISION_RISK_FIELD_A1_DEVELOPMENT_FAIL"
        )
        self.assertEqual(
            set(result["arms"]), {"geometry_only", "geometry_plus_motion"}
        )


if __name__ == "__main__":
    unittest.main()
