import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_public_silver_synthetic_pair_delta_probe as probe


class SyntheticPairDeltaProbeTest(unittest.TestCase):
    def test_prototype_direction_uses_unit_pair_contributions(self):
        direction = probe.prototype_direction([np.asarray([100.0, 0.0]), np.asarray([0.0, 1.0])])
        self.assertAlmostEqual(direction[0], direction[1])

    def test_fold_excludes_synthetic_descendants_of_holdout(self):
        real = [
            {"pair_id": "r1", "parent_source_id": "s1", "delta": np.asarray([1.0, 0.0])},
            {"pair_id": "r2", "parent_source_id": "s2", "delta": np.asarray([1.0, 0.1])},
            {"pair_id": "r3", "parent_source_id": "s3", "delta": np.asarray([1.0, -0.1])},
        ]
        synthetic = [
            {"pair_id": "a1", "parent_source_id": "s1", "delta": np.asarray([1.0, 0.0])},
            {"pair_id": "a2", "parent_source_id": "s2", "delta": np.asarray([1.0, 0.0])},
            {"pair_id": "a3", "parent_source_id": "s3", "delta": np.asarray([1.0, 0.0])},
        ]
        folds = probe.fold_rows(real, synthetic)
        for fold in folds:
            blocked = fold["held_out_parent_source_id"]
            self.assertTrue(fold["held_out_source_descendants_excluded"])
            self.assertNotIn(f"a{blocked[-1]}", fold["training_synthetic_pair_ids"])


if __name__ == "__main__":
    unittest.main()
