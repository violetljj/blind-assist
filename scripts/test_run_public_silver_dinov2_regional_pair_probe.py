import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_public_silver_dinov2_regional_pair_probe as probe


class DinoV2RegionalPairProbeTest(unittest.TestCase):
    def test_regional_vector_uses_five_hidden_blocks_and_is_unit_normalized(self):
        tokens = np.arange(257 * 3, dtype=np.float64).reshape(257, 3) + 1.0
        vector = probe.regional_vector(tokens)
        self.assertEqual((15,), vector.shape)
        self.assertAlmostEqual(1.0, float(np.linalg.norm(vector)), places=12)

    def test_fold_excludes_held_parent_descendants(self):
        real = [
            {"kind": "real", "pair_id": "r1", "parent_source_id": "s1", "delta": np.array([1.0, 0.0])},
            {"kind": "real", "pair_id": "r2", "parent_source_id": "s2", "delta": np.array([1.0, 0.0])},
        ]
        auxiliary = [
            {"kind": "synthetic", "pair_id": "x1", "parent_source_id": "s1", "delta": np.array([1.0, 0.0])},
            {"kind": "inverse", "pair_id": "i1", "parent_source_id": "s3", "delta": np.array([1.0, 0.0])},
        ]
        rows = probe.evaluate_folds(real, auxiliary)
        held = next(row for row in rows if row["held_out_parent_source_id"] == "s1")
        self.assertTrue(held["held_out_source_descendants_excluded"])
        self.assertNotIn("x1", held["training_forward_synthetic_pair_ids"])
        self.assertIn("i1", held["training_inverse_pair_ids"])


if __name__ == "__main__":
    unittest.main()
