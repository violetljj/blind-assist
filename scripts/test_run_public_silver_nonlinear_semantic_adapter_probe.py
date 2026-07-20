import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_public_silver_nonlinear_semantic_adapter_probe as probe


class NonlinearSemanticAdapterProbeTest(unittest.TestCase):
    def test_fit_is_deterministic_and_learns_pair_order(self):
        pairs = [
            {"clear": np.array([0.0, 0.2]), "risk": np.array([1.0, 0.8])},
            {"clear": np.array([0.1, 0.0]), "risk": np.array([0.9, 1.0])},
        ]
        first, mean1, scale1, losses1 = probe.fit_adapter(pairs, steps=40)
        second, mean2, scale2, losses2 = probe.fit_adapter(pairs, steps=40)
        self.assertTrue(np.array_equal(mean1, mean2))
        self.assertTrue(np.array_equal(scale1, scale2))
        self.assertEqual(losses1, losses2)
        self.assertLess(losses1[-1], losses1[0])
        for row in pairs:
            self.assertGreater(probe.score(first, row["risk"], mean1, scale1), probe.score(first, row["clear"], mean1, scale1))

    def test_real_fold_excludes_all_held_source_descendants(self):
        real = [
            {"kind": "real", "pair_id": "r1", "parent_source_id": "s1", "clear": np.array([0.0, 0.0]), "risk": np.array([1.0, 1.0])},
            {"kind": "real", "pair_id": "r2", "parent_source_id": "s2", "clear": np.array([0.0, 0.1]), "risk": np.array([0.9, 1.0])},
        ]
        synthetic = [
            {"kind": "synthetic", "pair_id": "x1", "parent_source_id": "s1", "clear": np.array([0.0, 0.0]), "risk": np.array([1.0, 0.8])},
            {"kind": "synthetic", "pair_id": "x2", "parent_source_id": "s3", "clear": np.array([0.1, 0.0]), "risk": np.array([0.8, 1.0])},
            {"kind": "inverse", "pair_id": "i1", "parent_source_id": "s4", "clear": np.array([0.0, 0.1]), "risk": np.array([1.0, 0.9])},
        ]
        rows = probe.evaluate_real_folds(real, synthetic)
        held_s1 = next(row for row in rows if row["held_out_parent_source_id"] == "s1")
        self.assertTrue(held_s1["held_out_source_descendants_excluded"])
        self.assertNotIn("x1", held_s1["training_synthetic_pair_ids"])
        self.assertIn("x2", held_s1["training_synthetic_pair_ids"])
        self.assertIn("i1", held_s1["training_inverse_pair_ids"])


if __name__ == "__main__":
    unittest.main()
