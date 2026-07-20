import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_public_silver_full_semantic_pair_adapter_probe as probe


class FullSemanticPairAdapterProbeTest(unittest.TestCase):
    def test_semantic_vector_has_three_regions_per_class(self):
        probabilities = np.zeros((4, 16, 16), dtype=np.float64)
        probabilities[0] = 1.0
        vector = probe.semantic_vector(probabilities)
        self.assertEqual((12,), vector.shape)
        self.assertTrue(np.isfinite(vector).all())

    def test_folds_exclude_all_held_source_descendants(self):
        real = [
            {"kind":"real","pair_id":"r1","parent_source_id":"s1","delta":np.asarray([1.0])},
            {"kind":"real","pair_id":"r2","parent_source_id":"s2","delta":np.asarray([1.0])},
            {"kind":"real","pair_id":"r3","parent_source_id":"s3","delta":np.asarray([1.0])},
        ]
        synthetic = [
            {"kind":"synthetic","pair_id":"a1","parent_source_id":"s1","delta":np.asarray([1.0])},
            {"kind":"synthetic","pair_id":"a2","parent_source_id":"s2","delta":np.asarray([1.0])},
            {"kind":"synthetic","pair_id":"a3","parent_source_id":"s3","delta":np.asarray([1.0])},
        ]
        self.assertTrue(all(row["held_out_source_descendants_excluded"] for row in probe.evaluate_folds(real, synthetic)))


if __name__ == "__main__":
    unittest.main()
