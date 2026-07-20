import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_public_silver_dinov2_bootstrap_short_runs as probe


class DinoV2BootstrapShortRunsTest(unittest.TestCase):
    def test_fit_is_deterministic_and_reduces_loss(self):
        pairs = [
            {"parent_source_id": "s1", "delta": np.array([1.0, 0.1])},
            {"parent_source_id": "s2", "delta": np.array([0.8, 0.2])},
            {"parent_source_id": "s3", "delta": np.array([0.9, -0.1])},
        ]
        first = probe.fit_pair_head(pairs, seed=7, steps=20)
        second = probe.fit_pair_head(pairs, seed=7, steps=20)
        self.assertTrue(np.array_equal(first["direction"], second["direction"]))
        self.assertEqual(first["sampled_source_ids"], second["sampled_source_ids"])
        self.assertLess(first["final_loss"], first["initial_loss"])

    def test_seed_gate_uses_real_folds_and_rice(self):
        real = [
            {"kind": "real", "pair_id": "r1", "parent_source_id": "s1", "delta": np.array([1.0, 0.0])},
            {"kind": "real", "pair_id": "r2", "parent_source_id": "s2", "delta": np.array([1.0, 0.0])},
        ]
        auxiliary = [{"kind": "inverse", "pair_id": "i1", "parent_source_id": "s3", "delta": np.array([1.0, 0.0])}]
        row = probe.evaluate_seed(real, auxiliary, np.array([1.0, 0.0]), np.array([1.0, 0.0]), seed=3)
        self.assertTrue(row["run_gate_passed"])
        self.assertEqual(1.0, row["real_pair_ordering_rate"])


if __name__ == "__main__":
    unittest.main()
