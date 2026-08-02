import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose_stage_c_d6_sanpo_direction_profile import (
    pairwise_auc,
    profile_channels,
)


class SanpoDirectionProfileTest(unittest.TestCase):
    def test_pairwise_auc_counts_ties(self):
        self.assertEqual(1.0, pairwise_auc([2.0, 3.0], [0.0, 1.0]))
        self.assertEqual(0.5, pairwise_auc([1.0], [1.0]))
        self.assertEqual(0.0, pairwise_auc([0.0], [1.0]))

    def test_central_minus_lateral_profile(self):
        risk = np.zeros((1, 3, 3, 6, 6), dtype=np.float32)
        known = np.ones_like(risk)
        risk[:, 1:, 1:, 2:4, :] = 0.8
        risk[:, 1:, 1:, (0, 1, 4, 5), :] = 0.2
        channels = profile_channels(risk, known)
        self.assertAlmostEqual(
            0.6,
            float(
                channels[
                    "risk_mean/central_minus_lateral_mean"
                ][0]
            ),
            places=6,
        )
        self.assertAlmostEqual(
            0.8,
            float(channels["body_k3_support/central_mean"][0]),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
