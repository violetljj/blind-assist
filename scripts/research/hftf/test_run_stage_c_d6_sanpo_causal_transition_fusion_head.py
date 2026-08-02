import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_stage_c_d6_sanpo_causal_transition_fusion_head import (
    causal_transition_matrix,
    temporal_feature_names,
)


class SanpoCausalTransitionFusionHeadTest(unittest.TestCase):
    def test_transition_features_are_causal_and_one_second_bounded(self):
        event = {
            "frames": [
                {"timestamp_ms": index * 100}
                for index in range(14)
            ]
        }
        matrix = np.arange(28, dtype=np.float64).reshape(14, 2)
        output = causal_transition_matrix(event, matrix)
        self.assertEqual((14, 6), output.shape)
        # Sampled 5 Hz indices are 0, 2, ..., 12. At frame 12 the
        # one-second window starts five sampled steps earlier at frame 2.
        expected = np.concatenate(
            [
                matrix[12],
                matrix[12] - matrix[2],
                matrix[[2, 4, 6, 8, 10, 12]].mean(axis=0),
            ]
        )
        np.testing.assert_allclose(expected, output[12])
        np.testing.assert_allclose(np.zeros(6), output[11])

    def test_feature_names_follow_matrix_family_order(self):
        self.assertEqual(
            [
                "current/a",
                "current/b",
                "delta-1s/a",
                "delta-1s/b",
                "mean-prefix-1s/a",
                "mean-prefix-1s/b",
            ],
            temporal_feature_names(["a", "b"]),
        )


if __name__ == "__main__":
    unittest.main()
