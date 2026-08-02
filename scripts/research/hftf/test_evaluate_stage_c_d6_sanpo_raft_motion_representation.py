import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_stage_c_d6_sanpo_raft_motion_representation import (
    grid_features,
    remove_global_affine,
)


class SanpoRaftMotionRepresentationTest(unittest.TestCase):
    def test_global_affine_flow_is_removed(self):
        height, width = 128, 224
        y, x = np.mgrid[0:height, 0:width]
        flow = np.stack(
            (
                4.0 + 0.002 * x,
                -2.0 + 0.002 * y,
            )
        ).astype(np.float32)
        flow[:, 40:70, 80:120] += np.asarray(
            [3.0, 2.0],
            dtype=np.float32,
        )[:, None, None]

        residual, diagnostic = remove_global_affine(flow)

        self.assertIsNotNone(residual)
        self.assertEqual(diagnostic["reason"], "ok")
        background = np.ones((height, width), dtype=bool)
        background[40:70, 80:120] = False
        background_magnitude = np.sqrt(
            np.square(residual[0]) + np.square(residual[1])
        )[background].mean()
        foreground_magnitude = np.sqrt(
            np.square(residual[0]) + np.square(residual[1])
        )[40:70, 80:120].mean()
        self.assertLess(background_magnitude, 0.1)
        self.assertGreater(foreground_magnitude, 3.0)

    def test_grid_feature_shape_is_fixed(self):
        values = np.ones((128, 224), dtype=np.float32)
        features = grid_features(values, 0.5)

        self.assertEqual(features.shape, (54,))
        self.assertTrue(np.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
