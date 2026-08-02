import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_stage_c_d6_sanpo_motion_alignment_separability import (
    align_previous,
    residual_features,
)


class SanpoMotionAlignmentSeparabilityTest(unittest.TestCase):
    def test_alignment_reduces_synthetic_camera_translation(self):
        previous = np.zeros((128, 224), dtype=np.float32)
        for y in range(12, 118, 16):
            for x in range(12, 214, 18):
                cv2.circle(previous, (x, y), 3, 1.0, -1)
        matrix = np.asarray(
            [[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]],
            dtype=np.float32,
        )
        current = cv2.warpAffine(
            previous,
            matrix,
            (224, 128),
        )

        aligned, diagnostic = align_previous(previous, current)

        self.assertIsNotNone(aligned)
        self.assertEqual(diagnostic["reason"], "ok")
        aligned_image, valid = aligned
        raw_error = np.abs(current - previous).mean()
        aligned_error = np.abs(current - aligned_image)[
            valid > 0.5
        ].mean()
        self.assertLess(aligned_error, raw_error * 0.1)

    def test_residual_feature_shape_is_fixed(self):
        residual = np.linspace(
            0.0,
            1.0,
            128 * 224,
            dtype=np.float32,
        ).reshape(128, 224)
        features = residual_features(
            residual,
            np.ones_like(residual),
        )

        self.assertEqual(features.shape, (54,))
        self.assertTrue(np.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
