from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).with_name("build_dav2_android_parity_corpus_r0.py")
SPEC = importlib.util.spec_from_file_location("dav2_parity_corpus", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Dav2AndroidParityCorpusR0Test(unittest.TestCase):
    def test_center_crop_is_camera_shaped_and_contiguous(self) -> None:
        source = np.arange(640 * 640 * 3, dtype=np.uint8).reshape(640, 640, 3)
        cropped = MODULE.center_crop_camera_frame(source)
        self.assertEqual(cropped.shape, (480, 640, 3))
        self.assertTrue(cropped.flags.c_contiguous)
        np.testing.assert_array_equal(cropped[0], source[80])
        np.testing.assert_array_equal(cropped[-1], source[559])

    def test_stress_transforms_match_frozen_parameters(self) -> None:
        impulse = np.zeros((480, 640, 3), dtype=np.uint8)
        impulse[240, 320] = 255
        gaussian = MODULE.perturb_bgr(impulse, "gaussian_sigma3")
        motion = MODULE.perturb_bgr(impulse, "motion_horizontal_length17")
        self.assertGreater(np.count_nonzero(gaussian), 17 * 3)
        self.assertEqual(np.count_nonzero(motion[240, :, 0]), 17)
        self.assertEqual(np.count_nonzero(motion[:, 320, 0]), 1)

    def test_downstream_depth_perturbations(self) -> None:
        depth = np.ones((480, 640), dtype=np.float32)
        masked = MODULE.mask_lower_roi_half(depth)
        self.assertEqual(int(np.sum(~np.isfinite(masked))), 108 * 640)
        local = MODULE.local_horizontal_linear(depth)
        self.assertAlmostEqual(float(np.median(local)), 1.0)
        self.assertAlmostEqual(float(local[0, 0]), 0.8)
        self.assertAlmostEqual(float(local[0, -1]), 1.2)

    def test_parity_metrics_detect_exact_and_mismatch(self) -> None:
        reference = np.asarray([[1.0, 2.0]], dtype=np.float32)
        exact = MODULE.parity_metrics(reference, reference.copy())
        self.assertTrue(exact["allclose_rtol_1e_4_atol_1e_4"])
        self.assertEqual(exact["maximum_absolute_error_m"], 0.0)
        mismatch = MODULE.parity_metrics(reference, reference + 0.01)
        self.assertFalse(mismatch["allclose_rtol_1e_4_atol_1e_4"])
        self.assertGreater(mismatch["root_mean_squared_error_m"], 0.009)


if __name__ == "__main__":
    unittest.main()
