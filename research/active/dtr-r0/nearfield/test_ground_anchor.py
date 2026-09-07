import unittest
import numpy as np
import torch
from near_field import Camera, NearFieldEncoder
from ground_anchor import fit_ground, arm_inputs


class GroundTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.camera = Camera(320, 240, 90, 1.7, -10)
        self.encoder = NearFieldEncoder(self.camera)

    def floor(self, a=0., b=0.):
        den = (self.encoder.ray_z-a*self.encoder.ray_x-b*self.encoder.ray_y).numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(den < 0, -1.7/den, np.nan).astype(np.float32)

    def test_scale_recovery_and_input_unchanged(self):
        depth = self.floor()*1.25
        before = depth.copy()
        fit = fit_ground(depth, self.camera)
        self.assertEqual(fit.status, "ACCEPTED")
        self.assertAlmostEqual(fit.scale, .8, places=5)
        np.testing.assert_array_equal(depth, before)

    def test_sloped_plane_and_vertical_height(self):
        depth = self.floor(.12, -.06)*1.25
        fit = fit_ground(depth, self.camera)
        np.testing.assert_allclose(fit.plane, (.12, -.06, -2.125), atol=1e-5)
        corrected, plane = arm_inputs(depth, fit, "scale_and_ground")
        self.assertFalse(self.encoder.encode(corrected, ground_plane=plane).alerts().any())

    def test_invalid_fit_keeps_dependent_arms_unknown(self):
        depth = np.full((240,320), np.nan, np.float32)
        fit = fit_ground(depth, self.camera)
        self.assertEqual(fit.status, "UNKNOWN")
        for arm in ("scale_only", "ground_only", "scale_and_ground"):
            self.assertIsNone(arm_inputs(depth, fit, arm))
        self.assertIs(arm_inputs(depth, fit, "raw")[0], depth)

    def test_horizontal_ground_default_parity(self):
        depth = np.full((240,320), 2., np.float32)
        raw = self.encoder.encode(depth)
        relative = self.encoder.encode(depth, ground_plane=(0.,0.,-1.7))
        np.testing.assert_array_equal(raw.region_distance_m, relative.region_distance_m)
        self.assertEqual(raw.region_state, relative.region_state)


if __name__ == "__main__":
    unittest.main()
