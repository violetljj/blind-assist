import math
import unittest
import numpy as np
from foundation_geometry_infer import disparity_to_depth


class DepthTests(unittest.TestCase):
    def test_original_pixel_units_and_visibility(self):
        rig = dict(width=640, height=360, hfov_deg=70, baseline_m=.1)
        fx = 320/math.tan(math.radians(35))
        disp = np.full((360, 640), fx*.1/2, np.float32)
        raw, clipped = disparity_to_depth(disp, rig)
        np.testing.assert_allclose(raw, 2, rtol=1e-6)
        self.assertTrue(np.isnan(clipped[:, :23]).all())
        np.testing.assert_allclose(clipped[:, 23:], 2, rtol=1e-6)

    def test_raw_far_and_invalid_values(self):
        rig = dict(width=640, height=360, hfov_deg=70, baseline_m=.1)
        fb = 32/math.tan(math.radians(35))
        disp = np.full((360, 640), fb/8, np.float32)
        disp[0, 0:4] = [0, -1, np.nan, np.inf]
        raw, clipped = disparity_to_depth(disp, rig)
        self.assertTrue(np.isnan(raw[0, :4]).all())
        np.testing.assert_allclose(raw[1:], 8, rtol=1e-6)
        self.assertTrue(np.isnan(clipped).all())


if __name__ == '__main__': unittest.main()
