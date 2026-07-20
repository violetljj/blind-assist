import unittest

import numpy as np
from PIL import Image, ImageDraw

import build_public_video_path_intrusion_counterfactuals as counterfactuals


class PathIntrusionCounterfactualTest(unittest.TestCase):
    def test_composition_preserves_pixels_outside_intervention_mask(self) -> None:
        parent = Image.new("RGB", (640, 360), (30, 60, 90))
        asset = Image.new("RGBA", (20, 50), (0, 0, 0, 0))
        ImageDraw.Draw(asset).polygon([(10, 0), (3, 45), (17, 45)], fill=(255, 100, 0, 255))
        edited, mask = counterfactuals.compose(parent, asset, [(320, 280, 50)])
        passed, outside = counterfactuals.unchanged_outside_mask(parent, edited, mask)
        self.assertTrue(passed)
        self.assertEqual(0, outside)
        self.assertGreater(np.count_nonzero(np.any(np.asarray(parent) != np.asarray(edited), axis=2)), 0)


if __name__ == "__main__":
    unittest.main()
