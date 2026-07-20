#!/usr/bin/env python3

import unittest

import numpy as np

from prepare_synthetic_chroma_asset import prepare


class ChromaAssetTest(unittest.TestCase):
    def test_extracts_largest_non_magenta_component_with_binary_alpha(self) -> None:
        image = np.zeros((40, 50, 3), dtype=np.uint8)
        image[:] = [230, 20, 230]
        image[10:35, 15:35] = [20, 120, 240]
        image[1:3, 1:3] = [0, 0, 0]
        rgba, bounds = prepare(image, chroma_margin=100, crop_margin=2)
        self.assertEqual(sorted(np.unique(rgba[..., 3]).tolist()), [0, 255])
        self.assertEqual(bounds["foreground_pixel_count"], 500)
        self.assertEqual(rgba.shape[2], 4)


if __name__ == "__main__":
    unittest.main()
