#!/usr/bin/env python3

import unittest

import numpy as np

from scripts.research.hftf.deployment.depthart.prepare_depth_anything_v2_qairt_canary import (
    INPUT_SHAPE,
    INPUT_ZERO_POINT,
    quantize_input,
)


class PrepareDepthAnythingV2QairtCanaryTest(unittest.TestCase):
    def test_zero_maps_to_published_zero_point(self) -> None:
        values = np.zeros(INPUT_SHAPE, dtype=np.float32)
        quantized = quantize_input(values)
        self.assertEqual(quantized.dtype, np.uint16)
        self.assertTrue(np.all(quantized == INPUT_ZERO_POINT))

    def test_shape_is_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            quantize_input(np.zeros((1, 1, 1, 3), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()

