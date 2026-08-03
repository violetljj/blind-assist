#!/usr/bin/env python3

import unittest

import numpy as np
from export_depth_anything_v2_metric_onnx import (
    make_validation_input,
    validate_input_shape,
)


class ExportDepthAnythingV2MetricOnnxTest(unittest.TestCase):
    def test_validation_input_is_deterministic_normalized_nchw(self) -> None:
        first = make_validation_input(392, 518)
        second = make_validation_input(392, 518)
        self.assertEqual(first.shape, (1, 3, 392, 518))
        self.assertEqual(first.dtype, np.float32)
        np.testing.assert_array_equal(first, second)

    def test_shape_requires_patch_multiple(self) -> None:
        validate_input_shape(518, 686)
        with self.assertRaises(ValueError):
            validate_input_shape(518, 685)

    def test_shape_requires_positive_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            validate_input_shape(0, 518)


if __name__ == "__main__":
    unittest.main()
