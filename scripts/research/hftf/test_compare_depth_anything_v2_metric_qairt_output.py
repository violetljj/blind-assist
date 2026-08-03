#!/usr/bin/env python3

import unittest

import numpy as np
from compare_depth_anything_v2_metric_qairt_output import compare


class CompareDepthAnythingV2MetricQairtOutputTest(unittest.TestCase):
    def test_compare_reports_metric_and_relative_difference(self) -> None:
        reference = np.asarray([[[1.0, 2.0]]], dtype=np.float32)
        candidate = np.asarray([[[1.1, 1.8]]], dtype=np.float32)
        result = compare(reference, candidate)
        self.assertAlmostEqual(result["mean_abs_difference_m"], 0.15, places=6)
        self.assertAlmostEqual(
            result["mean_relative_abs_difference"], 0.10, places=6
        )

    def test_compare_rejects_shape_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            compare(np.ones((1, 2)), np.ones((2, 1)))


if __name__ == "__main__":
    unittest.main()
