import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx

from .prepare_depthart_selective_scan_canary import SHAPES, cases, make_model, reference


class PrepareDepthArtSelectiveScanCanaryTest(unittest.TestCase):
    def test_model_contract(self) -> None:
        model = make_model()
        onnx.checker.check_model(model)
        self.assertEqual(len(model.graph.node), 1)
        self.assertEqual(model.graph.node[0].domain, "com.depthart")
        self.assertEqual(model.graph.node[0].op_type, "SelectiveScan")

    def test_cases_are_deterministic_and_finite(self) -> None:
        first = cases(20260808)
        second = cases(20260808)
        self.assertEqual(list(first), ["nominal", "accumulation", "softplus_extremes"])
        for name in first:
            for tensor_name, shape in SHAPES.items():
                self.assertEqual(first[name][tensor_name].shape, shape)
                np.testing.assert_array_equal(first[name][tensor_name], second[name][tensor_name])
            expected = reference(first[name])
            self.assertEqual(expected.shape, SHAPES["u"])
            self.assertTrue(np.isfinite(expected).all())


if __name__ == "__main__":
    unittest.main()
