#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import numpy as np
from onnx import TensorProto, helper, numpy_helper

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from scripts.research.hftf.deployment.depthart.generate_dav2_selective_w8a16_overrides_a5_r0 import (
    select_static_linear_weights,
    symmetric_int8_encoding,
)


class SelectiveW8A16OverrideTest(unittest.TestCase):
    def test_symmetric_axis_one_encoding(self) -> None:
        weight = np.array([[-127.0, 0.0], [127.0, 2.54]], dtype=np.float32)
        encoding = symmetric_int8_encoding("w", weight)
        self.assertEqual(encoding["axis"], 1)
        self.assertEqual(encoding["output_dtype"], "int8")
        self.assertAlmostEqual(encoding["y_scale"][0], 1.0)
        self.assertAlmostEqual(encoding["y_scale"][1], 0.02, places=6)
        self.assertEqual(encoding["y_zero_point"], [0, 0])

    def test_selector_rejects_partial_graph(self) -> None:
        weight = numpy_helper.from_array(np.ones((2, 2), dtype=np.float32), name="w")
        node = helper.make_node(
            "MatMul",
            ["x", "w"],
            ["y"],
            name="/blocks.0/attn/qkv/MatMul",
        )
        graph = helper.make_graph(
            [node],
            "partial",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
            [weight],
        )
        with self.assertRaisesRegex(ValueError, "exactly 48"):
            select_static_linear_weights(helper.make_model(graph))


if __name__ == "__main__":
    unittest.main()

