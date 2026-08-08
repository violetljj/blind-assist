import unittest

import onnx
from onnx import TensorProto, helper

from .rewrite_depthart_layernorm_custom_onnx import rewrite_model


class RewriteDepthArtLayerNormCustomOnnxTest(unittest.TestCase):
    def test_rewrites_frozen_last_axis_contract(self) -> None:
        node = helper.make_node("LayerNormalization", ["x", "w", "b"], ["y"], name="ln", axis=-1, epsilon=1e-5)
        model = helper.make_model(helper.make_graph(
            [node], "g",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8, 128]),
             helper.make_tensor_value_info("w", TensorProto.FLOAT, [128]),
             helper.make_tensor_value_info("b", TensorProto.FLOAT, [128])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 128])],
        ), opset_imports=[helper.make_opsetid("", 17)])
        rewritten, records = rewrite_model(model)
        self.assertEqual(len(records), 1)
        result = rewritten.graph.node[0]
        self.assertEqual((result.domain, result.op_type), ("com.depthart", "DepthArtLayerNorm"))
        self.assertEqual(len(result.output), 1)
        self.assertAlmostEqual(helper.get_attribute_value(result.attribute[0]), 1e-5, places=10)

    def test_rejects_non_last_axis(self) -> None:
        node = helper.make_node("LayerNormalization", ["x", "w", "b"], ["y"], axis=1)
        model = helper.make_model(helper.make_graph([node], "g", [], []), opset_imports=[helper.make_opsetid("", 17)])
        with self.assertRaisesRegex(ValueError, "unsupported LayerNormalization contract"):
            rewrite_model(model)


if __name__ == "__main__":
    unittest.main()
