import unittest

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from .rewrite_depthart_layernorm_rank4_onnx import rewrite_model


class RewriteDepthArtLayerNormRank4OnnxTest(unittest.TestCase):
    def test_rank4_wrapper_preserves_output(self) -> None:
        rng = np.random.default_rng(31)
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8, 128])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 128])
        weight = numpy_helper.from_array(rng.normal(size=128).astype(np.float32), "weight")
        bias = numpy_helper.from_array(rng.normal(size=128).astype(np.float32), "bias")
        node = helper.make_node(
            "LayerNormalization", ["x", "weight", "bias"], ["y"], axis=-1, epsilon=1e-5
        )
        graph = helper.make_graph([node], "layernorm", [x], [y], [weight, bias])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
        rewritten, receipt = rewrite_model(model)
        values = {"x": rng.normal(size=(1, 8, 128)).astype(np.float32)}
        expected = ort.InferenceSession(model.SerializeToString()).run(None, values)[0]
        actual = ort.InferenceSession(rewritten.SerializeToString()).run(None, values)[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(receipt["layernorm_nodes_wrapped"], 1)
        self.assertEqual([node.op_type for node in rewritten.graph.node], ["Unsqueeze", "LayerNormalization", "Squeeze"])

    def test_rejects_non_last_axis_rank3_layernorm(self) -> None:
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8, 128])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 128])
        node = helper.make_node("LayerNormalization", ["x"], ["y"], axis=1)
        model = helper.make_model(
            helper.make_graph([node], "layernorm", [x], [y]),
            opset_imports=[helper.make_opsetid("", 17)],
            ir_version=10,
        )
        with self.assertRaisesRegex(ValueError, "expected rank-3 last-axis"):
            rewrite_model(model)


if __name__ == "__main__":
    unittest.main()
