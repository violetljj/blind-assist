import unittest

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from .lower_depthart_layernorm_onnx import lower_model


class LowerDepthArtLayerNormOnnxTest(unittest.TestCase):
    def test_lowered_graph_matches_layernorm(self) -> None:
        rng = np.random.default_rng(37)
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 8, 128])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 8, 128])
        weight = numpy_helper.from_array(rng.normal(size=128).astype(np.float32), "weight")
        bias = numpy_helper.from_array(rng.normal(size=128).astype(np.float32), "bias")
        node = helper.make_node(
            "LayerNormalization", ["x", "weight", "bias"], ["y"], axis=-1, epsilon=1e-5
        )
        model = helper.make_model(
            helper.make_graph([node], "layernorm", [x], [y], [weight, bias]),
            opset_imports=[helper.make_opsetid("", 17)],
            ir_version=10,
        )
        lowered, receipt = lower_model(model)
        values = {"x": rng.normal(size=(2, 8, 128)).astype(np.float32)}
        expected = ort.InferenceSession(model.SerializeToString()).run(None, values)[0]
        actual = ort.InferenceSession(lowered.SerializeToString()).run(None, values)[0]
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
        self.assertEqual(receipt["layernorm_nodes_lowered"], 1)
        self.assertEqual(receipt["remaining_layernorm_nodes"], 0)

    def test_rejects_non_last_axis(self) -> None:
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8, 128])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 128])
        node = helper.make_node("LayerNormalization", ["x", "w", "b"], ["y"], axis=1)
        model = helper.make_model(
            helper.make_graph([node], "layernorm", [x], [y]),
            opset_imports=[helper.make_opsetid("", 17)],
            ir_version=10,
        )
        with self.assertRaisesRegex(ValueError, "unsupported LayerNormalization contract"):
            lower_model(model)


if __name__ == "__main__":
    unittest.main()
